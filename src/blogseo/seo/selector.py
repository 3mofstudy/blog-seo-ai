"""review 階段的挑選邏輯。

這個模組完全不呼叫任何 LLM，也不修改使用者的檔案：它只讀 analyze 產生的
JSON、整理出可挑選的候選、檢查衝突，然後把使用者的決定組成 selection 物件。

互動式的輸出入留在 :mod:`blogseo.cli`，這裡一律是純函式，方便單獨測試。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from blogseo import __version__
from blogseo.errors import AnalysisLoadError, MarkdownParseError, SelectionError
from blogseo.markdown.parser import parse_article
from blogseo.schemas.result import AnalysisResult, ImageEntry
from blogseo.schemas.selection import (
    SELECTION_SCHEMA_VERSION,
    Selection,
    SelectionImage,
)

#: 能被 review 讀取的最低 analysis 結構版本。
MIN_ANALYSIS_SCHEMA: Final[tuple[int, int]] = (1, 2)

#: 從這個版本起 ``article_info`` 才有 ``content_hash``。更舊的檔案仍可挑選，
#: 但無從得知文章是否已被編輯，因此雜湊會留空並在流程中提醒。
CONTENT_HASH_SCHEMA: Final[tuple[int, int]] = (1, 3)

#: 從這個版本起 metadata 才有 ``alt_max_chars``。
ALT_MAX_SCHEMA: Final[tuple[int, int]] = (1, 4)

#: 1.4 之前寫死的 alt 字元上限，用來回填舊檔案，讓當時的長度顯示對得上。
LEGACY_ALT_MAX_CHARS: Final[int] = 125


@dataclass(frozen=True)
class KeywordCandidate:
    """某個模型提出的一組關鍵字。"""

    alias: str
    model_id: str
    keywords: list[str]


@dataclass(frozen=True)
class SummaryCandidate:
    """某個模型提出的其中一則摘要。"""

    alias: str
    model_id: str
    variant: int
    text: str

    @property
    def length(self) -> int:
        """摘要字元數，以 Python ``len()`` 計算。"""
        return len(self.text)


@dataclass(frozen=True)
class ImageCandidate:
    """某個模型對某張圖片提出的建議。

    檔名與 alt 一起產生，但挑選時是兩個獨立的決定：某個模型的檔名取得漂亮，
    alt 卻可能又臭又長。
    """

    alias: str
    model_id: str
    filename: str
    alt: str

    @property
    def alt_length(self) -> int:
        """alt 的字元數，以 Python ``len()`` 計算。

        直接從文字算，而不是讀 JSON 裡的 ``alt_length``，這樣舊版本的檔案
        （還沒有那個欄位）也能顯示正確的長度。
        """
        return len(self.alt)


@dataclass(frozen=True)
class Conflict:
    """兩個以上的決定指向同一個檔案路徑。"""

    directory: str
    filename: str
    sources: list[str]
    reason: str


def _parse_schema_version(raw: str) -> tuple[int, int]:
    """把 ``"1.3"`` 這種版本字串拆成數字。

    Args:
        raw: 版本字串。

    Returns:
        (主版本, 次版本)。

    Raises:
        AnalysisLoadError: 格式無法解析。
    """
    pieces = raw.split(".")
    try:
        return int(pieces[0]), int(pieces[1]) if len(pieces) > 1 else 0
    except (ValueError, IndexError) as exc:
        raise AnalysisLoadError(f"無法解析 schema_version：{raw!r}") from exc


def _backfill_legacy_fields(payload: dict[str, Any], version: tuple[int, int]) -> None:
    """替舊版本的 JSON 補上後來才加的欄位。

    讓早期跑出來的分析結果不必重跑一次 analyze 就能繼續挑選。

    Args:
        payload: 原始 JSON 內容，會就地修改。
        version: 這份檔案的結構版本。
    """
    if version < CONTENT_HASH_SCHEMA:
        info = payload.get("article_info")
        if isinstance(info, dict):
            info.setdefault("content_hash", "")

    if version < ALT_MAX_SCHEMA:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.setdefault("alt_max_chars", LEGACY_ALT_MAX_CHARS)


def load_analysis(path: Path) -> AnalysisResult:
    """讀取並驗證 analyze 產生的 JSON。

    這份檔案是使用者可以手動編輯的外部輸入，所以一律走 Pydantic 驗證，
    版本不相容時給出可行動的錯誤訊息而不是硬吃。

    Args:
        path: analysis JSON 的路徑。

    Returns:
        驗證後的分析結果。

    Raises:
        AnalysisLoadError: 檔案不存在、不是合法 JSON、版本過舊或結構不符。
    """
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise AnalysisLoadError(f"找不到分析結果檔案：{path}") from exc
    except OSError as exc:
        raise AnalysisLoadError(f"無法讀取 {path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisLoadError(f"{path} 不是合法的 JSON：{exc}") from exc

    if not isinstance(payload, dict):
        raise AnalysisLoadError(f"{path} 的頂層必須是物件")

    raw_version = payload.get("schema_version")
    if not isinstance(raw_version, str):
        raise AnalysisLoadError(f"{path} 缺少 schema_version，可能不是本工具的產物")

    version = _parse_schema_version(raw_version)
    if version[0] != MIN_ANALYSIS_SCHEMA[0] or version < MIN_ANALYSIS_SCHEMA:
        expected = ".".join(str(part) for part in MIN_ANALYSIS_SCHEMA)
        raise AnalysisLoadError(
            f"{path} 的結構版本是 {raw_version}，review 需要 {expected} 以上，"
            "請重新執行 analyze 產生新的分析結果"
        )

    _backfill_legacy_fields(payload, version)

    try:
        return AnalysisResult.model_validate(payload)
    except ValidationError as exc:
        raise AnalysisLoadError(f"{path} 的結構不符合預期：{exc.error_count()} 處錯誤") from exc


def keyword_candidates(result: AnalysisResult) -> list[KeywordCandidate]:
    """列出所有可挑選的關鍵字組。

    Args:
        result: 分析結果。

    Returns:
        依模型別名順序排列的候選；失敗或沒產關鍵字的模型不會出現。
    """
    candidates: list[KeywordCandidate] = []
    for alias, entry in result.keyword_summary.items():
        if entry.error is not None or entry.result is None:
            continue
        if not entry.result.keywords:
            continue
        candidates.append(
            KeywordCandidate(
                alias=alias,
                model_id=entry.model_id,
                keywords=list(entry.result.keywords),
            )
        )
    return candidates


def summary_candidates(result: AnalysisResult) -> list[SummaryCandidate]:
    """把所有模型的所有摘要版本攤平成一份候選清單。

    Args:
        result: 分析結果。

    Returns:
        依模型別名、版本順序排列的候選；失敗的模型不會出現。
    """
    candidates: list[SummaryCandidate] = []
    for alias, entry in result.keyword_summary.items():
        if entry.error is not None or entry.result is None:
            continue
        for variant, text in enumerate(entry.result.summaries, start=1):
            candidates.append(
                SummaryCandidate(
                    alias=alias,
                    model_id=entry.model_id,
                    variant=variant,
                    text=text,
                )
            )
    return candidates


def image_candidates(entry: ImageEntry) -> list[ImageCandidate]:
    """列出某張圖片可挑選的建議。

    Args:
        entry: 圖片條目。

    Returns:
        依模型別名順序排列的候選；失敗的格子不會出現。
    """
    candidates: list[ImageCandidate] = []
    for alias, analysis in entry.models.items():
        if analysis.error is not None or analysis.result is None:
            continue
        if not analysis.final_filename:
            continue
        candidates.append(
            ImageCandidate(
                alias=alias,
                model_id=analysis.model_id,
                filename=analysis.final_filename,
                alt=analysis.result.alt,
            )
        )
    return candidates


def selectable_images(result: AnalysisResult) -> dict[str, ImageEntry]:
    """挑出真正可以套用的圖片。

    被略過的圖片（外部連結、站台絕對路徑、本地找不到）與所有模型都失敗的
    圖片都不該進入挑選流程。

    Args:
        result: 分析結果。

    Returns:
        依原順序排列、可挑選的圖片條目。
    """
    return {
        key: entry
        for key, entry in result.images.items()
        if entry.skipped_reason is None
        and entry.resolved_path is not None
        and image_candidates(entry)
    }


def default_alias(result: AnalysisResult) -> str | None:
    """猜一個合理的預設模型別名。

    以有成功產出文章分析的模型優先，其次是圖片分析有成功的模型。

    Args:
        result: 分析結果。

    Returns:
        別名；沒有任何模型成功時為 ``None``。
    """
    for alias, entry in result.keyword_summary.items():
        if entry.error is None and entry.result is not None:
            return alias
    for entry in result.images.values():
        for candidate in image_candidates(entry):
            return candidate.alias
    return None


def find_conflicts(images: dict[str, SelectionImage]) -> list[Conflict]:
    """檢查最終檔名是否互相衝突或撞到既有檔案。

    比對時把檔名轉小寫，因為 Windows 與 macOS 的檔案系統預設不分大小寫，
    ``Chart.png`` 與 ``chart.png`` 會互相覆蓋。只改 alt 不改名的條目不參與比對。

    Args:
        images: 每張圖片的最終決定。

    Returns:
        所有衝突；沒有衝突時為空清單。
    """
    renames = {
        key: item for key, item in images.items() if item.new_filename is not None
    }
    conflicts: list[Conflict] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    #: 會被改名讓出來的舊路徑，撞到這些不算衝突。
    vacated: set[str] = set()

    for key, item in renames.items():
        assert item.new_filename is not None
        current = Path(item.resolved_path)
        target = current.with_name(item.new_filename)
        grouped.setdefault(
            (str(target.parent), item.new_filename.casefold()), []
        ).append(key)
        if target != current:
            vacated.add(str(current).casefold())

    for (directory, filename), sources in grouped.items():
        if len(sources) > 1:
            conflicts.append(
                Conflict(
                    directory=directory,
                    filename=filename,
                    sources=sorted(sources),
                    reason="多張圖片選到相同檔名",
                )
            )

    for key, item in renames.items():
        assert item.new_filename is not None
        current = Path(item.resolved_path)
        target = current.with_name(item.new_filename)
        if target == current:
            continue
        if not target.exists():
            continue
        if str(target).casefold() in vacated:
            continue
        conflicts.append(
            Conflict(
                directory=str(target.parent),
                filename=item.new_filename,
                sources=[key],
                reason="目標檔名已經存在，且不會被其他改名讓出來",
            )
        )

    return conflicts


def build_selection(
    result: AnalysisResult,
    *,
    source_analysis: Path,
    keywords: list[str],
    summary: str,
    images: dict[str, SelectionImage],
) -> Selection:
    """把各項決定組成可落地的 selection 物件。

    Args:
        result: 來源分析結果。
        source_analysis: analysis JSON 的路徑。
        keywords: 最終採用的關鍵字。
        summary: 最終採用的摘要。
        images: 每張圖片的最終決定。

    Returns:
        可直接寫成 JSON 的選擇結果。

    Raises:
        SelectionError: 三個項目全空，寫出來也沒有東西可套用。
    """
    if not keywords and not summary and not images:
        raise SelectionError("沒有挑選任何項目，不產生選擇檔")

    return Selection(
        schema_version=SELECTION_SCHEMA_VERSION,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        tool_version=__version__,
        source_analysis=str(source_analysis.resolve()),
        article_path=result.article_info.path,
        content_hash=result.article_info.content_hash,
        keywords=keywords,
        summary=summary,
        images=images,
    )


def article_changed(result: AnalysisResult) -> bool | None:
    """判斷文章在 analyze 之後是否被編輯過。

    只比對正文，front matter 的變動不影響圖片引用的字元位置。

    Args:
        result: 分析結果。

    Returns:
        內容不一致為 ``True``，一致為 ``False``；無從判斷為 ``None``，
        可能是文章已不存在，或分析結果是沒有 ``content_hash`` 的舊版本。
    """
    if not result.article_info.content_hash:
        return None
    try:
        article = parse_article(Path(result.article_info.path))
    except MarkdownParseError:
        return None
    return article.content_hash != result.article_info.content_hash
