"""apply 階段：把 selection JSON 真正寫回文章與圖片。

只讀選擇檔與當下的 Markdown，不呼叫任何 LLM。套用前會重新解析文章並比對
``content_hash``，正文被改過就拒絕，避免拿失效的圖片位置去替換。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

from blogseo.errors import ApplyError, SelectionLoadError
from blogseo.image.renamer import rename_files
from blogseo.markdown.parser import (
    ImageTarget,
    ParsedArticle,
    _strip_path_decorations,
    parse_article,
)
from blogseo.markdown.updater import (
    apply_spans,
    join_source_path,
    rewrite_image_markup,
    splice_body,
)
from blogseo.schemas.selection import (
    SELECTION_SCHEMA_VERSION,
    Selection,
    SelectionImage,
)
from blogseo.seo.selector import find_conflicts

if TYPE_CHECKING:
    from blogseo.obsidian.gitless_sync import GitlessSyncOutcome

#: 能被 apply 讀取的最低 selection 結構版本。
MIN_SELECTION_SCHEMA: Final[tuple[int, int]] = (1, 0)


@dataclass(frozen=True)
class MarkupChange:
    """正文中一次圖片語法的替換。"""

    key: str
    line: int
    old: str
    new: str


@dataclass(frozen=True)
class RenameChange:
    """磁碟上一次圖片改名。"""

    key: str
    source: Path
    dest: Path


@dataclass
class ApplyPlan:
    """套用前的完整計畫，dry-run 與實際執行共用同一份。"""

    article_path: Path
    original_text: str
    new_text: str
    markup_changes: list[MarkupChange] = field(default_factory=list)
    renames: list[RenameChange] = field(default_factory=list)
    hash_mismatch: bool = False

    @property
    def is_empty(self) -> bool:
        """沒有任何要寫入的變更。"""
        return not self.markup_changes and not self.renames


def _parse_schema_version(raw: str) -> tuple[int, int]:
    """把 ``"1.1"`` 這種版本字串拆成數字。

    Args:
        raw: 版本字串。

    Returns:
        (主版本, 次版本)。

    Raises:
        SelectionLoadError: 格式無法解析。
    """
    pieces = raw.split(".")
    try:
        return int(pieces[0]), int(pieces[1]) if len(pieces) > 1 else 0
    except (ValueError, IndexError) as exc:
        raise SelectionLoadError(f"無法解析 schema_version：{raw!r}") from exc


def _backfill_legacy_selection(payload: dict[str, Any], version: tuple[int, int]) -> None:
    """1.0 的 ``source_model`` 同時代表檔名與 alt 的來源。

    Args:
        payload: 原始 JSON，會就地修改。
        version: 這份檔案的結構版本。
    """
    if version >= (1, 1):
        return
    images = payload.get("images")
    if not isinstance(images, dict):
        return
    for item in images.values():
        if not isinstance(item, dict):
            continue
        source_model = item.get("source_model")
        if isinstance(source_model, str):
            item.setdefault("filename_model", source_model)
            item.setdefault("alt_model", source_model)


def load_selection(path: Path) -> Selection:
    """讀取並驗證 review 產生的選擇檔。

    Args:
        path: selection JSON 的路徑。

    Returns:
        驗證後的選擇結果。

    Raises:
        SelectionLoadError: 檔案不存在、不是合法 JSON、版本過舊或結構不符。
    """
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise SelectionLoadError(f"找不到選擇檔：{path}") from exc
    except OSError as exc:
        raise SelectionLoadError(f"無法讀取 {path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise SelectionLoadError(f"{path} 不是合法的 JSON：{exc}") from exc

    if not isinstance(payload, dict):
        raise SelectionLoadError(f"{path} 的頂層必須是物件")

    raw_version = payload.get("schema_version")
    if not isinstance(raw_version, str):
        raise SelectionLoadError(f"{path} 缺少 schema_version，可能不是本工具的產物")

    version = _parse_schema_version(raw_version)
    expected = ".".join(str(part) for part in MIN_SELECTION_SCHEMA)
    if version[0] != MIN_SELECTION_SCHEMA[0] or version < MIN_SELECTION_SCHEMA:
        raise SelectionLoadError(
            f"{path} 的結構版本是 {raw_version}，apply 需要 {expected} 以上，"
            "請重新執行 review"
        )
    current = ".".join(str(part) for part in _parse_schema_version(SELECTION_SCHEMA_VERSION))
    if version[0] > _parse_schema_version(SELECTION_SCHEMA_VERSION)[0]:
        raise SelectionLoadError(
            f"{path} 的結構版本是 {raw_version}，比這個版本的工具支援的 {current} 還新"
        )

    _backfill_legacy_selection(payload, version)

    try:
        return Selection.model_validate(payload)
    except ValidationError as exc:
        raise SelectionLoadError(
            f"{path} 的結構不符合預期：{exc.error_count()} 處錯誤"
        ) from exc


def _match_target(article: ParsedArticle, key: str, item: SelectionImage) -> ImageTarget:
    """在當下的文章裡找到選擇檔對應的那張圖。

    Args:
        article: 重新解析後的文章。
        key: 選擇檔裡的圖片鍵。
        item: 該張圖的決定。

    Returns:
        對應的圖片 target。

    Raises:
        ApplyError: 找不到、檔案不在，或無法改名。
    """
    by_key = {target.key: target for target in article.targets}
    target = by_key.get(key)
    if target is None:
        by_source = {target.source: target for target in article.targets}
        target = by_source.get(item.source)
    if target is None:
        raise ApplyError(f"選擇檔中的圖片 {key} 在文章裡找不到對應引用")
    if target.resolved_path is None or not target.exists:
        reason = target.skipped_reason or "本地找不到這個檔案"
        raise ApplyError(f"無法套用 {key}：{reason}")
    return target


def _markup_path(reference_source: str, new_filename: str | None) -> str | None:
    """決定正文裡要寫的新路徑；檔名沒變時回傳 ``None``。"""
    if new_filename is None:
        return None
    stripped = _strip_path_decorations(reference_source)
    if Path(stripped).name == new_filename:
        return None
    return join_source_path(reference_source, new_filename)


def build_plan(selection: Selection, *, force: bool = False) -> ApplyPlan:
    """讀取文章、核對雜湊，組出要寫入的變更。

    Args:
        selection: 選擇結果。
        force: 正文雜湊不一致時是否繼續。以當下重新解析的位置替換，
            而不是分析當時記下的位置。

    Returns:
        套用計畫。

    Raises:
        ApplyError: 文章找不到、雜湊不一致且未指定 force、圖片對不上或檔名衝突。
        MarkdownParseError: 文章無法解析。
    """
    article_path = Path(selection.article_path)
    try:
        original_text = article_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ApplyError(f"找不到文章：{article_path}") from exc
    except OSError as exc:
        raise ApplyError(f"無法讀取文章 {article_path}：{exc}") from exc

    article = parse_article(article_path)

    hash_mismatch = bool(
        selection.content_hash and article.content_hash != selection.content_hash
    )
    if hash_mismatch and not force:
        raise ApplyError(
            "原始文章在分析之後被編輯過，圖片位置可能已經失效。"
            "重新執行 analyze 與 review，或加上 --force 用當下的內容套用"
        )
    if not selection.content_hash and not force:
        raise ApplyError(
            "這份選擇檔沒有記錄正文雜湊，無法確認文章有沒有被改過。"
            "加上 --force 略過這項檢查"
        )

    live_images: dict[str, SelectionImage] = {}
    replacements: list[tuple[int, int, str]] = []
    markup_changes: list[MarkupChange] = []
    renames: list[RenameChange] = []

    for key, item in selection.images.items():
        target = _match_target(article, key, item)
        assert target.resolved_path is not None
        live_images[key] = item.model_copy(
            update={"resolved_path": str(target.resolved_path)}
        )

        dest: Path | None = None
        if item.new_filename is not None:
            dest = target.resolved_path.with_name(item.new_filename)
            if dest != target.resolved_path:
                renames.append(
                    RenameChange(key=key, source=target.resolved_path, dest=dest)
                )

        for reference in target.references:
            new_path = _markup_path(reference.source, item.new_filename)
            if new_path is None and item.alt is None:
                continue
            current = article.content[reference.start : reference.end]
            if current != reference.raw:
                raise ApplyError(
                    f"{key} 在第 {reference.line} 行的內容與解析結果不一致，已中止"
                )
            new_markup = rewrite_image_markup(
                reference.raw,
                reference.syntax,
                new_path=new_path,
                new_alt=item.alt,
            )
            if new_markup == reference.raw:
                continue
            replacements.append((reference.start, reference.end, new_markup))
            markup_changes.append(
                MarkupChange(
                    key=key, line=reference.line, old=reference.raw, new=new_markup
                )
            )

    conflicts = find_conflicts(live_images)
    if conflicts:
        details = "；".join(
            f"{conflict.filename}（{conflict.reason}：{'、'.join(conflict.sources)}）"
            for conflict in conflicts
        )
        raise ApplyError(f"檔名衝突，未套用任何變更：{details}")

    new_body = apply_spans(article.content, replacements) if replacements else article.content
    new_text = (
        splice_body(original_text, article.content, new_body)
        if replacements
        else original_text
    )

    return ApplyPlan(
        article_path=article_path,
        original_text=original_text,
        new_text=new_text,
        markup_changes=markup_changes,
        renames=renames,
        hash_mismatch=hash_mismatch,
    )


def execute_plan(plan: ApplyPlan, *, backup: bool = False) -> GitlessSyncOutcome | None:
    """依計畫寫入文章並改名圖片。

    先改圖片檔名，再寫 Markdown；若寫入失敗會嘗試把檔名改回去。
    寫入成功後，若文章在 Obsidian 庫裡，會同步更新 Gitless Sync 清單。

    Args:
        plan: :func:`build_plan` 的結果。
        backup: 寫入前是否把原文複製成 ``.bak``。

    Returns:
        Gitless Sync 清單的更新結果；沒有庫或沒有變更時為 ``None``。

    Raises:
        ApplyError: 寫入或改名失敗。
    """
    if plan.is_empty:
        return None

    pairs = [(item.source, item.dest) for item in plan.renames]
    renamed = False
    try:
        if pairs:
            rename_files(pairs)
            renamed = True
        if backup and plan.new_text != plan.original_text:
            backup_path = plan.article_path.with_name(plan.article_path.name + ".bak")
            shutil.copy2(plan.article_path, backup_path)
        if plan.new_text != plan.original_text:
            _atomic_write(plan.article_path, plan.new_text)
    except OSError as exc:
        if renamed:
            try:
                rename_files([(item.dest, item.source) for item in reversed(plan.renames)])
            except OSError:
                raise ApplyError(
                    f"寫入文章失敗，且圖片檔名無法還原：{exc}"
                ) from exc
        raise ApplyError(f"套用失敗：{exc}") from exc

    from blogseo.obsidian.gitless_sync import patch_gitless_sync_metadata

    return patch_gitless_sync_metadata(plan)


def _atomic_write(path: Path, text: str) -> None:
    """先寫到同目錄暫存檔再取代，避免寫到一半留下半份檔案。

    Args:
        path: 目標路徑。
        text: 完整檔案內容。
    """
    temp = path.with_name(f"{path.stem}.blogseo-new{path.suffix}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temp.replace(path)
