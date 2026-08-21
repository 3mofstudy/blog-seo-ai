"""Markdown 解析：拆出 front matter、擷取圖片引用、推算本地路徑。

設計重點：

* 圖片引用會連同**在正文中的字元位置**一起記錄，apply 階段才能做精準替換，
  不會像全域字串取代那樣誤傷同名文字或互為前綴的檔名。
* 程式碼區塊（圍籬與行內）內的圖片語法會被忽略，避免教學文章裡的範例被改到。
* 位置皆相對於「去掉 front matter 之後的正文」，回寫時把新正文接回原檔，
  既有的 front matter 原樣保留。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlsplit

import frontmatter

from blogseo.config import CONTEXT_RADIUS
from blogseo.errors import MarkdownParseError
from blogseo.image.renamer import IMAGE_EXTENSIONS

_FENCED_CODE = re.compile(
    r"^(?P<fence>```+|~~~+)[^\n]*\n.*?(?:^(?P=fence)[^\n]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)

_MARKDOWN_IMAGE = re.compile(
    r"!\[(?P<alt>(?:[^\[\]\\]|\\.)*)\]"
    r"\(\s*(?P<path><[^>\n]*>|[^\s()]+)"
    r"(?:\s+(?P<title>\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_HTML_IMAGE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE | re.DOTALL)
_HTML_ATTR = re.compile(
    r"""\b(?P<name>src|alt)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)

_HEADING = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*#*\s*$", re.MULTILINE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

#: 視為外部資源、不做本地處理的 URL scheme。
_REMOTE_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "data", "ftp"})


@dataclass(frozen=True)
class ImageReference:
    """正文中的一次圖片引用。

    Attributes:
        raw: 原始的完整語法字串。
        source: 語法中寫的路徑字串（尚未 URL 解碼）。
        alt: 原本的 alt 文字。
        start: 於正文中的起始字元位置。
        end: 於正文中的結束字元位置（不含）。
        line: 於正文中的行號，從 1 起算。
        syntax: ``markdown`` 或 ``html``。
    """

    raw: str
    source: str
    alt: str
    start: int
    end: int
    line: int
    syntax: str


@dataclass
class ImageTarget:
    """一張實體圖片，以及它在正文中的所有引用。

    同一張圖被引用多次時會合併成一個 target，避免重複呼叫模型。
    """

    key: str
    source: str
    resolved_path: Path | None
    exists: bool
    skipped_reason: str | None = None
    references: list[ImageReference] = field(default_factory=list)

    @property
    def extension(self) -> str:
        """原始副檔名，含點；無法判斷時為空字串。"""
        if self.resolved_path is not None:
            return self.resolved_path.suffix.lower()
        return Path(self.source).suffix.lower()

    @property
    def analyzable(self) -> bool:
        """是否應該送進模型分析。"""
        return self.exists and self.skipped_reason is None


@dataclass
class ParsedArticle:
    """解析完成的文章。

    Attributes:
        path: Markdown 檔案路徑。
        metadata: front matter 內容。
        content: 去掉 front matter 之後的正文。
        title: 文章標題，取自 front matter 或第一個 H1。
        language: 偵測到的主要語言（``zh`` 或 ``en``）。
        word_count: 正文字數。
        targets: 相異圖片清單。
    """

    path: Path
    metadata: dict[str, Any]
    content: str
    title: str | None
    language: str
    word_count: int
    targets: list[ImageTarget]

    @property
    def content_hash(self) -> str:
        """正文的 sha256，見 :func:`compute_content_hash`。"""
        return compute_content_hash(self.content)


def compute_content_hash(content: str) -> str:
    """計算正文的 sha256。

    只涵蓋去掉 front matter 之後的正文，因為 apply 只改正文裡的圖片語法，
    不改 front matter。

    Args:
        content: 去掉 front matter 之後的正文。

    Returns:
        小寫十六進位的雜湊字串。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _mask_code(content: str) -> str:
    """把程式碼區塊換成等長空白，保留換行以維持字元位置與行號。

    Args:
        content: 原始正文。

    Returns:
        與輸入等長、程式碼區域已被遮蔽的字串。
    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    masked = _FENCED_CODE.sub(blank, content)
    return _INLINE_CODE.sub(blank, masked)


def _strip_path_decorations(raw_path: str) -> str:
    """去掉角括號包裝並做 URL 解碼。

    Args:
        raw_path: 語法中擷取到的原始路徑字串。

    Returns:
        可直接拿去組路徑的字串。
    """
    candidate = raw_path.strip()
    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1]
    return unquote(candidate).strip()


def _classify_source(source: str) -> str | None:
    """判斷這個路徑是否需要略過。

    Args:
        source: 已解碼的路徑字串。

    Returns:
        需要略過時回傳原因字串，需要處理則回傳 ``None``。
    """
    if not source:
        return "路徑為空"
    scheme = urlsplit(source).scheme.lower()
    if scheme in _REMOTE_SCHEMES:
        return "外部連結，不做本地改名"
    if source.startswith("//"):
        return "外部連結，不做本地改名"
    if source.startswith("/"):
        return "站台絕對路徑，超出本工具處理範圍"
    if Path(source).suffix.lower() not in IMAGE_EXTENSIONS:
        return "副檔名不是支援的圖片格式"
    return None


def _resolve_path(markdown_path: Path, source: str) -> Path:
    """以 Markdown 檔案所在目錄為基準推算圖片絕對路徑。

    Args:
        markdown_path: Markdown 檔案路徑。
        source: 已解碼的相對路徑。

    Returns:
        絕對路徑（未保證存在）。
    """
    candidate = Path(source)
    if candidate.is_absolute():
        return candidate
    return (markdown_path.parent / candidate).resolve()


def _iter_markdown_images(masked: str, content: str) -> list[ImageReference]:
    """擷取標準 Markdown 圖片語法。

    Args:
        masked: 已遮蔽程式碼的正文，用來比對。
        content: 原始正文，用來取回真實字串。

    Returns:
        圖片引用清單。
    """
    references: list[ImageReference] = []
    for match in _MARKDOWN_IMAGE.finditer(masked):
        start, end = match.span()
        references.append(
            ImageReference(
                raw=content[start:end],
                source=match.group("path"),
                alt=match.group("alt") or "",
                start=start,
                end=end,
                line=content.count("\n", 0, start) + 1,
                syntax="markdown",
            )
        )
    return references


def _iter_html_images(masked: str, content: str) -> list[ImageReference]:
    """擷取 HTML ``<img>`` 標籤。

    Args:
        masked: 已遮蔽程式碼的正文，用來比對。
        content: 原始正文，用來取回真實字串。

    Returns:
        圖片引用清單；沒有 ``src`` 屬性的標籤會被忽略。
    """
    references: list[ImageReference] = []
    for match in _HTML_IMAGE.finditer(masked):
        attrs = {
            m.group("name").lower(): m.group("value")
            for m in _HTML_ATTR.finditer(match.group(0))
        }
        src = attrs.get("src")
        if not src:
            continue
        start, end = match.span()
        references.append(
            ImageReference(
                raw=content[start:end],
                source=src,
                alt=attrs.get("alt", ""),
                start=start,
                end=end,
                line=content.count("\n", 0, start) + 1,
                syntax="html",
            )
        )
    return references


def _group_targets(markdown_path: Path, references: list[ImageReference]) -> list[ImageTarget]:
    """把引用依實體圖片分組。

    Args:
        markdown_path: Markdown 檔案路徑，用來推算相對路徑。
        references: 所有圖片引用。

    Returns:
        依首次出現順序排列的圖片清單。
    """
    targets: dict[str, ImageTarget] = {}
    for reference in references:
        source = _strip_path_decorations(reference.source)
        skipped = _classify_source(source)

        if skipped is not None:
            key = source or reference.raw
            target = targets.get(key)
            if target is None:
                target = ImageTarget(
                    key=key,
                    source=source,
                    resolved_path=None,
                    exists=False,
                    skipped_reason=skipped,
                )
                targets[key] = target
            target.references.append(reference)
            continue

        resolved = _resolve_path(markdown_path, source)
        try:
            key = resolved.relative_to(markdown_path.parent.resolve()).as_posix()
        except ValueError:
            key = resolved.as_posix()

        target = targets.get(key)
        if target is None:
            exists = resolved.is_file()
            target = ImageTarget(
                key=key,
                source=source,
                resolved_path=resolved,
                exists=exists,
                skipped_reason=None if exists else "本地找不到這個檔案",
            )
            targets[key] = target
        target.references.append(reference)

    return list(targets.values())


def _detect_language(text: str) -> str:
    """以 CJK 字元比例粗略判斷語言。

    Args:
        text: 正文。

    Returns:
        ``zh`` 或 ``en``。
    """
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    total = cjk + latin
    if total == 0:
        return "en"
    return "zh" if cjk / total > 0.1 else "en"


def _count_words(text: str) -> int:
    """計算字數：中文以字為單位，英文以詞為單位。

    Args:
        text: 正文。

    Returns:
        字數。
    """
    return len(_CJK.findall(text)) + len(_LATIN_WORD.findall(text))


def _extract_title(metadata: dict[str, Any], content: str) -> str | None:
    """取得文章標題。

    Args:
        metadata: front matter 內容。
        content: 正文。

    Returns:
        front matter 的 title，其次是第一個標題行；都沒有則為 ``None``。
    """
    raw_title = metadata.get("title")
    if isinstance(raw_title, str) and raw_title.strip():
        return raw_title.strip()
    match = _HEADING.search(_mask_code(content))
    if match is not None:
        return match.group("text").strip()
    return None


def parse_article(markdown_path: Path) -> ParsedArticle:
    """讀取並解析一篇 Markdown 文章。

    Args:
        markdown_path: Markdown 檔案路徑。

    Returns:
        解析結果。

    Raises:
        MarkdownParseError: 檔案不存在、不是檔案，或內容無法解析。
    """
    if not markdown_path.is_file():
        raise MarkdownParseError(f"找不到 Markdown 檔案：{markdown_path}")

    try:
        with markdown_path.open("r", encoding="utf-8") as handle:
            post = frontmatter.load(handle)
    except UnicodeDecodeError as exc:
        raise MarkdownParseError(
            f"{markdown_path} 不是 UTF-8 編碼，請先轉檔後再試"
        ) from exc
    except OSError as exc:
        raise MarkdownParseError(f"無法讀取 {markdown_path}：{exc}") from exc
    except Exception as exc:  # frontmatter 會把 YAML 錯誤原樣拋出
        raise MarkdownParseError(f"解析 front matter 失敗：{exc}") from exc

    content = post.content
    masked = _mask_code(content)
    references = _iter_markdown_images(masked, content) + _iter_html_images(masked, content)
    references.sort(key=lambda ref: ref.start)

    metadata = dict(post.metadata)
    return ParsedArticle(
        path=markdown_path.resolve(),
        metadata=metadata,
        content=content,
        title=_extract_title(metadata, content),
        language=_detect_language(content),
        word_count=_count_words(content),
        targets=_group_targets(markdown_path.resolve(), references),
    )


def extract_context(article: ParsedArticle, target: ImageTarget, radius: int = CONTEXT_RADIUS) -> str:
    """擷取圖片周圍的段落，幫助模型理解這張圖在文章中的作用。

    會一併附上文章標題與圖片所在章節的標題，實測比只丟圖片明顯提升 alt 品質。

    Args:
        article: 已解析的文章。
        target: 目標圖片。
        radius: 圖片前後各取多少字元。

    Returns:
        組好的上下文文字；找不到引用位置時回傳空字串。
    """
    if not target.references:
        return ""

    reference = target.references[0]
    start = max(0, reference.start - radius)
    end = min(len(article.content), reference.end + radius)
    excerpt = article.content[start:end].strip()

    parts: list[str] = []
    if article.title:
        parts.append(f"文章標題：{article.title}")

    section = None
    for match in _HEADING.finditer(_mask_code(article.content)):
        if match.start() > reference.start:
            break
        section = match.group("text").strip()
    if section and section != article.title:
        parts.append(f"所在章節：{section}")

    if reference.alt.strip():
        parts.append(f"原本的 alt：{reference.alt.strip()}")

    parts.append(f"圖片周圍內容：\n{excerpt}")
    return "\n".join(parts)
