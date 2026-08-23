"""把選擇結果寫回 Markdown：改圖片語法，並可把關鍵字／摘要寫進 front matter。

圖片引用依字元位置替換，不用全域字串取代，避免誤傷同名路徑或教學範例。
位置來自重新解析當下的文章，因此 apply 前必須先確認正文雜湊仍一致。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter

from blogseo.errors import ApplyError
from blogseo.markdown.parser import _MARKDOWN_IMAGE, _strip_path_decorations

_HTML_SRC = re.compile(
    r"""(\bsrc\s*=\s*)(?P<quote>["'])(?P<value>.*?)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ALT = re.compile(
    r"""(\balt\s*=\s*)(?P<quote>["'])(?P<value>.*?)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_HTML_CLOSING = re.compile(r"(\s*/\s*)?>\s*$", re.DOTALL)


def join_source_path(original_source: str, new_filename: str) -> str:
    """在原始相對路徑的同一層目錄換上新檔名。

    Args:
        original_source: Markdown 裡寫的路徑，可能帶角括號或百分號編碼。
        new_filename: 含副檔名的新檔名。

    Returns:
        以 ``/`` 分隔的相對路徑。
    """
    stripped = _strip_path_decorations(original_source)
    parent = Path(stripped).parent
    if str(parent) in (".", ""):
        return new_filename
    return f"{parent.as_posix()}/{new_filename}"


def _escape_markdown_alt(text: str) -> str:
    """去掉會打斷 ``![alt](path)`` 語法的字元。"""
    return text.replace("]", "")


def _escape_html_attr(text: str, quote: str) -> str:
    """依所用引號跳脫 HTML 屬性值。"""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;")
    if quote == '"':
        return escaped.replace('"', "&quot;")
    return escaped.replace("'", "&#39;")


def rewrite_markdown_image(
    raw: str, *, new_path: str | None = None, new_alt: str | None = None
) -> str:
    """改寫一則 Markdown 圖片語法，保留 title 與角括號包裝。

    Args:
        raw: 原始的完整 ``![alt](path)`` 字串。
        new_path: 新路徑；``None`` 表示不改。
        new_alt: 新 alt；``None`` 表示不改。

    Returns:
        改寫後的語法。

    Raises:
        ApplyError: 原始字串不是可辨識的 Markdown 圖片語法。
    """
    match = _MARKDOWN_IMAGE.fullmatch(raw)
    if match is None:
        raise ApplyError(f"無法解析 Markdown 圖片語法：{raw}")

    alt = match.group("alt") if new_alt is None else _escape_markdown_alt(new_alt)
    path = match.group("path")
    if new_path is not None:
        stripped = path.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            path = f"<{new_path}>"
        else:
            path = new_path
    title = match.group("title")
    if title:
        return f"![{alt}]({path} {title})"
    return f"![{alt}]({path})"


def rewrite_html_image(
    raw: str, *, new_path: str | None = None, new_alt: str | None = None
) -> str:
    """改寫一個 ``<img>`` 標籤，保留 width 等其他屬性。

    Args:
        raw: 原始的完整標籤。
        new_path: 新的 ``src``；``None`` 表示不改。
        new_alt: 新的 ``alt``；``None`` 表示不改。沒有 alt 屬性時會補上。

    Returns:
        改寫後的標籤。

    Raises:
        ApplyError: 要改路徑卻找不到 ``src`` 屬性。
    """
    result = raw
    if new_path is not None:
        updated, count = _HTML_SRC.subn(
            lambda match: f"{match.group(1)}{match.group('quote')}{new_path}{match.group('quote')}",
            result,
            count=1,
        )
        if count == 0:
            raise ApplyError(f"HTML 圖片沒有 src 屬性：{raw}")
        result = updated

    if new_alt is None:
        return result

    alt_match = _HTML_ALT.search(result)
    if alt_match is not None:
        quote = alt_match.group("quote")
        escaped = _escape_html_attr(new_alt, quote)
        return _HTML_ALT.sub(
            lambda match: f"{match.group(1)}{quote}{escaped}{quote}",
            result,
            count=1,
        )

    escaped = _escape_html_attr(new_alt, '"')
    closing = _HTML_CLOSING.search(result)
    if closing is None:
        raise ApplyError(f"無法在 HTML 圖片標籤補上 alt：{raw}")
    insert_at = closing.start()
    return f'{result[:insert_at]} alt="{escaped}"{result[insert_at:]}'


def rewrite_image_markup(
    raw: str, syntax: str, *, new_path: str | None = None, new_alt: str | None = None
) -> str:
    """依語法種類改寫一次圖片引用。

    Args:
        raw: 原始語法字串。
        syntax: ``markdown`` 或 ``html``。
        new_path: 新路徑；``None`` 表示不改。
        new_alt: 新 alt；``None`` 表示不改。

    Returns:
        改寫後的語法；兩項都不改時原樣回傳。
    """
    if new_path is None and new_alt is None:
        return raw
    if syntax == "html":
        return rewrite_html_image(raw, new_path=new_path, new_alt=new_alt)
    return rewrite_markdown_image(raw, new_path=new_path, new_alt=new_alt)


def apply_spans(content: str, replacements: list[tuple[int, int, str]]) -> str:
    """依字元位置由後往前替換，避免前面的替換把後面的位置帶偏。

    Args:
        content: 正文。
        replacements: ``(start, end, 新字串)`` 清單。

    Returns:
        替換後的正文。
    """
    result = content
    for start, end, text in sorted(replacements, key=lambda item: item[0], reverse=True):
        result = result[:start] + text + result[end:]
    return result


def splice_body(original_text: str, original_body: str, new_body: str) -> str:
    """把新的正文接回原檔，front matter 原樣保留。

    用正文在原檔中最後一次出現的位置來切，避免 YAML 裡碰巧有相同字串時切錯。

    Args:
        original_text: 磁碟上的完整檔案內容。
        original_body: 解析後的原文正文（不含 front matter）。
        new_body: 替換圖片語法之後的正文。

    Returns:
        完整檔案內容。

    Raises:
        ApplyError: 原檔裡找不到對應的正文，無法安全回寫。
    """
    if not original_body:
        return original_text

    index = original_text.rfind(original_body)
    if index < 0:
        raise ApplyError("無法把正文接回原檔，front matter 與正文的邊界無法對上")

    prefix = original_text[:index]
    suffix = original_text[index + len(original_body) :]
    return prefix + new_body + suffix


def rewrite_front_matter(
    original_text: str,
    *,
    keywords: list[str] | None = None,
    summary: str | None = None,
    keywords_key: str = "keywords",
    summary_key: str = "description",
) -> str:
    """把關鍵字與摘要寫進 YAML front matter，其他欄位保留。

    沒有 front matter 時會新建一塊。關鍵字為空清單、摘要為空字串時，
    該項不會寫入，避免用空值蓋掉原文。

    Args:
        original_text: 完整檔案內容（可含已改過的圖片語法）。
        keywords: 要寫入的關鍵字；空或 ``None`` 表示不改這個欄位。
        summary: 要寫入的摘要（meta description）；空或 ``None`` 表示不改。
        keywords_key: front matter 裡關鍵字的欄位名。
        summary_key: front matter 裡摘要的欄位名。

    Returns:
        更新後的完整檔案內容。
    """
    updates: dict[str, Any] = {}
    if keywords:
        updates[keywords_key] = list(keywords)
    if summary:
        updates[summary_key] = summary
    if not updates:
        return original_text

    post = frontmatter.loads(original_text)
    post.metadata.update(updates)
    dumped = frontmatter.dumps(post)
    if not dumped.endswith("\n"):
        dumped += "\n"
    return dumped
