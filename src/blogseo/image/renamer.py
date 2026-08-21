"""圖片檔名的正規化與衝突處理。

實際的改名動作屬於 ``apply`` 階段；此模組目前只負責純函式部分，
確保 analyze 階段就能把模型建議的檔名正規化成安全的形式。
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Final

#: 允許的圖片副檔名（小寫，含點）。
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
)

#: 檔名主體的長度上限，避免產生過長的路徑。
MAX_STEM_LENGTH: Final[int] = 60

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_EDGE_HYPHENS = re.compile(r"^-+|-+$")


def slugify(text: str, *, fallback: str = "image") -> str:
    """把任意字串轉成 URL 安全的小寫連字號檔名主體。

    會先去除副檔名，再以 NFKD 正規化剝掉重音符號；無法轉成 ASCII 的字元
    （例如中文）會被丟棄，因此呼叫端應該要求模型直接產生英文檔名。

    Args:
        text: 模型建議的檔名，可包含副檔名。
        fallback: 正規化後為空字串時使用的預設值。

    Returns:
        只含 ``a-z``、``0-9`` 與 ``-`` 的檔名主體，不含副檔名。
    """
    candidate = text.strip()
    suffix = Path(candidate).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        candidate = candidate[: -len(suffix)]

    normalized = unicodedata.normalize("NFKD", candidate)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _NON_SLUG_CHARS.sub("-", ascii_only)
    slug = _EDGE_HYPHENS.sub("", slug)

    if len(slug) > MAX_STEM_LENGTH:
        slug = slug[:MAX_STEM_LENGTH]
        slug = slug.rsplit("-", 1)[0] if "-" in slug else slug
        slug = _EDGE_HYPHENS.sub("", slug)

    return slug or fallback


def build_filename(stem: str, extension: str) -> str:
    """把檔名主體與副檔名組成完整檔名。

    Args:
        stem: 已正規化的檔名主體。
        extension: 原始副檔名，含點；大小寫不拘。

    Returns:
        完整檔名，副檔名一律轉小寫。
    """
    suffix = extension.lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{stem}{suffix}"
