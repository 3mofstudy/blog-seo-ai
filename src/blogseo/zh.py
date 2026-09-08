"""中文用字轉換。

大陸視覺模型常無視 prompt、直接輸出簡體。語言代碼是繁體中文時，
把 alt 再轉成台灣正體，避免落地簡體字。
"""

from __future__ import annotations

from typing import Final

from zhconv import convert

#: 這個專案把 ``zh`` 視為繁體中文（台灣正體）。
_TRADITIONAL_CODES: Final[frozenset[str]] = frozenset(
    {"zh", "zh-hant", "zh-tw", "zh-hk", "zh-mo"}
)


def wants_traditional_chinese(code: str) -> bool:
    """判斷這個語言代碼是否要求繁體中文。

    Args:
        code: 語言代碼，例如 ``zh``、``en``。

    Returns:
        需要繁體中文時為 True。
    """
    normalized = code.strip().lower().replace("_", "-")
    return normalized in _TRADITIONAL_CODES or normalized.startswith(
        ("zh-hant", "zh-tw")
    )


def to_traditional_chinese(text: str) -> str:
    """把簡體中文轉成台灣繁體；已是繁體或非中文時維持原樣。

    Args:
        text: 原始文字。

    Returns:
        轉換後的文字。
    """
    return convert(text, "zh-tw")
