"""繁簡轉換的單元測試。"""

from __future__ import annotations

import pytest

from blogseo.zh import to_traditional_chinese, wants_traditional_chinese


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("zh", True),
        ("ZH", True),
        ("zh-TW", True),
        ("zh-Hant", True),
        ("en", False),
        ("ja", False),
        ("zh-CN", False),
        ("zh-Hans", False),
    ],
)
def test_wants_traditional_chinese(code: str, expected: bool) -> None:
    assert wants_traditional_chinese(code) is expected


def test_simplified_alt_converts_to_taiwan_traditional() -> None:
    assert to_traditional_chinese("Python调用R的ggplot2生成折线图") == (
        "Python調用R的ggplot2生成折線圖"
    )


def test_already_traditional_stays_put() -> None:
    text = "虛擬機概觀頁面"
    assert to_traditional_chinese(text) == text


def test_english_is_unchanged() -> None:
    text = "Python calling R ggplot2"
    assert to_traditional_chinese(text) == text
