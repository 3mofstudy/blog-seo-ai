"""Prompt 組裝與結果結構的單元測試。"""

from __future__ import annotations

import pytest

from blogseo.llm.prompts import text_system_prompt
from blogseo.schemas.result import (
    AnalysisField,
    KeywordsOnlySchema,
    KeywordSummaryResult,
    KeywordSummarySchema,
    SummariesOnlySchema,
    text_output_schema,
)


def test_prompt_states_the_configured_range() -> None:
    prompt = text_system_prompt("zh", summary_min_chars=80, summary_max_chars=120)

    assert "80 到 120 個字元" in prompt
    assert "標點符號與空白各算一個字元" in prompt


def test_prompt_lists_distinct_angles_for_each_variant() -> None:
    prompt = text_system_prompt("zh", summary_count=3)

    assert "3 則不同的" in prompt
    for index in (1, 2, 3):
        assert f"第 {index} 則：" in prompt
    assert "第 4 則：" not in prompt


def test_prompt_omits_angle_section_for_single_summary() -> None:
    prompt = text_system_prompt("zh", summary_count=1)

    assert "一則 meta description" in prompt
    assert "第 1 則：" not in prompt


def test_prompt_extends_beyond_predefined_angles() -> None:
    prompt = text_system_prompt("zh", summary_count=5)

    assert "第 5 則：" in prompt


def test_prompt_follows_article_language() -> None:
    assert "繁體中文" in text_system_prompt("zh")
    assert "English" in text_system_prompt("en")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["  前後空白  "], ["前後空白"]),
        (["換\n行\n壓平"], ["換 行 壓平"]),
        (["重複", "重複", "不同"], ["重複", "不同"]),
        (["", "   ", "有內容"], ["有內容"]),
    ],
)
def test_summaries_are_cleaned(raw: list[str], expected: list[str]) -> None:
    result = KeywordSummaryResult(keywords=["a"], summaries=raw)
    assert result.summaries == expected


def test_keywords_are_capped_and_deduped() -> None:
    result = KeywordSummaryResult(
        keywords=["#一", "一", "二", "三", "四", "五", "六"],
        summaries=["摘要"],
    )
    assert result.keywords == ["一", "二", "三", "四", "五"]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({AnalysisField.KEYWORDS}, KeywordsOnlySchema),
        ({AnalysisField.SUMMARY}, SummariesOnlySchema),
        ({AnalysisField.KEYWORDS, AnalysisField.SUMMARY}, KeywordSummarySchema),
        ({AnalysisField.KEYWORDS, AnalysisField.IMAGES}, KeywordsOnlySchema),
    ],
)
def test_output_schema_matches_requested_fields(
    fields: set[AnalysisField], expected: type
) -> None:
    assert text_output_schema(frozenset(fields)) is expected


def test_output_schema_rejects_image_only() -> None:
    with pytest.raises(ValueError, match="至少要有一項"):
        text_output_schema(frozenset({AnalysisField.IMAGES}))


def test_partial_schema_omits_the_other_field() -> None:
    assert "summaries" not in KeywordsOnlySchema.model_json_schema()["properties"]
    assert "keywords" not in SummariesOnlySchema.model_json_schema()["properties"]


def test_prompt_omits_summary_rules_when_not_requested() -> None:
    prompt = text_system_prompt("zh", include_summaries=False)

    assert "關鍵字" in prompt
    assert "摘要" not in prompt
    assert "meta description" not in prompt


def test_prompt_omits_keyword_rules_when_not_requested() -> None:
    prompt = text_system_prompt("zh", include_keywords=False)

    assert "meta description" in prompt
    assert "關鍵字" not in prompt


def test_prompt_rejects_empty_request() -> None:
    with pytest.raises(ValueError, match="至少要產生一項"):
        text_system_prompt("zh", include_keywords=False, include_summaries=False)


def test_from_payload_fills_missing_field_with_empty_list() -> None:
    result = KeywordSummaryResult.from_payload(
        KeywordsOnlySchema(keywords=["一", "二"])
    )
    assert result.keywords == ["一", "二"]
    assert result.summaries == []
