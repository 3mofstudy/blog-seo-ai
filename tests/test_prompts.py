"""Prompt 組裝與結果結構的單元測試。"""

from __future__ import annotations

import pytest

from blogseo.llm.prompts import image_system_prompt, text_system_prompt
from blogseo.schemas.result import (
    AnalysisField,
    ImageAnalysisResult,
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


def test_prompt_states_keyword_count() -> None:
    prompt = text_system_prompt("zh", keyword_count=8)
    assert "8 個關鍵字" in prompt
    assert "中間不能有空白" in prompt
    assert "plotnine教學" in prompt


def test_prompt_lists_distinct_angles_for_each_variant() -> None:
    prompt = text_system_prompt("zh", summary_count=3)

    assert "3 則不同的" in prompt
    for index in (1, 2, 3):
        assert f"第 {index} 則：" in prompt
    assert "第 4 則：" not in prompt


def test_prompt_omits_angle_section_for_single_summary() -> None:
    prompt = text_system_prompt("zh", summary_count=1)

    assert "一則文章摘要" in prompt
    assert "第 1 則：" not in prompt


def test_prompt_describes_vocus_list_summary_style() -> None:
    prompt = text_system_prompt("zh")

    assert "方格子" in prompt
    assert "https://vocus.cc/salon/lucy-r" in prompt
    assert "主要介紹" in prompt
    assert "本文為 MLflow 實戰系列的完結篇" in prompt
    assert "不要用「本文將介紹」" not in prompt
    assert "100 到 150 個字元" in prompt


def test_prompt_extends_beyond_predefined_angles() -> None:
    prompt = text_system_prompt("zh", summary_count=5)

    assert "第 5 則：" in prompt


def test_prompt_follows_article_language() -> None:
    assert "繁體中文" in text_system_prompt("zh")
    assert "繁體中文" in text_system_prompt("zh-TW")
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


def test_keywords_are_deduped() -> None:
    result = KeywordSummaryResult(
        keywords=["#一", "一", "二", "三", "四", "五", "六"],
        summaries=["摘要"],
    )
    assert result.keywords == ["一", "二", "三", "四", "五", "六"]


def test_keywords_internal_spaces_are_removed() -> None:
    result = KeywordSummaryResult(
        keywords=["Python plotnine 繪圖", "ggplot2 轉 Python", "plotnine教學"],
        summaries=["摘要"],
    )
    assert result.keywords == ["Pythonplotnine繪圖", "ggplot2轉Python", "plotnine教學"]


def test_keywords_with_and_without_spaces_are_deduped() -> None:
    result = KeywordSummaryResult(
        keywords=["Azure NSG", "AzureNSG", "白名單"],
        summaries=["摘要"],
    )
    assert result.keywords == ["AzureNSG", "白名單"]


def test_keywords_are_capped_at_schema_max() -> None:
    result = KeywordSummaryResult(
        keywords=[str(index) for index in range(25)],
        summaries=["摘要"],
    )
    assert result.keywords == [str(index) for index in range(20)]


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
    assert "方格子" not in prompt
    assert "meta description" not in prompt


def test_prompt_omits_keyword_rules_when_not_requested() -> None:
    prompt = text_system_prompt("zh", include_keywords=False)

    assert "文章摘要" in prompt
    assert "方格子" in prompt
    assert "關鍵字" not in prompt


def test_prompt_rejects_empty_request() -> None:
    with pytest.raises(ValueError, match="至少要產生一項"):
        text_system_prompt("zh", include_keywords=False, include_summaries=False)


def test_image_prompt_states_alt_limit() -> None:
    prompt = image_system_prompt("zh", alt_max_chars=30)

    assert "不得超過 30 個字元" in prompt
    assert "繁體中文" in prompt
    assert "禁止使用簡體字" in prompt
    assert "一句話講完" in prompt


def test_image_prompt_follows_alt_language() -> None:
    english = image_system_prompt("en", alt_max_chars=40)
    assert "English" in english
    assert "不得超過 40 個字元" in english
    assert "禁止使用簡體字" not in english


def test_from_payload_fills_missing_field_with_empty_list() -> None:
    result = KeywordSummaryResult.from_payload(
        KeywordsOnlySchema(keywords=["一", "二"])
    )
    assert result.keywords == ["一", "二"]
    assert result.summaries == []


def test_image_prompt_states_json_field_names() -> None:
    prompt = image_system_prompt("zh")
    assert "suggested_filename" in prompt
    assert "不要改成 filename" in prompt


@pytest.mark.parametrize(
    "payload",
    [
        {"suggested_filename": "AWS MFA Settings", "alt": "安全憑證頁面"},
        {"filename": "AWS MFA Settings", "alt": "安全憑證頁面"},
        {"file_name": "AWS MFA Settings", "alt_text": "安全憑證頁面"},
    ],
)
def test_image_result_accepts_common_aliases(payload: dict[str, str]) -> None:
    result = ImageAnalysisResult.model_validate(payload)
    assert result.suggested_filename == "aws-mfa-settings"
    assert result.alt == "安全憑證頁面"
