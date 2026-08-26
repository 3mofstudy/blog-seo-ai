"""Gemini provider 的單元測試。

不打真實 API：回傳結構用假 client 餵，thinking 參數與圖片格式單獨測。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from blogseo.errors import ProviderError
from blogseo.llm.gemini import GeminiProvider, _thinking_kwargs
from blogseo.schemas.result import (
    AnalysisField,
    ImageAnalysisResult,
    KeywordSummarySchema,
)


class _FakeModels:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


def _parsed(
    payload: Any,
    *,
    prompt_tokens: int = 10,
    candidate_tokens: int = 5,
    thoughts_tokens: int = 0,
    finish_reason: str = "STOP",
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed=payload,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            thoughts_token_count=thoughts_tokens,
        ),
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=None,
    )


def _provider(fake: _FakeModels) -> GeminiProvider:
    provider = GeminiProvider(api_key="gemini-test-key")
    provider._client = _FakeClient(fake)
    return provider


def test_analyze_text_uses_generate_content_and_schema() -> None:
    payload = KeywordSummarySchema(
        keywords=["Azure NSG", "白名單"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeModels(_parsed(payload))
    result = _provider(fake).analyze_text("文章內容")

    assert result.keywords[:2] == ["Azure NSG", "白名單"]
    assert result.summaries
    call = fake.calls[0]
    assert call["model"] == GeminiProvider.default_model
    assert call["config"].response_schema is KeywordSummarySchema
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].thinking_config.thinking_level.value.lower() == "low"
    assert "以下是文章內容" in call["contents"]


def test_analyze_image_sends_inline_bytes() -> None:
    payload = ImageAnalysisResult(
        suggested_filename="Azure VM Overview", alt="虛擬機概觀頁面"
    )
    fake = _FakeModels(_parsed(payload))
    result = _provider(fake).analyze_image(
        b"fake-bytes",
        "上下文",
        media_type="image/png",
        original_filename="pic.png",
    )

    assert isinstance(result, ImageAnalysisResult)
    assert result.suggested_filename == "azure-vm-overview"
    contents = fake.calls[0]["contents"]
    assert contents[0].inline_data.mime_type == "image/png"
    assert contents[0].inline_data.data == b"fake-bytes"
    assert "原始檔名" in contents[1]


def test_analyze_text_records_api_failure() -> None:
    fake = _FakeModels(error=errors.APIError(429, {"error": {"message": "quota"}}))
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="quota"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1
    assert provider.usage.calls == 1


def test_empty_parsed_output_is_failure() -> None:
    fake = _FakeModels(_parsed(None, finish_reason="MAX_TOKENS"))
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="token 額度"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1


def test_keywords_only_schema_omits_summaries() -> None:
    from blogseo.schemas.result import KeywordsOnlySchema

    fake = _FakeModels(
        _parsed(KeywordsOnlySchema(keywords=["甲", "乙", "丙", "丁", "戊"]))
    )
    provider = GeminiProvider(
        api_key="gemini-test-key",
        fields=frozenset({AnalysisField.KEYWORDS}),
    )
    provider._client = _FakeClient(fake)
    result = provider.analyze_text("文章")

    assert fake.calls[0]["config"].response_schema is KeywordsOnlySchema
    assert result.summaries == []


def test_usage_includes_thought_tokens() -> None:
    payload = KeywordSummarySchema(
        keywords=["一"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeModels(
        _parsed(payload, prompt_tokens=100, candidate_tokens=20, thoughts_tokens=8)
    )
    provider = _provider(fake)
    provider.analyze_text("文章")

    assert provider.usage.calls == 1
    assert provider.usage.failed_calls == 0
    assert provider.usage.input_tokens == 100
    assert provider.usage.output_tokens == 28


def test_thinking_kwargs_by_model_family() -> None:
    gemini3 = _thinking_kwargs("gemini-3.5-flash", "low")
    assert gemini3["thinking_config"].thinking_level.value.lower() == "low"
    gemini25 = _thinking_kwargs("gemini-2.5-flash", "low")
    assert gemini25["thinking_config"].thinking_budget == 0
    assert _thinking_kwargs("unknown-model", "low") == {}


def test_legacy_25_model_disables_thinking() -> None:
    payload = KeywordSummarySchema(
        keywords=["一"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeModels(_parsed(payload))
    provider = GeminiProvider(api_key="gemini-test-key", model="gemini-2.5-flash")
    provider._client = _FakeClient(fake)
    provider.analyze_text("文章")
    assert fake.calls[0]["config"].thinking_config.thinking_budget == 0


def test_create_provider_gemini() -> None:
    from blogseo.llm.registry import create_provider

    provider = create_provider("gemini", api_key="gemini-test-key")
    assert isinstance(provider, GeminiProvider)
    assert provider.model == GeminiProvider.default_model

    overridden = create_provider("google:gemini-2.5-flash", api_key="gemini-test-key")
    assert overridden.model == "gemini-2.5-flash"


def test_default_model_has_pricing() -> None:
    provider = GeminiProvider(api_key="gemini-test-key")
    assert provider.price_per_mtok(provider.model) == (1.50, 9.00)
