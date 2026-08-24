"""OpenAI provider 的單元測試。

不打真實 API：回傳結構用假 client 餵，reasoning 參數與圖片格式單獨測。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from blogseo.errors import ProviderError
from blogseo.llm.openai import OpenAIProvider, _reasoning_kwargs
from blogseo.schemas.result import (
    AnalysisField,
    ImageAnalysisResult,
    KeywordSummarySchema,
)


class _FakeResponses:
    """把預先準備好的回應或例外丟給 provider。"""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _parsed(
    payload: Any,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    status: str = "completed",
    incomplete_reason: str | None = None,
) -> SimpleNamespace:
    incomplete = (
        SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
    )
    return SimpleNamespace(
        output_parsed=payload,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        status=status,
        incomplete_details=incomplete,
    )


def _provider(fake: _FakeResponses) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-test")
    provider._client = _FakeClient(fake)
    return provider


def test_analyze_text_uses_responses_parse_and_schema() -> None:
    payload = KeywordSummarySchema(
        keywords=["Azure NSG", "白名單"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeResponses(_parsed(payload))
    result = _provider(fake).analyze_text("文章內容")

    assert result.keywords[:2] == ["Azure NSG", "白名單"]
    assert result.summaries
    call = fake.calls[0]
    assert call["model"] == OpenAIProvider.default_model
    assert call["text_format"] is KeywordSummarySchema
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "low"}
    assert "以下是文章內容" in call["input"]


def test_analyze_image_sends_data_url() -> None:
    payload = ImageAnalysisResult(
        suggested_filename="Azure VM Overview", alt="虛擬機概觀頁面"
    )
    fake = _FakeResponses(_parsed(payload))
    result = _provider(fake).analyze_image(
        b"fake-bytes",
        "上下文",
        media_type="image/png",
        original_filename="pic.png",
    )

    assert isinstance(result, ImageAnalysisResult)
    assert result.suggested_filename == "azure-vm-overview"
    content = fake.calls[0]["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "input_text"


def test_analyze_text_records_api_failure() -> None:
    fake = _FakeResponses(error=openai.APITimeoutError(request=None))
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="timed out|Timeout"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1
    assert provider.usage.calls == 1


def test_analyze_text_records_truncated_output() -> None:
    fake = _FakeResponses(
        error=openai.LengthFinishReasonError(
            completion=SimpleNamespace(usage=SimpleNamespace())
        )
    )
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="token 額度"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1


def test_empty_parsed_output_is_failure() -> None:
    fake = _FakeResponses(
        _parsed(None, status="incomplete", incomplete_reason="max_output_tokens")
    )
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="token 額度"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1


def test_keywords_only_schema_omits_summaries() -> None:
    from blogseo.schemas.result import KeywordsOnlySchema

    fake = _FakeResponses(
        _parsed(KeywordsOnlySchema(keywords=["甲", "乙", "丙", "丁", "戊"]))
    )
    provider = OpenAIProvider(
        api_key="sk-test",
        fields=frozenset({AnalysisField.KEYWORDS}),
    )
    provider._client = _FakeClient(fake)
    result = provider.analyze_text("文章")

    assert fake.calls[0]["text_format"] is KeywordsOnlySchema
    assert result.summaries == []


def test_usage_is_recorded_on_success() -> None:
    payload = KeywordSummarySchema(
        keywords=["一"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeResponses(_parsed(payload, input_tokens=100, output_tokens=20))
    provider = _provider(fake)
    provider.analyze_text("文章")

    assert provider.usage.calls == 1
    assert provider.usage.failed_calls == 0
    assert provider.usage.input_tokens == 100
    assert provider.usage.output_tokens == 20


def test_reasoning_kwargs_only_for_gpt5_family() -> None:
    assert _reasoning_kwargs("gpt-5.6-terra", "low") == {"reasoning": {"effort": "low"}}
    assert _reasoning_kwargs("gpt-4o", "low") == {}
    assert _reasoning_kwargs("o3-mini", "low") == {"reasoning": {"effort": "low"}}


def test_legacy_model_does_not_send_reasoning() -> None:
    payload = KeywordSummarySchema(
        keywords=["一"],
        summaries=["這是一則超過六十字元下限的摘要內容用來通過驗證。"],
    )
    fake = _FakeResponses(_parsed(payload))
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
    provider._client = _FakeClient(fake)
    provider.analyze_text("文章")
    assert "reasoning" not in fake.calls[0]


def test_create_provider_openai() -> None:
    from blogseo.llm.registry import create_provider

    provider = create_provider("gpt", api_key="sk-test")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == OpenAIProvider.default_model

    overridden = create_provider("gpt:gpt-5.6-sol", api_key="sk-test")
    assert overridden.model == "gpt-5.6-sol"


def test_default_model_has_pricing() -> None:
    provider = OpenAIProvider(api_key="sk-test")
    assert provider.price_per_mtok(provider.model) == (2.0, 12.0)
