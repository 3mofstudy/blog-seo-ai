"""Hugging Face provider 的單元測試。

不打真實 Inference API：回傳結構用假 client 餵，JSON 解析單獨測。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from blogseo.errors import ProviderError
from blogseo.llm.huggingface import (
    HuggingFaceProvider,
    explain_hf_error,
    loads_json_payload,
)
from blogseo.schemas.result import AnalysisField, ImageAnalysisResult


class _FakeClient:
    """把預先準備好的回應或例外丟給 provider。"""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _completion(content: str, *, prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _provider(fake: _FakeClient) -> HuggingFaceProvider:
    provider = HuggingFaceProvider(api_key="hf-test-token")
    provider._text_client = fake
    provider._image_client = fake
    return provider


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"keywords": ["一"]}', {"keywords": ["一"]}),
        ('```json\n{"alt": "圖"}\n```', {"alt": "圖"}),
        ('說明如下：\n{"suggested_filename": "chart"}\n以上', {"suggested_filename": "chart"}),
    ],
)
def test_loads_json_payload(raw: str, expected: dict[str, str | list[str]]) -> None:
    assert loads_json_payload(raw) == expected


def test_loads_json_payload_rejects_array() -> None:
    with pytest.raises(TypeError, match="JSON 物件"):
        loads_json_payload("[1, 2]")


def test_analyze_text_validates_schema() -> None:
    fake = _FakeClient(
        _completion(
            '{"keywords": ["Azure NSG", "白名單"],'
            '"summaries": ["這是一則超過六十字元下限的摘要內容用來通過驗證。"]}'
        )
    )
    result = _provider(fake).analyze_text("文章內容")

    assert result.keywords[:2] == ["Azure NSG", "白名單"]
    assert result.summaries
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[0]["model"] == HuggingFaceProvider.default_model


def test_analyze_image_sends_data_url() -> None:
    fake = _FakeClient(
        _completion(
            '{"suggested_filename": "Azure VM Overview", "alt": "虛擬機概觀頁面"}'
        )
    )
    result = _provider(fake).analyze_image(
        b"fake-bytes",
        "上下文",
        media_type="image/png",
        original_filename="pic.png",
    )

    assert isinstance(result, ImageAnalysisResult)
    assert result.suggested_filename == "azure-vm-overview"
    assert fake.calls[0]["model"] == HuggingFaceProvider.default_image_model
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    content = fake.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_explain_hf_error_for_disabled_provider() -> None:
    message = explain_hf_error(
        RuntimeError(
            "Bad request: {'message': \"The requested model "
            "'Qwen/Qwen2.5-VL-7B-Instruct' is not supported by any provider "
            "you have enabled.\", 'code': 'model_not_supported'}"
        )
    )
    assert "Nscale" in message
    assert "huggingface.co/settings/inference-providers" in message


def test_text_and_image_use_different_models() -> None:
    provider = HuggingFaceProvider(api_key="hf-test-token")
    assert provider.text_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert provider.image_model == "zai-org/GLM-4.6V-Flash"
    assert provider.text_model != provider.image_model


def test_analyze_image_accepts_filename_alias() -> None:
    fake = _FakeClient(
        _completion('{"filename": "AWS Console MFA", "alt": "安全憑證選項"}')
    )
    result = _provider(fake).analyze_image(b"fake-bytes", "上下文")
    assert result.suggested_filename == "aws-console-mfa"
    assert result.alt == "安全憑證選項"


def test_analyze_text_records_http_failure() -> None:
    fake = _FakeClient(error=TimeoutError("inference timed out"))
    provider = _provider(fake)

    with pytest.raises(ProviderError, match="timed out"):
        provider.analyze_text("文章")
    assert provider.usage.failed_calls == 1
    assert provider.usage.calls == 1


def test_keywords_only_schema_omits_summaries() -> None:
    fake = _FakeClient(_completion('{"keywords": ["甲", "乙", "丙", "丁", "戊"]}'))
    provider = HuggingFaceProvider(
        api_key="hf-test-token",
        fields=frozenset({AnalysisField.KEYWORDS}),
    )
    provider._text_client = fake
    provider._image_client = fake
    result = provider.analyze_text("文章")

    schema = fake.calls[0]["response_format"]["json_schema"]["schema"]
    assert "keywords" in schema["properties"]
    assert "summaries" not in schema.get("properties", {})
    assert result.summaries == []
