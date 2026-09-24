"""Hugging Face Inference Providers。

文字與圖片拆成兩顆較小、較不容易塞車的模型，都吃 Hugging Face 帳號的免費額度：

* 關鍵字 / 摘要：``Qwen/Qwen3-4B-Instruct-2507``（Nscale）
* 圖片檔名 / alt：``zai-org/GLM-4.6V-Flash``（Novita）

個別模型失敗只寫進該格 ``error``，不會讓整個 analyze 掛掉。
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, ClassVar, TypeVar

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from pydantic import BaseModel, ValidationError

from blogseo.errors import ProviderError
from blogseo.llm.base import BaseProvider
from blogseo.llm.prompts import (
    image_system_prompt,
    image_user_prompt,
    text_system_prompt,
    text_user_prompt,
)
from blogseo.schemas.result import (
    AnalysisField,
    ImageAnalysisResult,
    KeywordSummaryResult,
    text_output_schema,
)

ResultT = TypeVar("ResultT", bound=BaseModel)

#: 模型偶爾會把 JSON 包在 markdown 圍籬裡。
_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

#: Hugging Face 帳號要在這裡開啟要使用的 Inference Provider。
_PROVIDER_SETTINGS_URL = "https://huggingface.co/settings/inference-providers"


class HuggingFaceProvider(BaseProvider):
    """呼叫 Hugging Face Inference Providers 上的開放權重模型。"""

    name: ClassVar[str] = "huggingface"
    #: 中文指令跟隨穩、體積小，走 Nscale，不跟熱門視覺模型搶 Featherless 名額。
    default_model: ClassVar[str] = "Qwen/Qwen3-4B-Instruct-2507"
    #: 專門看圖，走 Novita，免費額度比較吃得下。
    default_image_model: ClassVar[str] = "zai-org/GLM-4.6V-Flash"
    #: 走免費額度，不估美元費用，表格會顯示「—」。
    pricing: ClassVar[dict[str, tuple[float, float]]] = {}

    _TEXT_MAX_TOKENS: ClassVar[int] = 2048
    _IMAGE_MAX_TOKENS: ClassVar[int] = 1024

    def __init__(self, *, image_model: str | None = None, **kwargs: Any) -> None:
        """初始化 provider，分別建立文字與圖片的 Hub client。

        Args:
            image_model: 覆寫圖片模型 id；未指定時使用 :attr:`default_image_model`。
            **kwargs: 傳給 :class:`~blogseo.llm.base.BaseProvider` 的參數。
        """
        super().__init__(**kwargs)
        self._image_model = image_model or self.default_image_model
        self._text_client = InferenceClient(
            provider="nscale",
            api_key=self._api_key,
            timeout=self.timeout,
        )
        self._image_client = InferenceClient(
            provider="novita",
            api_key=self._api_key,
            timeout=self.timeout,
        )

    @property
    def image_model(self) -> str:
        """圖片分析實際使用的模型 id。"""
        return self._image_model

    def analyze_text(self, content: str) -> KeywordSummaryResult:
        """分析文章內容，產生關鍵字與摘要。

        Args:
            content: 文章正文。

        Returns:
            關鍵字與摘要。

        Raises:
            ProviderError: API 呼叫失敗或回傳內容不符合結構。
        """
        schema = text_output_schema(self.fields)
        payload = self._complete(
            client=self._text_client,
            model=self.text_model,
            messages=[
                {
                    "role": "system",
                    "content": text_system_prompt(
                        self.language,
                        include_keywords=AnalysisField.KEYWORDS in self.fields,
                        include_summaries=AnalysisField.SUMMARY in self.fields,
                        keyword_count=self.keyword_count,
                        keyword_max_chars=self.keyword_max_chars,
                        summary_count=self.summary_count,
                        summary_min_chars=self.summary_min_chars,
                        summary_max_chars=self.summary_max_chars,
                    ),
                },
                {"role": "user", "content": text_user_prompt(None, content)},
            ],
            output_format=schema,
            max_tokens=self._TEXT_MAX_TOKENS,
            response_format=_json_schema_format(schema),
        )
        return self._limit_text_result(KeywordSummaryResult.from_payload(payload))

    def analyze_image(
        self,
        image_bytes: bytes,
        context: str,
        *,
        media_type: str = "image/png",
        original_filename: str = "",
    ) -> ImageAnalysisResult:
        """分析單張圖片，產生建議檔名與 alt 文字。

        Args:
            image_bytes: 圖片位元組。
            context: 圖片在文章中的上下文。
            media_type: 圖片的 MIME type。
            original_filename: 原始檔名，供模型參考。

        Returns:
            建議檔名與 alt 文字。

        Raises:
            ProviderError: API 呼叫失敗或回傳內容不符合結構。
        """
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return self._complete(
            client=self._image_client,
            model=self.image_model,
            messages=[
                {
                    "role": "system",
                    "content": image_system_prompt(
                        self.alt_language, alt_max_chars=self.alt_max_chars
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}"
                            },
                        },
                        {
                            "type": "text",
                            "text": image_user_prompt(context, original_filename),
                        },
                    ],
                },
            ],
            output_format=ImageAnalysisResult,
            max_tokens=self._IMAGE_MAX_TOKENS,
            # Novita / GLM 不支援 json_schema，只接受 json_object。
            response_format={"type": "json_object"},
        )

    def _complete(
        self,
        *,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        output_format: type[ResultT],
        max_tokens: int,
        response_format: dict[str, Any],
    ) -> ResultT:
        """送出一次 chat completion 並把內容驗證成 Pydantic 模型。

        Hugging Face 的結構化輸出是 ``response_format``（文字用
        ``json_schema``，圖片用 ``json_object``），回傳仍是 JSON 字串，
        再由我們用同一份 schema 驗證——不是手寫正則去挖欄位。

        Args:
            client: 對應供應商的 Hub client。
            model: 實際呼叫的模型 id。
            messages: chat 訊息。
            output_format: 期望的輸出結構。
            max_tokens: 輸出 token 上限。
            response_format: Hub 的 ``response_format``；視覺模型常用
                ``json_object``，因為多家供應商不支援 ``json_schema``。

        Returns:
            驗證後的結果。

        Raises:
            ProviderError: 網路錯誤、空回應，或 JSON 對不上 schema。
        """
        started = time.perf_counter()
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format=response_format,
            )
        except (
            HfHubHTTPError,
            InferenceTimeoutError,
            TimeoutError,
            OSError,
            httpx.HTTPError,
        ) as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(self.name, model, explain_hf_error(exc)) from exc

        elapsed = _elapsed_ms(started)
        raw = _message_content(response)
        if not raw:
            self.usage.record_failure(elapsed_ms=elapsed)
            raise ProviderError(self.name, model, "回應是空的")

        try:
            parsed = output_format.model_validate(loads_json_payload(raw))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            self.usage.record_failure(elapsed_ms=elapsed)
            raise ProviderError(
                self.name, model, f"回傳內容不符合預期結構：{exc}"
            ) from exc

        usage = getattr(response, "usage", None)
        self.usage.record_success(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            elapsed_ms=elapsed,
        )
        return parsed


def _json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    """把 Pydantic model 包成 Inference Providers 要的 response_format。

    Args:
        schema: 輸出結構。

    Returns:
        ``chat_completion`` 的 ``response_format`` 參數。
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": False,
        },
    }


def _message_content(response: Any) -> str:
    """從 Hub 回傳物件取出文字。

    Args:
        response: ``chat_completion`` 的回傳值。

    Returns:
        助理訊息內容；取不到時為空字串。
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content.strip() if isinstance(content, str) else ""


def loads_json_payload(raw: str) -> dict[str, Any]:
    """把模型回傳的文字解析成 JSON 物件。

    允許包在 markdown 圍籬裡，或前後夾雜說明文字。

    Args:
        raw: 模型回傳的字串。

    Returns:
        JSON 物件。

    Raises:
        TypeError: 解析出來不是 JSON 物件。
        json.JSONDecodeError: 內容看起來像 JSON 但解析失敗。
    """
    stripped = raw.strip()
    fenced = _FENCED_JSON.search(stripped)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("結構化輸出必須是 JSON 物件")
    return payload


def explain_hf_error(exc: BaseException) -> str:
    """把 Hub 的原始錯誤翻成使用者看得懂的說明。

    Args:
        exc: SDK 丟出的例外。

    Returns:
        給 ``ProviderError`` 用的訊息。
    """
    text = str(exc).strip()
    response = getattr(exc, "response", None)
    body = ""
    if response is not None:
        body = (getattr(response, "text", None) or "").strip()
    if body and body not in text:
        text = f"{text}\n{body}"
    lowered = text.lower()
    if "certificate verify failed" in lowered or "self-signed certificate" in lowered:
        return (
            "連線被憑證驗證擋下（憑證鏈裡有自簽憑證）。"
            "通常是防毒、公司代理或 VPN 在攔截 HTTPS。"
            "請確認該憑證已安裝在 Windows 的受信任根憑證，然後再試一次。"
        )
    if "json_schema" in lowered and "does not support" in lowered:
        return "這個供應商的視覺模型不支援 json_schema，請改用 json_object。"
    if "capacity_exhausted" in lowered or "temporarily at capacity" in lowered:
        return "這個模型目前額滿，請稍後再試，或改用其他供應商上的模型。"
    if "model_not_supported" in lowered or "not supported by any provider" in lowered:
        return (
            "這個模型目前沒有你已啟用的供應商在託管。"
            f"請到 {_PROVIDER_SETTINGS_URL} 確認 Nscale、Novita 為開啟，"
            "且該列不要填 Hugging Face 的 hf_ token。"
        )
    return text


def _elapsed_ms(started: float) -> int:
    """計算從 ``started`` 到現在經過的毫秒數。

    Args:
        started: :func:`time.perf_counter` 的起始值。

    Returns:
        經過的毫秒數。
    """
    return int((time.perf_counter() - started) * 1000)
