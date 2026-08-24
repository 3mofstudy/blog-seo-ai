"""OpenAI (GPT) provider。

使用 SDK 的 ``responses.parse`` 原生 structured output，直接把 Pydantic
model 當成輸出 schema，不需要自己解析 JSON 或處理模型回傳額外文字的情況。
"""

from __future__ import annotations

import base64
import time
from typing import Any, ClassVar, Final, TypeVar

import openai
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

#: GPT-5 / o 系列才支援 reasoning.effort；舊模型帶這個參數會直接 400。
_REASONING_PREFIXES: Final[tuple[str, ...]] = ("gpt-5", "o1", "o3", "o4")


class OpenAIProvider(BaseProvider):
    """呼叫 GPT 系列模型。"""

    name: ClassVar[str] = "openai"
    #: 平衡智力與費用，對應 Claude Sonnet 那一檔。
    default_model: ClassVar[str] = "gpt-5.6-terra"

    # 每百萬 token 的（輸入, 輸出）美元單價，取自官方價目表，2026-08-24 查核。
    # 只記標準短上下文價；OpenAI 調價時要更新這裡，查不到的模型會顯示費用未知。
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "gpt-5.6-sol": (4.0, 20.0),
        "gpt-5.6": (4.0, 20.0),
        "gpt-5.6-terra": (2.0, 12.0),
        "gpt-5.6-luna": (0.20, 1.20),
    }

    _TEXT_MAX_TOKENS: ClassVar[int] = 2048
    _IMAGE_MAX_TOKENS: ClassVar[int] = 1024

    # GPT-5.6 預設 reasoning effort 是 medium。抽關鍵字與寫摘要不需要深度
    # 推理，跟 Claude 一樣明確降到 low，避免思考 token 吃掉輸出額度。
    _REASONING_EFFORT: ClassVar[str] = "low"

    def __init__(self, **kwargs: Any) -> None:
        """初始化 provider 並建立 SDK client。

        Args:
            **kwargs: 傳給 :class:`~blogseo.llm.base.BaseProvider` 的參數。
        """
        super().__init__(**kwargs)
        self._client = openai.OpenAI(
            api_key=self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _parse(
        self,
        *,
        instructions: str,
        user_input: str | list[dict[str, Any]],
        text_format: type[ResultT],
        max_output_tokens: int,
    ) -> ResultT:
        """送出一次請求並取回結構化結果。

        Args:
            instructions: system prompt，對應 Responses API 的 ``instructions``。
            user_input: user 訊息；純文字分析是字串，圖片分析是帶圖的 input 列表。
            text_format: 期望的輸出結構。
            max_output_tokens: 輸出 token 上限。

        Returns:
            解析後的結果物件。

        Raises:
            ProviderError: API 呼叫失敗、逾時，或回傳內容不符合結構。
        """
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "text_format": text_format,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        request.update(_reasoning_kwargs(self.model, self._REASONING_EFFORT))
        try:
            response = self._client.responses.parse(**request)
        except openai.LengthFinishReasonError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(
                self.name,
                self.model,
                f"輸出在寫完 JSON 前就用完 {max_output_tokens} 個 token 額度，"
                "可能是文章過長或摘要要求太多，試試減少 --summaries 或降低 --summary-max",
            ) from exc
        except openai.OpenAIError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(self.name, self.model, str(exc)) from exc
        except ValidationError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(
                self.name, self.model, f"回傳內容不符合預期結構：{exc}"
            ) from exc

        elapsed = _elapsed_ms(started)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            self.usage.record_failure(elapsed_ms=elapsed)
            raise ProviderError(
                self.name, self.model, _describe_empty_response(response, max_output_tokens)
            )

        usage = getattr(response, "usage", None)
        self.usage.record_success(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            elapsed_ms=elapsed,
        )
        return parsed

    def analyze_text(self, content: str) -> KeywordSummaryResult:
        """分析文章內容，產生關鍵字與摘要。

        Args:
            content: 文章正文。

        Returns:
            關鍵字與摘要。

        Raises:
            ProviderError: 呼叫失敗或回傳內容無法解析。
        """
        payload = self._parse(
            instructions=text_system_prompt(
                self.language,
                include_keywords=AnalysisField.KEYWORDS in self.fields,
                include_summaries=AnalysisField.SUMMARY in self.fields,
                keyword_count=self.keyword_count,
                summary_count=self.summary_count,
                summary_min_chars=self.summary_min_chars,
                summary_max_chars=self.summary_max_chars,
            ),
            user_input=text_user_prompt(None, content),
            text_format=text_output_schema(self.fields),
            max_output_tokens=self._TEXT_MAX_TOKENS,
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
            ProviderError: 呼叫失敗或回傳內容無法解析。
        """
        encoded = base64.b64encode(image_bytes).decode("ascii")
        user_input: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{encoded}",
                    },
                    {
                        "type": "input_text",
                        "text": image_user_prompt(context, original_filename),
                    },
                ],
            }
        ]
        return self._parse(
            instructions=image_system_prompt(
                self.alt_language, alt_max_chars=self.alt_max_chars
            ),
            user_input=user_input,
            text_format=ImageAnalysisResult,
            max_output_tokens=self._IMAGE_MAX_TOKENS,
        )


def _reasoning_kwargs(model: str, effort: str) -> dict[str, Any]:
    """只在支援 reasoning 的模型上帶 ``reasoning.effort``。

    Args:
        model: 實際呼叫的模型 id。
        effort: 思考深度，例如 ``low``。

    Returns:
        要併進 ``responses.parse`` 的參數；不支援時為空 dict。
    """
    ident = model.casefold()
    if ident.startswith(_REASONING_PREFIXES):
        return {"reasoning": {"effort": effort}}
    return {}


def _describe_empty_response(response: Any, max_output_tokens: int) -> str:
    """把沒有結構化輸出的回應翻成使用者看得懂的說明。

    Args:
        response: SDK 回傳的 ``ParsedResponse``。
        max_output_tokens: 這次請求的輸出 token 上限。

    Returns:
        給 ``ProviderError`` 用的訊息。
    """
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    if reason == "max_output_tokens":
        return (
            f"輸出在寫完 JSON 前就用完 {max_output_tokens} 個 token 額度，"
            "可能是文章過長或摘要要求太多，試試減少 --summaries 或降低 --summary-max"
        )
    if reason == "content_filter":
        return "回應被內容過濾擋下，沒有結構化輸出"
    status = getattr(response, "status", None)
    return f"回應中沒有結構化輸出（status={status}）"


def _elapsed_ms(started: float) -> int:
    """計算從 ``started`` 到現在經過的毫秒數。

    Args:
        started: :func:`time.perf_counter` 的起始值。

    Returns:
        經過的毫秒數。
    """
    return int((time.perf_counter() - started) * 1000)
