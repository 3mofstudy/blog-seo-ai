"""Google Gemini provider。

使用 ``google-genai`` 的 ``generate_content`` 加上 ``response_schema``，
直接把 Pydantic model 當成輸出 schema。
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, TypeVar

from google import genai
from google.genai import errors, types
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


class GeminiProvider(BaseProvider):
    """呼叫 Gemini 系列模型。"""

    name: ClassVar[str] = "gemini"
    #: 目前的 Flash 主力，對應 Claude Sonnet / GPT terra 那一檔。
    default_model: ClassVar[str] = "gemini-3.5-flash"

    # 每百萬 token 的（輸入, 輸出）美元單價，取自官方價目表，2026-08-26 查核。
    # 長上下文加價不記；本工具送出的內容遠低於 200k。查不到的模型費用顯示未知。
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "gemini-3.5-flash": (1.50, 9.00),
        "gemini-3.1-flash-lite": (0.25, 1.50),
        "gemini-3.1-pro-preview": (2.00, 12.00),
        "gemini-3-flash-preview": (0.50, 3.00),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
    }

    _TEXT_MAX_TOKENS: ClassVar[int] = 2048
    _IMAGE_MAX_TOKENS: ClassVar[int] = 1024
    _THINKING_LEVEL: ClassVar[str] = "low"

    def __init__(self, **kwargs: Any) -> None:
        """初始化 provider 並建立 SDK client。

        Args:
            **kwargs: 傳給 :class:`~blogseo.llm.base.BaseProvider` 的參數。
        """
        super().__init__(**kwargs)
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )

    def _generate(
        self,
        *,
        system: str,
        contents: str | list[Any],
        output_format: type[ResultT],
        max_output_tokens: int,
    ) -> ResultT:
        """送出一次請求並取回結構化結果。"""
        started = time.perf_counter()
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=output_format,
            **_thinking_kwargs(self.model, self._THINKING_LEVEL),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(self.name, self.model, str(exc)) from exc
        except ValidationError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(
                self.name, self.model, f"回傳內容不符合預期結構：{exc}"
            ) from exc

        elapsed = _elapsed_ms(started)
        parsed = _coerce_parsed(response, output_format)
        if parsed is None:
            self.usage.record_failure(elapsed_ms=elapsed)
            raise ProviderError(
                self.name, self.model, _describe_empty_response(response, max_output_tokens)
            )

        usage = getattr(response, "usage_metadata", None)
        input_tokens, output_tokens = _usage_tokens(usage)
        self.usage.record_success(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        )
        return parsed

    def analyze_text(self, content: str) -> KeywordSummaryResult:
        """分析文章內容，產生關鍵字與摘要。"""
        payload = self._generate(
            system=text_system_prompt(
                self.language,
                include_keywords=AnalysisField.KEYWORDS in self.fields,
                include_summaries=AnalysisField.SUMMARY in self.fields,
                keyword_count=self.keyword_count,
                keyword_max_chars=self.keyword_max_chars,
                summary_count=self.summary_count,
                summary_min_chars=self.summary_min_chars,
                summary_max_chars=self.summary_max_chars,
            ),
            contents=text_user_prompt(None, content),
            output_format=text_output_schema(self.fields),
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
        """分析單張圖片，產生建議檔名與 alt 文字。"""
        return self._generate(
            system=image_system_prompt(
                self.alt_language, alt_max_chars=self.alt_max_chars
            ),
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                image_user_prompt(context, original_filename),
            ],
            output_format=ImageAnalysisResult,
            max_output_tokens=self._IMAGE_MAX_TOKENS,
        )


def _thinking_kwargs(model: str, level: str) -> dict[str, Any]:
    """Gemini 3 用 thinking_level；2.5 用 thinking_budget=0 關掉思考。"""
    ident = model.casefold()
    if ident.startswith("gemini-3"):
        return {"thinking_config": types.ThinkingConfig(thinking_level=level)}
    if ident.startswith("gemini-2.5"):
        return {"thinking_config": types.ThinkingConfig(thinking_budget=0)}
    return {}


def _coerce_parsed(response: Any, output_format: type[ResultT]) -> ResultT | None:
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        return None
    if isinstance(parsed, output_format):
        return parsed
    try:
        return output_format.model_validate(parsed)
    except (ValidationError, TypeError):
        return None


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "response_token_count", None)
        or 0
    )
    thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    return input_tokens, output_tokens + thoughts


def _describe_empty_response(response: Any, max_output_tokens: int) -> str:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    reason_text = str(reason or "")
    if "MAX_TOKENS" in reason_text:
        return (
            f"輸出在寫完 JSON 前就用完 {max_output_tokens} 個 token 額度，"
            "可能是文章過長或摘要要求太多，試試減少 --summaries 或降低 --summary-max"
        )
    if "SAFETY" in reason_text:
        return "回應被內容過濾擋下，沒有結構化輸出"
    feedback = getattr(response, "prompt_feedback", None)
    blocked = getattr(feedback, "block_reason", None) if feedback is not None else None
    if blocked:
        return f"回應被內容過濾擋下（{blocked}）"
    return f"回應中沒有結構化輸出（finish_reason={reason}）"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
