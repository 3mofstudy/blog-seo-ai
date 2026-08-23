"""Anthropic (Claude) provider。

使用 SDK 1.x 的 ``messages.parse`` 原生 structured output，直接把 Pydantic
model 當成輸出 schema，不需要自己解析 JSON 或處理模型回傳額外文字的情況。
"""

from __future__ import annotations

import base64
import time
from typing import Any, ClassVar, TypeVar

import anthropic
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


class AnthropicProvider(BaseProvider):
    """呼叫 Claude 系列模型。"""

    name: ClassVar[str] = "anthropic"
    default_model: ClassVar[str] = "claude-sonnet-5"

    # 每百萬 token 的（輸入, 輸出）美元單價，取自官方價目表，2026-08-21 查核。
    # Anthropic 調價時要更新這裡，查不到的模型會顯示費用未知而不是估錯。
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "claude-fable-5": (10.0, 50.0),
        "claude-mythos-5": (10.0, 50.0),
        "claude-opus-5": (5.0, 25.0),
        "claude-opus-4-8": (5.0, 25.0),
        "claude-opus-4-7": (5.0, 25.0),
        "claude-opus-4-6": (5.0, 25.0),
        "claude-opus-4-5": (5.0, 25.0),
        "claude-sonnet-5": (2.0, 10.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
    }

    _TEXT_MAX_TOKENS: ClassVar[int] = 2048
    _IMAGE_MAX_TOKENS: ClassVar[int] = 1024

    # Claude Sonnet 5 預設會做 extended thinking，實測光思考就吃掉三千多個
    # output token、單次呼叫拉長到 30 秒，而且會把 max_tokens 用完導致沒有
    # 額度輸出 JSON。抽關鍵字與寫摘要不需要深度推理，實測 low 的品質與
    # medium 相當但快四倍，因此明確關掉。
    _THINKING_EFFORT: ClassVar[str] = "low"

    def __init__(self, **kwargs: Any) -> None:
        """初始化 provider 並建立 SDK client。

        Args:
            **kwargs: 傳給 :class:`~blogseo.llm.base.BaseProvider` 的參數。
        """
        super().__init__(**kwargs)
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _parse(
        self,
        *,
        system: str,
        blocks: list[dict[str, Any]],
        output_format: type[ResultT],
        max_tokens: int,
    ) -> ResultT:
        """送出一次請求並取回結構化結果。

        Args:
            system: system prompt。
            blocks: user 訊息的內容區塊。
            output_format: 期望的輸出結構。
            max_tokens: 輸出 token 上限。

        Returns:
            解析後的結果物件。

        Raises:
            ProviderError: API 呼叫失敗、逾時，或回傳內容不符合結構。
        """
        started = time.perf_counter()
        try:
            message = self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": blocks}],
                output_format=output_format,
                output_config={"effort": self._THINKING_EFFORT},
            )
        except anthropic.AnthropicError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(self.name, self.model, str(exc)) from exc
        except ValidationError as exc:
            self.usage.record_failure(elapsed_ms=_elapsed_ms(started))
            raise ProviderError(
                self.name, self.model, f"回傳內容不符合預期結構：{exc}"
            ) from exc

        elapsed = _elapsed_ms(started)
        parsed = _extract_parsed(message)
        if parsed is None:
            self.usage.record_failure(elapsed_ms=elapsed)
            if message.stop_reason == "max_tokens":
                raise ProviderError(
                    self.name,
                    self.model,
                    f"輸出在寫完 JSON 前就用完 {max_tokens} 個 token 額度，"
                    "可能是文章過長或摘要要求太多，試試減少 --summaries 或降低 --summary-max",
                )
            raise ProviderError(
                self.name,
                self.model,
                f"回應中沒有結構化輸出（stop_reason={message.stop_reason}）",
            )

        self.usage.record_success(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
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
            system=text_system_prompt(
                self.language,
                include_keywords=AnalysisField.KEYWORDS in self.fields,
                include_summaries=AnalysisField.SUMMARY in self.fields,
                keyword_count=self.keyword_count,
                summary_count=self.summary_count,
                summary_min_chars=self.summary_min_chars,
                summary_max_chars=self.summary_max_chars,
            ),
            blocks=[{"type": "text", "text": text_user_prompt(None, content)}],
            output_format=text_output_schema(self.fields),
            max_tokens=self._TEXT_MAX_TOKENS,
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
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded},
            },
            {"type": "text", "text": image_user_prompt(context, original_filename)},
        ]
        return self._parse(
            system=image_system_prompt(
                self.alt_language, alt_max_chars=self.alt_max_chars
            ),
            blocks=blocks,
            output_format=ImageAnalysisResult,
            max_tokens=self._IMAGE_MAX_TOKENS,
        )


def _elapsed_ms(started: float) -> int:
    """計算從 ``started`` 到現在經過的毫秒數。

    Args:
        started: :func:`time.perf_counter` 的起始值。

    Returns:
        經過的毫秒數。
    """
    return int((time.perf_counter() - started) * 1000)


def _extract_parsed(message: Any) -> Any:
    """從回應的內容區塊中取出結構化輸出。

    Args:
        message: SDK 回傳的 ``ParsedMessage``。

    Returns:
        解析後的物件；找不到時為 ``None``。
    """
    for block in message.content:
        parsed = getattr(block, "parsed_output", None)
        if parsed is not None:
            return parsed
    return None
