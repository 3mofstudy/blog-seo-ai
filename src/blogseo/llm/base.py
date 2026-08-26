"""所有 LLM provider 的共同介面。

新增 provider 時繼承 :class:`BaseProvider`，實作兩個抽象方法，並到
:mod:`blogseo.llm.registry` 註冊。金鑰讀取統一由這裡的 ``__init__`` 處理，
子類別不得自行存取環境變數。
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Final

from blogseo.config import (
    ALT_MAX_CHARS,
    KEYWORD_COUNT,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
    get_api_key,
)
from blogseo.schemas.result import (
    DEFAULT_FIELDS,
    AnalysisField,
    ImageAnalysisResult,
    KeywordSummaryResult,
)

#: 模型 id 常見的日期後綴，例如 ``claude-sonnet-4-5-20250929``。
_DATE_SUFFIX: Final[re.Pattern[str]] = re.compile(r"-\d{8}$")


@dataclass
class UsageStats:
    """單一 provider 實例的累計用量。

    同一個 provider 實例會被多個工作執行緒共用（一篇文章的多張圖平行分析），
    因此累加動作需要加鎖。
    """

    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self, *, input_tokens: int, output_tokens: int, elapsed_ms: int) -> None:
        """累計一次成功呼叫。

        Args:
            input_tokens: 本次輸入 token 數。
            output_tokens: 本次輸出 token 數。
            elapsed_ms: 本次耗時（毫秒）。
        """
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.elapsed_ms += elapsed_ms

    def record_failure(self, *, elapsed_ms: int) -> None:
        """累計一次失敗呼叫。

        Args:
            elapsed_ms: 本次耗時（毫秒）。
        """
        with self._lock:
            self.calls += 1
            self.failed_calls += 1
            self.elapsed_ms += elapsed_ms


class BaseProvider(ABC):
    """LLM provider 的抽象基底。

    Attributes:
        name: provider 名稱，需與 :data:`blogseo.config.API_KEY_ENV_VARS` 的鍵一致。
        default_model: 未指定模型時使用的預設模型 id。
        pricing: 模型 id 對應到「每百萬 token 的輸入、輸出美元單價」。
            價格是寫死的，各家調價時要記得更新；查不到的模型費用會顯示為未知，
            而不是猜一個數字。
    """

    name: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    pricing: ClassVar[dict[str, tuple[float, float]]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        language: str = "zh",
        alt_language: str = "zh",
        fields: frozenset[AnalysisField] = DEFAULT_FIELDS,
        keyword_count: int = KEYWORD_COUNT,
        summary_count: int = SUMMARY_VARIANT_COUNT,
        summary_min_chars: int = SUMMARY_MIN_CHARS,
        summary_max_chars: int = SUMMARY_MAX_CHARS,
        alt_max_chars: int = ALT_MAX_CHARS,
        timeout: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        """初始化 provider。

        Args:
            api_key: 直接指定金鑰；未指定時從環境變數讀取。
            model: 指定模型 id；未指定時使用 :attr:`default_model`。
            language: 文章語言代碼，決定關鍵字與摘要的語言。
            alt_language: alt 文字的語言代碼。
            fields: 這次要產生的項目，決定送給模型的 schema 與 prompt 內容。
            keyword_count: 要產生幾個關鍵字。
            summary_count: 要產生幾個不同角度的摘要版本。
            summary_min_chars: 每則摘要的字元下限。
            summary_max_chars: 每則摘要的字元上限。
            alt_max_chars: alt 文字的字元上限。
            timeout: 單次請求逾時秒數。
            max_retries: SDK 層級的重試次數。

        Raises:
            ConfigError: 找不到金鑰。
        """
        self.model = model or self.default_model
        self.language = language
        self.alt_language = alt_language
        self.fields = fields
        self.keyword_count = keyword_count
        self.summary_count = summary_count
        self.summary_min_chars = summary_min_chars
        self.summary_max_chars = summary_max_chars
        self.alt_max_chars = alt_max_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = UsageStats()
        self._api_key = api_key or get_api_key(self.name)

    @property
    def text_model(self) -> str:
        """文章分析實際使用的模型 id。"""
        return self.model

    @property
    def image_model(self) -> str:
        """圖片分析實際使用的模型 id。"""
        return self.model

    @classmethod
    def price_per_mtok(cls, model: str) -> tuple[float, float] | None:
        """查詢某個模型每百萬 token 的輸入與輸出單價。

        會先試完整 id，再去掉 ``-YYYYMMDD`` 日期後綴重試，這樣
        ``claude-sonnet-4-5-20250929`` 也能對到 ``claude-sonnet-4-5``。

        Args:
            model: 模型 id。

        Returns:
            ``(輸入單價, 輸出單價)``；價目表裡沒有就回傳 ``None``。
        """
        if model in cls.pricing:
            return cls.pricing[model]
        return cls.pricing.get(_DATE_SUFFIX.sub("", model))

    def estimate_cost_usd(self) -> float | None:
        """依累計用量估算本次花費。

        Returns:
            估算的美元金額；模型不在價目表中時為 ``None``。
        """
        prices = self.price_per_mtok(self.model)
        if prices is None:
            return None
        input_price, output_price = prices
        return (
            self.usage.input_tokens * input_price
            + self.usage.output_tokens * output_price
        ) / 1_000_000

    def _limit_text_result(self, result: KeywordSummaryResult) -> KeywordSummaryResult:
        """依本次設定裁切關鍵字與摘要數量。模型多給的直接捨去，不足時不補。"""
        updates: dict[str, list[str]] = {}
        if len(result.keywords) > self.keyword_count:
            updates["keywords"] = result.keywords[: self.keyword_count]
        if len(result.summaries) > self.summary_count:
            updates["summaries"] = result.summaries[: self.summary_count]
        if not updates:
            return result
        return result.model_copy(update=updates)

    @abstractmethod
    def analyze_text(self, content: str) -> KeywordSummaryResult:
        """分析文章內容，產生關鍵字與摘要。

        Args:
            content: 文章正文。

        Returns:
            關鍵字與摘要。

        Raises:
            ProviderError: 呼叫失敗或回傳內容無法解析。
        """

    @abstractmethod
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

    def __repr__(self) -> str:
        """回傳便於除錯的字串表示。"""
        return f"{type(self).__name__}(model={self.model!r})"
