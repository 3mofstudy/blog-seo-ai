"""專案自訂例外。

所有對外拋出的錯誤都繼承 :class:`BlogSeoError`，CLI 最外層只需攔截這一個型別，
其餘情況一律讓例外浮出，避免吞掉非預期的錯誤。
"""

from __future__ import annotations


class BlogSeoError(Exception):
    """本工具所有自訂例外的基底。"""


class ConfigError(BlogSeoError):
    """設定或環境變數缺漏、格式錯誤。"""


class UnknownProviderError(ConfigError):
    """使用者指定了未註冊的 provider 或模型別名。"""


class MarkdownParseError(BlogSeoError):
    """Markdown 檔案無法讀取或解析。"""


class ImageNotFoundError(BlogSeoError):
    """Markdown 中引用的圖片在本地找不到。"""


class ImageProcessingError(BlogSeoError):
    """圖片讀取、解碼或縮圖失敗。"""


class AnalysisLoadError(BlogSeoError):
    """analysis JSON 讀不到、不是合法 JSON，或結構版本不相容。"""


class SelectionError(BlogSeoError):
    """review 階段無法產生有效的選擇結果。

    例如沒有任何可挑的候選，或檔名衝突未解決。
    """


class SelectionLoadError(BlogSeoError):
    """selection JSON 讀不到、不是合法 JSON，或結構版本不相容。"""


class ApplyError(BlogSeoError):
    """apply 階段無法安全套用。

    例如文章在分析之後被改過、圖片檔找不到，或目標檔名會覆蓋既有檔案。
    """


class ProviderError(BlogSeoError):
    """LLM provider 呼叫失敗。

    依專案硬性原則，個別模型失敗不可中斷整體流程，因此這個例外通常會在
    :mod:`blogseo.seo.analyzer` 被攔下並轉成結果 JSON 裡的 ``error`` 欄位。

    Attributes:
        provider: 發生錯誤的 provider 名稱。
        model: 實際使用的模型 id。
    """

    def __init__(self, provider: str, model: str, message: str) -> None:
        """初始化。

        Args:
            provider: provider 名稱，例如 ``anthropic``。
            model: 模型 id，例如 ``claude-sonnet-5``。
            message: 人類可讀的錯誤描述。
        """
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")
