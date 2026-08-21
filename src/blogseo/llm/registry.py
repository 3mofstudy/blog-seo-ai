"""Provider 註冊表。

把使用者在 CLI 打的模型別名對應到實際的 provider 類別。採用延遲匯入，
因此尚未安裝的 SDK 不會影響其他 provider 的使用。

別名可用 ``別名:模型id`` 的形式覆寫模型，例如 ``claude:claude-opus-5``，
如此也能在同一次執行中比較同一家的不同模型。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Final

from blogseo.errors import ConfigError, UnknownProviderError
from blogseo.llm.base import BaseProvider


@dataclass(frozen=True)
class ProviderSpec:
    """一個 provider 的註冊資訊。

    Attributes:
        key: provider 正式名稱，與金鑰設定的鍵一致。
        module: 實作所在的模組路徑。
        class_name: 實作的類別名稱。
        package: 對應的 PyPI 套件名稱，用於產生安裝提示。
        aliases: 使用者可以在 CLI 輸入的別名。
    """

    key: str
    module: str
    class_name: str
    package: str
    aliases: tuple[str, ...]


_SPECS: Final[tuple[ProviderSpec, ...]] = (
    ProviderSpec(
        key="anthropic",
        module="blogseo.llm.anthropic",
        class_name="AnthropicProvider",
        package="anthropic",
        aliases=("claude", "anthropic"),
    ),
    ProviderSpec(
        key="huggingface",
        module="blogseo.llm.huggingface",
        class_name="HuggingFaceProvider",
        package="huggingface_hub",
        aliases=("hf", "huggingface", "qwen"),
    ),
    ProviderSpec(
        key="openai",
        module="blogseo.llm.openai",
        class_name="OpenAIProvider",
        package="openai",
        aliases=("gpt", "openai"),
    ),
    ProviderSpec(
        key="gemini",
        module="blogseo.llm.gemini",
        class_name="GeminiProvider",
        package="google-genai",
        aliases=("gemini", "google"),
    ),
)

_ALIAS_INDEX: Final[dict[str, ProviderSpec]] = {
    alias: spec for spec in _SPECS for alias in spec.aliases
}


def known_aliases() -> list[str]:
    """列出所有可用的別名。

    Returns:
        別名清單，依註冊順序。
    """
    return list(_ALIAS_INDEX)


def parse_token(token: str) -> tuple[ProviderSpec, str | None]:
    """解析 ``別名`` 或 ``別名:模型id`` 形式的字串。

    Args:
        token: 使用者輸入的模型指定字串。

    Returns:
        對應的 spec，以及模型覆寫值（未指定時為 ``None``）。

    Raises:
        UnknownProviderError: 別名未註冊。
    """
    alias, _, model = token.strip().partition(":")
    alias = alias.strip().lower()
    model = model.strip()

    spec = _ALIAS_INDEX.get(alias)
    if spec is None:
        available = "、".join(known_aliases())
        raise UnknownProviderError(f"未知的模型別名「{alias}」，可用的有：{available}")
    return spec, model or None


def create_provider(token: str, **kwargs: Any) -> BaseProvider:
    """依別名建立 provider 實例。

    Args:
        token: ``別名`` 或 ``別名:模型id``。
        **kwargs: 傳給 provider 建構子的參數。

    Returns:
        provider 實例。

    Raises:
        UnknownProviderError: 別名未註冊。
        ConfigError: 對應的 SDK 尚未安裝、實作尚未完成，或金鑰缺漏。
    """
    spec, model_override = parse_token(token)

    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise ConfigError(
            f"{spec.key} 尚未可用：{exc}。請先安裝對應 SDK：uv add {spec.package}"
        ) from exc

    provider_class = getattr(module, spec.class_name, None)
    if provider_class is None:
        raise ConfigError(f"{spec.key} 的實作尚未完成（找不到 {spec.class_name}）")

    if model_override is not None:
        kwargs["model"] = model_override
    return provider_class(**kwargs)


def parse_model_tokens(raw: str) -> list[str]:
    """把 CLI 的 ``--models`` 參數拆成 token 清單。

    Args:
        raw: 以逗號分隔的字串。

    Returns:
        去除空白與重複後的 token 清單，保留輸入順序。

    Raises:
        ConfigError: 未指定任何模型。
    """
    tokens: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        alias, separator, model = piece.partition(":")
        alias = alias.strip().lower()
        if not alias:
            continue
        # 只把別名折成小寫。Hugging Face 的模型 id 大小寫有意義，
        # 例如 Qwen/Qwen2.5-VL-7B-Instruct。
        token = f"{alias}:{model.strip()}" if separator else alias
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        raise ConfigError("請至少指定一個模型，例如 --models hf")
    return tokens
