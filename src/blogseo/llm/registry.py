"""Provider 註冊表。

把使用者在 CLI 打的模型別名對應到實際的 provider 類別。採用延遲匯入，
因此尚未安裝的 SDK 不會影響其他 provider 的使用。

別名可用 ``別名:模型id`` 的形式覆寫模型，例如 ``claude:claude-opus-5``，
如此也能在同一次執行中比較同一家的不同模型。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Final, Literal

from blogseo.errors import ConfigError, UnknownProviderError
from blogseo.llm.base import BaseProvider

ModelRole = Literal["text", "image"]

_PROVIDER_LABELS: Final[dict[str, str]] = {
    "anthropic": "Claude",
    "huggingface": "Hugging Face",
    "openai": "OpenAI",
    "gemini": "Gemini",
}


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


@dataclass(frozen=True)
class ModelPreset:
    """設定選單上的一筆已實作預設模型。

    Attributes:
        token: 寫進設定檔的字串，含別名與型號，例如 ``hf:Qwen/Qwen3-4B-Instruct-2507``。
        provider_key: provider 正式名稱。
        provider_label: 給人看的供應商名稱。
        alias: 第一個別名（選單用，避免 hf / huggingface 重複）。
        model_id: 實際模型型號。
        note: 補充說明，例如免費額度。
    """

    token: str
    provider_key: str
    provider_label: str
    alias: str
    model_id: str
    note: str = ""


def _load_provider_class(spec: ProviderSpec) -> type[BaseProvider] | None:
    """載入已實作的 provider 類別；模組或類別不存在時回傳 ``None``。"""
    try:
        module = importlib.import_module(spec.module)
    except ImportError:
        return None
    provider_class = getattr(module, spec.class_name, None)
    if provider_class is None:
        return None
    return provider_class


def _default_model_id(spec: ProviderSpec, role: ModelRole) -> str:
    """讀取 provider 類別上的預設型號。尚未實作時回傳空字串。"""
    provider_class = _load_provider_class(spec)
    if provider_class is None:
        return ""
    if role == "image":
        return getattr(provider_class, "default_image_model", None) or provider_class.default_model
    return provider_class.default_model


def list_model_presets(role: ModelRole) -> list[ModelPreset]:
    """列出已實作 provider 的預設型號，每個供應商一筆。

    設定選單用這個清單，而不是把 hf、huggingface、qwen 等別名全列出來。
    """
    presets: list[ModelPreset] = []
    for spec in _SPECS:
        model_id = _default_model_id(spec, role)
        if not model_id:
            continue
        alias = spec.aliases[0]
        note = "免費額度" if spec.key == "huggingface" else ""
        presets.append(
            ModelPreset(
                token=f"{alias}:{model_id}",
                provider_key=spec.key,
                provider_label=_PROVIDER_LABELS.get(spec.key, spec.key),
                alias=alias,
                model_id=model_id,
                note=note,
            )
        )
    return presets


def format_model_token(token: str, *, role: ModelRole = "text") -> str:
    """把設定值顯示成「別名（型號）」。

    只寫 ``hf`` 時會補上該用途的預設型號，讓選單看得出實際用哪一顆。
    """
    spec, model = parse_token(token)
    resolved = model or _default_model_id(spec, role) or "（尚未實作）"
    alias = spec.aliases[0]
    return f"{alias}（{resolved}）"


def token_matches_preset(token: str, preset: ModelPreset, *, role: ModelRole) -> bool:
    """判斷目前設定是否屬於這個供應商（型號可以不同）。"""
    del role
    spec, _model = parse_token(token)
    return spec.key == preset.provider_key


def suggested_model_id(token: str, preset: ModelPreset, *, role: ModelRole) -> str:
    """輸入框的預設值：已選這家就沿用目前型號，否則用內建預設。"""
    spec, model = parse_token(token)
    if spec.key == preset.provider_key and model:
        return model
    return preset.model_id


def normalize_typed_model_id(raw: str, preset: ModelPreset) -> str:
    """把使用者打的型號整理成純模型 id。

    可只打 ``claude-opus-5``，也可打 ``claude:claude-opus-5``。
    若冠上別家供應商的別名則拒絕。
    """
    text = raw.strip()
    if not text:
        raise ConfigError("請輸入型號名稱")
    if ":" not in text:
        return text
    spec, model = parse_token(text)
    if spec.key != preset.provider_key:
        raise ConfigError(
            f"這是 {preset.provider_label} 的型號，不能填 {spec.aliases[0]} 的值"
        )
    if not model:
        raise ConfigError("請輸入型號名稱")
    return model


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


def create_provider(token: str, *, role: str = "both", **kwargs: Any) -> BaseProvider:
    """依別名建立 provider 實例。

    Args:
        token: ``別名`` 或 ``別名:模型id``。
        role: ``text``、``image`` 或 ``both``。Hugging Face 的模型覆寫
            在 ``image`` 時套到圖片模型，其餘套到文字模型。
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
        if role == "image" and spec.key == "huggingface":
            kwargs.setdefault("image_model", model_override)
        else:
            kwargs.setdefault("model", model_override)
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
