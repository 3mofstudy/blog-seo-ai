"""集中管理環境變數、金鑰與全域常數。

金鑰只在這裡讀取，provider 實作不得自行碰觸 ``os.environ``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import find_dotenv, load_dotenv

from blogseo.errors import ConfigError

#: 每個 provider 可接受的環境變數名稱，依序嘗試，先找到的優先。
API_KEY_ENV_VARS: Final[dict[str, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

#: 送進 LLM 前，圖片長邊縮到的上限（px）。Claude 對超過 1568px 的圖會自行縮放，
#: 先在本地縮好可以省下上傳流量與 token。
MAX_IMAGE_EDGE: Final[int] = 1568

#: 單張圖片編碼後的大小上限（bytes），超過就降品質重新編碼。
MAX_IMAGE_BYTES: Final[int] = 4 * 1024 * 1024

#: 要求模型產生的關鍵字數量。
KEYWORD_COUNT: Final[int] = 5

#: meta description 的字元上限。字元一律以 Python ``len()`` 計算，
#: 中文字、標點、空白各算一個。Google 桌面版搜尋結果約在 150 到 160 之間截斷。
SUMMARY_MAX_CHARS: Final[int] = 150

#: meta description 的字元下限。太短的摘要撐不滿搜尋結果版位，資訊量也不足。
SUMMARY_MIN_CHARS: Final[int] = 60

#: 每個模型要產生幾個不同角度的摘要版本供挑選。
SUMMARY_VARIANT_COUNT: Final[int] = 3

#: alt 文字的字元上限。
ALT_MAX_CHARS: Final[int] = 125

#: 擷取圖片上下文時，前後各取的字元數。
CONTEXT_RADIUS: Final[int] = 600

#: 費用換算用的美元兌新台幣匯率預設值（2026-08-21 收盤 31.848）。
#: 匯率會變動，需要精確數字時用 ``--twd-rate`` 覆寫。
USD_TO_TWD_RATE: Final[float] = 31.8

#: analyze 結果的預設輸出目錄（相對於執行時的工作目錄）。
DEFAULT_OUTPUT_DIR: Final[Path] = Path("output")

_dotenv_loaded = False


def load_environment() -> None:
    """載入 ``.env``（只會實際執行一次）。

    從目前工作目錄往上尋找 ``.env``，找不到就靜默略過，讓使用者仍可用系統
    環境變數提供金鑰。
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    _dotenv_loaded = True


def get_api_key(provider: str) -> str:
    """取得指定 provider 的 API key。

    Args:
        provider: provider 名稱，需為 :data:`API_KEY_ENV_VARS` 的鍵。

    Returns:
        金鑰字串。

    Raises:
        ConfigError: 找不到對應的環境變數，或 provider 名稱未知。
    """
    load_environment()
    try:
        candidates = API_KEY_ENV_VARS[provider]
    except KeyError as exc:
        raise ConfigError(f"未知的 provider：{provider}") from exc

    for name in candidates:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    joined = " 或 ".join(candidates)
    raise ConfigError(
        f"找不到 {provider} 的 API key，請在 .env 或環境變數設定 {joined}"
    )


def has_api_key(provider: str) -> bool:
    """檢查某個 provider 的金鑰是否已設定。

    Args:
        provider: provider 名稱。

    Returns:
        金鑰存在時為 ``True``。
    """
    try:
        get_api_key(provider)
    except ConfigError:
        return False
    return True
