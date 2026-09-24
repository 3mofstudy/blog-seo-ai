"""集中管理環境變數、金鑰與全域常數。

金鑰只在這裡讀取，provider 實作不得自行碰觸 ``os.environ``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import find_dotenv, load_dotenv

from blogseo.errors import ConfigError

#: 使用者層級設定目錄名稱（位於 APPDATA 或 XDG_CONFIG_HOME 底下）。
_USER_CONFIG_DIRNAME: Final[str] = "blogseo"

#: 每個 provider 可接受的環境變數名稱，依序嘗試，先找到的優先。
API_KEY_ENV_VARS: Final[dict[str, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"),
}

#: 設定選單顯示的供應商名稱。
PROVIDER_DISPLAY_NAMES: Final[dict[str, str]] = {
    "huggingface": "Hugging Face",
    "anthropic": "Claude",
    "openai": "OpenAI",
    "gemini": "Gemini",
}

#: 金鑰選單的供應商順序（預設的 Hugging Face 放最前面）。
PROVIDER_KEY_ORDER: Final[tuple[str, ...]] = (
    "huggingface",
    "anthropic",
    "openai",
    "gemini",
)

#: 已設定金鑰時，畫面只顯示開頭幾碼作為識別。
API_KEY_PREFIX_LEN: Final[int] = 10

#: CLI ``analyze`` 未指定 ``--models`` 時使用的別名。走 Hugging Face 免費額度。
DEFAULT_MODEL_TOKEN: Final[str] = "hf"

#: 送進 LLM 前，圖片長邊縮到的上限（px）。Claude 對超過 1568px 的圖會自行縮放，
#: 先在本地縮好可以省下上傳流量與 token。
MAX_IMAGE_EDGE: Final[int] = 1568

#: 單張圖片編碼後的大小上限（bytes），超過就降品質重新編碼。
MAX_IMAGE_BYTES: Final[int] = 4 * 1024 * 1024

#: 要求模型產生的關鍵字數量。
KEYWORD_COUNT: Final[int] = 5

#: 每個關鍵字的字元上限。字元一律以 Python ``len()`` 計算，
#: 中文字、英文字母、數字各算一個。
KEYWORD_MAX_CHARS: Final[int] = 6

#: 方格子列表摘要的字元上限。字元一律以 Python ``len()`` 計算，
#: 中文字、標點、空白各算一個。超過會在沙龍列表被截斷。
SUMMARY_MAX_CHARS: Final[int] = 150

#: 方格子列表摘要的字元下限。少於這個長度通常講不完文章重點。
SUMMARY_MIN_CHARS: Final[int] = 100

#: 每個模型要產生幾個不同角度的摘要版本供挑選。
SUMMARY_VARIANT_COUNT: Final[int] = 3

#: alt 文字的字元上限。螢幕閱讀器是逐字唸出 alt 的，過長的描述聽起來很痛苦；
#: 圖片本身的細節應該寫在正文，alt 只要一句話交代這張圖在講什麼。
#: 需要更長的描述時用 ``--alt-max`` 放寬。
ALT_MAX_CHARS: Final[int] = 30

#: 擷取圖片上下文時，前後各取的字元數。
CONTEXT_RADIUS: Final[int] = 600

#: 費用換算用的美元兌新台幣匯率預設值（2026-08-21 收盤 31.848）。
#: 匯率會變動，需要精確數字時用 ``--twd-rate`` 覆寫。
USD_TO_TWD_RATE: Final[float] = 31.8

#: analyze 結果的預設輸出目錄（相對於執行時的工作目錄）。
DEFAULT_OUTPUT_DIR: Final[Path] = Path("output")

_dotenv_loaded = False


def user_config_dir() -> Path:
    """使用者層級設定目錄。

    套件安裝成全域指令後，金鑰與 ``blogseo.json`` 預設放這裡，與工作目錄無關。

    - Windows：``%APPDATA%\\blogseo``
    - 其他：``$XDG_CONFIG_HOME/blogseo`` 或 ``~/.config/blogseo``
    """
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata) / _USER_CONFIG_DIRNAME
        return Path.home() / "AppData" / "Roaming" / _USER_CONFIG_DIRNAME
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg) / _USER_CONFIG_DIRNAME
    return Path.home() / ".config" / _USER_CONFIG_DIRNAME


def user_env_path() -> Path:
    """使用者層級 ``.env`` 路徑。"""
    return user_config_dir() / ".env"


def load_environment() -> None:
    """載入 ``.env``（只會實際執行一次）。

    已存在的環境變數不會被覆蓋。載入順序：

    1. 從目前工作目錄往上找專案 ``.env``（部落格專案專用金鑰優先）
    2. 使用者設定目錄的 ``.env``（全域安裝後的預設位置）

    兩處都沒有就略過，讓系統環境變數仍可提供金鑰。
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    project_env = find_dotenv(usecwd=True)
    if project_env:
        load_dotenv(project_env)
    user_env = user_env_path()
    if user_env.is_file():
        load_dotenv(user_env)
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
        f"找不到 {provider} 的 API key，請在環境變數、專案 .env，"
        f"其他設定的供應商金鑰，或 {user_env_path()} 設定 {joined}"
    )


def peek_api_key(provider: str) -> str | None:
    """讀取金鑰；尚未設定時回傳 ``None``，不丟例外。

    Args:
        provider: provider 名稱，需為 :data:`API_KEY_ENV_VARS` 的鍵。

    Returns:
        金鑰字串，或 ``None``。

    Raises:
        ConfigError: provider 名稱未知。
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
    return None


def mask_api_key(key: str | None, *, prefix_len: int = API_KEY_PREFIX_LEN) -> str:
    """只留下金鑰開頭幾碼，供畫面識別。未設定則空白。"""
    if not key:
        return ""
    return key[:prefix_len]


def save_api_key(provider: str, key: str) -> Path:
    """把金鑰寫進使用者 ``.env``，並立刻套用到目前行程。

    不會寫進 ``blogseo.json``。空白金鑰會拒絕，以免誤刪。

    Args:
        provider: provider 名稱。
        key: 金鑰內容。

    Returns:
        實際寫入的 ``.env`` 路徑。

    Raises:
        ConfigError: provider 未知、金鑰空白，或無法寫檔。
    """
    cleaned = key.strip()
    if not cleaned:
        raise ConfigError("金鑰不能是空白")
    try:
        names = API_KEY_ENV_VARS[provider]
    except KeyError as exc:
        raise ConfigError(f"未知的 provider：{provider}") from exc
    canonical = names[0]
    path = user_env_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _upsert_env_file(path, canonical, cleaned)
    except OSError as exc:
        raise ConfigError(f"無法寫入金鑰檔 {path}：{exc}") from exc
    os.environ[canonical] = cleaned
    return path


def _format_env_value(value: str) -> str:
    """必要時替 .env 值加上引號。"""
    if any(char in value for char in " \t#\"'"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _upsert_env_file(path: Path, name: str, value: str) -> None:
    """在 .env 裡新增或覆寫一個變數，保留其他行。"""
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)
    else:
        lines = []
    assignment = f"{name}={_format_env_value(value)}\n"
    prefix = f"{name}="
    export_prefix = f"export {name}="
    replaced = False
    output: list[str] = []
    for line in lines:
        check = line.lstrip(" \t")
        if check.startswith(prefix) or check.startswith(export_prefix):
            output.append(assignment)
            replaced = True
            continue
        output.append(line if line.endswith("\n") else f"{line}\n")
    if not replaced:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        output.append(assignment)
    path.write_text("".join(output), encoding="utf-8")
