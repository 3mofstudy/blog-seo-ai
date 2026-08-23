"""使用者可調預設值。

所有人工可改的預設都寫在 ``blogseo.json``。沒開 CLI 也能直接改這個檔；
程式選單「其他設定」則是把變更寫回同一份 JSON。

尋找順序：

1. 環境變數 ``BLOGSEO_SETTINGS`` 指定的路徑
2. 從目前工作目錄往上找 ``blogseo.json``（部落格專案專用）
3. 使用者設定目錄的 ``blogseo.json``（全域安裝後的預設位置）
4. 都沒有就用內建預設（與 :mod:`blogseo.config` 的常數一致）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blogseo.config import (
    ALT_MAX_CHARS,
    CONTEXT_RADIUS,
    DEFAULT_MODEL_TOKEN,
    DEFAULT_OUTPUT_DIR,
    KEYWORD_COUNT,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
    USD_TO_TWD_RATE,
    user_config_dir,
)
from blogseo.errors import ConfigError

#: 設定檔檔名，放在專案根目錄或任何工作目錄的上層。
SETTINGS_FILENAME: Final[str] = "blogseo.json"

#: 可用環境變數覆寫設定檔路徑。
SETTINGS_PATH_ENV: Final[str] = "BLOGSEO_SETTINGS"

#: 寫進 JSON 給人看的說明。程式讀檔時會忽略未知欄位。
_README: Final[str] = (
    "可直接編輯此檔來調整預設值，不必開啟 CLI。"
    "未寫的欄位會用內建預設。"
    "也可在程式裡選「6. 其他設定」寫回這裡。"
    "模型寫法與 --models 相同，例如 hf、claude、hf:Qwen/Qwen3-8B。"
)


class ModelsSettings(BaseModel):
    """分析時使用的模型別名。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    text: str = Field(
        default=DEFAULT_MODEL_TOKEN,
        description="關鍵字與摘要使用的模型，例如 hf:Qwen/Qwen3-4B-Instruct-2507",
    )
    image: str = Field(
        default=DEFAULT_MODEL_TOKEN,
        description="圖片檔名與 alt 使用的模型，例如 hf:zai-org/GLM-4.6V-Flash",
    )


class KeywordsSettings(BaseModel):
    """關鍵字產出規則。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    count: int = Field(default=KEYWORD_COUNT, ge=1, le=20)


class SummarySettings(BaseModel):
    """摘要產出規則。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    min_chars: int = Field(default=SUMMARY_MIN_CHARS, ge=10)
    max_chars: int = Field(default=SUMMARY_MAX_CHARS, ge=20)
    count: int = Field(default=SUMMARY_VARIANT_COUNT, ge=1, le=5)

    @model_validator(mode="after")
    def min_must_be_below_max(self) -> SummarySettings:
        """摘要下限必須小於上限。"""
        if self.min_chars >= self.max_chars:
            raise ValueError(
                f"summary.min_chars（{self.min_chars}）必須小於 summary.max_chars（{self.max_chars}）"
            )
        return self


class AltSettings(BaseModel):
    """圖片 alt 文字規則。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    max_chars: int = Field(default=ALT_MAX_CHARS, ge=10)
    language: str = Field(default="auto")


class AnalyzeSettings(BaseModel):
    """analyze 階段的執行參數。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    concurrency: int = Field(default=4, ge=1, le=16)
    timeout_seconds: float = Field(default=90.0, ge=5.0)
    max_images: int | None = Field(default=None, ge=1)
    context_radius: int = Field(default=CONTEXT_RADIUS, ge=50)


class OutputSettings(BaseModel):
    """中間產物的落地位置。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    dir: str = Field(default=str(DEFAULT_OUTPUT_DIR))


class CostSettings(BaseModel):
    """花費估算。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    usd_to_twd_rate: float = Field(default=USD_TO_TWD_RATE, gt=0)


class ApplySettings(BaseModel):
    """套用到 Markdown 時的行為。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    backup: bool = False
    write_front_matter: bool = True
    keywords_key: str = Field(default="keywords", min_length=1)
    summary_key: str = Field(default="description", min_length=1)


class AppSettings(BaseModel):
    """``blogseo.json`` 的完整結構。缺漏欄位用內建預設補齊。"""

    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, validate_assignment=True
    )

    schema_version: str = "1.0"
    readme: str = Field(default=_README, alias="_readme")
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    keywords: KeywordsSettings = Field(default_factory=KeywordsSettings)
    summary: SummarySettings = Field(default_factory=SummarySettings)
    alt: AltSettings = Field(default_factory=AltSettings)
    analyze: AnalyzeSettings = Field(default_factory=AnalyzeSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    apply: ApplySettings = Field(default_factory=ApplySettings)

    def to_json(self) -> str:
        """序列化成適合手改的 JSON 字串。"""
        payload = self.model_dump(by_alias=True)
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @property
    def output_dir(self) -> Path:
        """中間產物目錄。相對路徑以目前工作目錄為準。"""
        return Path(self.output.dir)


def find_settings_path(*, start: Path | None = None) -> Path | None:
    """尋找設定檔路徑；找不到時回傳 ``None``。

    Args:
        start: 搜尋起點；預設為目前工作目錄。

    Returns:
        既有設定檔路徑，或 ``None``。
    """
    env = os.environ.get(SETTINGS_PATH_ENV, "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / SETTINGS_FILENAME
        if candidate.is_file():
            return candidate

    user_path = user_config_dir() / SETTINGS_FILENAME
    if user_path.is_file():
        return user_path
    return None


def default_save_path(*, start: Path | None = None) -> Path:
    """決定要把設定寫到哪裡。

    已有設定檔就覆寫它；否則寫到使用者設定目錄，讓全域指令不必依賴工作目錄。
    """
    existing = find_settings_path(start=start)
    if existing is not None:
        return existing
    return user_config_dir() / SETTINGS_FILENAME


def load_settings(path: Path | None = None) -> AppSettings:
    """讀取設定。檔案不存在時回傳內建預設，不會報錯。

    Args:
        path: 指定檔案；未指定則依 :func:`find_settings_path` 尋找。

    Returns:
        驗證後的設定。缺漏欄位已用預設補齊。

    Raises:
        ConfigError: 檔案存在但不是合法 JSON，或欄位型別／範圍不對。
    """
    target = path if path is not None else find_settings_path()
    if target is None:
        return AppSettings()
    return load_settings_file(target)


def load_settings_file(path: Path) -> AppSettings:
    """讀取並驗證一份設定檔。

    Args:
        path: JSON 檔路徑。

    Returns:
        驗證後的設定。

    Raises:
        ConfigError: 讀不到、不是物件，或欄位不合法。
    """
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except FileNotFoundError as exc:
        raise ConfigError(f"找不到設定檔：{path}") from exc
    except OSError as exc:
        raise ConfigError(f"無法讀取設定檔 {path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} 不是合法的 JSON：{exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigError(f"{path} 的頂層必須是物件")

    try:
        return AppSettings.model_validate(payload)
    except Exception as exc:
        raise ConfigError(f"{path} 的設定不合法：{exc}") from exc


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    """把設定寫回 JSON。

    Args:
        settings: 要落地的設定。
        path: 目標路徑；未指定則覆寫現有檔，或寫到工作目錄。

    Returns:
        實際寫入的路徑。

    Raises:
        ConfigError: 無法寫入。
    """
    target = path if path is not None else default_save_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(settings.to_json(), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"無法寫入設定檔 {target}：{exc}") from exc
    return target


def settings_from_mapping(payload: dict[str, Any]) -> AppSettings:
    """從部分欄位組出設定，缺的用預設。供測試使用。"""
    return AppSettings.model_validate(payload)
