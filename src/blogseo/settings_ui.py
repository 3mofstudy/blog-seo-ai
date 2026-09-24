"""設定選單：把預設值寫進 ``blogseo.json``，金鑰寫進使用者 ``.env``。"""

from __future__ import annotations

import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.table import Table

from blogseo.config import (
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_KEY_ORDER,
    mask_api_key,
    peek_api_key,
    save_api_key,
    user_env_path,
)
from blogseo.display import ask_choice, console, error_console
from blogseo.errors import BlogSeoError, ConfigError, UnknownProviderError
from blogseo.llm.registry import (
    ModelPreset,
    ModelRole,
    format_model_token,
    list_model_presets,
    normalize_typed_model_id,
    parse_model_tokens,
    suggested_model_id,
    token_matches_preset,
)
from blogseo.settings import (
    AppSettings,
    default_save_path,
    find_settings_path,
    load_settings,
    save_settings,
)


def _print_settings_overview(settings: AppSettings) -> None:
    """列出目前的預設值。"""
    table = Table(title="目前預設值", title_justify="left", header_style="bold cyan")
    table.add_column("項目", no_wrap=True)
    table.add_column("值", overflow="fold")
    table.add_row("關鍵字／摘要模型", format_model_token(settings.models.text, role="text"))
    table.add_row("圖片辨識模型", format_model_token(settings.models.image, role="image"))
    table.add_row(
        "關鍵字",
        f"{settings.keywords.count} 個，每個最多 {settings.keywords.max_chars} 字",
    )
    table.add_row(
        "摘要字數",
        f"{settings.summary.min_chars}–{settings.summary.max_chars}　{settings.summary.count} 則",
    )
    table.add_row("alt 字數上限", str(settings.alt.max_chars))
    table.add_row("alt 語言", settings.alt.language)
    table.add_row("平行呼叫", str(settings.analyze.concurrency))
    table.add_row("逾時秒數", str(settings.analyze.timeout_seconds))
    table.add_row(
        "最多分析圖片",
        "不限" if settings.analyze.max_images is None else str(settings.analyze.max_images),
    )
    table.add_row("輸出目錄", settings.output.dir)
    table.add_row("匯率 USD→TWD", str(settings.cost.usd_to_twd_rate))
    table.add_row(
        "寫入 front matter",
        "是" if settings.apply.write_front_matter else "否",
    )
    table.add_row("關鍵字欄位", settings.apply.keywords_key)
    table.add_row("摘要欄位", settings.apply.summary_key)
    table.add_row("套用前備份", "是" if settings.apply.backup else "否")
    for provider in PROVIDER_KEY_ORDER:
        label = f"{PROVIDER_DISPLAY_NAMES[provider]} 金鑰"
        table.add_row(label, mask_api_key(peek_api_key(provider)))
    console.print(table)


def _prompt_model_id_for_provider(
    preset: ModelPreset, current: str, *, role: ModelRole
) -> str | None:
    """進入供應商後請使用者輸入型號；``0`` 回到供應商清單。"""
    suggested = suggested_model_id(current, preset, role=role)
    lines = [
        f"內建預設：{preset.model_id}",
        "可改成同一家的其他型號，不限於內建預設。",
        "輸入 0 返回供應商選單。",
    ]
    if preset.note:
        lines.insert(1, preset.note)
    console.print(Panel("\n".join(lines), title=preset.provider_label, expand=False))
    while True:
        raw = typer.prompt("請輸入型號名稱", default=suggested)
        stripped = raw.strip()
        if stripped == "0":
            return None
        try:
            model_id = normalize_typed_model_id(stripped, preset)
            token = f"{preset.alias}:{model_id}"
            parse_model_tokens(token)
        except (ConfigError, UnknownProviderError) as exc:
            error_console.print(f"[red]錯誤：[/red]{exc}")
            continue
        return token


def _prompt_model_token(label: str, current: str, *, role: ModelRole) -> str | None:
    """先選供應商，再輸入該家的型號。

    內建預設只是建議值，Claude 不必永遠用 ``claude-sonnet-5``，
    Hugging Face 也不必永遠用同一顆。

    Args:
        label: 選單標題。
        current: 目前設定值。
        role: ``text`` 或 ``image``，決定顯示哪顆內建預設。

    Returns:
        ``別名:型號``。選 ``0`` 返回主選單時回傳 ``None``，不改設定。
    """
    presets = list_model_presets(role)
    while True:
        table = Table(title=label, title_justify="left")
        table.add_column("編號", justify="right", no_wrap=True)
        table.add_column("供應商", no_wrap=True)
        table.add_column("型號", overflow="fold")
        table.add_column("說明")
        default_choice = 1
        for index, preset in enumerate(presets, start=1):
            is_current = token_matches_preset(current, preset, role=role)
            if is_current:
                default_choice = index
            model_id = suggested_model_id(current, preset, role=role)
            marker = "（目前）" if is_current else "內建預設"
            note = " ".join(part for part in (preset.note, marker) if part)
            table.add_row(str(index), preset.provider_label, model_id, note)
        table.add_row("0", "返回主選單", "", "")
        console.print(table)
        choice = ask_choice(len(presets), default=default_choice, allow_zero=True)
        if choice == 0:
            return None
        token = _prompt_model_id_for_provider(presets[choice - 1], current, role=role)
        if token is not None:
            return token


def _persist(settings: AppSettings) -> AppSettings:
    """驗證後寫回 JSON。"""
    try:
        validated = AppSettings.model_validate(settings.model_dump(by_alias=True))
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    path = save_settings(validated)
    console.print(f"[green]已寫入[/green] {path}")
    return load_settings(path)


def _prompt_api_keys() -> None:
    """讓使用者為各供應商輸入金鑰，寫進使用者 ``.env``。

    畫面只顯示已設定金鑰的前十碼；尚未設定的欄位留白。直接 Enter 不更改。
    """
    while True:
        table = Table(title="供應商金鑰", title_justify="left", header_style="bold cyan")
        table.add_column("編號", justify="right", no_wrap=True)
        table.add_column("供應商", no_wrap=True)
        table.add_column("金鑰前十碼")
        for index, provider in enumerate(PROVIDER_KEY_ORDER, start=1):
            table.add_row(
                str(index),
                PROVIDER_DISPLAY_NAMES[provider],
                mask_api_key(peek_api_key(provider)),
            )
        table.add_row("0", "返回", "")
        console.print(table)
        console.print(f"[dim]金鑰寫在 {user_env_path()}，不會進 blogseo.json。[/dim]")
        choice = ask_choice(len(PROVIDER_KEY_ORDER), default=0, allow_zero=True)
        if choice == 0:
            return
        provider = PROVIDER_KEY_ORDER[choice - 1]
        label = PROVIDER_DISPLAY_NAMES[provider]
        prefix = mask_api_key(peek_api_key(provider))
        status = prefix if prefix else "尚未設定"
        console.print(f"{label} 目前：{status}")
        console.print("[dim]直接 Enter 則不更改。輸入時不會顯示內容。[/dim]")
        raw = typer.prompt("請輸入金鑰", default="", hide_input=True)
        cleaned = raw.strip()
        if not cleaned:
            console.print("[dim]未更改。[/dim]")
            continue
        path = save_api_key(provider, cleaned)
        console.print(
            f"[green]已寫入[/green] {path}　識別：{mask_api_key(cleaned)}"
        )


def run_settings_menu() -> AppSettings:
    """其他設定的子選單，改完立刻寫進 JSON。"""
    settings = load_settings()
    while True:
        _print_settings_overview(settings)
        path = find_settings_path() or default_save_path()
        lines = [
            f"設定檔：{path}",
            "1. 預設關鍵字／摘要模型",
            "2. 預設圖片辨識模型",
            "3. 預設摘要字數（下限、上限、則數）",
            "4. 預設圖片 alt 字數上限",
            "5. 供應商金鑰",
            "6. 更多預設值",
            "0. 返回主選單",
        ]
        console.print(Panel("\n".join(lines), title="其他設定", expand=False))
        choice = ask_choice(6, default=0, allow_zero=True)
        if choice == 0:
            return settings
        try:
            if choice == 1:
                token = _prompt_model_token(
                    "預設關鍵字／摘要模型", settings.models.text, role="text"
                )
                if token is None:
                    return settings
                settings.models.text = token
                settings = _persist(settings)
            elif choice == 2:
                token = _prompt_model_token(
                    "預設圖片辨識模型", settings.models.image, role="image"
                )
                if token is None:
                    return settings
                settings.models.image = token
                settings = _persist(settings)
            elif choice == 3:
                settings.summary.min_chars = typer.prompt(
                    "摘要字元下限", default=settings.summary.min_chars, type=int
                )
                settings.summary.max_chars = typer.prompt(
                    "摘要字元上限", default=settings.summary.max_chars, type=int
                )
                settings.summary.count = typer.prompt(
                    "摘要則數（1–5）", default=settings.summary.count, type=int
                )
                settings = _persist(settings)
            elif choice == 4:
                settings.alt.max_chars = typer.prompt(
                    "alt 字元上限", default=settings.alt.max_chars, type=int
                )
                settings = _persist(settings)
            elif choice == 5:
                _prompt_api_keys()
            elif choice == 6:
                settings = _run_more_settings(settings)
        except (BlogSeoError, ValidationError) as exc:
            error_console.print(f"[red]錯誤：[/red]{exc}")
            settings = load_settings()


def _run_more_settings(settings: AppSettings) -> AppSettings:
    """較少改到的預設值。"""
    lines = [
        "1. alt 語言（auto / zh / en）",
        "2. 關鍵字數量與字數上限",
        "3. 平行呼叫數",
        "4. 請求逾時秒數",
        "5. 最多分析幾張圖（0 = 不限）",
        "6. 輸出目錄",
        "7. 匯率 USD→TWD",
        "8. 套用時寫入 front matter",
        "9. front matter 欄位名",
        "10. 套用前備份 .bak",
        "0. 返回",
    ]
    console.print(Panel("\n".join(lines), title="更多預設值", expand=False))
    choice = ask_choice(10, default=0, allow_zero=True)
    if choice == 0:
        return settings
    if choice == 1:
        settings.alt.language = typer.prompt("alt 語言", default=settings.alt.language)
    elif choice == 2:
        settings.keywords.count = typer.prompt(
            "關鍵字數量", default=settings.keywords.count, type=int
        )
        settings.keywords.max_chars = typer.prompt(
            "每個關鍵字字數上限", default=settings.keywords.max_chars, type=int
        )
    elif choice == 3:
        settings.analyze.concurrency = typer.prompt(
            "平行呼叫數", default=settings.analyze.concurrency, type=int
        )
    elif choice == 4:
        settings.analyze.timeout_seconds = typer.prompt(
            "逾時秒數", default=settings.analyze.timeout_seconds, type=float
        )
    elif choice == 5:
        raw = typer.prompt(
            "最多分析幾張圖（0 = 不限）",
            default=settings.analyze.max_images or 0,
            type=int,
        )
        settings.analyze.max_images = None if raw <= 0 else raw
    elif choice == 6:
        settings.output.dir = typer.prompt("輸出目錄", default=settings.output.dir)
    elif choice == 7:
        settings.cost.usd_to_twd_rate = typer.prompt(
            "匯率", default=settings.cost.usd_to_twd_rate, type=float
        )
    elif choice == 8:
        settings.apply.write_front_matter = typer.confirm(
            "套用時把關鍵字／摘要寫進 front matter？",
            default=settings.apply.write_front_matter,
        )
    elif choice == 9:
        settings.apply.keywords_key = typer.prompt(
            "關鍵字欄位名", default=settings.apply.keywords_key
        )
        settings.apply.summary_key = typer.prompt(
            "摘要欄位名", default=settings.apply.summary_key
        )
    elif choice == 10:
        settings.apply.backup = typer.confirm(
            "套用前把原文複製成 .bak？", default=settings.apply.backup
        )
    return _persist(settings)
