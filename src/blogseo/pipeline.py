"""互動式一次跑完：選單 → 分析 → 挑選 → 套用。

中間產物寫進 ``output/``（路徑由 ``blogseo.json`` 決定），分析 JSON 與
選擇 JSON 都可以在下一步之前手動改。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from pydantic import ValidationError

from blogseo.display import (
    console,
    error_console,
    format_length,
    print_analysis_sections,
    print_apply_plan,
    print_article_panel,
    print_gitless_outcome,
    print_review_header,
    print_selection_summary,
)
from blogseo.config import (
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_KEY_ORDER,
    mask_api_key,
    peek_api_key,
    save_api_key,
    user_env_path,
)
from blogseo.errors import BlogSeoError, ConfigError, UnknownProviderError
from blogseo.image.renamer import build_filename, slugify
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
from blogseo.markdown.parser import parse_article
from blogseo.schemas.result import DEFAULT_FIELDS, AnalysisField, AnalysisResult, ImageEntry
from blogseo.schemas.selection import MANUAL_SOURCE, Selection, SelectionImage
from blogseo.settings import (
    AppSettings,
    default_save_path,
    find_settings_path,
    load_settings,
    save_settings,
)
from blogseo.seo.analyzer import AnalyzeOptions, analyze_article, options_from_settings
from blogseo.seo.applier import build_plan, execute_plan
from blogseo.seo.selector import (
    ImageCandidate,
    KeywordCandidate,
    SummaryCandidate,
    article_changed,
    build_selection,
    default_alias,
    find_conflicts,
    image_candidates,
    keyword_candidates,
    selectable_images,
    summary_candidates,
)

#: 檔名衝突最多請使用者重改幾輪。
_MAX_CONFLICT_ROUNDS = 5

#: 主選單：編號 → (fields, 說明)
ANALYSIS_MENU: tuple[tuple[int, frozenset[AnalysisField], str], ...] = (
    (1, frozenset(AnalysisField), "分析後，給出所有建議（摘要、關鍵字、圖片 SEO）"),
    (2, frozenset({AnalysisField.SUMMARY}), "分析後，給出摘要"),
    (3, frozenset({AnalysisField.KEYWORDS}), "分析後，給出關鍵字"),
    (4, frozenset({AnalysisField.KEYWORDS, AnalysisField.SUMMARY}), "分析後，給出摘要 + 關鍵字"),
    (5, frozenset({AnalysisField.IMAGES}), "分析後，只圖片 SEO 建議"),
)

SETTINGS_MENU_ITEM = 6
QUIT_MENU_ITEM = 0


def fields_from_result(result: AnalysisResult) -> frozenset[AnalysisField]:
    """從分析結果讀出本次要求的項目；舊檔沒記則視為全部。"""
    raw = result.metadata.requested_fields
    if not raw:
        return DEFAULT_FIELDS
    selected = {AnalysisField(name) for name in raw if name in AnalysisField}
    return frozenset(selected) if selected else DEFAULT_FIELDS


def resolve_output_path(
    source_path: Path, out: Path | None, *, suffix: str, output_dir: Path
) -> Path:
    """決定 JSON 的輸出路徑。"""
    default_name = f"{source_path.stem.removesuffix('-analysis')}{suffix}"
    if out is None:
        return output_dir / default_name
    if out.is_dir() or out.suffix == "":
        return out / default_name
    return out


def ask_choice(maximum: int, *, default: int) -> int:
    """要求使用者輸入一個編號。"""
    while True:
        value = typer.prompt("請選擇編號", default=default, type=int)
        if 1 <= value <= maximum:
            return value
        error_console.print(f"[red]請輸入 1 到 {maximum} 之間的數字。[/red]")


def _ask_menu_choice(maximum: int, *, default: int, allow_zero: bool = True) -> int:
    """主選單／設定選單用的編號，可接受 0 代表返回或結束。"""
    lowest = 0 if allow_zero else 1
    while True:
        value = typer.prompt("請選擇編號", default=default, type=int)
        if lowest <= value <= maximum:
            return value
        error_console.print(f"[red]請輸入 {lowest} 到 {maximum} 之間的數字。[/red]")


def _strip_path_input(raw: str) -> Path:
    """處理使用者貼上的路徑，去掉包住的引號。"""
    cleaned = raw.strip().strip('"').strip("'")
    return Path(cleaned).expanduser()


def _prompt_markdown_path(current: Path | None) -> Path:
    """詢問要分析的 Markdown 檔案。"""
    default = str(current) if current is not None else ""
    while True:
        raw = typer.prompt("請輸入 Markdown 檔案路徑", default=default or None)
        path = _strip_path_input(str(raw))
        if path.is_file():
            return path
        error_console.print(f"[red]找不到檔案：[/red]{path}")


def _preferred_index(candidates: Sequence[Any], preferred: str | None) -> int:
    """找出偏好模型在候選清單中的位置。"""
    for index, candidate in enumerate(candidates):
        if candidate.alias == preferred:
            return index
    return 0


def _note_fallback(item: str, preferred: str | None, actual: str) -> None:
    """提醒使用者這一項改用了其他模型的結果。"""
    if preferred is None or preferred == actual:
        return
    console.print(
        f"[yellow]提醒：[/yellow]{item} 沒有 {preferred} 的可用結果，改用 {actual}。"
    )


def confirm_freshness(result: AnalysisResult, *, assume_yes: bool, force: bool) -> None:
    """確認文章在分析之後沒有被改過。"""
    changed = article_changed(result)
    if changed is False:
        return

    if changed is None and not result.article_info.content_hash:
        message = (
            "這份分析結果是舊版本產生的，沒有記錄正文雜湊，"
            "無法確認文章在分析之後有沒有被改過"
        )
    elif changed is None:
        message = "找不到或無法讀取原始文章，無法確認內容是否有變動"
    else:
        message = "原始文章在分析之後被編輯過，圖片位置可能已經失效"
    error_console.print(f"[yellow]警告：[/yellow]{message}")

    if force:
        error_console.print("[dim]已指定 --force，繼續進行。[/dim]")
        return
    if assume_yes:
        error_console.print(
            "[dim]非互動模式下不會自動繼續。重新執行分析取得最新結果，"
            "或加上 --force 略過這項檢查。[/dim]"
        )
        raise typer.Exit(code=1)
    if not typer.confirm("仍要繼續挑選嗎？", default=False):
        raise typer.Exit(code=1)


def choose_keywords(
    candidates: list[KeywordCandidate], *, assume_yes: bool, preferred: str | None
) -> list[str]:
    """挑選最終要採用的關鍵字。"""
    if not candidates:
        return []

    default_index = _preferred_index(candidates, preferred)
    if assume_yes:
        chosen = candidates[default_index]
        _note_fallback("關鍵字", preferred, chosen.alias)
        return list(chosen.keywords)

    table = Table(title="關鍵字候選", title_justify="left", header_style="bold cyan")
    table.add_column("編號", justify="right", no_wrap=True)
    table.add_column("模型", no_wrap=True)
    table.add_column("關鍵字", overflow="fold")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(str(index), candidate.alias, "、".join(candidate.keywords))
    manual_choice = len(candidates) + 1
    skip_choice = len(candidates) + 2
    table.add_row(str(manual_choice), "—", "手動輸入")
    table.add_row(str(skip_choice), "—", "不採用")
    console.print(table)

    choice = ask_choice(skip_choice, default=default_index + 1)
    if choice == skip_choice:
        return []
    if choice == manual_choice:
        raw = typer.prompt("請輸入關鍵字，以逗號分隔")
        return [piece.strip() for piece in raw.split(",") if piece.strip()]
    return list(candidates[choice - 1].keywords)


def choose_summary(
    candidates: list[SummaryCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    variant: int,
    min_chars: int,
    max_chars: int,
) -> str:
    """挑選最終要採用的摘要。"""
    if not candidates:
        return ""

    default_index = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.alias == preferred and candidate.variant == variant
        ),
        _preferred_index(candidates, preferred),
    )
    if assume_yes:
        chosen = candidates[default_index]
        _note_fallback("摘要", preferred, chosen.alias)
        if chosen.variant != variant:
            console.print(
                f"[yellow]提醒：[/yellow]沒有第 {variant} 則摘要，"
                f"改用 {chosen.alias} 的第 {chosen.variant} 則。"
            )
        return chosen.text

    table = Table(
        title=f"摘要候選（analyze 當時要求 {min_chars}–{max_chars} 字元）",
        title_justify="left",
        header_style="bold cyan",
    )
    table.add_column("編號", justify="right", no_wrap=True)
    table.add_column("模型", no_wrap=True)
    table.add_column("版本", justify="right", no_wrap=True)
    table.add_column("字數", justify="right", no_wrap=True)
    table.add_column("內容", overflow="fold")
    for index, candidate in enumerate(candidates, start=1):
        in_range = min_chars <= candidate.length <= max_chars
        length_cell = (
            str(candidate.length) if in_range else f"[yellow]{candidate.length}[/yellow]"
        )
        table.add_row(
            str(index),
            candidate.alias,
            str(candidate.variant),
            length_cell,
            candidate.text,
        )
    manual_choice = len(candidates) + 1
    skip_choice = len(candidates) + 2
    table.add_row(str(manual_choice), "—", "—", "—", "手動輸入")
    table.add_row(str(skip_choice), "—", "—", "—", "不採用")
    console.print(table)

    choice = ask_choice(skip_choice, default=default_index + 1)
    if choice == skip_choice:
        return ""
    if choice == manual_choice:
        return " ".join(typer.prompt("請輸入摘要").split())
    return candidates[choice - 1].text


def _prompt_filename(extension: str, *, default_stem: str) -> str:
    """要求使用者輸入檔名主體，並正規化成安全的檔名。"""
    raw = typer.prompt("請輸入檔名（不含副檔名）", default=default_stem)
    return build_filename(slugify(raw), extension)


def choose_filename(
    key: str,
    entry: ImageEntry,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
) -> tuple[str | None, str | None]:
    """挑選某張圖片要改成的檔名。"""
    default_index = _preferred_index(candidates, preferred)
    if assume_yes:
        chosen = candidates[default_index]
        _note_fallback(f"圖片 {key}", preferred, chosen.alias)
        return chosen.filename, chosen.alias

    table = Table(
        title=f"圖片 {key}：檔名", title_justify="left", header_style="bold magenta"
    )
    table.add_column("編號", justify="right", no_wrap=True)
    table.add_column("模型", no_wrap=True)
    table.add_column("建議檔名", overflow="fold")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(str(index), candidate.alias, candidate.filename)
    manual_choice = len(candidates) + 1
    keep_choice = len(candidates) + 2
    table.add_row(str(manual_choice), "—", "手動輸入")
    table.add_row(str(keep_choice), "—", f"保持原檔名（{Path(key).name}）")
    console.print(table)

    choice = ask_choice(keep_choice, default=default_index + 1)
    if choice == keep_choice:
        return None, None
    if choice == manual_choice:
        default_stem = Path(candidates[default_index].filename).stem
        filename = _prompt_filename(entry.extension, default_stem=default_stem)
        return filename, MANUAL_SOURCE
    chosen = candidates[choice - 1]
    return chosen.filename, chosen.alias


def choose_alt(
    key: str,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    max_chars: int,
) -> tuple[str | None, str | None]:
    """挑選某張圖片要寫入的 alt 文字。"""
    default_index = _preferred_index(candidates, preferred)
    if assume_yes:
        chosen = candidates[default_index]
        return chosen.alt, chosen.alias

    table = Table(
        title=f"圖片 {key}：alt（上限 {max_chars} 字元）",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("編號", justify="right", no_wrap=True)
    table.add_column("模型", no_wrap=True)
    table.add_column("字數", justify="right", no_wrap=True)
    table.add_column("alt", overflow="fold")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            candidate.alias,
            format_length(candidate.alt_length, max_chars),
            candidate.alt,
        )
    manual_choice = len(candidates) + 1
    keep_choice = len(candidates) + 2
    table.add_row(str(manual_choice), "—", "—", "手動輸入")
    table.add_row(str(keep_choice), "—", "—", "不改 alt")
    console.print(table)

    choice = ask_choice(keep_choice, default=default_index + 1)
    if choice == keep_choice:
        return None, None
    if choice == manual_choice:
        raw = typer.prompt("請輸入 alt 文字", default=candidates[default_index].alt)
        return " ".join(raw.split()), MANUAL_SOURCE
    chosen = candidates[choice - 1]
    return chosen.alt, chosen.alias


def choose_image(
    key: str,
    entry: ImageEntry,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    alt_max_chars: int,
) -> SelectionImage | None:
    """挑選某張圖片的檔名與 alt。"""
    assert entry.resolved_path is not None
    filename, filename_model = choose_filename(
        key, entry, candidates, assume_yes=assume_yes, preferred=preferred
    )
    alt, alt_model = choose_alt(
        key,
        candidates,
        assume_yes=assume_yes,
        preferred=preferred,
        max_chars=alt_max_chars,
    )
    selection = SelectionImage(
        source=entry.source,
        resolved_path=entry.resolved_path,
        new_filename=filename,
        filename_model=filename_model,
        alt=alt,
        alt_model=alt_model,
    )
    return None if selection.is_empty else selection


def resolve_conflicts(
    images: dict[str, SelectionImage], *, assume_yes: bool
) -> dict[str, SelectionImage]:
    """檢查檔名衝突，並在互動模式下請使用者當場改名。"""
    for _ in range(_MAX_CONFLICT_ROUNDS):
        conflicts = find_conflicts(images)
        if not conflicts:
            return images

        table = Table(title="檔名衝突", title_justify="left", header_style="bold red")
        table.add_column("檔名", overflow="fold")
        table.add_column("涉及的圖片", overflow="fold")
        table.add_column("原因", overflow="fold")
        for conflict in conflicts:
            table.add_row(conflict.filename, "、".join(conflict.sources), conflict.reason)
        console.print(table)

        if assume_yes:
            error_console.print(
                "[red]錯誤：[/red]非互動模式下無法解決檔名衝突，"
                "請改用互動模式或指定其他模型。"
            )
            raise typer.Exit(code=1)

        for conflict in conflicts:
            for key in conflict.sources:
                item = images.get(key)
                if item is None or item.new_filename is None:
                    continue
                console.print(f"重新命名 [bold]{key}[/bold]（留空則保持原檔名）")
                extension = Path(item.new_filename).suffix
                raw = typer.prompt(
                    "請輸入檔名（不含副檔名）",
                    default=Path(item.new_filename).stem,
                )
                if not raw.strip():
                    updated = item.model_copy(
                        update={"new_filename": None, "filename_model": None}
                    )
                    if updated.is_empty:
                        del images[key]
                    else:
                        images[key] = updated
                    continue
                images[key] = item.model_copy(
                    update={"new_filename": build_filename(slugify(raw), extension)}
                )

    error_console.print("[red]錯誤：[/red]檔名衝突仍未解決，已中止。")
    raise typer.Exit(code=1)


def run_review_session(
    result: AnalysisResult,
    analysis_path: Path,
    *,
    assume_yes: bool = False,
    force: bool = False,
    pick: str | None = None,
    summary_index: int = 1,
) -> Selection:
    """從分析結果挑選，回傳選擇物件（尚未寫檔）。"""
    print_review_header(result, analysis_path.name)
    confirm_freshness(result, assume_yes=assume_yes, force=force)

    available = sorted(result.metadata.models)
    if pick is not None and pick not in result.metadata.models:
        error_console.print(
            f"[red]錯誤：[/red]這份分析結果沒有 {pick} 的資料；"
            f"可用的有：{'、'.join(available) or '（無）'}"
        )
        raise typer.Exit(code=1)

    preferred = pick or default_alias(result)
    if preferred is None:
        error_console.print("[red]錯誤：[/red]這份分析結果沒有任何成功的項目可挑選。")
        raise typer.Exit(code=1)

    fields = fields_from_result(result)
    keywords: list[str] = []
    summary = ""
    if AnalysisField.KEYWORDS in fields:
        keywords = choose_keywords(
            keyword_candidates(result), assume_yes=assume_yes, preferred=preferred
        )
    if AnalysisField.SUMMARY in fields:
        summary = choose_summary(
            summary_candidates(result),
            assume_yes=assume_yes,
            preferred=preferred,
            variant=summary_index,
            min_chars=result.metadata.summary_min_chars,
            max_chars=result.metadata.summary_max_chars,
        )

    alt_max_chars = result.metadata.alt_max_chars
    images: dict[str, SelectionImage] = {}
    if AnalysisField.IMAGES in fields:
        for key, entry in selectable_images(result).items():
            chosen = choose_image(
                key,
                entry,
                image_candidates(entry),
                assume_yes=assume_yes,
                preferred=preferred,
                alt_max_chars=alt_max_chars,
            )
            if chosen is not None:
                images[key] = chosen
        images = resolve_conflicts(images, assume_yes=assume_yes)

    selection = build_selection(
        result,
        source_analysis=analysis_path,
        keywords=keywords,
        summary=summary,
        images=images,
    )
    print_selection_summary(keywords, summary, images, alt_max_chars=alt_max_chars)
    return selection


def run_analyze_session(
    markdown_path: Path,
    options: AnalyzeOptions,
    *,
    output_dir: Path,
    out: Path | None = None,
) -> tuple[AnalysisResult, Path]:
    """分析文章、顯示結果、寫出 analysis JSON。"""
    article = parse_article(markdown_path)
    tokens = list(
        dict.fromkeys([*options.tokens_for_text(), *options.tokens_for_image()])
    )
    print_article_panel(article, tokens, options.fields)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("分析中", total=None)

        def on_progress(done: int, total: int, label: str) -> None:
            progress.update(task_id, completed=done, total=total, description=f"完成 {label}")

        result = analyze_article(article, options, progress=on_progress)

    output_path = resolve_output_path(
        markdown_path, out, suffix="-analysis.json", output_dir=output_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_json(), encoding="utf-8")

    print_analysis_sections(result, options.fields)
    console.print(f"\n結果已寫入 [bold green]{output_path}[/bold green]")
    console.print("[dim]原始檔案尚未修改。可先編輯這份 JSON 再繼續挑選。[/dim]")
    return result, output_path


def run_apply_session(
    selection: Selection,
    settings: AppSettings,
    *,
    assume_yes: bool = False,
    force: bool = False,
    dry_run: bool = False,
    backup: bool | None = None,
    write_front_matter: bool | None = None,
) -> None:
    """依選擇結果預覽並寫入 Markdown。"""
    use_backup = settings.apply.backup if backup is None else backup
    use_front_matter = (
        settings.apply.write_front_matter
        if write_front_matter is None
        else write_front_matter
    )
    plan = build_plan(
        selection,
        force=force,
        write_front_matter=use_front_matter,
        keywords_key=settings.apply.keywords_key,
        summary_key=settings.apply.summary_key,
    )
    if plan.hash_mismatch:
        error_console.print(
            "[yellow]警告：[/yellow]文章正文與分析當時不一致，已依 --force 用當下內容套用。"
        )
    if plan.is_empty:
        console.print("沒有需要套用的變更。關鍵字與摘要已留在選擇檔裡。")
        return

    print_apply_plan(plan)
    if dry_run:
        console.print("[dim]這是預覽，原始檔案未被修改。[/dim]")
        return
    if not assume_yes and not typer.confirm("確定寫入這些變更？", default=True):
        console.print("已取消套用。選擇檔仍保留，之後可再執行 apply。")
        raise typer.Exit(code=1)

    gitless = execute_plan(plan, backup=use_backup)
    console.print(f"\n已套用 [bold green]{plan.article_path}[/bold green]")
    if use_backup and plan.new_text != plan.original_text:
        console.print(f"[dim]原文備份：{plan.article_path.name}.bak[/dim]")
    print_gitless_outcome(gitless)


def _print_main_menu(settings: AppSettings) -> None:
    """顯示啟動後的主選單。"""
    path = find_settings_path()
    location = str(path) if path is not None else f"（尚未建立，將寫到 {default_save_path()}）"
    lines = [f"{number}. {label}" for number, _fields, label in ANALYSIS_MENU]
    lines.append(f"{SETTINGS_MENU_ITEM}. 其他設定")
    lines.append(f"{QUIT_MENU_ITEM}. 結束")
    lines.append("")
    lines.append(f"文字模型：{format_model_token(settings.models.text, role='text')}")
    lines.append(f"圖片模型：{format_model_token(settings.models.image, role='image')}")
    lines.append(f"設定檔　：{location}")
    console.print(Panel("\n".join(lines), title="blog-seo-ai", expand=False))


def _print_settings_overview(settings: AppSettings) -> None:
    """列出目前的預設值。"""
    table = Table(title="目前預設值", title_justify="left", header_style="bold cyan")
    table.add_column("項目", no_wrap=True)
    table.add_column("值", overflow="fold")
    table.add_row("關鍵字／摘要模型", format_model_token(settings.models.text, role="text"))
    table.add_row("圖片辨識模型", format_model_token(settings.models.image, role="image"))
    table.add_row("關鍵字數量", str(settings.keywords.count))
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
        choice = _ask_menu_choice(len(presets), default=default_choice)
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
        choice = _ask_menu_choice(len(PROVIDER_KEY_ORDER), default=0)
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
        choice = _ask_menu_choice(6, default=0)
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
        "2. 關鍵字數量",
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
    choice = _ask_menu_choice(10, default=0)
    if choice == 0:
        return settings
    if choice == 1:
        settings.alt.language = typer.prompt("alt 語言", default=settings.alt.language)
    elif choice == 2:
        settings.keywords.count = typer.prompt(
            "關鍵字數量", default=settings.keywords.count, type=int
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


def run_full_pipeline(markdown_path: Path, fields: frozenset[AnalysisField]) -> None:
    """分析 → 挑選 → 套用，中間產物寫進 output/。"""
    settings = load_settings()
    options = options_from_settings(settings, fields=fields)
    try:
        result, analysis_path = run_analyze_session(
            markdown_path, options, output_dir=settings.output_dir
        )
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    if not typer.confirm("開始挑選要採用的結果？", default=True):
        console.print(
            f"已停在分析結果。[dim]之後可執行：blog-seo-ai review {analysis_path}[/dim]"
        )
        return

    try:
        selection = run_review_session(result, analysis_path)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    selection_path = resolve_output_path(
        analysis_path, None, suffix="-selection.json", output_dir=settings.output_dir
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(selection.to_json(), encoding="utf-8")
    console.print(f"選擇已寫入 [bold green]{selection_path}[/bold green]")

    if not typer.confirm("接著套用到 Markdown？", default=True):
        console.print(
            f"已停在選擇檔。[dim]之後可執行：blog-seo-ai apply {selection_path}[/dim]"
        )
        return

    try:
        run_apply_session(selection, settings)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc


def run_interactive(markdown_path: Path | None = None) -> None:
    """啟動後的主迴圈：先選要做什麼，再一次跑完。"""
    current_path = markdown_path
    while True:
        try:
            settings = load_settings()
        except BlogSeoError as exc:
            error_console.print(f"[red]錯誤：[/red]{exc}")
            raise typer.Exit(code=1) from exc

        _print_main_menu(settings)
        choice = _ask_menu_choice(SETTINGS_MENU_ITEM, default=1)
        if choice == QUIT_MENU_ITEM:
            console.print("再見。")
            return
        if choice == SETTINGS_MENU_ITEM:
            try:
                run_settings_menu()
            except BlogSeoError as exc:
                error_console.print(f"[red]錯誤：[/red]{exc}")
            continue

        fields = next(item[1] for item in ANALYSIS_MENU if item[0] == choice)
        current_path = _prompt_markdown_path(current_path)
        run_full_pipeline(current_path, fields)
        if not typer.confirm("回到主選單？", default=True):
            return
