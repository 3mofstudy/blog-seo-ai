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

from blogseo.display import (
    ask_choice,
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
from blogseo.errors import BlogSeoError
from blogseo.image.renamer import build_filename, slugify
from blogseo.llm.registry import format_model_token
from blogseo.markdown.parser import parse_article
from blogseo.schemas.result import DEFAULT_FIELDS, AnalysisField, AnalysisResult, ImageEntry
from blogseo.schemas.selection import MANUAL_SOURCE, Selection, SelectionImage
from blogseo.settings import (
    AppSettings,
    default_save_path,
    find_settings_path,
    load_settings,
)
from blogseo.settings_ui import run_settings_menu
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
        choice = ask_choice(SETTINGS_MENU_ITEM, default=1, allow_zero=True)
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
