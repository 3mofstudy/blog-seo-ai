"""命令列進入點。

目前提供三個指令：

* ``analyze``：讀取文章與圖片、呼叫指定模型、把結果落地成 JSON。
* ``review``：讀回那份 JSON，讓使用者從多個模型的結果中挑選，寫出 selection JSON。
* ``apply``：依選擇檔改寫 Markdown 圖片語法，並把圖片改名。

``analyze`` 與 ``review`` 不會修改使用者的原始檔案；``apply`` 才是真正動手的那一步。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from blogseo import __version__
from blogseo.config import (
    ALT_MAX_CHARS,
    DEFAULT_MODEL_TOKEN,
    DEFAULT_OUTPUT_DIR,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
    USD_TO_TWD_RATE,
)
from blogseo.errors import BlogSeoError
from blogseo.image.renamer import build_filename, slugify
from blogseo.llm.registry import known_aliases, parse_model_tokens
from blogseo.markdown.parser import ParsedArticle, parse_article
from blogseo.schemas.result import AnalysisField, AnalysisResult, ImageEntry
from blogseo.schemas.selection import MANUAL_SOURCE, SelectionImage
from blogseo.seo.analyzer import AnalyzeOptions, analyze_article, parse_fields
from blogseo.seo.applier import ApplyPlan, build_plan, execute_plan, load_selection
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
    load_analysis,
    selectable_images,
    summary_candidates,
)

#: ``--fields`` 的預設值，字串形式方便直接顯示在說明裡。
_DEFAULT_FIELDS_ARG = "keywords,summary,images"

#: 檔名衝突最多請使用者重改幾輪，避免改到又撞在一起時無限循環。
_MAX_CONFLICT_ROUNDS = 5

app = typer.Typer(
    name="blog-seo-ai",
    help="用多家 LLM 分析部落格文章，產生 SEO 關鍵字、摘要與圖片 alt 建議。",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    """處理 ``--version`` 旗標。

    Args:
        value: 是否指定了該旗標。

    Raises:
        typer.Exit: 顯示版本後結束。
    """
    if value:
        console.print(f"blog-seo-ai {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="顯示版本"),
    ] = False,
) -> None:
    """blog-seo-ai 的共用選項。"""


def _print_article_panel(
    article: ParsedArticle, tokens: list[str], fields: frozenset[AnalysisField]
) -> None:
    """顯示文章基本資訊。

    Args:
        article: 已解析的文章。
        tokens: 這次要使用的模型別名。
        fields: 這次要產生的項目。
    """
    analyzable = sum(1 for target in article.targets if target.analyzable)
    lines = [
        f"標題　　：{article.title or '（無）'}",
        f"語言　　：{article.language}　字數：{article.word_count}",
        f"圖片　　：共 {len(article.targets)} 張，可分析 {analyzable} 張",
        f"模型　　：{'、'.join(tokens)}",
        f"產出項目：{'、'.join(sorted(field.value for field in fields))}",
    ]
    console.print(Panel("\n".join(lines), title=article.path.name, expand=False))


def _print_skipped(result: AnalysisResult) -> None:
    """列出被略過的圖片與原因。

    Args:
        result: 分析結果。
    """
    skipped = {
        key: entry.skipped_reason
        for key, entry in result.images.items()
        if entry.skipped_reason
    }
    if not skipped:
        return
    table = Table(title="略過的圖片", title_justify="left", header_style="yellow")
    table.add_column("圖片", overflow="fold")
    table.add_column("原因", overflow="fold")
    for key, reason in skipped.items():
        table.add_row(key, reason)
    console.print(table)


def _print_keyword_table(result: AnalysisResult) -> None:
    """顯示各模型的關鍵字。

    Args:
        result: 分析結果。
    """
    table = Table(title="關鍵字", title_justify="left", header_style="bold cyan")
    table.add_column("模型", no_wrap=True)
    table.add_column("關鍵字", overflow="fold")

    for alias, entry in result.keyword_summary.items():
        if entry.error is not None:
            table.add_row(alias, f"[red]失敗：{entry.error}[/red]")
            continue
        assert entry.result is not None
        table.add_row(alias, "、".join(entry.result.keywords))
    console.print(table)


def _print_summary_table(result: AnalysisResult) -> None:
    """顯示各模型的多個摘要版本，並標出超出字數範圍的版本。

    Args:
        result: 分析結果。
    """
    low = result.metadata.summary_min_chars
    high = result.metadata.summary_max_chars

    table = Table(
        title=f"摘要候選（要求 {low}–{high} 字元）",
        title_justify="left",
        header_style="bold cyan",
    )
    table.add_column("模型", no_wrap=True)
    table.add_column("版本", justify="right", no_wrap=True)
    table.add_column("字數", justify="right", no_wrap=True)
    table.add_column("內容", overflow="fold")

    for alias, entry in result.keyword_summary.items():
        if entry.error is not None or entry.result is None:
            table.add_row(alias, "-", "-", f"[red]失敗：{entry.error}[/red]")
            continue
        for index, text in enumerate(entry.result.summaries, start=1):
            length = len(text)
            in_range = low <= length <= high
            length_cell = str(length) if in_range else f"[yellow]{length}[/yellow]"
            table.add_row(alias if index == 1 else "", str(index), length_cell, text)
    console.print(table)


def _format_length(length: int, limit: int) -> str:
    """把字數格式化成表格用的字串，超出上限標成黃色。

    Args:
        length: 實際字元數。
        limit: 要求的上限。

    Returns:
        顯示字串。
    """
    return str(length) if length <= limit else f"[yellow]{length}[/yellow]"


def _print_image_tables(result: AnalysisResult) -> None:
    """顯示各模型對每張圖的建議。

    Args:
        result: 分析結果。
    """
    limit = result.metadata.alt_max_chars
    for key, entry in result.images.items():
        if not entry.models:
            continue
        table = Table(title=f"圖片：{key}", title_justify="left", header_style="bold magenta")
        table.add_column("模型", no_wrap=True)
        table.add_column("建議檔名", overflow="fold")
        table.add_column("字數", justify="right", no_wrap=True)
        table.add_column(f"alt（上限 {limit}）", overflow="fold")
        for alias, analysis in entry.models.items():
            if analysis.error is not None:
                table.add_row(alias, "[red]失敗[/red]", "-", f"[red]{analysis.error}[/red]")
                continue
            assert analysis.result is not None
            table.add_row(
                alias,
                analysis.final_filename or "",
                _format_length(len(analysis.result.alt), limit),
                analysis.result.alt,
            )
        console.print(table)


def _format_twd(amount: float | None) -> str:
    """把新台幣金額格式化成表格用的字串。

    金額通常只有幾角，小數點後兩位不夠看，所以未滿一元時多留兩位。

    Args:
        amount: 金額；未知為 ``None``。

    Returns:
        顯示字串。
    """
    if amount is None:
        return "—"
    if amount < 1:
        return f"{amount:.4f}"
    return f"{amount:,.2f}"


def _print_usage_table(result: AnalysisResult) -> None:
    """顯示 token 用量、耗時與估算花費。

    Args:
        result: 分析結果。
    """
    metadata = result.metadata
    table = Table(title="用量", title_justify="left", header_style="dim")
    table.add_column("模型", no_wrap=True)
    table.add_column("模型 id", no_wrap=True)
    table.add_column("呼叫", justify="right")
    table.add_column("失敗", justify="right")
    table.add_column("輸入 tokens", justify="right")
    table.add_column("輸出 tokens", justify="right")
    table.add_column("花費 NT$", justify="right")

    for alias, usage in metadata.models.items():
        table.add_row(
            alias,
            usage.model_id,
            str(usage.calls),
            str(usage.failed_calls),
            f"{usage.input_tokens:,}",
            f"{usage.output_tokens:,}",
            _format_twd(usage.estimated_cost_twd),
        )

    caption = f"整體耗時 {metadata.total_elapsed_ms / 1000:.1f} 秒"
    if metadata.total_cost_twd is not None:
        caption += (
            f"　合計 NT$ {_format_twd(metadata.total_cost_twd)}"
            f"（US$ {metadata.total_cost_usd:.4f}，匯率 {metadata.usd_to_twd_rate}）"
        )
    table.caption = caption
    console.print(table)
    if any(usage.estimated_cost_twd is None for usage in metadata.models.values()):
        console.print(
            "[dim]部分模型不在內建價目表中，費用顯示為「—」。"
            "價目表在 llm/anthropic.py 的 pricing。[/dim]"
        )


def _warn_about_unused_options(
    fields: frozenset[AnalysisField],
    *,
    summaries: int,
    summary_min: int,
    summary_max: int,
    alt_max: int,
    max_images: int | None,
) -> None:
    """提醒使用者哪些參數在這次的 ``--fields`` 之下不會生效。

    靜靜忽略參數比報錯更容易讓人困惑，所以這裡明講。

    Args:
        fields: 這次要產生的項目。
        summaries: ``--summaries`` 的值。
        summary_min: ``--summary-min`` 的值。
        summary_max: ``--summary-max`` 的值。
        alt_max: ``--alt-max`` 的值。
        max_images: ``--max-images`` 的值。
    """
    unused: list[str] = []

    if AnalysisField.SUMMARY not in fields:
        if summaries != SUMMARY_VARIANT_COUNT:
            unused.append("--summaries")
        if summary_min != SUMMARY_MIN_CHARS:
            unused.append("--summary-min")
        if summary_max != SUMMARY_MAX_CHARS:
            unused.append("--summary-max")

    if AnalysisField.IMAGES not in fields:
        if max_images is not None:
            unused.append("--max-images")
        if alt_max != ALT_MAX_CHARS:
            unused.append("--alt-max")

    if unused:
        console.print(
            f"[yellow]提醒：[/yellow]{'、'.join(unused)} 在本次 --fields 之下不會生效。"
        )


def _resolve_output_path(source_path: Path, out: Path | None, *, suffix: str) -> Path:
    """決定 JSON 的輸出路徑。

    來源如果本身就是 ``xxx-analysis.json``，會先把後綴去掉再接上新的，
    這樣 review 的產物是 ``xxx-selection.json`` 而不是 ``xxx-analysis-selection.json``。

    Args:
        source_path: 來源檔案路徑。
        out: 使用者指定的輸出路徑或目錄。
        suffix: 預設檔名的後綴，含副檔名。

    Returns:
        實際要寫入的檔案路徑。
    """
    default_name = f"{source_path.stem.removesuffix('-analysis')}{suffix}"
    if out is None:
        return DEFAULT_OUTPUT_DIR / default_name
    if out.is_dir() or out.suffix == "":
        return out / default_name
    return out


@app.command()
def analyze(
    markdown_path: Annotated[
        Path, typer.Argument(help="要分析的 Markdown 檔案路徑", show_default=False)
    ],
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="以逗號分隔的模型別名，可用「別名:模型id」覆寫文字模型，例如 hf:Qwen/Qwen3-8B",
        ),
    ] = DEFAULT_MODEL_TOKEN,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="輸出 JSON 的路徑或目錄", show_default=False),
    ] = None,
    fields: Annotated[
        str,
        typer.Option(
            "--fields",
            "-f",
            help=(
                "要產生哪些項目，逗號分隔。可用 keywords、summary、images。"
                "例如只要關鍵字就用 --fields keywords"
            ),
        ),
    ] = _DEFAULT_FIELDS_ARG,
    max_images: Annotated[
        int | None,
        typer.Option("--max-images", help="最多分析幾張圖片，用來控制花費", show_default=False),
    ] = None,
    alt_lang: Annotated[
        str, typer.Option("--alt-lang", help="alt 文字語言，auto 表示跟隨文章語言")
    ] = "auto",
    alt_max: Annotated[
        int,
        typer.Option("--alt-max", help="alt 文字的字元上限（含標點與空白）", min=10),
    ] = ALT_MAX_CHARS,
    summaries: Annotated[
        int,
        typer.Option("--summaries", help="每個模型產生幾個不同角度的摘要版本", min=1, max=5),
    ] = SUMMARY_VARIANT_COUNT,
    summary_min: Annotated[
        int,
        typer.Option("--summary-min", help="摘要字元下限（含標點與空白）", min=10),
    ] = SUMMARY_MIN_CHARS,
    summary_max: Annotated[
        int,
        typer.Option("--summary-max", help="摘要字元上限（含標點與空白）", min=20),
    ] = SUMMARY_MAX_CHARS,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="平行呼叫上限", min=1, max=16)
    ] = 4,
    timeout: Annotated[
        float, typer.Option("--timeout", help="單次 API 請求逾時秒數", min=5.0)
    ] = 90.0,
    twd_rate: Annotated[
        float,
        typer.Option("--twd-rate", help="費用換算成新台幣使用的匯率", min=0.1),
    ] = USD_TO_TWD_RATE,
) -> None:
    """分析文章並輸出 JSON，不會修改任何原始檔案。"""
    if summary_min >= summary_max:
        error_console.print(
            f"[red]錯誤：[/red]--summary-min（{summary_min}）必須小於 --summary-max（{summary_max}）"
        )
        raise typer.Exit(code=1)

    try:
        tokens = parse_model_tokens(models)
        selected_fields = parse_fields(fields)
        article = parse_article(markdown_path)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    _warn_about_unused_options(
        selected_fields,
        summaries=summaries,
        summary_min=summary_min,
        summary_max=summary_max,
        alt_max=alt_max,
        max_images=max_images,
    )
    _print_article_panel(article, tokens, selected_fields)

    options = AnalyzeOptions(
        model_tokens=tokens,
        fields=selected_fields,
        alt_language=alt_lang,
        summary_count=summaries,
        summary_min_chars=summary_min,
        summary_max_chars=summary_max,
        alt_max_chars=alt_max,
        max_images=max_images,
        concurrency=concurrency,
        timeout=timeout,
        usd_to_twd_rate=twd_rate,
    )

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

        try:
            result = analyze_article(article, options, progress=on_progress)
        except BlogSeoError as exc:
            progress.stop()
            error_console.print(f"[red]錯誤：[/red]{exc}")
            raise typer.Exit(code=1) from exc

    output_path = _resolve_output_path(markdown_path, out, suffix="-analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(result.to_json())

    if AnalysisField.KEYWORDS in selected_fields:
        _print_keyword_table(result)
    if AnalysisField.SUMMARY in selected_fields:
        _print_summary_table(result)
    if AnalysisField.IMAGES in selected_fields:
        _print_image_tables(result)
        _print_skipped(result)
    _print_usage_table(result)
    console.print(f"\n結果已寫入 [bold green]{output_path}[/bold green]")
    console.print("[dim]原始檔案未被修改。接著執行 review 挑選要採用的結果：[/dim]")
    console.print(f"[dim]  uv run blog-seo-ai review {output_path}[/dim]")


def _print_review_header(result: AnalysisResult, source: Path) -> None:
    """顯示這份分析結果的基本資訊。

    Args:
        result: 分析結果。
        source: analysis JSON 的路徑。
    """
    info = result.article_info
    lines = [
        f"文章　　：{info.title or '（無）'}",
        f"路徑　　：{info.path}",
        f"分析時間：{result.metadata.generated_at}",
        f"模型　　：{'、'.join(result.metadata.models) or '（無）'}",
        f"產出項目：{'、'.join(result.metadata.requested_fields)}",
    ]
    console.print(Panel("\n".join(lines), title=source.name, expand=False))


def _confirm_freshness(result: AnalysisResult, *, assume_yes: bool, force: bool) -> None:
    """確認文章在分析之後沒有被改過。

    圖片引用的字元位置是相對於當時的正文，文章一改就會失效，套用時可能改到
    錯誤的位置，所以這裡先擋一次。

    Args:
        result: 分析結果。
        assume_yes: 是否為非互動模式。
        force: 是否強制略過這項檢查。

    Raises:
        typer.Exit: 使用者選擇不繼續，或非互動模式下未指定 ``--force``。
    """
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
            "[dim]非互動模式下不會自動繼續。重新執行 analyze 取得最新結果，"
            "或加上 --force 略過這項檢查。[/dim]"
        )
        raise typer.Exit(code=1)
    if not typer.confirm("仍要繼續挑選嗎？", default=False):
        raise typer.Exit(code=1)


def _ask_choice(maximum: int, *, default: int) -> int:
    """要求使用者輸入一個編號。

    Args:
        maximum: 可接受的最大編號。
        default: 直接按 Enter 時採用的編號。

    Returns:
        使用者選擇的編號，從 1 起算。
    """
    while True:
        value = typer.prompt("請選擇編號", default=default, type=int)
        if 1 <= value <= maximum:
            return value
        error_console.print(f"[red]請輸入 1 到 {maximum} 之間的數字。[/red]")


def _preferred_index(candidates: Sequence[Any], preferred: str | None) -> int:
    """找出偏好模型在候選清單中的位置。

    Args:
        candidates: 候選清單，元素需有 ``alias`` 屬性。
        preferred: 偏好的模型別名。

    Returns:
        對應的索引；找不到時回傳 0。
    """
    for index, candidate in enumerate(candidates):
        if candidate.alias == preferred:
            return index
    return 0


def _note_fallback(item: str, preferred: str | None, actual: str) -> None:
    """提醒使用者這一項改用了其他模型的結果。

    Args:
        item: 項目名稱，用於訊息。
        preferred: 原本偏好的模型別名。
        actual: 實際採用的模型別名。
    """
    if preferred is None or preferred == actual:
        return
    console.print(
        f"[yellow]提醒：[/yellow]{item} 沒有 {preferred} 的可用結果，改用 {actual}。"
    )


def _choose_keywords(
    candidates: list[KeywordCandidate], *, assume_yes: bool, preferred: str | None
) -> list[str]:
    """挑選最終要採用的關鍵字。

    Args:
        candidates: 可挑選的關鍵字組。
        assume_yes: 非互動模式時直接採用偏好模型的結果。
        preferred: 偏好的模型別名。

    Returns:
        最終關鍵字；不採用時為空清單。
    """
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

    choice = _ask_choice(skip_choice, default=default_index + 1)
    if choice == skip_choice:
        return []
    if choice == manual_choice:
        raw = typer.prompt("請輸入關鍵字，以逗號分隔")
        return [piece.strip() for piece in raw.split(",") if piece.strip()]
    return list(candidates[choice - 1].keywords)


def _choose_summary(
    candidates: list[SummaryCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    variant: int,
    min_chars: int,
    max_chars: int,
) -> str:
    """挑選最終要採用的摘要。

    Args:
        candidates: 攤平後的摘要候選。
        assume_yes: 非互動模式時直接採用偏好模型的指定版本。
        preferred: 偏好的模型別名。
        variant: 非互動模式要採用第幾則，從 1 起算。
        min_chars: 字元下限，僅用於顯示。
        max_chars: 字元上限，僅用於顯示。

    Returns:
        最終摘要；不採用時為空字串。
    """
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

    choice = _ask_choice(skip_choice, default=default_index + 1)
    if choice == skip_choice:
        return ""
    if choice == manual_choice:
        return " ".join(typer.prompt("請輸入摘要").split())
    return candidates[choice - 1].text


def _prompt_filename(extension: str, *, default_stem: str) -> str:
    """要求使用者輸入檔名主體，並正規化成安全的檔名。

    Args:
        extension: 原始副檔名，含點。
        default_stem: 直接按 Enter 時採用的主體。

    Returns:
        含副檔名的完整檔名。
    """
    raw = typer.prompt("請輸入檔名（不含副檔名）", default=default_stem)
    return build_filename(slugify(raw), extension)


def _choose_filename(
    key: str,
    entry: ImageEntry,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
) -> tuple[str | None, str | None]:
    """挑選某張圖片要改成的檔名。

    Args:
        key: 圖片在 JSON 中的鍵。
        entry: 圖片條目。
        candidates: 可挑選的建議。
        assume_yes: 非互動模式時直接採用偏好模型的結果。
        preferred: 偏好的模型別名。

    Returns:
        ``(檔名, 來源別名)``；選擇不改名時兩者皆為 ``None``。
    """
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

    choice = _ask_choice(keep_choice, default=default_index + 1)
    if choice == keep_choice:
        return None, None
    if choice == manual_choice:
        default_stem = Path(candidates[default_index].filename).stem
        filename = _prompt_filename(entry.extension, default_stem=default_stem)
        return filename, MANUAL_SOURCE
    chosen = candidates[choice - 1]
    return chosen.filename, chosen.alias


def _choose_alt(
    key: str,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    max_chars: int,
) -> tuple[str | None, str | None]:
    """挑選某張圖片要寫入的 alt 文字。

    跟檔名分開問，因為兩者的品質彼此無關：檔名取得漂亮的模型，alt 可能寫得
    又臭又長，反過來也一樣。

    Args:
        key: 圖片在 JSON 中的鍵。
        candidates: 可挑選的建議。
        assume_yes: 非互動模式時直接採用偏好模型的結果。
        preferred: 偏好的模型別名。
        max_chars: alt 的字元上限，僅用於顯示。

    Returns:
        ``(alt, 來源別名)``；選擇不改 alt 時兩者皆為 ``None``。
    """
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
            _format_length(candidate.alt_length, max_chars),
            candidate.alt,
        )
    manual_choice = len(candidates) + 1
    keep_choice = len(candidates) + 2
    table.add_row(str(manual_choice), "—", "—", "手動輸入")
    table.add_row(str(keep_choice), "—", "—", "不改 alt")
    console.print(table)

    choice = _ask_choice(keep_choice, default=default_index + 1)
    if choice == keep_choice:
        return None, None
    if choice == manual_choice:
        raw = typer.prompt("請輸入 alt 文字", default=candidates[default_index].alt)
        return " ".join(raw.split()), MANUAL_SOURCE
    chosen = candidates[choice - 1]
    return chosen.alt, chosen.alias


def _choose_image(
    key: str,
    entry: ImageEntry,
    candidates: list[ImageCandidate],
    *,
    assume_yes: bool,
    preferred: str | None,
    alt_max_chars: int,
) -> SelectionImage | None:
    """挑選某張圖片的檔名與 alt。

    Args:
        key: 圖片在 JSON 中的鍵。
        entry: 圖片條目。
        candidates: 可挑選的建議。
        assume_yes: 非互動模式時直接採用偏好模型的結果。
        preferred: 偏好的模型別名。
        alt_max_chars: alt 的字元上限，僅用於顯示。

    Returns:
        這張圖的決定；檔名與 alt 都不動時為 ``None``。
    """
    assert entry.resolved_path is not None
    filename, filename_model = _choose_filename(
        key, entry, candidates, assume_yes=assume_yes, preferred=preferred
    )
    alt, alt_model = _choose_alt(
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


def _resolve_conflicts(
    images: dict[str, SelectionImage], *, assume_yes: bool
) -> dict[str, SelectionImage]:
    """檢查檔名衝突，並在互動模式下請使用者當場改名。

    衝突留到 apply 才發現就只能中止，這裡還救得回來。

    Args:
        images: 每張圖片的決定。
        assume_yes: 非互動模式時直接報錯。

    Returns:
        沒有衝突的決定。

    Raises:
        typer.Exit: 非互動模式下有衝突，或互動模式下改了幾輪仍未解決。
    """
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


def _print_selection_summary(
    keywords: list[str],
    summary: str,
    images: dict[str, SelectionImage],
    *,
    alt_max_chars: int,
) -> None:
    """顯示最終的挑選結果。

    Args:
        keywords: 最終關鍵字。
        summary: 最終摘要。
        images: 每張圖片的決定。
        alt_max_chars: alt 的字元上限，僅用於顯示。
    """
    renames = sum(1 for item in images.values() if item.new_filename is not None)
    alts = sum(1 for item in images.values() if item.alt is not None)
    lines = [
        f"關鍵字：{'、'.join(keywords) if keywords else '（不採用）'}",
        f"摘要　：{summary or '（不採用）'}",
        f"圖片　：{renames} 張要改名，{alts} 張要改 alt",
    ]
    console.print(Panel("\n".join(lines), title="最終選擇", expand=False))

    if not images:
        return
    table = Table(title="圖片", title_justify="left", header_style="bold green")
    table.add_column("原路徑", overflow="fold")
    table.add_column("新檔名", overflow="fold")
    table.add_column("字數", justify="right", no_wrap=True)
    table.add_column("alt", overflow="fold")
    for key, item in images.items():
        if item.new_filename is None:
            filename_cell = "[dim]保持原檔名[/dim]"
        else:
            filename_cell = f"{item.new_filename} [dim]({item.filename_model})[/dim]"
        if item.alt is None:
            length_cell, alt_cell = "-", "[dim]不改[/dim]"
        else:
            length_cell = _format_length(len(item.alt), alt_max_chars)
            alt_cell = f"{item.alt} [dim]({item.alt_model})[/dim]"
        table.add_row(key, filename_cell, length_cell, alt_cell)
    console.print(table)


@app.command()
def review(
    analysis_path: Annotated[
        Path, typer.Argument(help="analyze 產生的 JSON 路徑", show_default=False)
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="輸出選擇檔的路徑或目錄", show_default=False),
    ] = None,
    pick: Annotated[
        str | None,
        typer.Option(
            "--pick",
            help="偏好的模型別名；互動模式下當作預設選項，搭配 --yes 則直接採用",
            show_default=False,
        ),
    ] = None,
    summary_index: Annotated[
        int,
        typer.Option("--summary-index", help="搭配 --yes 時採用第幾則摘要", min=1),
    ] = 1,
    assume_yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="不詢問，直接採用 --pick 指定模型的結果"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="即使文章在分析後被改過也繼續"),
    ] = False,
) -> None:
    """從分析結果中挑選要採用的內容，輸出選擇檔。

    全程只讀寫本地 JSON，不呼叫任何模型，因此不會產生費用，重跑幾次都可以。
    """
    try:
        result = load_analysis(analysis_path)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    _print_review_header(result, analysis_path)
    _confirm_freshness(result, assume_yes=assume_yes, force=force)

    available = sorted(result.metadata.models)
    if pick is not None and pick not in result.metadata.models:
        error_console.print(
            f"[red]錯誤：[/red]這份分析結果沒有 {pick} 的資料；"
            f"可用的有：{'、'.join(available) or '（無）'}"
        )
        raise typer.Exit(code=1)

    preferred = pick or default_alias(result)
    if preferred is None:
        error_console.print(
            "[red]錯誤：[/red]這份分析結果沒有任何成功的項目可挑選。"
        )
        raise typer.Exit(code=1)

    keywords = _choose_keywords(
        keyword_candidates(result), assume_yes=assume_yes, preferred=preferred
    )
    summary = _choose_summary(
        summary_candidates(result),
        assume_yes=assume_yes,
        preferred=preferred,
        variant=summary_index,
        min_chars=result.metadata.summary_min_chars,
        max_chars=result.metadata.summary_max_chars,
    )

    alt_max_chars = result.metadata.alt_max_chars
    images: dict[str, SelectionImage] = {}
    for key, entry in selectable_images(result).items():
        chosen = _choose_image(
            key,
            entry,
            image_candidates(entry),
            assume_yes=assume_yes,
            preferred=preferred,
            alt_max_chars=alt_max_chars,
        )
        if chosen is not None:
            images[key] = chosen
    images = _resolve_conflicts(images, assume_yes=assume_yes)

    try:
        selection = build_selection(
            result,
            source_analysis=analysis_path,
            keywords=keywords,
            summary=summary,
            images=images,
        )
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    _print_selection_summary(keywords, summary, images, alt_max_chars=alt_max_chars)

    output_path = _resolve_output_path(analysis_path, out, suffix="-selection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(selection.to_json())

    console.print(f"\n選擇已寫入 [bold green]{output_path}[/bold green]")
    console.print("[dim]原始檔案仍未被修改。接著執行 apply 才會真正改檔：[/dim]")
    console.print(f"[dim]  uv run blog-seo-ai apply {output_path}[/dim]")


def _print_apply_plan(plan: ApplyPlan) -> None:
    """顯示即將寫入的變更。

    Args:
        plan: 套用計畫。
    """
    lines = [
        f"文章　　：{plan.article_path}",
        f"圖片語法：{len(plan.markup_changes)} 處",
        f"檔案改名：{len(plan.renames)} 張",
    ]
    console.print(Panel("\n".join(lines), title="即將套用", expand=False))

    if plan.markup_changes:
        table = Table(title="Markdown 變更", title_justify="left", header_style="bold cyan")
        table.add_column("行", justify="right", no_wrap=True)
        table.add_column("圖片", overflow="fold")
        table.add_column("寫成", overflow="fold")
        for change in plan.markup_changes:
            table.add_row(str(change.line), change.key, change.new)
        console.print(table)

    if plan.renames:
        table = Table(title="檔案改名", title_justify="left", header_style="bold magenta")
        table.add_column("原檔名", overflow="fold")
        table.add_column("新檔名", overflow="fold")
        for change in plan.renames:
            table.add_row(change.source.name, change.dest.name)
        console.print(table)


def _print_gitless_outcome(outcome) -> None:
    """說明 Gitless Sync 清單是否已更新，以及接下來要不要重載。"""
    if outcome is None:
        return
    if outcome.warning:
        error_console.print(f"[yellow]警告：[/yellow]{outcome.warning}")
        return
    if outcome.metadata_path is None:
        return

    parts: list[str] = []
    if outcome.marked_deleted:
        parts.append(f"舊圖 {len(outcome.marked_deleted)} 張標刪除")
    if outcome.added:
        parts.append(f"新檔 {len(outcome.added)} 個待上傳")
    if outcome.marked_dirty:
        parts.append(f"既有檔 {len(outcome.marked_dirty)} 個標為已改")
    if parts:
        console.print("[dim]已更新 Gitless Sync 清單：" + "、".join(parts) + "。[/dim]")
    console.print(
        "[dim]套用時若 Obsidian 是關著的：直接開啟後按同步即可。"
        "若當時開著：Ctrl+P → Reload app without saving，再同步。[/dim]"
    )


@app.command()
def apply(
    selection_path: Annotated[
        Path, typer.Argument(help="review 產生的選擇檔路徑", show_default=False)
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="只顯示將要做的變更，不寫入任何檔案"),
    ] = False,
    assume_yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="不詢問，直接套用"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="即使文章在分析後被改過、或缺少雜湊也繼續"),
    ] = False,
    backup: Annotated[
        bool,
        typer.Option("--backup", help="寫入前把原文複製成 .bak"),
    ] = False,
) -> None:
    """依選擇檔改寫圖片語法並把圖片改名。

    這是唯一會修改使用者原始檔案的指令。不會改 front matter。
    建議先用 ``--dry-run`` 看過計畫。
    """
    try:
        selection = load_selection(selection_path)
        plan = build_plan(selection, force=force)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    if plan.hash_mismatch:
        error_console.print(
            "[yellow]警告：[/yellow]文章正文與分析當時不一致，已依 --force 用當下內容套用。"
        )

    if plan.is_empty:
        console.print("沒有需要套用的變更。")
        return

    _print_apply_plan(plan)

    if dry_run:
        console.print("[dim]這是預覽，原始檔案未被修改。[/dim]")
        return

    if not assume_yes and not typer.confirm("確定寫入這些變更？", default=False):
        console.print("已取消。")
        raise typer.Exit(code=1)

    try:
        gitless = execute_plan(plan, backup=backup)
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"\n已套用 [bold green]{plan.article_path}[/bold green]")
    if backup and plan.new_text != plan.original_text:
        console.print(f"[dim]原文備份：{plan.article_path.name}.bak[/dim]")
    _print_gitless_outcome(gitless)


@app.command("providers")
def list_providers() -> None:
    """列出可用的模型別名。"""
    table = Table(title="可用的模型別名", title_justify="left")
    table.add_column("別名")
    for alias in known_aliases():
        table.add_row(alias)
    console.print(table)
    console.print(
        "[dim]預設 --models hf（文字 Qwen3-4B、圖片 GLM-4.6V-Flash）。"
        "也可用 --models claude，或 --models hf:Qwen/Qwen3-8B[/dim]"
    )
