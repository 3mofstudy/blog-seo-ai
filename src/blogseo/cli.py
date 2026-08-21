"""命令列進入點。

目前提供 ``analyze``：讀取文章與圖片、呼叫指定模型、把結果落地成 JSON。
依專案硬性原則，這個階段完全不會修改使用者的任何原始檔案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
    DEFAULT_OUTPUT_DIR,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
    USD_TO_TWD_RATE,
)
from blogseo.errors import BlogSeoError
from blogseo.llm.registry import known_aliases, parse_model_tokens
from blogseo.markdown.parser import ParsedArticle, parse_article
from blogseo.schemas.result import AnalysisField, AnalysisResult
from blogseo.seo.analyzer import AnalyzeOptions, analyze_article, parse_fields

#: ``--fields`` 的預設值，字串形式方便直接顯示在說明裡。
_DEFAULT_FIELDS_ARG = "keywords,summary,images"

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


def _print_image_tables(result: AnalysisResult) -> None:
    """顯示各模型對每張圖的建議。

    Args:
        result: 分析結果。
    """
    for key, entry in result.images.items():
        if not entry.models:
            continue
        table = Table(title=f"圖片：{key}", title_justify="left", header_style="bold magenta")
        table.add_column("模型", no_wrap=True)
        table.add_column("建議檔名", overflow="fold")
        table.add_column("alt", overflow="fold")
        for alias, analysis in entry.models.items():
            if analysis.error is not None:
                table.add_row(alias, "[red]失敗[/red]", f"[red]{analysis.error}[/red]")
                continue
            assert analysis.result is not None
            table.add_row(alias, analysis.final_filename or "", analysis.result.alt)
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
    max_images: int | None,
) -> None:
    """提醒使用者哪些參數在這次的 ``--fields`` 之下不會生效。

    靜靜忽略參數比報錯更容易讓人困惑，所以這裡明講。

    Args:
        fields: 這次要產生的項目。
        summaries: ``--summaries`` 的值。
        summary_min: ``--summary-min`` 的值。
        summary_max: ``--summary-max`` 的值。
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

    if AnalysisField.IMAGES not in fields and max_images is not None:
        unused.append("--max-images")

    if unused:
        console.print(
            f"[yellow]提醒：[/yellow]{'、'.join(unused)} 在本次 --fields 之下不會生效。"
        )


def _resolve_output_path(markdown_path: Path, out: Path | None) -> Path:
    """決定 JSON 的輸出路徑。

    Args:
        markdown_path: 來源 Markdown 路徑。
        out: 使用者指定的輸出路徑或目錄。

    Returns:
        實際要寫入的檔案路徑。
    """
    default_name = f"{markdown_path.stem}-analysis.json"
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
            help="以逗號分隔的模型別名，可用「別名:模型id」覆寫，例如 claude:claude-opus-5",
        ),
    ] = "claude",
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

    output_path = _resolve_output_path(markdown_path, out)
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
    console.print("[dim]原始檔案未被修改；挑選與套用會由後續的 apply 指令處理。[/dim]")


@app.command("providers")
def list_providers() -> None:
    """列出可用的模型別名。"""
    table = Table(title="可用的模型別名", title_justify="left")
    table.add_column("別名")
    for alias in known_aliases():
        table.add_row(alias)
    console.print(table)
    console.print("[dim]用法：--models claude 或 --models claude:claude-opus-5[/dim]")
