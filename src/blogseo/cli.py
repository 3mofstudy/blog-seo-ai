"""命令列進入點。

預設 ``blog-seo-ai [文章.md]`` 會開互動選單，一次跑完
分析 → 挑選 → 套用。``analyze`` / ``review`` / ``apply`` 仍保留給腳本使用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from blogseo import __version__
from blogseo.display import console, error_console
from blogseo.errors import BlogSeoError
from blogseo.llm.registry import list_model_presets, parse_model_tokens
from blogseo.pipeline import (
    resolve_output_path,
    run_analyze_session,
    run_apply_session,
    run_interactive,
    run_review_session,
)
from blogseo.schemas.result import AnalysisField
from blogseo.seo.analyzer import AnalyzeOptions, options_from_settings, parse_fields
from blogseo.seo.applier import load_selection
from blogseo.seo.selector import load_analysis
from blogseo.settings import load_settings
from blogseo.settings_ui import run_settings_menu

_SUBCOMMANDS = frozenset({"analyze", "review", "apply", "providers", "run", "settings"})

app = typer.Typer(
    name="blog-seo-ai",
    help=(
        "用 LLM 分析部落格文章，產生 SEO 關鍵字、摘要與圖片 alt。"
        "不加子指令時會進入互動選單，一次跑完分析、挑選與套用。"
    ),
    no_args_is_help=False,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """處理 ``--version`` 旗標。"""
    if value:
        console.print(f"blog-seo-ai {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="顯示版本"),
    ] = False,
) -> None:
    """blog-seo-ai 的共用選項。"""


@app.command()
def run(
    markdown_path: Annotated[
        Path | None,
        typer.Argument(help="要分析的 Markdown 檔案；也可之後在選單裡再填", show_default=False),
    ] = None,
) -> None:
    """互動式一次跑完：選項目 → 分析 → 挑選 → 套用。"""
    run_interactive(markdown_path)


@app.command("settings")
def edit_settings() -> None:
    """開啟設定選單，把預設值寫進 blogseo.json。"""
    try:
        run_settings_menu()
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc


def _warn_about_unused_options(
    fields: frozenset[AnalysisField],
    options: AnalyzeOptions,
    settings_summary_count: int,
    settings_summary_min: int,
    settings_summary_max: int,
    settings_alt_max: int,
) -> None:
    """提醒哪些參數在這次的 ``--fields`` 之下不會生效。"""
    unused: list[str] = []
    if AnalysisField.SUMMARY not in fields:
        if options.summary_count != settings_summary_count:
            unused.append("--summaries")
        if options.summary_min_chars != settings_summary_min:
            unused.append("--summary-min")
        if options.summary_max_chars != settings_summary_max:
            unused.append("--summary-max")
    if AnalysisField.IMAGES not in fields:
        if options.max_images is not None:
            unused.append("--max-images")
        if options.alt_max_chars != settings_alt_max:
            unused.append("--alt-max")
    if unused:
        console.print(
            f"[yellow]提醒：[/yellow]{'、'.join(unused)} 在本次 --fields 之下不會生效。"
        )


@app.command()
def analyze(
    markdown_path: Annotated[
        Path, typer.Argument(help="要分析的 Markdown 檔案路徑", show_default=False)
    ],
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            "-m",
            help="以逗號分隔的模型別名；未指定時分別使用 blogseo.json 的文字／圖片模型",
            show_default=False,
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="輸出 JSON 的路徑或目錄", show_default=False),
    ] = None,
    fields: Annotated[
        str,
        typer.Option(
            "--fields",
            "-f",
            help="要產生哪些項目，逗號分隔。可用 keywords、summary、images。",
        ),
    ] = "keywords,summary,images",
    max_images: Annotated[
        int | None,
        typer.Option("--max-images", help="最多分析幾張圖片，用來控制花費", show_default=False),
    ] = None,
    alt_lang: Annotated[
        str | None,
        typer.Option("--alt-lang", help="alt 文字語言，auto 表示跟隨文章語言", show_default=False),
    ] = None,
    alt_max: Annotated[
        int | None,
        typer.Option("--alt-max", help="alt 文字的字元上限", min=10, show_default=False),
    ] = None,
    summaries: Annotated[
        int | None,
        typer.Option("--summaries", help="每個模型產生幾個摘要版本", min=1, max=5, show_default=False),
    ] = None,
    summary_min: Annotated[
        int | None,
        typer.Option("--summary-min", help="摘要字元下限", min=10, show_default=False),
    ] = None,
    summary_max: Annotated[
        int | None,
        typer.Option("--summary-max", help="摘要字元上限", min=20, show_default=False),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", "-c", help="平行呼叫上限", min=1, max=16, show_default=False),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="單次 API 請求逾時秒數", min=5.0, show_default=False),
    ] = None,
    twd_rate: Annotated[
        float | None,
        typer.Option("--twd-rate", help="費用換算成新台幣使用的匯率", min=0.1, show_default=False),
    ] = None,
) -> None:
    """分析文章並輸出 JSON，不會修改任何原始檔案。"""
    try:
        settings = load_settings()
        selected_fields = parse_fields(fields)
        options = options_from_settings(settings, fields=selected_fields)
        if models is not None:
            tokens = parse_model_tokens(models)
            options.model_tokens = tokens
            options.text_tokens = None
            options.image_tokens = None
        if alt_lang is not None:
            options.alt_language = alt_lang
        if summaries is not None:
            options.summary_count = summaries
        if summary_min is not None:
            options.summary_min_chars = summary_min
        if summary_max is not None:
            options.summary_max_chars = summary_max
        if alt_max is not None:
            options.alt_max_chars = alt_max
        if max_images is not None:
            options.max_images = max_images
        if concurrency is not None:
            options.concurrency = concurrency
        if timeout is not None:
            options.timeout = timeout
        if twd_rate is not None:
            options.usd_to_twd_rate = twd_rate

        if options.summary_min_chars >= options.summary_max_chars:
            error_console.print(
                f"[red]錯誤：[/red]摘要下限（{options.summary_min_chars}）"
                f"必須小於上限（{options.summary_max_chars}）"
            )
            raise typer.Exit(code=1)

    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    _warn_about_unused_options(
        selected_fields,
        options,
        settings_summary_count=settings.summary.count,
        settings_summary_min=settings.summary.min_chars,
        settings_summary_max=settings.summary.max_chars,
        settings_alt_max=settings.alt.max_chars,
    )

    try:
        _result, output_path = run_analyze_session(
            markdown_path, options, output_dir=settings.output_dir, out=out
        )
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    # run_analyze_session 已印過文章資訊一次；這裡避免重複，只補下一步提示。
    console.print("[dim]接著可執行 review 挑選，或用互動模式一次跑完：[/dim]")
    console.print(f"[dim]  blog-seo-ai review {output_path}[/dim]")


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
    """從分析結果中挑選要採用的內容，輸出選擇檔。"""
    try:
        settings = load_settings()
        result = load_analysis(analysis_path)
        selection = run_review_session(
            result,
            analysis_path,
            assume_yes=assume_yes,
            force=force,
            pick=pick,
            summary_index=summary_index,
        )
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc

    output_path = resolve_output_path(
        analysis_path, out, suffix="-selection.json", output_dir=settings.output_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(selection.to_json(), encoding="utf-8")
    console.print(f"\n選擇已寫入 [bold green]{output_path}[/bold green]")
    console.print("[dim]原始檔案仍未被修改。接著執行 apply 才會真正改檔：[/dim]")
    console.print(f"[dim]  blog-seo-ai apply {output_path}[/dim]")


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
        bool | None,
        typer.Option("--backup/--no-backup", help="寫入前把原文複製成 .bak", show_default=False),
    ] = None,
    front_matter: Annotated[
        bool | None,
        typer.Option(
            "--front-matter/--no-front-matter",
            help="是否把關鍵字與摘要寫進 front matter；未指定則讀取 blogseo.json",
            show_default=False,
        ),
    ] = None,
) -> None:
    """依選擇檔改寫圖片語法、檔名，並可把關鍵字／摘要寫進 front matter。"""
    try:
        settings = load_settings()
        selection = load_selection(selection_path)
        run_apply_session(
            selection,
            settings,
            assume_yes=assume_yes,
            force=force,
            dry_run=dry_run,
            backup=backup,
            write_front_matter=front_matter,
        )
    except BlogSeoError as exc:
        error_console.print(f"[red]錯誤：[/red]{exc}")
        raise typer.Exit(code=1) from exc


@app.command("providers")
def list_providers() -> None:
    """列出已實作的預設模型型號。"""
    table = Table(title="已實作的預設模型", title_justify="left")
    table.add_column("用途")
    table.add_column("別名")
    table.add_column("型號")
    table.add_column("說明")
    for role, usage in (("text", "關鍵字／摘要"), ("image", "圖片辨識")):
        for preset in list_model_presets(role):
            table.add_row(usage, preset.alias, preset.model_id, preset.note)
    console.print(table)
    console.print(
        "[dim]上列是各供應商的內建預設型號，設定選單可改成同一家的其他型號。"
        "hf、huggingface、qwen 都是 Hugging Face；claude、anthropic 都是 Claude；"
        "gpt、openai 都是 OpenAI；gemini、google 都是 Gemini。[/dim]"
    )


def _inject_run_command() -> None:
    """沒有子指令時，把路徑或空白參數導向互動式 ``run``。"""
    args = sys.argv[1:]
    if not args:
        sys.argv.append("run")
        return
    first = args[0]
    if first in _SUBCOMMANDS or first.startswith("-"):
        return
    sys.argv.insert(1, "run")


def main() -> None:
    """套件進入點。"""
    # httpx 預設只信 certifi。Windows 上防毒或代理裝的憑證在系統憑證庫，
    # 不注入的話會以 self-signed certificate in certificate chain 中斷連線。
    import truststore

    truststore.inject_into_ssl()
    _inject_run_command()
    app()
