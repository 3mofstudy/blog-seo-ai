"""終端機表格與提示輸出。

顯示邏輯集中在這裡，方便 CLI 指令與互動式流程共用，也讓單元測試
不必透過 typer 才能檢查字串格式。
"""

from __future__ import annotations

from blogseo.markdown.parser import ParsedArticle
from blogseo.schemas.result import AnalysisField, AnalysisResult
from blogseo.schemas.selection import SelectionImage
from blogseo.seo.applier import ApplyPlan
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def format_length(length: int, limit: int) -> str:
    """把字數格式化成表格用的字串，超出上限標成黃色。"""
    return str(length) if length <= limit else f"[yellow]{length}[/yellow]"


def format_twd(amount: float | None) -> str:
    """把新台幣金額格式化成表格用的字串。"""
    if amount is None:
        return "—"
    if amount < 1:
        return f"{amount:.4f}"
    return f"{amount:,.2f}"


def print_article_panel(
    article: ParsedArticle, tokens: list[str], fields: frozenset[AnalysisField]
) -> None:
    """顯示文章基本資訊。"""
    analyzable = sum(1 for target in article.targets if target.analyzable)
    lines = [
        f"標題　　：{article.title or '（無）'}",
        f"語言　　：{article.language}　字數：{article.word_count}",
        f"圖片　　：共 {len(article.targets)} 張，可分析 {analyzable} 張",
        f"模型　　：{'、'.join(tokens)}",
        f"產出項目：{'、'.join(sorted(field.value for field in fields))}",
    ]
    console.print(Panel("\n".join(lines), title=article.path.name, expand=False))


def print_skipped(result: AnalysisResult) -> None:
    """列出被略過的圖片與原因。"""
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


def print_keyword_table(result: AnalysisResult) -> None:
    """顯示各模型的關鍵字。"""
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


def print_summary_table(result: AnalysisResult) -> None:
    """顯示各模型的多個摘要版本，並標出超出字數範圍的版本。"""
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


def print_image_tables(result: AnalysisResult) -> None:
    """顯示各模型對每張圖的建議。"""
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
                format_length(len(analysis.result.alt), limit),
                analysis.result.alt,
            )
        console.print(table)


def print_usage_table(result: AnalysisResult) -> None:
    """顯示 token 用量、耗時與估算花費。"""
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
            format_twd(usage.estimated_cost_twd),
        )

    caption = f"整體耗時 {metadata.total_elapsed_ms / 1000:.1f} 秒"
    if metadata.total_cost_twd is not None:
        caption += (
            f"　合計 NT$ {format_twd(metadata.total_cost_twd)}"
            f"（US$ {metadata.total_cost_usd:.4f}，匯率 {metadata.usd_to_twd_rate}）"
        )
    table.caption = caption
    console.print(table)
    if any(usage.estimated_cost_twd is None for usage in metadata.models.values()):
        console.print(
            "[dim]部分模型不在內建價目表中，費用顯示為「—」。"
            "價目表在 llm/anthropic.py 的 pricing。[/dim]"
        )


def print_analysis_sections(result: AnalysisResult, fields: frozenset[AnalysisField]) -> None:
    """依本次要求的項目顯示分析結果。"""
    if AnalysisField.KEYWORDS in fields:
        print_keyword_table(result)
    if AnalysisField.SUMMARY in fields:
        print_summary_table(result)
    if AnalysisField.IMAGES in fields:
        print_image_tables(result)
        print_skipped(result)
    print_usage_table(result)


def print_review_header(result: AnalysisResult, source: str) -> None:
    """顯示這份分析結果的基本資訊。"""
    info = result.article_info
    lines = [
        f"文章　　：{info.title or '（無）'}",
        f"路徑　　：{info.path}",
        f"分析時間：{result.metadata.generated_at}",
        f"模型　　：{'、'.join(result.metadata.models) or '（無）'}",
        f"產出項目：{'、'.join(result.metadata.requested_fields)}",
    ]
    console.print(Panel("\n".join(lines), title=source, expand=False))


def print_selection_summary(
    keywords: list[str],
    summary: str,
    images: dict[str, SelectionImage],
    *,
    alt_max_chars: int,
) -> None:
    """顯示最終的挑選結果。"""
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
            length_cell = format_length(len(item.alt), alt_max_chars)
            alt_cell = f"{item.alt} [dim]({item.alt_model})[/dim]"
        table.add_row(key, filename_cell, length_cell, alt_cell)
    console.print(table)


def print_apply_plan(plan: ApplyPlan) -> None:
    """顯示即將寫入的變更。"""
    lines = [
        f"文章　　：{plan.article_path}",
        f"圖片語法：{len(plan.markup_changes)} 處",
        f"檔案改名：{len(plan.renames)} 張",
    ]
    if plan.front_matter_updates:
        lines.append(f"front matter：{len(plan.front_matter_updates)} 個欄位")
    console.print(Panel("\n".join(lines), title="即將套用", expand=False))

    if plan.front_matter_updates:
        table = Table(title="Front matter", title_justify="left", header_style="bold yellow")
        table.add_column("欄位", no_wrap=True)
        table.add_column("寫入", overflow="fold")
        for key, value in plan.front_matter_updates.items():
            if isinstance(value, list):
                cell = "、".join(str(item) for item in value)
            else:
                cell = str(value)
            table.add_row(key, cell)
        console.print(table)

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


def print_gitless_outcome(outcome) -> None:
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
