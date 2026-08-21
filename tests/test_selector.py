"""review 階段的單元測試。

重點在三件事：分析結果是外部輸入，要能擋掉不相容的檔案；候選列舉不能把
失敗的格子端出來讓人挑；檔名衝突必須在寫出選擇檔之前就被抓到。
全程不呼叫任何 API，也不修改測試以外的檔案。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from blogseo.cli import app
from blogseo.errors import AnalysisLoadError, SelectionError
from blogseo.markdown.parser import parse_article
from blogseo.schemas.result import (
    SCHEMA_VERSION,
    AnalysisResult,
    ArticleInfo,
    ImageAnalysisResult,
    ImageEntry,
    ImageOccurrence,
    KeywordSummaryResult,
    Metadata,
    ModelImageAnalysis,
    ModelKeywordSummary,
    ModelUsage,
)
from blogseo.schemas.selection import Selection, SelectionImage
from blogseo.seo.selector import (
    LEGACY_ALT_MAX_CHARS,
    article_changed,
    build_selection,
    find_conflicts,
    image_candidates,
    keyword_candidates,
    load_analysis,
    selectable_images,
    summary_candidates,
)

runner = CliRunner()


def _make_article(tmp_path: Path, image_count: int = 2) -> Path:
    """建立含真實 PNG 的測試文章。

    Args:
        tmp_path: pytest 暫存目錄。
        image_count: 要產生的圖片數量。

    Returns:
        Markdown 檔案路徑。
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 測試文章", "", "一些內容。", ""]
    for index in range(image_count):
        Image.new("RGB", (40, 30), (index * 40, 100, 200)).save(images_dir / f"pic{index}.png")
        lines += [f"![圖 {index}](images/pic{index}.png)", ""]

    markdown_path = tmp_path / "post.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def _make_analysis(markdown_path: Path, *, fail_gpt_text: bool = False) -> AnalysisResult:
    """依真實文章手工組出一份分析結果。

    直接建構而不跑 analyze，測試才不必依賴 provider 的行為。

    Args:
        markdown_path: 測試文章路徑。
        fail_gpt_text: 是否讓 gpt 的文章分析標記為失敗。

    Returns:
        兩個模型並列的分析結果。
    """
    article = parse_article(markdown_path)
    keyword_summary = {
        "claude": ModelKeywordSummary(
            model_id="claude-sonnet-5",
            result=KeywordSummaryResult(
                keywords=["甲", "乙", "丙"], summaries=["第一則摘要", "第二則摘要"]
            ),
            summary_lengths=[5, 5],
        ),
        "gpt": ModelKeywordSummary(
            model_id="gpt-5",
            result=None if fail_gpt_text else KeywordSummaryResult(
                keywords=["丁", "戊"], summaries=["另一則摘要"]
            ),
            error="連線失敗" if fail_gpt_text else None,
        ),
    }

    images: dict[str, ImageEntry] = {}
    for index, target in enumerate(article.targets):
        images[target.key] = ImageEntry(
            source=target.source,
            resolved_path=str(target.resolved_path),
            exists=True,
            extension=".png",
            occurrences=[
                ImageOccurrence(
                    line=ref.line,
                    start=ref.start,
                    end=ref.end,
                    syntax=ref.syntax,
                    raw=ref.raw,
                    original_alt=ref.alt,
                )
                for ref in target.references
            ],
            models={
                "claude": ModelImageAnalysis(
                    model_id="claude-sonnet-5",
                    result=ImageAnalysisResult(
                        suggested_filename=f"claude-chart-{index}", alt=f"圖 {index}"
                    ),
                    final_filename=f"claude-chart-{index}.png",
                ),
                "gpt": ModelImageAnalysis(model_id="gpt-5", error="連線失敗"),
            },
        )

    return AnalysisResult(
        article_info=ArticleInfo(
            path=str(article.path),
            title=article.title,
            language=article.language,
            word_count=article.word_count,
            image_count=len(article.targets),
            content_hash=article.content_hash,
            frontmatter_keys=[],
        ),
        keyword_summary=keyword_summary,
        images=images,
        metadata=Metadata(
            generated_at="2026-08-21T09:00:00+00:00",
            tool_version="0.1.0",
            total_elapsed_ms=1000,
            requested_fields=["images", "keywords", "summary"],
            alt_language="zh",
            summary_min_chars=60,
            summary_max_chars=150,
            summary_count=2,
            alt_max_chars=30,
            usd_to_twd_rate=31.8,
            models={
                "claude": ModelUsage(model_id="claude-sonnet-5", calls=3),
                "gpt": ModelUsage(model_id="gpt-5", calls=3, failed_calls=3),
            },
        ),
    )


def _write_analysis(tmp_path: Path, result: AnalysisResult) -> Path:
    """把分析結果寫成檔案。

    Args:
        tmp_path: pytest 暫存目錄。
        result: 分析結果。

    Returns:
        JSON 檔案路徑。
    """
    path = tmp_path / "post-analysis.json"
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AnalysisLoadError, match="找不到"):
        load_analysis(tmp_path / "nope.json")


def test_load_rejects_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ 不是 JSON", encoding="utf-8")

    with pytest.raises(AnalysisLoadError, match="不是合法的 JSON"):
        load_analysis(path)


def _write_legacy_analysis(tmp_path: Path, markdown_path: Path) -> Path:
    """寫出一份沒有 content_hash 的 1.2 版分析結果。

    Args:
        tmp_path: pytest 暫存目錄。
        markdown_path: 測試文章路徑。

    Returns:
        JSON 檔案路徑。
    """
    payload = json.loads(_make_analysis(markdown_path).to_json())
    payload["schema_version"] = "1.2"
    payload["metadata"]["schema_version"] = "1.2"
    del payload["article_info"]["content_hash"]
    del payload["metadata"]["alt_max_chars"]
    path = tmp_path / "legacy-analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_rejects_old_schema(tmp_path: Path) -> None:
    payload = json.loads(_make_analysis(_make_article(tmp_path)).to_json())
    payload["schema_version"] = "1.1"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AnalysisLoadError, match="請重新執行 analyze"):
        load_analysis(path)


def test_load_accepts_1_2_without_hash(tmp_path: Path) -> None:
    path = _write_legacy_analysis(tmp_path, _make_article(tmp_path))
    loaded = load_analysis(path)

    assert loaded.article_info.content_hash == ""
    assert loaded.metadata.alt_max_chars == LEGACY_ALT_MAX_CHARS
    assert article_changed(loaded) is None


def test_load_round_trips(tmp_path: Path) -> None:
    result = _make_analysis(_make_article(tmp_path))
    loaded = load_analysis(_write_analysis(tmp_path, result))

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.article_info.content_hash == result.article_info.content_hash


def test_candidates_skip_failed_models(tmp_path: Path) -> None:
    result = _make_analysis(_make_article(tmp_path), fail_gpt_text=True)

    assert [item.alias for item in keyword_candidates(result)] == ["claude"]
    assert {item.alias for item in summary_candidates(result)} == {"claude"}
    for entry in result.images.values():
        assert [item.alias for item in image_candidates(entry)] == ["claude"]


def test_summary_candidates_are_flattened(tmp_path: Path) -> None:
    candidates = summary_candidates(_make_analysis(_make_article(tmp_path)))

    assert [(item.alias, item.variant) for item in candidates] == [
        ("claude", 1),
        ("claude", 2),
        ("gpt", 1),
    ]
    assert candidates[0].length == len("第一則摘要")


def test_selectable_images_excludes_skipped_and_failed(tmp_path: Path) -> None:
    result = _make_analysis(_make_article(tmp_path))
    keys = list(result.images)
    result.images[keys[0]].skipped_reason = "外部連結，不做本地改名"
    result.images[keys[1]].models["claude"] = ModelImageAnalysis(
        model_id="claude-sonnet-5", error="壞掉了"
    )

    assert selectable_images(result) == {}


def _selection_image(path: Path, filename: str) -> SelectionImage:
    """組出一筆圖片決定。

    Args:
        path: 圖片目前的絕對路徑。
        filename: 要改成的檔名。

    Returns:
        圖片決定。
    """
    return SelectionImage(
        source=path.name,
        resolved_path=str(path),
        new_filename=filename,
        filename_model="claude",
        alt="替代文字",
        alt_model="claude",
    )


def test_conflict_when_two_images_share_a_name(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (4, 4)).save(first)
    Image.new("RGB", (4, 4)).save(second)

    conflicts = find_conflicts(
        {
            "a.png": _selection_image(first, "chart.png"),
            "b.png": _selection_image(second, "chart.png"),
        }
    )

    assert len(conflicts) == 1
    assert conflicts[0].sources == ["a.png", "b.png"]


def test_conflict_when_target_already_exists(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(source)
    Image.new("RGB", (4, 4)).save(tmp_path / "chart.png")

    conflicts = find_conflicts({"a.png": _selection_image(source, "chart.png")})

    assert len(conflicts) == 1
    assert "已經存在" in conflicts[0].reason


def test_swapping_names_is_not_a_conflict(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (4, 4)).save(first)
    Image.new("RGB", (4, 4)).save(second)

    conflicts = find_conflicts(
        {
            "a.png": _selection_image(first, "b.png"),
            "b.png": _selection_image(second, "a.png"),
        }
    )

    assert conflicts == []


def test_same_directory_only(tmp_path: Path) -> None:
    left = tmp_path / "one"
    right = tmp_path / "two"
    left.mkdir()
    right.mkdir()
    Image.new("RGB", (4, 4)).save(left / "a.png")
    Image.new("RGB", (4, 4)).save(right / "a.png")

    conflicts = find_conflicts(
        {
            "one/a.png": _selection_image(left / "a.png", "chart.png"),
            "two/a.png": _selection_image(right / "a.png", "chart.png"),
        }
    )

    assert conflicts == []


def test_build_selection_rejects_empty(tmp_path: Path) -> None:
    result = _make_analysis(_make_article(tmp_path))

    with pytest.raises(SelectionError, match="沒有挑選任何項目"):
        build_selection(
            result,
            source_analysis=tmp_path / "post-analysis.json",
            keywords=[],
            summary="",
            images={},
        )


def test_article_changed_detects_edit(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path)
    result = _make_analysis(markdown_path)
    assert article_changed(result) is False

    markdown_path.write_text("完全不一樣的內容", encoding="utf-8")
    assert article_changed(result) is True

    markdown_path.unlink()
    assert article_changed(result) is None


def test_frontmatter_edit_does_not_count_as_change(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path)
    result = _make_analysis(markdown_path)
    body = markdown_path.read_text(encoding="utf-8")
    markdown_path.write_text(f"---\ntitle: 新標題\n---\n{body}", encoding="utf-8")

    assert article_changed(result) is False


def test_review_writes_selection_without_prompting(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path)
    analysis_path = _write_analysis(tmp_path, _make_analysis(markdown_path))
    out_path = tmp_path / "post-selection.json"

    outcome = runner.invoke(
        app,
        [
            "review",
            str(analysis_path),
            "--pick",
            "claude",
            "--summary-index",
            "2",
            "--yes",
            "--out",
            str(out_path),
        ],
    )

    assert outcome.exit_code == 0, outcome.output
    selection = Selection.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert selection.keywords == ["甲", "乙", "丙"]
    assert selection.summary == "第二則摘要"
    assert len(selection.images) == 2
    assert selection.content_hash == parse_article(markdown_path).content_hash
    assert all(item.filename_model == "claude" for item in selection.images.values())
    assert all(item.alt_model == "claude" for item in selection.images.values())


def test_review_stops_when_article_changed(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path)
    analysis_path = _write_analysis(tmp_path, _make_analysis(markdown_path))
    markdown_path.write_text("改過了", encoding="utf-8")

    outcome = runner.invoke(
        app, ["review", str(analysis_path), "--pick", "claude", "--yes"]
    )

    assert outcome.exit_code == 1
    assert "被編輯過" in outcome.output


def test_review_warns_about_legacy_analysis(tmp_path: Path) -> None:
    path = _write_legacy_analysis(tmp_path, _make_article(tmp_path))
    out_path = tmp_path / "legacy-selection.json"

    blocked = runner.invoke(app, ["review", str(path), "--pick", "claude", "--yes"])
    assert blocked.exit_code == 1
    assert "沒有記錄正文雜湊" in blocked.output

    forced = runner.invoke(
        app,
        ["review", str(path), "--pick", "claude", "--yes", "--force", "--out", str(out_path)],
    )
    assert forced.exit_code == 0, forced.output
    assert Selection.model_validate_json(out_path.read_text(encoding="utf-8")).content_hash == ""


def test_review_rejects_unknown_alias(tmp_path: Path) -> None:
    analysis_path = _write_analysis(tmp_path, _make_analysis(_make_article(tmp_path)))

    outcome = runner.invoke(
        app, ["review", str(analysis_path), "--pick", "llama", "--yes"]
    )

    assert outcome.exit_code == 1
    assert "llama" in outcome.output


def test_review_interactive_choices(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path, image_count=1)
    analysis_path = _write_analysis(tmp_path, _make_analysis(markdown_path))
    out_path = tmp_path / "picked.json"

    # 關鍵字選 gpt（2）、摘要選 claude 的第二則（2）、
    # 檔名選 claude（1）、alt 選不改（3）。
    outcome = runner.invoke(
        app,
        ["review", str(analysis_path), "--out", str(out_path)],
        input="2\n2\n1\n3\n",
    )

    assert outcome.exit_code == 0, outcome.output
    selection = Selection.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert selection.keywords == ["丁", "戊"]
    assert selection.summary == "第二則摘要"
    chosen = selection.images["images/pic0.png"]
    assert chosen.new_filename == "claude-chart-0.png"
    assert chosen.alt is None
    assert chosen.alt_model is None


def test_review_can_take_alt_without_renaming(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path, image_count=1)
    analysis_path = _write_analysis(tmp_path, _make_analysis(markdown_path))
    out_path = tmp_path / "alt-only.json"

    # 關鍵字與摘要都不採用（4、5），檔名保持原樣（3），alt 選 claude（1）。
    outcome = runner.invoke(
        app,
        ["review", str(analysis_path), "--out", str(out_path)],
        input="4\n5\n3\n1\n",
    )

    assert outcome.exit_code == 0, outcome.output
    selection = Selection.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert selection.keywords == []
    assert selection.summary == ""
    chosen = selection.images["images/pic0.png"]
    assert chosen.new_filename is None
    assert chosen.filename_model is None
    assert chosen.alt == "圖 0"


def test_review_drops_images_left_untouched(tmp_path: Path) -> None:
    markdown_path = _make_article(tmp_path, image_count=1)
    analysis_path = _write_analysis(tmp_path, _make_analysis(markdown_path))
    out_path = tmp_path / "no-images.json"

    # 關鍵字選 claude（1）、摘要選第一則（1）、檔名與 alt 都不改（3、3）。
    outcome = runner.invoke(
        app,
        ["review", str(analysis_path), "--out", str(out_path)],
        input="1\n1\n3\n3\n",
    )

    assert outcome.exit_code == 0, outcome.output
    selection = Selection.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert selection.images == {}
