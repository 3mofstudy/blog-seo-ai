"""apply 階段的單元測試。

重點：真正改檔之前必須擋下雜湊不一致與檔名衝突；dry-run 與取消都不得
動到原始檔。全程不呼叫 LLM。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from blogseo.cli import app
from blogseo.errors import ApplyError, SelectionLoadError
from blogseo.image.renamer import rename_files
from blogseo.markdown.parser import parse_article
from blogseo.schemas.selection import Selection, SelectionImage
from blogseo.seo.applier import build_plan, execute_plan, load_selection

runner = CliRunner()


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")


def _make_article(
    tmp_path: Path,
    body: str,
    *,
    images: tuple[str, ...] = (),
    name: str = "post.md",
) -> Path:
    for relative in images:
        _write_png(tmp_path / relative)
    markdown_path = tmp_path / name
    markdown_path.write_text(body, encoding="utf-8")
    return markdown_path


def _selection_for(
    markdown_path: Path,
    *,
    keywords: list[str] | None = None,
    summary: str = "",
    images: dict[str, SelectionImage] | None = None,
    content_hash: str | None = None,
) -> Selection:
    article = parse_article(markdown_path)
    return Selection(
        generated_at="2026-08-21T12:00:00+00:00",
        tool_version="0.1.0",
        source_analysis=str(markdown_path.with_name("post-analysis.json")),
        article_path=str(article.path),
        content_hash=content_hash if content_hash is not None else article.content_hash,
        keywords=keywords or [],
        summary=summary,
        images=images or {},
    )


def _write_selection(tmp_path: Path, selection: Selection) -> Path:
    path = tmp_path / "post-selection.json"
    path.write_text(selection.to_json(), encoding="utf-8")
    return path


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SelectionLoadError, match="找不到選擇檔"):
        load_selection(tmp_path / "nope.json")


def test_load_accepts_legacy_source_model(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    payload = _selection_for(markdown_path).model_dump()
    payload["schema_version"] = "1.0"
    payload["images"] = {
        "images/a.png": {
            "source": "images/a.png",
            "resolved_path": str(tmp_path / "images" / "a.png"),
            "new_filename": "chart.png",
            "alt": "說明",
            "source_model": "claude",
        }
    }
    path = tmp_path / "legacy.json"
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    loaded = load_selection(path)
    item = loaded.images["images/a.png"]
    assert item.filename_model == "claude"
    assert item.alt_model == "claude"


def test_applies_image_rewrite_without_front_matter(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path,
        "# 標題\n\n內文。\n\n![](images/DSC_0001.png)\n",
        images=("images/DSC_0001.png",),
    )
    selection = _selection_for(
        markdown_path,
        keywords=["關鍵字一", "關鍵字二"],
        summary="這是一段剛好夠長的摘要內容用來當 meta description。",
        images={
            "images/DSC_0001.png": SelectionImage(
                source="images/DSC_0001.png",
                resolved_path=str(tmp_path / "images" / "DSC_0001.png"),
                new_filename="spatial-chart.png",
                filename_model="claude",
                alt="空間分布圖",
                alt_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))

    text = markdown_path.read_text(encoding="utf-8")
    assert not text.startswith("---")
    assert "keywords:" not in text
    assert "description:" not in text
    assert "![空間分布圖](images/spatial-chart.png)" in text
    assert (tmp_path / "images" / "spatial-chart.png").is_file()
    assert not (tmp_path / "images" / "DSC_0001.png").exists()


def test_alt_only_does_not_rename(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                alt="說明",
                alt_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))

    assert markdown_path.read_text(encoding="utf-8") == "![說明](images/a.png)\n"
    assert (tmp_path / "images" / "a.png").is_file()


def test_filename_only_keeps_original_alt(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![原本](images/a.png)\n", images=("images/a.png",)
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))

    assert markdown_path.read_text(encoding="utf-8") == "![原本](images/chart.png)\n"
    assert (tmp_path / "images" / "chart.png").is_file()


def test_rewrites_every_occurrence_and_html(tmp_path: Path) -> None:
    body = (
        "![一次](images/chart.png)\n\n"
        '<img src="images/chart.png" alt="舊" width="700">\n'
    )
    markdown_path = _make_article(tmp_path, body, images=("images/chart.png",))
    selection = _selection_for(
        markdown_path,
        images={
            "images/chart.png": SelectionImage(
                source="images/chart.png",
                resolved_path=str(tmp_path / "images" / "chart.png"),
                new_filename="result-map.png",
                filename_model="claude",
                alt="結果圖",
                alt_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))
    text = markdown_path.read_text(encoding="utf-8")
    assert "![結果圖](images/result-map.png)" in text
    assert '<img src="images/result-map.png" alt="結果圖" width="700">' in text


def test_leaves_fenced_code_examples_alone(tmp_path: Path) -> None:
    body = (
        "```markdown\n"
        "![範例](images/example.png)\n"
        "```\n\n"
        "![真圖](images/real.png)\n"
    )
    markdown_path = _make_article(
        tmp_path, body, images=("images/example.png", "images/real.png")
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/real.png": SelectionImage(
                source="images/real.png",
                resolved_path=str(tmp_path / "images" / "real.png"),
                new_filename="actual.png",
                filename_model="claude",
                alt="實際結果",
                alt_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))
    text = markdown_path.read_text(encoding="utf-8")
    assert "![範例](images/example.png)" in text
    assert "![實際結果](images/actual.png)" in text


def test_existing_front_matter_is_untouched(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path,
        "---\ntitle: 原標題\ndate: 2026-01-01\n---\n\n![](images/a.png)\n",
        images=("images/a.png",),
    )
    original_yaml = "---\ntitle: 原標題\ndate: 2026-01-01\n---"
    selection = _selection_for(
        markdown_path,
        keywords=["甲"],
        summary="這是摘要文字，不應該被寫進檔案。",
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                alt="說明",
                alt_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))
    text = markdown_path.read_text(encoding="utf-8")
    assert text.startswith(original_yaml)
    assert "keywords:" not in text
    assert "description:" not in text
    assert "![說明](images/a.png)" in text


def test_hash_mismatch_is_blocked(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
            )
        },
    )
    markdown_path.write_text("完全不一樣了\n![](images/a.png)\n", encoding="utf-8")

    with pytest.raises(ApplyError, match="被編輯過"):
        build_plan(selection)
    assert (tmp_path / "images" / "a.png").is_file()


def test_force_uses_current_article(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
                alt="說明",
                alt_model="claude",
            )
        },
    )
    markdown_path.write_text("改過了\n\n![](images/a.png)\n", encoding="utf-8")
    execute_plan(build_plan(selection, force=True))
    assert "![說明](images/chart.png)" in markdown_path.read_text(encoding="utf-8")


def test_conflict_with_existing_file_is_blocked(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png", "images/chart.png")
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
            )
        },
    )
    with pytest.raises(ApplyError, match="檔名衝突"):
        build_plan(selection)


def test_swapping_filenames(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    rename_files([(first, second), (second, first)])
    assert first.read_bytes() == b"two"
    assert second.read_bytes() == b"one"


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection_path = _write_selection(
        tmp_path,
        _selection_for(
            markdown_path,
            keywords=["甲"],
            images={
                "images/a.png": SelectionImage(
                    source="images/a.png",
                    resolved_path=str(tmp_path / "images" / "a.png"),
                    new_filename="chart.png",
                    filename_model="claude",
                    alt="說明",
                    alt_model="claude",
                )
            },
        ),
    )
    original = markdown_path.read_text(encoding="utf-8")
    outcome = runner.invoke(app, ["apply", str(selection_path), "--dry-run"])
    assert outcome.exit_code == 0, outcome.output
    assert "預覽" in outcome.output
    assert markdown_path.read_text(encoding="utf-8") == original
    assert (tmp_path / "images" / "a.png").is_file()


def test_cli_apply_yes(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection_path = _write_selection(
        tmp_path,
        _selection_for(
            markdown_path,
            keywords=["甲"],
            summary="這是一段用來寫進 description 的摘要內容。",
            images={
                "images/a.png": SelectionImage(
                    source="images/a.png",
                    resolved_path=str(tmp_path / "images" / "a.png"),
                    new_filename="chart.png",
                    filename_model="claude",
                    alt="說明",
                    alt_model="claude",
                )
            },
        ),
    )
    outcome = runner.invoke(
        app, ["apply", str(selection_path), "--yes", "--backup"]
    )
    assert outcome.exit_code == 0, outcome.output
    text = markdown_path.read_text(encoding="utf-8")
    assert "![說明](images/chart.png)" in text
    assert "keywords:" not in text
    assert "description:" not in text
    assert (tmp_path / "post.md.bak").is_file()
    assert (tmp_path / "images" / "chart.png").is_file()


def test_cli_refuses_without_force_when_changed(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(tmp_path / "images" / "a.png"),
                alt="說明",
                alt_model="claude",
            )
        },
    )
    selection_path = _write_selection(tmp_path, selection)
    markdown_path.write_text("改過了\n![](images/a.png)\n", encoding="utf-8")
    outcome = runner.invoke(app, ["apply", str(selection_path), "--yes"])
    assert outcome.exit_code == 1
    assert "被編輯過" in outcome.output
