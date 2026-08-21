"""Gitless Sync metadata 更新的單元測試。"""

from __future__ import annotations

import json
from pathlib import Path

from blogseo.markdown.parser import parse_article
from blogseo.obsidian.gitless_sync import (
    find_gitless_metadata,
    patch_gitless_sync_metadata,
)
from blogseo.schemas.selection import Selection, SelectionImage
from blogseo.seo.applier import ApplyPlan, build_plan, execute_plan


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
    images: dict[str, SelectionImage] | None = None,
) -> Selection:
    article = parse_article(markdown_path)
    return Selection(
        generated_at="2026-08-21T12:00:00+00:00",
        tool_version="0.1.0",
        source_analysis=str(markdown_path.with_name("post-analysis.json")),
        article_path=str(article.path),
        content_hash=article.content_hash,
        keywords=[],
        summary="",
        images=images or {},
    )


def _write_metadata(vault: Path, files: dict) -> Path:
    path = vault / ".obsidian" / "github-sync-metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"lastSync": 1, "files": files}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_find_metadata_walks_up(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    article_dir = vault / "posts"
    article_dir.mkdir(parents=True)
    metadata = _write_metadata(vault, {})
    article = article_dir / "a.md"
    article.write_text("x", encoding="utf-8")
    assert find_gitless_metadata(article) == metadata


def test_no_metadata_is_silent(tmp_path: Path) -> None:
    markdown_path = _make_article(
        tmp_path, "![](images/a.png)\n", images=("images/a.png",)
    )
    plan = ApplyPlan(
        article_path=markdown_path,
        original_text="![](images/a.png)\n",
        new_text="![說明](images/chart.png)\n",
    )
    outcome = patch_gitless_sync_metadata(plan)
    assert outcome.metadata_path is None


def test_apply_marks_rename_in_metadata(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    article_dir = vault / "posts"
    markdown_path = _make_article(
        article_dir,
        "![](images/a.png)\n",
        images=("images/a.png",),
    )
    old = "posts/images/a.png"
    article_rel = "posts/post.md"
    metadata = _write_metadata(
        vault,
        {
            old: {
                "path": old,
                "sha": "abc123",
                "dirty": False,
                "justDownloaded": False,
                "lastModified": 1,
            },
            article_rel: {
                "path": article_rel,
                "sha": "def456",
                "dirty": False,
                "justDownloaded": False,
                "lastModified": 1,
            },
        },
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(article_dir / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
                alt="說明",
                alt_model="claude",
            )
        },
    )
    outcome = execute_plan(build_plan(selection))
    assert outcome is not None
    assert outcome.metadata_path == metadata
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    files = payload["files"]
    assert files[old]["deleted"] is True
    assert files[old]["dirty"] is True
    assert files[old]["sha"] == "abc123"
    new = "posts/images/chart.png"
    assert files[new]["sha"] is None
    assert files[new]["dirty"] is True
    assert "deleted" not in files[new]
    assert files[article_rel]["dirty"] is True
    assert files[article_rel]["sha"] == "def456"


def test_deleted_without_sha_is_dropped(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    article_dir = vault / "posts"
    markdown_path = _make_article(
        article_dir,
        "![](images/a.png)\n",
        images=("images/a.png",),
    )
    old = "posts/images/a.png"
    metadata = _write_metadata(
        vault,
        {
            old: {
                "path": old,
                "sha": None,
                "dirty": True,
                "justDownloaded": False,
                "lastModified": 1,
            }
        },
    )
    selection = _selection_for(
        markdown_path,
        images={
            "images/a.png": SelectionImage(
                source="images/a.png",
                resolved_path=str(article_dir / "images" / "a.png"),
                new_filename="chart.png",
                filename_model="claude",
            )
        },
    )
    execute_plan(build_plan(selection))
    files = json.loads(metadata.read_text(encoding="utf-8"))["files"]
    assert old not in files
    assert files["posts/images/chart.png"]["sha"] is None
