"""Markdown / HTML 圖片語法改寫的單元測試。"""

from __future__ import annotations

import pytest

from blogseo.errors import ApplyError
from blogseo.markdown.updater import (
    apply_spans,
    join_source_path,
    rewrite_front_matter,
    rewrite_html_image,
    rewrite_image_markup,
    rewrite_markdown_image,
    splice_body,
)


def test_join_source_path_keeps_directory() -> None:
    assert (
        join_source_path("assets/post/file-123.png", "azure-vm-overview.png")
        == "assets/post/azure-vm-overview.png"
    )
    assert join_source_path("pic.png", "chart.png") == "chart.png"
    assert join_source_path("<images/a b.png>", "space-chart.png") == "images/space-chart.png"


def test_rewrite_markdown_changes_alt_and_path() -> None:
    raw = "![舊說明](images/DSC_0001.png)"
    assert (
        rewrite_markdown_image(raw, new_alt="新說明", new_path="images/chart.png")
        == "![新說明](images/chart.png)"
    )


def test_rewrite_markdown_preserves_title_and_brackets() -> None:
    raw = '![圖](<images/a b.png> "標題文字")'
    assert (
        rewrite_markdown_image(raw, new_path="images/space-chart.png")
        == '![圖](<images/space-chart.png> "標題文字")'
    )


def test_rewrite_markdown_alt_only() -> None:
    raw = "![](images/a.png)"
    assert rewrite_markdown_image(raw, new_alt="說明") == "![說明](images/a.png)"


def test_rewrite_html_keeps_other_attributes() -> None:
    raw = '<img src="images/banner.png" alt="橫幅" width="600">'
    assert (
        rewrite_html_image(raw, new_path="images/hero.png", new_alt="主視覺")
        == '<img src="images/hero.png" alt="主視覺" width="600">'
    )


def test_rewrite_html_inserts_missing_alt() -> None:
    raw = '<img src="images/a.png" width="600">'
    assert (
        rewrite_html_image(raw, new_alt="說明")
        == '<img src="images/a.png" width="600" alt="說明">'
    )


def test_rewrite_html_self_closing() -> None:
    raw = '<img src="images/a.png"/>'
    assert rewrite_html_image(raw, new_alt="說明") == '<img src="images/a.png" alt="說明"/>'


def test_rewrite_html_empty_alt_is_allowed() -> None:
    raw = '<img src="images/a.png" alt="舊">'
    assert rewrite_html_image(raw, new_alt="") == '<img src="images/a.png" alt="">'


def test_apply_spans_replaces_from_the_end() -> None:
    content = "AAA BBB CCC"
    # 兩個替換如果從前往後做，第二段位置會錯。
    result = apply_spans(content, [(0, 3, "XXXX"), (8, 11, "YY")])
    assert result == "XXXX BBB YY"


def test_rewrite_image_markup_dispatches() -> None:
    assert (
        rewrite_image_markup("![](a.png)", "markdown", new_alt="x") == "![x](a.png)"
    )
    assert (
        rewrite_image_markup('<img src="a.png">', "html", new_alt="x")
        == '<img src="a.png" alt="x">'
    )


def test_rewrite_markdown_rejects_unknown_syntax() -> None:
    with pytest.raises(ApplyError, match="無法解析"):
        rewrite_markdown_image("不是圖片", new_alt="x")


def test_splice_body_preserves_front_matter() -> None:
    original = "---\ntitle: 原標題\n---\n\n![](images/a.png)\n"
    body = "![](images/a.png)"
    result = splice_body(original, body, "![說明](images/chart.png)")
    assert result.startswith("---\ntitle: 原標題\n---\n")
    assert "![說明](images/chart.png)" in result
    assert "keywords:" not in result


def test_splice_body_without_front_matter() -> None:
    original = "![](images/a.png)\n"
    result = splice_body(original, "![](images/a.png)", "![說明](images/a.png)")
    assert result == "![說明](images/a.png)\n"


def test_rewrite_front_matter_creates_block() -> None:
    result = rewrite_front_matter(
        "# 標題\n\n內文。\n",
        keywords=["甲", "乙"],
        summary="這是摘要",
    )
    assert result.startswith("---")
    assert "甲" in result
    assert "乙" in result
    assert "這是摘要" in result
    assert "# 標題" in result


def test_rewrite_front_matter_keeps_other_keys() -> None:
    original = "---\ntitle: 原標題\ndate: 2026-01-01\n---\n\n正文\n"
    result = rewrite_front_matter(
        original, keywords=["關鍵字"], summary="摘要文字", keywords_key="tags"
    )
    assert "title: 原標題" in result or "title: '原標題'" in result or 'title: "原標題"' in result
    assert "date:" in result
    assert "關鍵字" in result
    assert "摘要文字" in result
    assert "正文" in result


def test_rewrite_front_matter_skips_empty_values() -> None:
    original = "# 標題\n"
    assert rewrite_front_matter(original, keywords=[], summary="") == original
