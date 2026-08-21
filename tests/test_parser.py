"""Markdown 解析的單元測試。

重點在於邊界情況：程式碼區塊裡的圖片不能被抓走、外部連結要略過、
同一張圖被多次引用要合併，以及字元位置必須精準指向原始語法。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blogseo.errors import MarkdownParseError
from blogseo.markdown.parser import extract_context, parse_article


def _write_article(tmp_path: Path, body: str, *, images: tuple[str, ...] = ()) -> Path:
    """建立測試用的文章與圖片檔。

    Args:
        tmp_path: pytest 提供的暫存目錄。
        body: Markdown 內容。
        images: 需要一併建立的圖片相對路徑。

    Returns:
        Markdown 檔案路徑。
    """
    for relative in images:
        image_path = tmp_path / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")

    markdown_path = tmp_path / "post.md"
    markdown_path.write_text(body, encoding="utf-8")
    return markdown_path


def test_parses_frontmatter_and_title(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        "---\ntitle: 空間內插實作\ndate: 2026-01-01\n---\n\n本文說明克利金法的應用。\n",
    )
    article = parse_article(path)

    assert article.title == "空間內插實作"
    assert sorted(article.metadata) == ["date", "title"]
    assert article.language == "zh"
    assert article.content.startswith("本文說明")


def test_falls_back_to_first_heading_for_title(tmp_path: Path) -> None:
    path = _write_article(tmp_path, "# Getting Started\n\nSome text here.\n")
    article = parse_article(path)

    assert article.title == "Getting Started"
    assert article.language == "en"


def test_extracts_image_with_position_and_alt(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        "第一段文字。\n\n![原本的說明](images/DSC_0001.png)\n\n結尾。\n",
        images=("images/DSC_0001.png",),
    )
    article = parse_article(path)

    assert len(article.targets) == 1
    target = article.targets[0]
    assert target.key == "images/DSC_0001.png"
    assert target.exists is True
    assert target.extension == ".png"
    assert target.analyzable is True

    reference = target.references[0]
    assert reference.alt == "原本的說明"
    assert reference.line == 3
    assert article.content[reference.start : reference.end] == reference.raw
    assert reference.raw == "![原本的說明](images/DSC_0001.png)"


def test_ignores_images_inside_fenced_code(tmp_path: Path) -> None:
    body = (
        "說明如何寫圖片語法：\n\n"
        "```markdown\n"
        "![範例](images/example.png)\n"
        "```\n\n"
        "實際的圖：\n\n"
        "![真圖](images/real.png)\n"
    )
    path = _write_article(tmp_path, body, images=("images/example.png", "images/real.png"))
    article = parse_article(path)

    assert [target.key for target in article.targets] == ["images/real.png"]


def test_ignores_images_inside_inline_code(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        "語法是 `![alt](images/inline.png)` 這樣寫。\n",
        images=("images/inline.png",),
    )
    article = parse_article(path)

    assert article.targets == []


def test_skips_remote_and_site_absolute_paths(tmp_path: Path) -> None:
    body = (
        "![遠端](https://example.com/a.png)\n\n"
        "![站台絕對](/images/b.png)\n\n"
        "![本地](images/c.png)\n"
    )
    path = _write_article(tmp_path, body, images=("images/c.png",))
    article = parse_article(path)

    reasons = {target.key: target.skipped_reason for target in article.targets}
    assert reasons["https://example.com/a.png"] == "外部連結，不做本地改名"
    assert reasons["/images/b.png"] == "站台絕對路徑，超出本工具處理範圍"
    assert reasons["images/c.png"] is None


def test_marks_missing_local_file(tmp_path: Path) -> None:
    path = _write_article(tmp_path, "![不存在](images/missing.png)\n")
    article = parse_article(path)

    target = article.targets[0]
    assert target.exists is False
    assert target.analyzable is False
    assert target.skipped_reason == "本地找不到這個檔案"


def test_groups_repeated_references_into_one_target(tmp_path: Path) -> None:
    body = "![第一次](images/chart.png)\n\n中間段落。\n\n![第二次](images/chart.png)\n"
    path = _write_article(tmp_path, body, images=("images/chart.png",))
    article = parse_article(path)

    assert len(article.targets) == 1
    assert len(article.targets[0].references) == 2
    assert [ref.line for ref in article.targets[0].references] == [1, 5]


def test_decodes_percent_encoded_path(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        "![圖](images/my%20chart.png)\n",
        images=("images/my chart.png",),
    )
    article = parse_article(path)

    assert article.targets[0].key == "images/my chart.png"
    assert article.targets[0].exists is True


def test_parses_angle_bracket_and_title_syntax(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        '![圖](<images/a b.png> "標題文字")\n',
        images=("images/a b.png",),
    )
    article = parse_article(path)

    assert article.targets[0].key == "images/a b.png"
    assert article.targets[0].exists is True


def test_extracts_html_img_tag(tmp_path: Path) -> None:
    path = _write_article(
        tmp_path,
        '<img src="images/banner.png" alt="橫幅" width="600">\n',
        images=("images/banner.png",),
    )
    article = parse_article(path)

    target = article.targets[0]
    assert target.key == "images/banner.png"
    assert target.references[0].syntax == "html"
    assert target.references[0].alt == "橫幅"


def test_context_includes_title_and_section(tmp_path: Path) -> None:
    body = (
        "---\ntitle: 空間分析\n---\n\n"
        "## 克利金法結果\n\n"
        "下圖是模型輸出的等值線圖。\n\n"
        "![](images/result.png)\n\n"
        "可以看出西南側誤差較大。\n"
    )
    path = _write_article(tmp_path, body, images=("images/result.png",))
    article = parse_article(path)
    context = extract_context(article, article.targets[0])

    assert "文章標題：空間分析" in context
    assert "所在章節：克利金法結果" in context
    assert "等值線圖" in context
    assert "西南側誤差較大" in context


def test_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MarkdownParseError, match="找不到 Markdown 檔案"):
        parse_article(tmp_path / "nope.md")


def test_raises_for_broken_frontmatter(tmp_path: Path) -> None:
    path = _write_article(tmp_path, "---\ntitle: [unclosed\n---\n\n內文\n")
    with pytest.raises(MarkdownParseError):
        parse_article(path)
