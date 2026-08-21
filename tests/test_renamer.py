"""檔名正規化的單元測試。"""

from __future__ import annotations

import pytest

from blogseo.image.renamer import build_filename, slugify


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kriging Spatial Model Result", "kriging-spatial-model-result"),
        ("kriging-spatial-model.png", "kriging-spatial-model"),
        ("  Trailing And Leading  ", "trailing-and-leading"),
        ("Multiple___Separators!!Here", "multiple-separators-here"),
        ("Café Résumé", "cafe-resume"),
        ("--already--slugged--", "already-slugged"),
        ("2024 Q3 報表", "2024-q3"),
    ],
)
def test_slugify_normalizes(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_slugify_falls_back_when_nothing_survives() -> None:
    assert slugify("純中文檔名") == "image"
    assert slugify("純中文檔名", fallback="figure") == "figure"


def test_slugify_truncates_on_word_boundary() -> None:
    slug = slugify("a" * 30 + " " + "b" * 40)
    assert len(slug) <= 60
    assert not slug.endswith("-")


def test_slugify_keeps_non_image_suffix() -> None:
    assert slugify("report.v2") == "report-v2"


def test_build_filename_lowercases_extension() -> None:
    assert build_filename("chart", ".PNG") == "chart.png"
    assert build_filename("chart", "jpg") == "chart.jpg"
    assert build_filename("chart", "") == "chart"
