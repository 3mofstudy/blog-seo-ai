"""流程調度的單元測試。

重點在驗證專案的硬性原則：個別模型或個別圖片失敗時，只影響該格結果，
不能中斷其他呼叫，也不能讓整份 JSON 產不出來。全程不呼叫真實 API。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from blogseo.errors import ConfigError, ProviderError
from blogseo.llm.base import BaseProvider
from blogseo.llm.registry import create_provider, parse_model_tokens, parse_token
from blogseo.markdown.parser import parse_article
from blogseo.schemas.result import (
    AnalysisField,
    ImageAnalysisResult,
    KeywordSummaryResult,
)
from blogseo.seo import analyzer as analyzer_module
from blogseo.seo.analyzer import AnalyzeOptions, analyze_article, parse_fields


class _StubProvider(BaseProvider):
    """不打網路的假 provider，可指定要在哪個環節失敗。"""

    name = "anthropic"
    default_model = "stub-model"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "claude": (2.0, 10.0),
        "gpt": (5.0, 25.0),
    }

    def __init__(self, *, fail_text: bool = False, fail_image: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("api_key", "stub-key")
        super().__init__(**kwargs)
        self._fail_text = fail_text
        self._fail_image = fail_image

    def analyze_text(self, content: str) -> KeywordSummaryResult:
        if self._fail_text:
            self.usage.record_failure(elapsed_ms=1)
            raise ProviderError(self.name, self.model, "文章分析壞掉了")
        self.usage.record_success(input_tokens=100, output_tokens=20, elapsed_ms=1)
        # 跟真實 provider 一樣，只產生被要求的項目。
        payload: dict[str, list[str]] = {}
        if AnalysisField.KEYWORDS in self.fields:
            payload["keywords"] = ["一", "二", "三", "四", "五"]
        if AnalysisField.SUMMARY in self.fields:
            # 故意多回幾則，用來驗證 BaseProvider 會裁到要求的版本數。
            payload["summaries"] = [
                f"第 {index} 個角度的摘要內容" for index in range(1, 6)
            ]
        return self._limit_text_result(KeywordSummaryResult(**payload))

    def analyze_image(
        self,
        image_bytes: bytes,
        context: str,
        *,
        media_type: str = "image/png",
        original_filename: str = "",
    ) -> ImageAnalysisResult:
        if self._fail_image:
            self.usage.record_failure(elapsed_ms=1)
            raise ProviderError(self.name, self.model, "圖片分析壞掉了")
        self.usage.record_success(input_tokens=500, output_tokens=30, elapsed_ms=1)
        return ImageAnalysisResult(suggested_filename="Stub Chart Result", alt="替代文字")


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


@pytest.fixture
def stub_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """把 registry 的 provider 建立流程換成可控的假實作。

    Args:
        monkeypatch: pytest 的 monkeypatch fixture。

    Returns:
        可寫入設定的字典，鍵為模型別名。
    """
    config: dict[str, Any] = {}

    def fake_create(token: str, **kwargs: Any) -> BaseProvider:
        kwargs.pop("role", None)
        setting = config.get(token, {})
        if "raises" in setting:
            raise setting["raises"]
        return _StubProvider(
            fail_text=setting.get("fail_text", False),
            fail_image=setting.get("fail_image", False),
            model=token,
            **kwargs,
        )

    monkeypatch.setattr(analyzer_module, "create_provider", fake_create)
    return config


def test_produces_grouped_result(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude", "gpt"]))

    assert set(result.keyword_summary) == {"claude", "gpt"}
    assert len(result.images) == 2
    for entry in result.images.values():
        assert set(entry.models) == {"claude", "gpt"}
        assert entry.models["claude"].final_filename == "stub-chart-result.png"
    assert result.metadata.models["claude"].calls == 3


def test_provider_creation_failure_is_isolated(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    stub_factory["gpt"] = {"raises": ConfigError("openai 的實作尚未完成")}
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude", "gpt"]))

    assert result.keyword_summary["gpt"].error is not None
    assert result.keyword_summary["claude"].result is not None
    for entry in result.images.values():
        assert entry.models["gpt"].error is not None
        assert entry.models["claude"].result is not None


def test_image_failure_does_not_affect_text(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    stub_factory["claude"] = {"fail_image": True}
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude"]))

    assert result.keyword_summary["claude"].result is not None
    for entry in result.images.values():
        assert entry.models["claude"].result is None
        assert "圖片分析壞掉了" in (entry.models["claude"].error or "")
    assert result.metadata.models["claude"].failed_calls == 2


def test_text_failure_does_not_affect_images(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    stub_factory["claude"] = {"fail_text": True}
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude"]))

    assert result.keyword_summary["claude"].result is None
    for entry in result.images.values():
        assert entry.models["claude"].result is not None


def test_all_providers_failing_raises(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    stub_factory["gpt"] = {"raises": ConfigError("沒有金鑰")}
    article = parse_article(_make_article(tmp_path))

    with pytest.raises(ConfigError, match="沒有任何可用的模型"):
        analyze_article(article, AnalyzeOptions(model_tokens=["gpt"]))


def test_max_images_limits_calls(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path, image_count=3))
    result = analyze_article(
        article, AnalyzeOptions(model_tokens=["claude"], max_images=1)
    )

    analyzed = [entry for entry in result.images.values() if entry.models]
    skipped = [entry for entry in result.images.values() if entry.skipped_reason]
    assert len(analyzed) == 1
    assert len(skipped) == 2
    assert "max-images" in (skipped[0].skipped_reason or "")


def test_text_only_fields_skip_images(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(
        article,
        AnalyzeOptions(
            model_tokens=["claude"],
            fields=frozenset({AnalysisField.KEYWORDS, AnalysisField.SUMMARY}),
        ),
    )

    assert result.keyword_summary["claude"].result is not None
    assert all(not entry.models for entry in result.images.values())
    assert result.metadata.models["claude"].calls == 1
    assert result.metadata.requested_fields == ["keywords", "summary"]
    for entry in result.images.values():
        assert entry.skipped_reason == "本次 --fields 未包含 images"


def test_keywords_only_produces_no_summaries(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(
        article,
        AnalyzeOptions(model_tokens=["claude"], fields=frozenset({AnalysisField.KEYWORDS})),
    )

    entry = result.keyword_summary["claude"]
    assert entry.result is not None
    assert entry.result.keywords
    assert entry.result.summaries == []
    assert entry.summary_lengths == []
    assert result.metadata.requested_fields == ["keywords"]


def test_summary_only_produces_no_keywords(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(
        article,
        AnalyzeOptions(model_tokens=["claude"], fields=frozenset({AnalysisField.SUMMARY})),
    )

    entry = result.keyword_summary["claude"]
    assert entry.result is not None
    assert entry.result.keywords == []
    assert len(entry.result.summaries) == 3


def test_images_only_skips_text_call(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(
        article,
        AnalyzeOptions(model_tokens=["claude"], fields=frozenset({AnalysisField.IMAGES})),
    )

    assert result.keyword_summary == {}
    assert result.metadata.models["claude"].calls == 2
    for entry in result.images.values():
        assert entry.models["claude"].result is not None


def test_provider_failure_with_images_only_has_no_text_entry(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    stub_factory["gpt"] = {"raises": ConfigError("沒有金鑰")}
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(
        article,
        AnalyzeOptions(
            model_tokens=["claude", "gpt"], fields=frozenset({AnalysisField.IMAGES})
        ),
    )

    assert result.keyword_summary == {}
    for entry in result.images.values():
        assert entry.models["gpt"].error is not None


def test_cost_is_estimated_from_pricing_table(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(
        article, AnalyzeOptions(model_tokens=["claude"], usd_to_twd_rate=30.0)
    )

    usage = result.metadata.models["claude"]
    # 1 次文章分析（100 in / 20 out）+ 2 次圖片（各 500 in / 30 out）
    assert usage.input_tokens == 1100
    assert usage.output_tokens == 80
    assert usage.estimated_cost_usd == pytest.approx(1100 * 2 / 1e6 + 80 * 10 / 1e6)
    assert usage.estimated_cost_twd == pytest.approx(usage.estimated_cost_usd * 30.0)
    assert result.metadata.total_cost_usd == pytest.approx(usage.estimated_cost_usd)
    assert result.metadata.usd_to_twd_rate == 30.0


def test_cost_is_unknown_for_unlisted_model(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["anthropic"]))

    usage = result.metadata.models["anthropic"]
    assert usage.estimated_cost_usd is None
    assert usage.estimated_cost_twd is None
    assert result.metadata.total_cost_usd is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("keywords", {AnalysisField.KEYWORDS}),
        (" Summary , IMAGES ", {AnalysisField.SUMMARY, AnalysisField.IMAGES}),
        ("keywords,summary,images", set(AnalysisField)),
        ("keywords,keywords", {AnalysisField.KEYWORDS}),
    ],
)
def test_parse_fields_accepts_valid_input(
    raw: str, expected: set[AnalysisField]
) -> None:
    assert parse_fields(raw) == frozenset(expected)


def test_parse_fields_rejects_unknown_and_empty() -> None:
    with pytest.raises(ConfigError, match="無效的 --fields 項目"):
        parse_fields("keywords,titles")
    with pytest.raises(ConfigError, match="至少要指定一個項目"):
        parse_fields("  ,  ")


def test_analyze_does_not_touch_source_files(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    markdown_path = _make_article(tmp_path)
    before = {
        path: path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    analyze_article(parse_article(markdown_path), AnalyzeOptions(model_tokens=["claude"]))

    after = {
        path: path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_alt_language_follows_article_by_default(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude"]))
    assert result.metadata.alt_language == article.language


def test_defaults_to_three_summaries(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude"]))

    entry = result.keyword_summary["claude"]
    assert entry.result is not None
    assert len(entry.result.summaries) == 3
    assert len(set(entry.result.summaries)) == 3
    assert result.metadata.summary_count == 3
    assert result.metadata.summary_min_chars == 60
    assert result.metadata.summary_max_chars == 150


def test_summary_count_is_configurable(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(
        article, AnalyzeOptions(model_tokens=["claude"], summary_count=1)
    )

    entry = result.keyword_summary["claude"]
    assert entry.result is not None
    assert len(entry.result.summaries) == 1


def test_summary_lengths_match_texts(tmp_path: Path, stub_factory: dict[str, Any]) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(article, AnalyzeOptions(model_tokens=["claude"]))

    entry = result.keyword_summary["claude"]
    assert entry.result is not None
    assert entry.summary_lengths == [len(text) for text in entry.result.summaries]


def test_summary_range_is_recorded_in_metadata(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path, image_count=0))
    result = analyze_article(
        article,
        AnalyzeOptions(
            model_tokens=["claude"], summary_min_chars=80, summary_max_chars=120
        ),
    )

    assert result.metadata.summary_min_chars == 80
    assert result.metadata.summary_max_chars == 120


def test_alt_max_is_recorded_and_lengths_match(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path, image_count=1))
    result = analyze_article(
        article, AnalyzeOptions(model_tokens=["claude"], alt_max_chars=30)
    )

    assert result.metadata.alt_max_chars == 30
    entry = next(iter(result.images.values()))
    analysis = entry.models["claude"]
    assert analysis.result is not None
    assert analysis.alt_length == len(analysis.result.alt)


def test_registry_rejects_unknown_alias() -> None:
    with pytest.raises(ConfigError, match="未知的模型別名"):
        parse_token("llama")


def test_registry_parses_model_override() -> None:
    spec, model = parse_token("claude:claude-opus-5")
    assert spec.key == "anthropic"
    assert model == "claude-opus-5"


def test_registry_parses_huggingface_alias() -> None:
    spec, model = parse_token("hf")
    assert spec.key == "huggingface"
    assert model is None
    spec, model = parse_token("qwen:Qwen/Qwen2.5-VL-3B-Instruct")
    assert spec.key == "huggingface"
    assert model == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_model_presets_one_row_per_implemented_provider_with_model_id() -> None:
    from blogseo.llm.huggingface import HuggingFaceProvider
    from blogseo.llm.openai import OpenAIProvider
    from blogseo.llm.registry import list_model_presets

    text = list_model_presets("text")
    image = list_model_presets("image")
    assert [preset.alias for preset in text] == ["claude", "hf", "gpt"]
    assert [preset.alias for preset in image] == ["claude", "hf", "gpt"]
    assert {preset.provider_key for preset in text} == {
        "anthropic",
        "huggingface",
        "openai",
    }
    hf_text = next(preset for preset in text if preset.alias == "hf")
    hf_image = next(preset for preset in image if preset.alias == "hf")
    gpt = next(preset for preset in text if preset.alias == "gpt")
    assert hf_text.model_id == HuggingFaceProvider.default_model
    assert hf_image.model_id == HuggingFaceProvider.default_image_model
    assert gpt.model_id == OpenAIProvider.default_model
    assert hf_text.token == f"hf:{HuggingFaceProvider.default_model}"
    assert hf_image.token == f"hf:{HuggingFaceProvider.default_image_model}"
    assert gpt.token == f"gpt:{OpenAIProvider.default_model}"


def test_format_model_token_fills_in_default_model_id() -> None:
    from blogseo.llm.huggingface import HuggingFaceProvider
    from blogseo.llm.registry import format_model_token

    assert HuggingFaceProvider.default_model in format_model_token("hf", role="text")
    assert HuggingFaceProvider.default_image_model in format_model_token(
        "huggingface", role="image"
    )
    assert "claude-sonnet-5" in format_model_token("claude", role="text")
    assert "claude-opus-5" in format_model_token("claude:claude-opus-5", role="text")
    assert "gpt-5.6-terra" in format_model_token("gpt", role="text")


def test_suggested_model_id_keeps_current_when_same_provider() -> None:
    from blogseo.llm.registry import list_model_presets, suggested_model_id

    claude = next(preset for preset in list_model_presets("text") if preset.alias == "claude")
    hf = next(preset for preset in list_model_presets("text") if preset.alias == "hf")
    assert (
        suggested_model_id("claude:claude-opus-5", claude, role="text")
        == "claude-opus-5"
    )
    assert suggested_model_id("claude:claude-opus-5", hf, role="text") == hf.model_id
    assert suggested_model_id("hf", hf, role="text") == hf.model_id


def test_normalize_typed_model_id_accepts_plain_or_prefixed() -> None:
    from blogseo.llm.registry import (
        list_model_presets,
        normalize_typed_model_id,
    )

    claude = next(preset for preset in list_model_presets("text") if preset.alias == "claude")
    assert normalize_typed_model_id("claude-opus-5", claude) == "claude-opus-5"
    assert (
        normalize_typed_model_id("claude:claude-opus-5", claude) == "claude-opus-5"
    )
    with pytest.raises(ConfigError, match="不能填"):
        normalize_typed_model_id("hf:Qwen/Qwen3-8B", claude)


def test_registry_reports_unimplemented_provider() -> None:
    with pytest.raises(ConfigError, match="尚未完成"):
        create_provider("gemini")


def test_parse_model_tokens_dedupes_and_requires_value() -> None:
    assert parse_model_tokens(" claude , GPT ,claude") == ["claude", "gpt"]
    assert parse_model_tokens("hf:Qwen/Qwen2.5-VL-7B-Instruct") == [
        "hf:Qwen/Qwen2.5-VL-7B-Instruct"
    ]
    with pytest.raises(ConfigError):
        parse_model_tokens("  ,  ")


def test_text_and_image_can_use_different_models(
    tmp_path: Path, stub_factory: dict[str, Any]
) -> None:
    article = parse_article(_make_article(tmp_path))
    result = analyze_article(
        article,
        AnalyzeOptions(
            model_tokens=["claude", "gpt"],
            text_tokens=["claude"],
            image_tokens=["gpt"],
        ),
    )

    assert set(result.keyword_summary) == {"claude"}
    assert result.keyword_summary["claude"].result is not None
    for entry in result.images.values():
        assert set(entry.models) == {"gpt"}
        assert entry.models["gpt"].result is not None
    assert result.metadata.models["claude"].calls == 1
    assert result.metadata.models["gpt"].calls == 2
