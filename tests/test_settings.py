"""設定檔讀寫與預設值合併。"""

from __future__ import annotations

from pathlib import Path

import pytest

from blogseo.errors import ConfigError
from blogseo.pipeline import ANALYSIS_MENU, fields_from_result
from blogseo.schemas.result import AnalysisField, AnalysisResult, ArticleInfo, Metadata
from blogseo.settings import (
    SETTINGS_FILENAME,
    default_save_path,
    find_settings_path,
    load_settings,
    load_settings_file,
    save_settings,
    settings_from_mapping,
)
from blogseo.seo.analyzer import AnalyzeOptions, options_from_settings
from blogseo.llm.registry import parse_model_tokens


def test_partial_json_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / SETTINGS_FILENAME
    path.write_text('{"models": {"text": "claude"}, "alt": {"max_chars": 40}}\n', encoding="utf-8")
    settings = load_settings_file(path)
    assert settings.models.text == "claude"
    assert settings.models.image == "hf"
    assert settings.alt.max_chars == 40
    assert settings.summary.min_chars == 60
    assert settings.apply.write_front_matter is True


def test_invalid_summary_range_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / SETTINGS_FILENAME
    path.write_text(
        '{"summary": {"min_chars": 200, "max_chars": 80}}\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="min_chars"):
        load_settings_file(path)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / SETTINGS_FILENAME
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="不是合法的 JSON"):
        load_settings_file(path)


def test_save_roundtrip(tmp_path: Path) -> None:
    settings = settings_from_mapping(
        {"models": {"text": "claude", "image": "hf:zai-org/GLM-4.6V-Flash"}}
    )
    path = save_settings(settings, tmp_path / SETTINGS_FILENAME)
    loaded = load_settings_file(path)
    assert loaded.models.text == "claude"
    assert loaded.models.image == "hf:zai-org/GLM-4.6V-Flash"
    assert "_readme" in path.read_text(encoding="utf-8")


def test_find_settings_walks_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / SETTINGS_FILENAME).write_text('{"models": {"text": "claude"}}\n', encoding="utf-8")
    monkeypatch.delenv("BLOGSEO_SETTINGS", raising=False)
    found = find_settings_path(start=nested)
    assert found == root / SETTINGS_FILENAME
    monkeypatch.chdir(nested)
    assert load_settings().models.text == "claude"


def test_find_settings_falls_back_to_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLOGSEO_SETTINGS", raising=False)
    user_dir = tmp_path / "user-config"
    user_dir.mkdir()
    (user_dir / SETTINGS_FILENAME).write_text(
        '{"models": {"text": "claude"}}\n', encoding="utf-8"
    )
    monkeypatch.setattr("blogseo.settings.user_config_dir", lambda: user_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    found = find_settings_path(start=empty)
    assert found == user_dir / SETTINGS_FILENAME
    monkeypatch.chdir(empty)
    assert load_settings().models.text == "claude"


def test_default_save_path_uses_user_config_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLOGSEO_SETTINGS", raising=False)
    user_dir = tmp_path / "user-config"
    monkeypatch.setattr("blogseo.settings.user_config_dir", lambda: user_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert default_save_path(start=empty) == user_dir / SETTINGS_FILENAME


def test_env_overrides_search_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text('{"models": {"image": "claude"}}\n', encoding="utf-8")
    monkeypatch.setenv("BLOGSEO_SETTINGS", str(custom))
    assert find_settings_path() == custom
    assert load_settings().models.image == "claude"


def test_options_from_settings_splits_models() -> None:
    settings = settings_from_mapping(
        {"models": {"text": "claude", "image": "hf"}, "summary": {"count": 1}}
    )
    options = options_from_settings(
        settings, fields=frozenset({AnalysisField.KEYWORDS, AnalysisField.IMAGES})
    )
    assert options.text_tokens == ["claude"]
    assert options.image_tokens == ["hf"]
    assert options.summary_count == 1
    assert options.tokens_for_text() == ["claude"]
    assert options.tokens_for_image() == ["hf"]


def test_analysis_menu_covers_requested_combinations() -> None:
    mapped = {number: fields for number, fields, _label in ANALYSIS_MENU}
    assert mapped[1] == frozenset(AnalysisField)
    assert mapped[2] == frozenset({AnalysisField.SUMMARY})
    assert mapped[3] == frozenset({AnalysisField.KEYWORDS})
    assert mapped[4] == frozenset({AnalysisField.KEYWORDS, AnalysisField.SUMMARY})
    assert mapped[5] == frozenset({AnalysisField.IMAGES})


def test_fields_from_result_reads_metadata() -> None:
    result = AnalysisResult(
        article_info=ArticleInfo(
            path="post.md",
            title=None,
            language="zh",
            word_count=1,
            image_count=0,
            content_hash="abc",
        ),
        metadata=Metadata(
            generated_at="2026-08-23T00:00:00+00:00",
            tool_version="0.1.0",
            total_elapsed_ms=1,
            requested_fields=["keywords"],
            alt_language="zh",
            summary_min_chars=60,
            summary_max_chars=150,
            summary_count=3,
            alt_max_chars=30,
            usd_to_twd_rate=31.8,
        ),
    )
    assert fields_from_result(result) == frozenset({AnalysisField.KEYWORDS})


def test_parse_model_tokens_still_accepts_comma_in_settings() -> None:
    assert parse_model_tokens("hf,claude") == ["hf", "claude"]


def test_default_analyze_options_keep_unified_models() -> None:
    options = AnalyzeOptions(model_tokens=["claude"])
    assert options.tokens_for_text() == ["claude"]
    assert options.tokens_for_image() == ["claude"]


def _sample_presets() -> list:
    from blogseo.llm.registry import ModelPreset

    return [
        ModelPreset(
            token="hf:Qwen/Qwen3-4B-Instruct-2507",
            provider_key="huggingface",
            provider_label="Hugging Face",
            alias="hf",
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            note="免費額度",
        ),
        ModelPreset(
            token="claude:claude-sonnet-5",
            provider_key="anthropic",
            provider_label="Claude",
            alias="claude",
            model_id="claude-sonnet-5",
        ),
    ]


def test_prompt_model_token_zero_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from blogseo.settings_ui import _prompt_model_token

    monkeypatch.setattr(
        "blogseo.settings_ui.list_model_presets", lambda role: _sample_presets()
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.ask_choice", lambda *args, **kwargs: 0
    )
    assert _prompt_model_token("預設關鍵字／摘要模型", "hf", role="text") is None


def test_prompt_model_token_provider_then_typed_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blogseo.settings_ui import _prompt_model_token

    monkeypatch.setattr(
        "blogseo.settings_ui.list_model_presets", lambda role: _sample_presets()
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.ask_choice", lambda *args, **kwargs: 2
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.typer.prompt",
        lambda *args, **kwargs: "claude-opus-5",
    )
    assert (
        _prompt_model_token("預設圖片辨識模型", "hf", role="image")
        == "claude:claude-opus-5"
    )


def test_prompt_model_token_keeps_current_id_on_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blogseo.settings_ui import _prompt_model_token

    monkeypatch.setattr(
        "blogseo.settings_ui.list_model_presets", lambda role: _sample_presets()
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.ask_choice", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.typer.prompt",
        lambda *args, **kwargs: kwargs["default"],
    )
    assert (
        _prompt_model_token(
            "預設關鍵字／摘要模型", "hf:Qwen/Qwen3-8B", role="text"
        )
        == "hf:Qwen/Qwen3-8B"
    )


def test_prompt_model_token_zero_on_model_id_then_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blogseo.settings_ui import _prompt_model_token

    menu_choices = iter([1, 0])
    monkeypatch.setattr(
        "blogseo.settings_ui.list_model_presets", lambda role: _sample_presets()
    )
    monkeypatch.setattr(
        "blogseo.settings_ui.ask_choice",
        lambda *args, **kwargs: next(menu_choices),
    )
    monkeypatch.setattr("blogseo.settings_ui.typer.prompt", lambda *args, **kwargs: "0")
    assert _prompt_model_token("預設關鍵字／摘要模型", "hf", role="text") is None
