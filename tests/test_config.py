"""使用者設定目錄與全域 .env 載入。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from blogseo import config as config_mod
from blogseo.config import user_config_dir, user_env_path
from blogseo.errors import ConfigError


def test_user_config_dir_uses_appdata_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert user_config_dir() == tmp_path / "Roaming" / "blogseo"
    assert user_env_path() == tmp_path / "Roaming" / "blogseo" / ".env"


def test_load_environment_reads_user_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "_dotenv_loaded", False)
    monkeypatch.setattr(config_mod, "user_config_dir", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "find_dotenv", lambda usecwd=True: "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_user\n", encoding="utf-8")

    config_mod.load_environment()

    assert os.environ["HF_TOKEN"] == "hf_from_user"


def test_missing_api_key_mentions_user_env_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "_dotenv_loaded", False)
    monkeypatch.setattr(config_mod, "user_config_dir", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "find_dotenv", lambda usecwd=True: "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    with pytest.raises(ConfigError) as exc_info:
        config_mod.get_api_key("huggingface")
    assert str(tmp_path / ".env") in str(exc_info.value)


def test_mask_api_key_shows_first_ten_chars() -> None:
    from blogseo.config import mask_api_key

    assert mask_api_key("hf_abcdefghij") == "hf_abcdefg"
    assert mask_api_key(None) == ""
    assert mask_api_key("") == ""
    assert mask_api_key("abc") == "abc"


def test_peek_api_key_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "_dotenv_loaded", False)
    monkeypatch.setattr(config_mod, "user_config_dir", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "find_dotenv", lambda usecwd=True: "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    assert config_mod.peek_api_key("huggingface") is None


def test_save_api_key_writes_user_env_and_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "user_config_dir", lambda: tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = config_mod.save_api_key("huggingface", " hf_secret123 ")
    assert path == tmp_path / ".env"
    assert "HF_TOKEN=hf_secret123" in path.read_text(encoding="utf-8")
    assert os.environ["HF_TOKEN"] == "hf_secret123"
    assert config_mod.peek_api_key("huggingface") == "hf_secret123"
    assert config_mod.mask_api_key(config_mod.peek_api_key("huggingface")) == "hf_secret1"


def test_save_api_key_updates_existing_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=oldtoken\nANTHROPIC_API_KEY=keep\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "user_config_dir", lambda: tmp_path)
    config_mod.save_api_key("huggingface", "newtoken")
    text = env_file.read_text(encoding="utf-8")
    assert "HF_TOKEN=newtoken" in text
    assert "ANTHROPIC_API_KEY=keep" in text
    assert "oldtoken" not in text


def test_save_api_key_rejects_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="空白"):
        config_mod.save_api_key("anthropic", "   ")
