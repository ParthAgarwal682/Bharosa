"""LLM wrapper tests — no live network calls."""

from __future__ import annotations

import os
import pytest

from bharosa.rag.llm import EnvLLM, EnvLLMConfig, LLMNotConfiguredError, read_llm_config

try:
    from tests.test_rag_fixtures import FakeLLM
except ImportError:
    from test_rag_fixtures import FakeLLM


def test_fake_llm_records_prompts() -> None:
    llm = FakeLLM(response="ok")
    assert llm.complete("hello") == "ok"
    assert llm.call_count == 1
    assert llm.prompts == ["hello"]


def test_env_llm_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        "bharosa.rag.llm.read_llm_config",
        lambda: None,
    )
    with pytest.raises(LLMNotConfiguredError):
        EnvLLM(config=None)


def test_read_llm_config_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    # Ensure dotenv does not load a local .env key during this test.
    monkeypatch.setattr("bharosa.rag.llm._load_dotenv", lambda: None)
    assert read_llm_config() is None
    assert os.getenv("LLM_API_KEY") in (None, "")


def test_env_llm_config_repr_masks_api_key() -> None:
    """EnvLLMConfig repr must never expose the API key."""
    secret = "sk-super-secret-key-12345"
    cfg = EnvLLMConfig(provider="openai", model="gpt-4o", api_key=secret)
    rep = repr(cfg)
    assert secret not in rep
    assert "openai" in rep
    assert "gpt-4o" in rep


def test_env_llm_unsupported_provider_raises_explicit_error() -> None:
    """Unsupported provider raises explicit LLMNotConfiguredError without fallback."""
    cfg = EnvLLMConfig(provider="invalid_provider", model="some-model", api_key="key")
    with pytest.raises(LLMNotConfiguredError, match="Unsupported LLM_PROVIDER"):
        EnvLLM(config=cfg)
