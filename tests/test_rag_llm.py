"""LLM wrapper tests — no live network calls."""

from __future__ import annotations

import json
import logging
import os
import traceback
import urllib.error
import urllib.request

import pytest

from bharosa.rag.llm import (
    REQUEST_TIMEOUT,
    EnvLLM,
    EnvLLMConfig,
    LLMNotConfiguredError,
    LLMProviderError,
    get_llm,
    read_llm_config,
)

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


def test_env_llm_wraps_errors_in_llm_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    from bharosa.rag.llm import LLMProviderError

    secret_key = "sk-secret-do-not-leak"
    cfg = EnvLLMConfig(provider="openai", model="gpt-4o", api_key=secret_key)
    client = EnvLLM(config=cfg)

    def fake_urlopen_err(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen_err)
    with pytest.raises(LLMProviderError) as exc_info:
        client.complete("prompt")
    assert secret_key not in str(exc_info.value)


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENROUTER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_OPENROUTER_KEY = "sk-openrouter-test-do-not-leak"
_PROMPT = (
    "Use only the labelled zones.\n"
    "[Z1] Families with annual income below two lakh rupees may be eligible."
)
_ANSWER = "Families with annual income below two lakh rupees may be eligible."


class _MockHTTPResponse:
    """Context manager with the same shape as a successful urlopen result."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _headers(req: urllib.request.Request) -> dict[str, str]:
    return {name.lower(): value for name, value in req.header_items()}


def _json_body(req: urllib.request.Request) -> dict[str, object]:
    assert isinstance(req.data, bytes)
    body = json.loads(req.data.decode("utf-8"))
    assert isinstance(body, dict)
    return body


def _openrouter_client(provider: str = "openrouter") -> EnvLLM:
    return EnvLLM(
        config=EnvLLMConfig(
            provider=provider,
            model=_OPENROUTER_MODEL,
            api_key=_OPENROUTER_KEY,
        )
    )


def _exception_output(exc: BaseException) -> str:
    parts = [str(exc), repr(exc), "".join(traceback.format_exception(exc))]
    cause = exc.__cause__
    if cause is not None:
        parts.extend([str(cause), repr(cause), "".join(traceback.format_exception(cause))])
        for attr in ("reason", "msg", "filename", "doc", "strerror"):
            value = getattr(cause, attr, None)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, BaseException):
                parts.append(str(value))
                parts.append(repr(value))
    return "\n".join(parts)


@pytest.mark.parametrize("missing", ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"])
def test_openrouter_missing_setting_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        raise AssertionError("unconfigured LLM called the network")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("bharosa.rag.llm._load_dotenv", lambda: None)
    values = {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": _OPENROUTER_MODEL,
        "LLM_API_KEY": _OPENROUTER_KEY,
    }
    for name in values:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        if name != missing:
            monkeypatch.setenv(name, value)

    assert read_llm_config() is None
    with pytest.raises(LLMNotConfiguredError, match="LLM not configured"):
        get_llm()


def test_openrouter_reads_provider_model_and_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bharosa.rag.llm._load_dotenv", lambda: None)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", _OPENROUTER_MODEL)
    monkeypatch.setenv("LLM_API_KEY", _OPENROUTER_KEY)

    config = read_llm_config()
    assert config is not None
    assert config.provider == "openrouter"
    assert config.model == _OPENROUTER_MODEL
    assert config.api_key == _OPENROUTER_KEY
    assert _OPENROUTER_KEY not in repr(config)

    client = get_llm()
    assert isinstance(client, EnvLLM)
    assert client.model == _OPENROUTER_MODEL
    assert _OPENROUTER_KEY not in repr(client)


@pytest.mark.parametrize("provider", ["openrouter", "OpenRouter"])
def test_openrouter_posts_prompt_and_parses_answer(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        calls.append((args, kwargs))
        payload = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": _ANSWER}},
                    {"message": {"role": "assistant", "content": "do not use this choice"}},
                ]
            }
        ).encode("utf-8")
        return _MockHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with caplog.at_level(logging.DEBUG):
        answer = _openrouter_client(provider).complete(_PROMPT)

    assert answer == _ANSWER
    assert len(calls) == 1
    args, kwargs = calls[0]
    req = args[0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == _OPENROUTER_URL
    assert req.get_method() == "POST"
    assert kwargs["timeout"] == REQUEST_TIMEOUT

    headers = _headers(req)
    assert headers["authorization"] == f"Bearer {_OPENROUTER_KEY}"
    assert "x-api-key" not in headers
    assert "anthropic-version" not in headers
    assert _json_body(req) == {
        "model": _OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": _PROMPT}],
        "temperature": 0,
    }
    assert req.data is not None
    assert _OPENROUTER_KEY not in req.full_url
    assert _OPENROUTER_KEY not in req.data.decode("utf-8")

    captured = capsys.readouterr()
    assert _OPENROUTER_KEY not in caplog.text
    assert _OPENROUTER_KEY not in captured.out
    assert _OPENROUTER_KEY not in captured.err


def test_openrouter_errors_are_wrapped_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        req = args[0]
        assert isinstance(req, urllib.request.Request)
        calls.append(req.full_url)
        raise urllib.error.URLError(f"network down {_OPENROUTER_KEY}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMProviderError) as exc_info:
            _openrouter_client().complete(_PROMPT)

    err = exc_info.value
    assert type(err) is LLMProviderError
    assert str(err).startswith("URLError:")
    assert "[REDACTED]" in str(err)
    assert calls == [_OPENROUTER_URL]
    captured = capsys.readouterr()
    rendered = "\n".join(
        [_exception_output(err), caplog.text, captured.out, captured.err]
    )
    assert _OPENROUTER_KEY not in rendered


@pytest.mark.parametrize(
    ("raw", "prefix"),
    [
        (b"{", "JSONDecodeError:"),
        (b"{}", "KeyError:"),
        (b'{"choices": []}', "IndexError:"),
    ],
)
def test_openrouter_bad_payload_uses_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    prefix: str,
) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        return _MockHTTPResponse(raw)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(LLMProviderError) as exc_info:
        _openrouter_client().complete(_PROMPT)
    assert str(exc_info.value).startswith(prefix)
    assert _OPENROUTER_KEY not in _exception_output(exc_info.value)


def test_openrouter_redacts_key_echoed_in_invalid_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        return _MockHTTPResponse(_OPENROUTER_KEY.encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(LLMProviderError) as exc_info:
        _openrouter_client().complete(_PROMPT)

    cause = exc_info.value.__cause__
    assert isinstance(cause, json.JSONDecodeError)
    assert cause.doc == "[REDACTED]"
    assert _OPENROUTER_KEY not in _exception_output(exc_info.value)


def test_openai_request_stays_on_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        calls.append((args, kwargs))
        return _MockHTTPResponse(
            json.dumps(
                {"choices": [{"message": {"content": _ANSWER}}]}
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = EnvLLM(
        config=EnvLLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-openai-test-do-not-leak",
        )
    )
    assert client.complete(_PROMPT) == _ANSWER
    args, kwargs = calls[0]
    req = args[0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == _OPENAI_URL
    assert kwargs["timeout"] == REQUEST_TIMEOUT
    assert _headers(req)["authorization"] == "Bearer sk-openai-test-do-not-leak"
    assert _json_body(req) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": _PROMPT}],
        "temperature": 0,
    }


def test_anthropic_request_stays_on_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_urlopen(*args: object, **kwargs: object) -> _MockHTTPResponse:
        calls.append((args, kwargs))
        payload = {
            "content": [
                {"type": "text", "text": "Official line."},
                {"type": "tool_use", "text": "ignore this part"},
            ]
        }
        return _MockHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = EnvLLM(
        config=EnvLLMConfig(
            provider="anthropic",
            model="claude-test",
            api_key="sk-anthropic-test-do-not-leak",
        )
    )
    assert client.complete(_PROMPT) == "Official line."
    args, kwargs = calls[0]
    req = args[0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == _ANTHROPIC_URL
    assert kwargs["timeout"] == REQUEST_TIMEOUT
    headers = _headers(req)
    assert headers["x-api-key"] == "sk-anthropic-test-do-not-leak"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert _json_body(req) == {
        "model": "claude-test",
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [{"role": "user", "content": _PROMPT}],
    }

