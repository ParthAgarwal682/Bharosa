"""Single LLM provider wrapper for Bharosa RAG (Modules 2 and 3 only).

Never call this module for medicine facts. Reads ``LLM_PROVIDER``,
``LLM_MODEL``, and ``LLM_API_KEY`` from the environment (via ``.env``
when python-dotenv is available).

API key is protected from repr/logs via field(repr=False). Bounded timeout
of 30 seconds is enforced on network requests. No fallback model is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

REQUEST_TIMEOUT: float = 30.0


class LLMNotConfiguredError(RuntimeError):
    """Raised when no usable LLM provider/key is configured."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call fails due to HTTP/network/timeout/JSON-shape issues."""


class LLMClient(Protocol):
    """Minimal completion interface used by ``answer`` and claim explain."""

    def complete(self, prompt: str) -> str:
        """Return the model text for ``prompt``."""
        ...


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@dataclass(frozen=True)
class EnvLLMConfig:
    """Resolved provider settings from the environment."""

    provider: str
    model: str
    api_key: str = field(repr=False)


def read_llm_config() -> EnvLLMConfig | None:
    """Return provider config, or None if any required value is missing."""
    _load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "").strip()
    model = (os.getenv("LLM_MODEL") or "").strip()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not provider or not model or not api_key:
        return None
    return EnvLLMConfig(provider=provider, model=model, api_key=api_key)


class EnvLLM:
    """Live LLM client selected by ``LLM_PROVIDER``.

    Supported providers:
    - ``openai``: HTTP chat completions API
    - ``anthropic``: HTTP messages API

    No fallback model is provided. Unsupported or unconfigured values
    raise ``LLMNotConfiguredError``.
    """

    def __init__(self, config: EnvLLMConfig | None = None) -> None:
        self._config = config if config is not None else read_llm_config()
        if self._config is None:
            raise LLMNotConfiguredError(
                "LLM not configured. Set LLM_PROVIDER, LLM_MODEL, and "
                "LLM_API_KEY in .env (see README)."
            )
        provider = self._config.provider.lower()
        if provider not in {"openai", "anthropic"}:
            raise LLMNotConfiguredError(
                f"Unsupported LLM_PROVIDER={self._config.provider!r}. "
                "Only 'openai' and 'anthropic' are supported."
            )
        if not self._config.model:
            raise LLMNotConfiguredError("LLM_MODEL cannot be empty.")

    @property
    def model(self) -> str:
        """Configured model name."""
        assert self._config is not None
        return self._config.model

    def complete(self, prompt: str) -> str:
        """Call the configured provider and return assistant text."""
        assert self._config is not None
        provider = self._config.provider.lower()
        if provider == "openai":
            return self._complete_openai(prompt)
        return self._complete_anthropic(prompt)

    def _sanitize_error(self, exc: Exception) -> str:
        msg = f"{type(exc).__name__}: {exc}"
        if self._config and self._config.api_key:
            msg = msg.replace(self._config.api_key, "[REDACTED]")
        return msg

    def _complete_openai(self, prompt: str) -> str:
        assert self._config is not None
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self._config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw_data = resp.read().decode("utf-8")
            payload = json.loads(raw_data)
            return str(payload["choices"][0]["message"]["content"])
        except Exception as exc:
            raise LLMProviderError(self._sanitize_error(exc)) from exc

    def _complete_anthropic(self, prompt: str) -> str:
        assert self._config is not None
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self._config.model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._config.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw_data = resp.read().decode("utf-8")
            payload = json.loads(raw_data)
            parts = payload.get("content") or []
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            return "\n".join(texts).strip()
        except Exception as exc:
            raise LLMProviderError(self._sanitize_error(exc)) from exc



def get_llm() -> LLMClient:
    """Build the default live LLM client from environment configuration.

    Raises LLMNotConfiguredError if configuration is missing.
    """
    return EnvLLM()
