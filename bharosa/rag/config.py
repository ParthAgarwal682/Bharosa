"""Config dataclass for RAG thresholds read from environment variables.

Thresholds are read at call time from env vars so that tests and callers
can configure them dynamically.

Default threshold values are untuned placeholders; final numbers must be
tuned on a human-labelled dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_REFUSAL_THRESHOLD: float = 0.15
DEFAULT_CITATION_THRESHOLD: float = 0.25
DEFAULT_MIN_RETRIEVAL_SCORE: float = 0.05
DEFAULT_CITATION_POLICY: str = "flag"


def _read_float_env(var_name: str, default: float) -> float:
    """Read a float environment variable or return default.

    Raises ValueError naming the variable if the value is not a valid float.
    """
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float for {var_name}: {raw!r}"
        ) from exc


@dataclass(frozen=True)
class RAGConfig:
    """RAG configuration dataclass populated from environment variables.

    Default values are untuned placeholders.
    """

    refusal_score_threshold: float = DEFAULT_REFUSAL_THRESHOLD
    citation_threshold: float = DEFAULT_CITATION_THRESHOLD
    min_retrieval_score: float = DEFAULT_MIN_RETRIEVAL_SCORE
    citation_policy: str = DEFAULT_CITATION_POLICY

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Read configuration values from environment variables at call time."""
        raw_policy = os.getenv("RAG_CITATION_POLICY", DEFAULT_CITATION_POLICY).strip().lower()
        policy = raw_policy if raw_policy in {"flag", "withhold"} else DEFAULT_CITATION_POLICY
        return cls(
            refusal_score_threshold=_read_float_env(
                "RAG_REFUSAL_THRESHOLD", DEFAULT_REFUSAL_THRESHOLD
            ),
            citation_threshold=_read_float_env(
                "RAG_CITATION_THRESHOLD", DEFAULT_CITATION_THRESHOLD
            ),
            min_retrieval_score=_read_float_env(
                "RAG_MIN_RETRIEVAL_SCORE", DEFAULT_MIN_RETRIEVAL_SCORE
            ),
            citation_policy=policy,
        )


def get_config() -> RAGConfig:
    """Return RAGConfig evaluated from environment at call time."""
    return RAGConfig.from_env()
