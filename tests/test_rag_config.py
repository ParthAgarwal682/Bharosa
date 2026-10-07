"""Tests for RAGConfig and call-time threshold evaluation.

Per project rules, test doubles and fixtures are defined locally in tests/
and never imported from bharosa.rag.fixtures.
"""

from __future__ import annotations

import pytest

from bharosa.rag.answer import should_refuse
from bharosa.rag.citations import check_citations
from bharosa.rag.config import (
    DEFAULT_CITATION_THRESHOLD,
    DEFAULT_MIN_RETRIEVAL_SCORE,
    DEFAULT_REFUSAL_THRESHOLD,
    RAGConfig,
    get_config,
)
from bharosa.rag.types import Answer, CitedSentence

try:
    from tests.test_rag_fixtures import SchemeHit, scheme_hit
except ImportError:
    from test_rag_fixtures import SchemeHit, scheme_hit


def _local_hit(net: float, text: str = "Sample scheme text") -> SchemeHit:
    """Local test hit shaped like Paridhi's ZoneHit."""
    return scheme_hit(
        doc_id="test-doc-1",
        url="https://example.invalid/scheme",
        zone="benefits",
        text=text,
        last_changed_at="2026-01-01T00:00:00Z",
        cosine=0.2,
        net=net,
        bm25_score=None,
        rank=1,
        source="test_local",
    )


def test_rag_config_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.refusal_score_threshold == DEFAULT_REFUSAL_THRESHOLD
    assert cfg.citation_threshold == DEFAULT_CITATION_THRESHOLD
    assert cfg.min_retrieval_score == DEFAULT_MIN_RETRIEVAL_SCORE


def test_rag_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_REFUSAL_THRESHOLD", "0.45")
    monkeypatch.setenv("RAG_CITATION_THRESHOLD", "0.55")
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", "0.12")

    cfg = RAGConfig.from_env()
    assert cfg.refusal_score_threshold == 0.45
    assert cfg.citation_threshold == 0.55
    assert cfg.min_retrieval_score == 0.12


def test_config_evaluated_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold changes in env must take effect immediately on next call."""
    hits = [_local_hit(net=0.20)]

    # With default threshold 0.15, score 0.20 should not refuse
    monkeypatch.delenv("RAG_REFUSAL_THRESHOLD", raising=False)
    assert should_refuse(hits) is None

    # Dynamically change env to 0.30 at call time
    monkeypatch.setenv("RAG_REFUSAL_THRESHOLD", "0.30")
    assert should_refuse(hits) == "low_retrieval_score"


def test_citation_threshold_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Citation threshold from env must take effect dynamically at call time."""
    hits = [_local_hit(net=0.9, text="Ayushman covers secondary and tertiary hospitalization.")]
    ans = Answer(
        query="what is covered?",
        refused=False,
        refusal_reason=None,
        sentences=[
            CitedSentence(
                text="Ayushman scheme covers hospital care.",
                cite_ids=["Z1"],
            )
        ],
        hit_labels={"Z1": "test-doc-1::benefits"},
    )

    # By default (0.25), this overlap should be supported
    monkeypatch.delenv("RAG_CITATION_THRESHOLD", raising=False)
    checks_default = check_citations(ans, hits)
    assert checks_default[0].supported is True

    # If env raises threshold to 0.99 at call time, it must report unsupported
    monkeypatch.setenv("RAG_CITATION_THRESHOLD", "0.99")
    checks_high = check_citations(ans, hits)
    assert checks_high[0].supported is False


def test_invalid_env_threshold_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric string in environment threshold variable raises ValueError naming the variable."""
    monkeypatch.setenv("RAG_REFUSAL_THRESHOLD", "not_a_number")
    with pytest.raises(ValueError, match="Invalid float for RAG_REFUSAL_THRESHOLD: 'not_a_number'"):
        RAGConfig.from_env()

    monkeypatch.delenv("RAG_REFUSAL_THRESHOLD", raising=False)
    monkeypatch.setenv("RAG_CITATION_THRESHOLD", "invalid_value")
    with pytest.raises(ValueError, match="Invalid float for RAG_CITATION_THRESHOLD: 'invalid_value'"):
        RAGConfig.from_env()

    monkeypatch.delenv("RAG_CITATION_THRESHOLD", raising=False)
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", "bad_score")
    with pytest.raises(ValueError, match="Invalid float for RAG_MIN_RETRIEVAL_SCORE: 'bad_score'"):
        RAGConfig.from_env()
