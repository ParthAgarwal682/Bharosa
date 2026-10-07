"""Shared RAG data types.

Evidence is Paridhi's ``bharosa.index.zones.ZoneHit``. This module does not
define a second hit dataclass. ``ZoneEvidence`` is the structural contract
RAG actually reads:

- ``doc_id``, ``text``, ``url``, ``zone``, ``last_changed_at``
- ``cosine``, ``g_component``, ``freshness``, ``zone_weight``
- ``net``, ``bm25_score``, ``rank``

The normal RAG path uses ``net``. ``bm25_score`` is the BM25 baseline and
is not a NetScore. A missing ``net`` is not ``0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

RefusalReason = Literal[
    "empty_hits",
    "low_retrieval_score",
    "empty_evidence",
    "invalid_generation",
    "empty_generation",
    "llm_not_configured",
    "llm_unavailable",
    "citation_validation_failed",
    "mock_evidence",
    "unsupported_bm25_evidence",
    "missing_net_score",
]

ClaimLabel = Literal[
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "NO_CLAIM_FOUND",
]


class ZoneEvidence(Protocol):
    """Fields RAG reads from ``bharosa.index.zones.ZoneHit``.

    Not a second hit type. Callers pass Paridhi's ``ZoneHit`` objects.
    ``net`` is the retrieval score on the normal path. ``bm25_score`` is
    set only on the BM25 baseline path, where ``net`` is ``None``.
    """

    doc_id: str
    text: str
    url: str
    zone: str
    last_changed_at: str | None
    cosine: float
    g_component: float
    freshness: float
    zone_weight: float
    net: float | None
    bm25_score: float | None
    rank: int


def evidence_net(hit: ZoneEvidence) -> float | None:
    """Return ``ZoneHit.net``.

    ``None`` means this hit has no NetScore. Callers must not replace that
    with ``0`` or with ``bm25_score``.
    """
    net = hit.net
    if net is None:
        return None
    if isinstance(net, bool) or not isinstance(net, (int, float)):
        raise TypeError(f"net must be float or None, got {type(net).__name__}")
    return float(net)


def hits_with_net(hits: Sequence[ZoneEvidence]) -> list[ZoneEvidence]:
    """Hits that carry a real NetScore, in the original order."""
    return [hit for hit in hits if evidence_net(hit) is not None]


def missing_net_reason(hits: Sequence[ZoneEvidence]) -> RefusalReason | None:
    """Why a hit list cannot be scored on the normal path.

    Returns None when ``hits`` is empty or at least one hit has ``net``.
    BM25-only evidence (``net is None`` and ``bm25_score`` is set) is
    ``unsupported_bm25_evidence``. Evidence with neither score is
    ``missing_net_score``. Neither case is reported as a low NetScore.
    """
    if not hits or any(evidence_net(hit) is not None for hit in hits):
        return None
    if any(hit.bm25_score is not None for hit in hits):
        return "unsupported_bm25_evidence"
    return "missing_net_score"


def _is_zone_label(token: str) -> bool:
    text = token.strip()
    return len(text) >= 2 and text[0] in "Zz" and text[1:].isdigit()


def normalize_zone_label(token: str) -> str:
    """Uppercase a ``Z1``-style label. Leave doc ids unchanged."""
    text = str(token).strip()
    if _is_zone_label(text):
        return text.upper()
    return text


def label_index(hit_labels: Mapping[str, str]) -> dict[str, str]:
    """Map canonical label keys to ``doc_id::zone`` values."""
    indexed: dict[str, str] = {}
    for key, value in hit_labels.items():
        indexed[normalize_zone_label(key)] = value
    return indexed


def canonical_cite(token: str, hit_labels: Mapping[str, str]) -> str | None:
    """Return the structured label for ``token``, or None if it is unknown.

    Accepts a ``Z1`` label, or the ``doc_id`` / ``doc_id::zone`` stored for
    exactly one label. Ambiguous doc ids are rejected.
    """
    indexed = label_index(hit_labels)
    normalized = normalize_zone_label(token)
    if _is_zone_label(normalized):
        if normalized in indexed:
            return normalized
        return None
    matches = [
        key
        for key, value in indexed.items()
        if normalized == value or normalized == value.split("::", 1)[0]
    ]
    if len(matches) == 1:
        return matches[0]
    return None


@dataclass(frozen=True)
class CitedSentence:
    """One claim sentence and the zone labels it cites."""

    text: str
    cite_ids: list[str]


@dataclass
class Answer:
    """Structured RAG answer, or a refusal with no LLM call."""

    query: str
    refused: bool
    refusal_reason: str | None
    sentences: list[CitedSentence]
    hit_labels: dict[str, str]
    model: str | None = None
    raw_llm: str | None = None
    refusal_message: str | None = None


@dataclass(frozen=True)
class SentenceCheck:
    """Citation support check for one answer sentence.

    Lexical overlap is a heuristic: the fraction of claim tokens found in
    the evidence. It is not cosine similarity and not logical entailment.
    """

    sentence_index: int
    cite_ids: list[str]
    lexical_overlap: float
    supported: bool
    flag: str | None

    @property
    def max_cosine(self) -> float:
        """Backwards-compatible alias for lexical_overlap."""
        return self.lexical_overlap


@dataclass
class Verdict:
    """Module 3 claim-check result against official zone text."""

    message: str
    claim_spans: list[str]
    label: ClaimLabel
    evidence_cite: str | None
    evidence_text: str | None
    explanation: str
    retrieval_score: float | None = None
    claim_amounts: list[float] = field(default_factory=list)
    refusal_reason: str | None = None
