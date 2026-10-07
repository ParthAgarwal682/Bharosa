"""Shared RAG data types (consumer contract for Module C evidence).

ZoneHit fields below are the evidence object Person D needs from
Paridhi's ``search_schemes``. They are proposed from README Section 10
(Zone document + ranked hits). If Paridhi's public ``ZoneHit`` differs,
stop and tell Paridhi — do not silently change their API.

``source`` is a RAG-side annotation for honesty: fixture hits used in
tests must be ``mock_fixture`` and must never enter final report tables
as live crawl evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ZoneName = Literal[
    "eligibility",
    "benefits",
    "documents",
    "how_to_apply",
    "other",
]

RefusalReason = Literal[
    "empty_hits",
    "low_retrieval_score",
    "no_matching_filter",
    "empty_evidence",
    "invalid_generation",
    "empty_generation",
    "llm_not_configured",
    "llm_unavailable",
    "citation_validation_failed",
    "mock_evidence",
]

ClaimLabel = Literal[
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "NO_CLAIM_FOUND",
]


@dataclass(frozen=True)
class ZoneHit:
    """One ranked zone returned by Module C retrieval.

    IR concept: a scored evidence segment. RAG may only cite ``text``.
    ``score`` drives the refusal gate. ``rank`` is 1-based among hits.
    """

    doc_id: str
    url: str
    domain: str
    title: str
    zone: ZoneName
    text: str
    state: str | None
    conditions: list[str]
    g_score: float
    crawled_at: str
    last_changed_at: str
    content_hash: str
    score: float
    rank: int
    source: str = "retriever"
    """Provenance tag. Use ``mock_fixture`` only for tests / dry runs."""


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

    Lexical overlap is the fraction of claim tokens found in the evidence,
    not cosine similarity and not logical entailment.
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
