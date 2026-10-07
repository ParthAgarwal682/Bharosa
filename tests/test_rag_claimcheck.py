"""Claim checker tests — pattern extract, retrieve callable, compare across hits."""

from __future__ import annotations

import pytest

from bharosa.rag.claimcheck import check_claim, extract_claim_spans
from bharosa.rag.config import get_config

try:
    from tests.test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit
except ImportError:
    from test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit


def test_extract_rupee_fee_claim() -> None:
    spans, amounts, claims_free, deadlines = extract_claim_spans(
        "pay Rs 500 to activate free health card"
    )
    assert amounts == [500.0]
    assert any("500" in s for s in spans)
    assert claims_free is True
    assert deadlines == []


def test_no_zones_is_insufficient_not_guess() -> None:
    verdict = check_claim(
        "pay Rs 500 to activate free health card",
        retrieve=lambda q: [],
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"
    assert "INSUFFICIENT INFORMATION" in verdict.explanation
    assert verdict.claim_amounts == [500.0]

    with pytest.raises(TypeError):
        check_claim("pay Rs 500", retrieve=None)  # type: ignore[arg-type]


def test_fee_claim_contradicts_official_no_fee() -> None:
    zones = mock_scheme_hits()
    verdict = check_claim(
        "pay Rs 500 to activate free health card",
        retrieve=lambda q: zones,
        allow_mock=True,
    )
    assert verdict.label == "CONTRADICTED"
    assert verdict.evidence_text is not None
    assert "registration fee" in verdict.evidence_text.casefold() or verdict.retrieval_score is not None
    assert all(z.source == "mock_fixture" for z in zones)


def test_free_claim_consistent_with_official() -> None:
    zones = mock_scheme_hits()
    verdict = check_claim(
        "Ayushman card activation is free, no registration fee",
        retrieve=lambda q: zones,
        allow_mock=True,
    )
    assert verdict.label == "SUPPORTED"
    assert verdict.evidence_cite is not None


def test_no_claim_found() -> None:
    verdict = check_claim(
        "hello how are you",
        retrieve=lambda q: mock_scheme_hits(),
        allow_mock=True,
    )
    assert verdict.label == "NO_CLAIM_FOUND"


def test_optional_llm_only_polishes_explanation() -> None:
    zones = mock_scheme_hits()
    llm = FakeLLM(response="Official sources say activation needs no fee.")
    verdict = check_claim(
        "pay ₹500 registration fee for health card",
        retrieve=lambda q: zones,
        llm=llm,
        allow_mock=True,
    )
    assert verdict.label == "CONTRADICTED"
    assert llm.call_count == 1
    assert verdict.explanation == "Official sources say activation needs no fee."


def test_compare_all_hits_catches_contradiction_in_secondary_hit() -> None:
    """Verifies that check_claim checks all hits, detecting contradiction even when another hit is present."""
    zones = mock_scheme_hits()
    verdict = check_claim(
        "pay ₹1000 charge before hospital admission under the scheme",
        retrieve=lambda q: zones,
        allow_mock=True,
    )
    assert verdict.label == "CONTRADICTED"
    assert verdict.evidence_cite == "Z2"
    assert "no registration fee" in (verdict.evidence_text or "").casefold()


def test_conflicting_zones_return_insufficient_with_both_evidence() -> None:
    """When retrieved zones disagree (one SUPPORTED, another CONTRADICTED), return INSUFFICIENT_EVIDENCE.

    Both zones must be above the minimum retrieval score.
    """
    min_score = get_config().min_retrieval_score
    zone_free = scheme_hit(
        doc_id="doc-free",
        url="https://example.invalid/free",
        zone="benefits",
        text="There is no registration fee to activate the card.",
        last_changed_at="2026-01-01T00:00:00Z",
        cosine=0.4,
        net=0.80,
        bm25_score=None,
        rank=1,
        source="test_local",
    )
    zone_fee = scheme_hit(
        doc_id="doc-fee",
        url="https://example.invalid/fee",
        zone="benefits",
        text="A registration fee of Rs 500 is required to activate.",
        last_changed_at="2026-01-01T00:00:00Z",
        cosine=0.4,
        net=0.75,
        bm25_score=None,
        rank=2,
        source="test_local",
    )
    assert zone_free.net is not None and zone_free.net >= min_score
    assert zone_fee.net is not None and zone_fee.net >= min_score

    # Message claims Rs 500 fee: zone_fee supports Rs 500, zone_free contradicts it (no fee)
    verdict = check_claim(
        "pay Rs 500 to activate free health card",
        retrieve=lambda q: [zone_free, zone_fee],
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"
    assert "Z1" in (verdict.evidence_cite or "")
    assert "Z2" in (verdict.evidence_cite or "")
    assert "disagree" in verdict.explanation.casefold()
    assert "CONTRADICTS" in (verdict.evidence_text or "")
    assert "SUPPORTS" in (verdict.evidence_text or "")


def test_item_5a_min_retrieval_score_ignored_low_scores() -> None:
    from dataclasses import replace
    hits = mock_scheme_hits()
    low_zone = replace(hits[1], text="Free sanitary napkins are distributed in schools.", net=0.001)
    verdict = check_claim(
        "Pay Rs 500 to activate health card",
        retrieve=lambda q: [low_zone],
        allow_mock=True,
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"


def test_item_5b_amount_context_fee_vs_cover() -> None:
    from dataclasses import replace
    hits = mock_scheme_hits()
    cover_zone = replace(hits[1], text="Health cover is up to Rs 5,00,000 per family per year.")
    verdict = check_claim(
        "Pay Rs 500000 registration fee to activate your card",
        retrieve=lambda q: [cover_zone],
        allow_mock=True,
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"


def test_item_5d_date_parsing_phrasings() -> None:
    from dataclasses import replace
    hits = mock_scheme_hits()
    date_zone = replace(hits[1], text="Registration closes by 01-01-2099.")
    for msg in [
        "Last date to register is 01-01-2099 or lose cover",
        "Register before 1-1-2099",
        "Register before 01-01-2099",
    ]:
        verdict = check_claim(msg, retrieve=lambda q: [date_zone], allow_mock=True)
        assert verdict.label == "SUPPORTED", f"Failed for phrasing: {msg}"


def test_claim_checker_uses_net() -> None:
    zone = scheme_hit(
        doc_id="net-fee",
        text="There is no registration fee to activate the card.",
        net=0.42,
        cosine=0.99,
        bm25_score=None,
        rank=1,
    )
    verdict = check_claim(
        "pay Rs 500 to activate the card",
        retrieve=lambda q: [zone],
    )
    assert verdict.label == "CONTRADICTED"
    assert verdict.retrieval_score == pytest.approx(0.42)


def test_bm25_missing_net_does_not_become_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """High or zero BM25 must not pass the NetScore gate, even at threshold 0."""
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", "0")
    zone = scheme_hit(
        doc_id="bm25-fee",
        text="There is no registration fee to activate the card.",
        net=None,
        bm25_score=0.0,
        cosine=0.99,
        rank=1,
    )
    verdict = check_claim(
        "pay Rs 500 to activate the card",
        retrieve=lambda q: [zone],
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"
    assert verdict.retrieval_score is None
    assert verdict.refusal_reason == "unsupported_bm25_evidence"

    strong_bm25 = scheme_hit(
        doc_id="bm25-fee-high",
        text="There is no registration fee to activate the card.",
        net=None,
        bm25_score=0.99,
        cosine=0.99,
        rank=1,
    )
    high = check_claim(
        "pay Rs 500 to activate the card",
        retrieve=lambda q: [strong_bm25],
    )
    assert high.label == "INSUFFICIENT_EVIDENCE"
    assert high.retrieval_score is None
    assert high.refusal_reason == "unsupported_bm25_evidence"


def test_missing_net_without_bm25_is_not_invented() -> None:
    zone = scheme_hit(
        doc_id="no-score",
        text="There is no registration fee to activate the card.",
        net=None,
        bm25_score=None,
        rank=1,
    )
    verdict = check_claim(
        "pay Rs 500 to activate the card",
        retrieve=lambda q: [zone],
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"
    assert verdict.retrieval_score is None
    assert verdict.refusal_reason == "missing_net_score"


def test_check_claim_default_refuses_mock_evidence() -> None:
    zones = mock_scheme_hits()
    verdict = check_claim(
        "pay Rs 500 to activate free health card",
        retrieve=lambda q: zones,
    )
    assert verdict.label == "INSUFFICIENT_EVIDENCE"
    assert verdict.refusal_reason == "mock_evidence"


