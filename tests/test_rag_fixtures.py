"""Hand-written mock zones and FakeLLM for tests only.

These are NOT live crawl evidence. Every hit has ``source='mock_fixture'``.
Must never be exported from bharosa/rag or used as production evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from bharosa.rag.types import ZoneHit


@dataclass
class FakeLLM:
    """Deterministic stand-in for unit tests only."""

    response: str = ""
    prompts: list[str] | None = None
    call_count: int = 0

    def __post_init__(self) -> None:
        if self.prompts is None:
            self.prompts = []

    def complete(self, prompt: str) -> str:
        """Return the configured response and record the prompt."""
        assert self.prompts is not None
        self.prompts.append(prompt)
        self.call_count += 1
        return self.response


def mock_scheme_hits() -> list[ZoneHit]:
    """Return a tiny fixed scheme corpus for unit tests."""
    return [
        ZoneHit(
            doc_id="fixture-ayushman-eligibility",
            url="https://example.invalid/ayushman/eligibility",
            domain="example.invalid",
            title="Ayushman Bharat eligibility (fixture)",
            zone="eligibility",
            text=(
                "Families with annual income below two lakh rupees in "
                "Uttar Pradesh may be eligible for Ayushman Bharat cover "
                "subject to the official beneficiary list."
            ),
            state="UP",
            conditions=["heart", "hospitalization"],
            g_score=0.9,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-elig",
            score=0.82,
            rank=1,
            source="mock_fixture",
        ),
        ZoneHit(
            doc_id="fixture-ayushman-benefits",
            url="https://example.invalid/ayushman/benefits",
            domain="example.invalid",
            title="Ayushman Bharat benefits (fixture)",
            zone="benefits",
            text=(
                "The scheme provides health cover up to five lakh rupees "
                "per family per year for listed secondary and tertiary care. "
                "There is no registration fee to activate the card."
            ),
            state="UP",
            conditions=["hospitalization"],
            g_score=0.9,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-ben",
            score=0.71,
            rank=2,
            source="mock_fixture",
        ),
        ZoneHit(
            doc_id="fixture-ayushman-documents",
            url="https://example.invalid/ayushman/documents",
            domain="example.invalid",
            title="Ayushman Bharat documents (fixture)",
            zone="documents",
            text=(
                "Carry a government photo identity proof and the beneficiary "
                "card at the empanelled hospital. Do not share Aadhaar OTP "
                "with unknown callers."
            ),
            state="UP",
            conditions=[],
            g_score=0.85,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-docs",
            score=0.55,
            rank=3,
            source="mock_fixture",
        ),
    ]


def weak_hits() -> list[ZoneHit]:
    """Hits with scores below the default refusal threshold."""
    base = mock_scheme_hits()[0]
    return [
        ZoneHit(
            doc_id=base.doc_id,
            url=base.url,
            domain=base.domain,
            title=base.title,
            zone=base.zone,
            text=base.text,
            state=base.state,
            conditions=list(base.conditions),
            g_score=base.g_score,
            crawled_at=base.crawled_at,
            last_changed_at=base.last_changed_at,
            content_hash=base.content_hash,
            score=0.02,
            rank=1,
            source="mock_fixture",
        )
    ]
