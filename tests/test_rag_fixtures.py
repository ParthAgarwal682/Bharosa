"""Hand-written scheme hits shaped like Paridhi's ZoneHit, plus FakeLLM.

These are NOT live crawl evidence. Hits used as mock evidence set
``source='mock_fixture'``. That attribute is test-only: Paridhi's
``bharosa.index.zones.ZoneHit`` does not have it.

Must never be exported from bharosa/rag or used as production evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SchemeHit:
    """Fixture with the fields ``bharosa.index.zones.ZoneHit`` actually has.

    ``source`` is not part of that contract. It exists so tests can mark
    mock evidence. RAG reads it only through ``getattr``.
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
    source: str | None = None


def scheme_hit(
    *,
    doc_id: str,
    text: str,
    url: str = "https://example.invalid/scheme",
    zone: str = "benefits",
    net: float | None = 0.8,
    last_changed_at: str | None = "2026-01-01T00:00:00+00:00",
    cosine: float = 0.4,
    bm25_score: float | None = None,
    rank: int = 1,
    g_component: float = 0.0,
    freshness: float = 0.0,
    zone_weight: float = 1.0,
    source: str | None = None,
) -> SchemeHit:
    """Build one actual-shaped ZoneHit fixture. ``bm25_score`` defaults to None."""
    return SchemeHit(
        doc_id=doc_id,
        text=text,
        url=url,
        zone=zone,
        last_changed_at=last_changed_at,
        cosine=cosine,
        g_component=g_component,
        freshness=freshness,
        zone_weight=zone_weight,
        net=net,
        bm25_score=bm25_score,
        rank=rank,
        source=source,
    )


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


def mock_scheme_hits() -> list[SchemeHit]:
    """Return a tiny fixed scheme corpus for unit tests.

    Each hit has a NetScore in ``net`` and ``bm25_score=None``.
    """
    return [
        scheme_hit(
            doc_id="fixture-ayushman-eligibility",
            url="https://example.invalid/ayushman/eligibility",
            zone="eligibility",
            text=(
                "Families with annual income below two lakh rupees in "
                "Uttar Pradesh may be eligible for Ayushman Bharat cover "
                "subject to the official beneficiary list."
            ),
            last_changed_at="2026-01-01T00:00:00+00:00",
            cosine=0.33,
            net=0.82,
            bm25_score=None,
            rank=1,
            source="mock_fixture",
        ),
        scheme_hit(
            doc_id="fixture-ayushman-benefits",
            url="https://example.invalid/ayushman/benefits",
            zone="benefits",
            text=(
                "The scheme provides health cover up to five lakh rupees "
                "per family per year for listed secondary and tertiary care. "
                "There is no registration fee to activate the card."
            ),
            last_changed_at="2026-01-01T00:00:00+00:00",
            cosine=0.41,
            net=0.71,
            bm25_score=None,
            rank=2,
            source="mock_fixture",
        ),
        scheme_hit(
            doc_id="fixture-ayushman-documents",
            url="https://example.invalid/ayushman/documents",
            zone="documents",
            text=(
                "Carry a government photo identity proof and the beneficiary "
                "card at the empanelled hospital. Do not share Aadhaar OTP "
                "with unknown callers."
            ),
            last_changed_at="2026-01-01T00:00:00+00:00",
            cosine=0.22,
            net=0.55,
            bm25_score=None,
            rank=3,
            source="mock_fixture",
        ),
    ]


def test_scheme_hit_exposes_zonehit_fields() -> None:
    hit = mock_scheme_hits()[0]
    assert hit.net == 0.82
    assert hit.bm25_score is None
    assert hit.rank == 1
    assert hit.last_changed_at == "2026-01-01T00:00:00+00:00"
    for name in (
        "doc_id",
        "text",
        "url",
        "zone",
        "net",
        "last_changed_at",
        "cosine",
        "bm25_score",
        "rank",
    ):
        assert hasattr(hit, name)
    for absent in ("title", "state", "conditions", "domain", "g_score", "crawled_at", "score"):
        assert not hasattr(hit, absent)


def weak_hits() -> list[SchemeHit]:
    """A hit whose net is below the default refusal threshold.

    Cosine is deliberately high so a gate that read cosine instead of net
    would not refuse.
    """
    base = mock_scheme_hits()[0]
    return [replace(base, net=0.02, cosine=0.99, bm25_score=None, rank=1)]
