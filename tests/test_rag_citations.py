"""Citation checker tests — lexical overlap heuristic, not logical entailment."""

from __future__ import annotations

from bharosa.rag.answer import label_hits
from bharosa.rag.citations import check_citations
from bharosa.rag.config import get_config
from bharosa.rag.similarity import text_cosine
from bharosa.rag.types import Answer, CitedSentence

try:
    from tests.test_rag_fixtures import mock_scheme_hits
except ImportError:
    from test_rag_fixtures import mock_scheme_hits


def test_supported_sentence_near_cited_zone() -> None:
    hits = mock_scheme_hits()
    ans = Answer(
        query="eligibility?",
        refused=False,
        refusal_reason=None,
        sentences=[
            CitedSentence(
                text=(
                    "Families with annual income below two lakh rupees in "
                    "Uttar Pradesh may be eligible for Ayushman Bharat cover"
                ),
                cite_ids=["Z1"],
            )
        ],
        hit_labels=label_hits(hits),
    )
    checks = check_citations(ans, hits)
    assert len(checks) == 1
    assert checks[0].flag is None
    assert checks[0].supported is True
    assert checks[0].max_cosine >= get_config().citation_threshold


def test_unsupported_sentence_low_overlap() -> None:
    hits = mock_scheme_hits()
    ans = Answer(
        query="eligibility?",
        refused=False,
        refusal_reason=None,
        sentences=[
            CitedSentence(
                text="Mars colony residents get free spaceship insurance forever.",
                cite_ids=["Z1"],
            )
        ],
        hit_labels=label_hits(hits),
    )
    checks = check_citations(ans, hits)
    assert checks[0].supported is False
    assert checks[0].flag == "unsupported"
    assert checks[0].max_cosine < get_config().citation_threshold


def test_missing_and_unknown_cite_flags() -> None:
    hits = mock_scheme_hits()
    ans = Answer(
        query="q",
        refused=False,
        refusal_reason=None,
        sentences=[
            CitedSentence(text="No cite here.", cite_ids=[]),
            CitedSentence(text="Bad id.", cite_ids=["Z99"]),
        ],
        hit_labels=label_hits(hits),
    )
    checks = check_citations(ans, hits)
    assert checks[0].flag == "missing_citation"
    assert checks[1].flag == "unknown_cite_id"


def test_refused_answer_yields_no_checks() -> None:
    hits = mock_scheme_hits()
    ans = Answer(
        query="q",
        refused=True,
        refusal_reason="empty_hits",
        sentences=[],
        hit_labels={},
    )
    assert check_citations(ans, hits) == []


def test_text_cosine_identical_higher_than_unrelated() -> None:
    text = mock_scheme_hits()[0].text
    same = text_cosine(text, text)
    other = text_cosine(text, "completely unrelated pineapple recipe")
    assert same > other
    assert same > 0.9
