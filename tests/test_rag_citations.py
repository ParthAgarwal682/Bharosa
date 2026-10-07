"""Citation checker tests — lexical overlap heuristic, not logical entailment."""

from __future__ import annotations

from bharosa.rag.answer import label_hits
from bharosa.rag.citations import check_citations
from bharosa.rag.config import get_config
from bharosa.rag.similarity import text_cosine
from bharosa.rag.types import Answer, CitedSentence

try:
    from tests.test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit
except ImportError:
    from test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit


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


def test_doc_id_cite_is_supported_and_unknown_doc_is_rejected() -> None:
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
                cite_ids=[hits[0].doc_id],
            ),
            CitedSentence(text="Not in the corpus.", cite_ids=["not-a-real-doc"]),
        ],
        hit_labels=label_hits(hits),
    )
    checks = check_citations(ans, hits)
    assert checks[0].flag is None
    assert checks[0].supported is True
    assert checks[1].flag == "unknown_cite_id"
    assert checks[1].supported is False


def test_mixed_list_citation_binds_label_to_doc_id() -> None:
    """Z1 is the net hit even when a BM25 hit sits first in the original list."""
    import json

    from bharosa.rag.answer import answer

    baseline = scheme_hit(
        doc_id="bm25-doc",
        text="Mars colony residents get free spaceship insurance forever.",
        net=None,
        bm25_score=0.99,
        cosine=0.99,
        rank=1,
    )
    official = scheme_hit(
        doc_id="net-doc",
        zone="eligibility",
        url="https://example.invalid/eligible",
        text=(
            "Families with annual income below two lakh rupees in "
            "Uttar Pradesh may be eligible for Ayushman Bharat cover."
        ),
        net=0.8,
        cosine=0.2,
        bm25_score=None,
        rank=2,
    )
    llm = FakeLLM(
        response=json.dumps(
            {
                "claims": [
                    {
                        "text": (
                            "Families with annual income below two lakh rupees "
                            "may be eligible."
                        ),
                        "cite_ids": ["Z1"],
                    }
                ]
            }
        )
    )
    result = answer("eligible?", [baseline, official], llm=llm)
    assert result.refused is False
    assert result.hit_labels["Z1"] == "net-doc::eligibility"
    assert llm.prompts is not None
    assert "bm25-doc" not in llm.prompts[0]
    checks = check_citations(result, [baseline, official])
    assert checks[0].supported is True
    assert checks[0].flag is None


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


def test_default_tokenizer_and_injected_tokenizer() -> None:
    from bharosa.rag.similarity import default_tokenize
    tokens = default_tokenize("Ayushman Bharat scheme covers ₹5,00,000 per family!")
    assert "ayushman" in tokens
    assert "₹5" in tokens or "₹" in tokens or "500000" in tokens
    custom_called = []
    def custom_tok(text: str) -> list[str]:
        custom_called.append(text)
        return text.split()
    score = text_cosine("foo bar", "foo bar baz", tokenizer=custom_tok)
    assert score == 1.0
    assert len(custom_called) == 2


def test_honest_naming_lexical_overlap() -> None:
    from bharosa.rag.similarity import lexical_overlap_score
    hits = mock_scheme_hits()
    ans = Answer(
        query="eligibility?",
        refused=False,
        refusal_reason=None,
        sentences=[CitedSentence(text="Income below two lakh in UP may be eligible.", cite_ids=["Z1"])],
        hit_labels=label_hits(hits),
    )
    checks = check_citations(ans, hits)
    assert checks[0].lexical_overlap > 0.0
    score = lexical_overlap_score("foo bar", "foo bar baz")
    assert score == 1.0


