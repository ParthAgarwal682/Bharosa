"""Refusal, prompt, and parsing tests for scheme RAG (no live API)."""

from __future__ import annotations

import json
import pytest

from bharosa.rag.answer import (
    INVALID_GENERATION_MESSAGE,
    LLMOutputValidationError,
    answer,
    build_prompt,
    parse_llm_answer,
    should_refuse,
)
from bharosa.rag.config import get_config
from bharosa.rag.types import CitedSentence

try:
    from tests.test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit, weak_hits
except ImportError:
    from test_rag_fixtures import FakeLLM, mock_scheme_hits, scheme_hit, weak_hits


def test_refuse_empty_hits_does_not_call_llm() -> None:
    llm = FakeLLM(response="should not run")
    result = answer("UP heart yojana?", [], llm=llm)
    assert result.refused is True
    assert result.refusal_reason == "empty_hits"
    assert result.sentences == []
    assert llm.call_count == 0
    assert result.refusal_message is not None


def test_refuse_low_score_does_not_call_llm() -> None:
    llm = FakeLLM(response="hallucination")
    hits = weak_hits()
    assert hits[0].net is not None
    assert hits[0].net < get_config().refusal_score_threshold
    assert hits[0].cosine >= get_config().refusal_score_threshold
    assert hits[0].bm25_score is None
    result = answer("UP heart yojana?", hits, llm=llm, allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "low_retrieval_score"
    assert llm.call_count == 0


def test_strong_net_reaches_llm_even_when_cosine_is_low() -> None:
    hit = scheme_hit(
        doc_id="net-strong",
        text="Families with annual income below two lakh rupees may be eligible.",
        zone="eligibility",
        url="https://example.invalid/eligible",
        net=0.9,
        cosine=0.01,
        bm25_score=None,
        last_changed_at="2026-01-01T00:00:00+00:00",
        rank=1,
    )
    llm = FakeLLM(
        response=json.dumps(
            {"claims": [{"text": "Income below two lakh may be eligible.", "cite_ids": ["Z1"]}]}
        )
    )
    result = answer("eligible?", [hit], llm=llm)
    assert result.refused is False
    assert llm.call_count == 1
    assert result.sentences[0].cite_ids == ["Z1"]


def test_bm25_without_net_refuses_before_llm() -> None:
    hit = scheme_hit(
        doc_id="bm25-only",
        text="There is no registration fee to activate the card.",
        net=None,
        bm25_score=0.99,
        cosine=0.99,
        rank=1,
    )
    llm = FakeLLM(response="should not run")
    result = answer("fee?", [hit], llm=llm)
    assert result.refused is True
    assert result.refusal_reason == "unsupported_bm25_evidence"
    assert llm.call_count == 0


def test_missing_net_does_not_become_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BM25 score of 0 must not pass a zero NetScore threshold as net=0."""
    monkeypatch.setenv("RAG_REFUSAL_THRESHOLD", "0")
    hit = scheme_hit(
        doc_id="bm25-zero",
        text="There is no registration fee to activate the card.",
        net=None,
        bm25_score=0.0,
        cosine=0.0,
        rank=1,
    )
    llm = FakeLLM(response="should not run")
    result = answer("fee?", [hit], llm=llm)
    assert result.refused is True
    assert result.refusal_reason == "unsupported_bm25_evidence"
    assert llm.call_count == 0


def test_high_bm25_does_not_rescue_weak_net() -> None:
    weak = scheme_hit(
        doc_id="net-weak",
        text="Families with annual income below two lakh rupees may be eligible.",
        net=0.02,
        bm25_score=None,
        cosine=0.1,
        rank=1,
    )
    baseline = scheme_hit(
        doc_id="bm25-strong",
        text="Families with annual income below two lakh rupees may be eligible.",
        net=None,
        bm25_score=0.99,
        cosine=0.99,
        rank=2,
    )
    llm = FakeLLM(response="should not run")
    result = answer("eligible?", [weak, baseline], llm=llm)
    assert result.refused is True
    assert result.refusal_reason == "low_retrieval_score"
    assert llm.call_count == 0



def test_build_prompt_labels_zones() -> None:
    hits = mock_scheme_hits()
    prompt = build_prompt("papa ke heart operation ke liye yojana UP", hits)
    assert "[Z1]" in prompt
    assert "[Z2]" in prompt
    assert "doc_id=fixture-ayushman-eligibility" in prompt
    assert "zone=eligibility" in prompt
    assert "url=https://example.invalid/ayushman/eligibility" in prompt
    assert "last_changed_at=2026-01-01T00:00:00+00:00" in prompt
    assert "title=" not in prompt
    assert "state=" not in prompt
    assert "conditions=" not in prompt
    assert "g_score=" not in prompt
    assert "crawled_at=" not in prompt
    assert "no registration fee" in prompt.casefold() or "five lakh" in prompt.casefold()
    assert "Question:" in prompt
    assert all(h.source == "mock_fixture" for h in hits)
    assert "JSON" in prompt


def test_answer_with_fake_llm_parses_citations() -> None:
    hits = mock_scheme_hits()
    llm = FakeLLM(
        response=json.dumps(
            {
                "claims": [
                    {
                        "text": "Income below two lakh may be eligible in UP.",
                        "cite_ids": ["Z1"],
                    },
                    {
                        "text": "Cover is up to five lakh per family per year.",
                        "cite_ids": ["Z2"],
                    },
                ]
            }
        )
    )
    result = answer(
        "papa ke heart operation ke liye yojana, UP, income 2 lakh",
        hits,
        llm=llm,
        allow_mock=True,
    )
    assert result.refused is False
    assert llm.call_count == 1
    assert len(result.sentences) == 2
    assert result.sentences[0].cite_ids == ["Z1"]
    assert result.sentences[1].cite_ids == ["Z2"]
    assert "Z1" in result.hit_labels


def test_parse_llm_answer_empty_raw() -> None:
    assert parse_llm_answer("", {"Z1": "x"}) == []
    assert parse_llm_answer("   ", {"Z1": "x"}) == []


def test_parse_llm_answer_malformed_json_raises() -> None:
    """Malformed JSON must be rejected with LLMOutputValidationError."""
    with pytest.raises(LLMOutputValidationError, match="Malformed JSON"):
        parse_llm_answer("not valid json at all", {"Z1": "x"})


def test_parse_llm_answer_empty_cite_ids_raises() -> None:
    """Claims missing cite_ids or with empty cite_ids must be rejected."""
    raw = json.dumps({"claims": [{"text": "This has no citation.", "cite_ids": []}]})
    with pytest.raises(LLMOutputValidationError, match="missing cite_ids"):
        parse_llm_answer(raw, {"Z1": "x"})


def test_parse_llm_answer_unknown_cite_id_raises() -> None:
    """Unknown cite IDs must be rejected with LLMOutputValidationError, not silently dropped."""
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "Eligible for coverage.",
                    "cite_ids": ["Z1", "Z99"],
                }
            ]
        }
    )
    with pytest.raises(LLMOutputValidationError, match="Unknown cite_id 'Z99'"):
        parse_llm_answer(raw, {"Z1": "doc1::eligibility"})


def test_answer_malformed_output_refuses() -> None:
    """Malformed LLM output produces a refusal answer with neutral user message and empty sentences."""
    hits = mock_scheme_hits()
    llm = FakeLLM(response="raw unformatted text that is not json")
    result = answer("UP yojana?", hits, llm=llm, allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "invalid_generation"
    assert result.sentences == []
    assert result.refusal_message == INVALID_GENERATION_MESSAGE
    assert result.raw_llm == "raw unformatted text that is not json"


def test_answer_unknown_cite_id_refuses() -> None:
    """LLM hallucinating unknown cite_ids produces a refusal answer, not an answer with dropped IDs."""
    hits = mock_scheme_hits()
    llm = FakeLLM(
        response=json.dumps(
            {
                "claims": [
                    {
                        "text": "Some claim.",
                        "cite_ids": ["Z99"],
                    }
                ]
            }
        )
    )
    result = answer("UP yojana?", hits, llm=llm, allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "invalid_generation"
    assert result.sentences == []
    assert result.refusal_message == INVALID_GENERATION_MESSAGE


def test_answer_empty_generation_refuses() -> None:
    hits = mock_scheme_hits()
    for raw in ['{"claims": []}', '', '   ', '{"claims":[{"text":"","cite_ids":["Z1"]}]}']:
        llm = FakeLLM(response=raw)
        result = answer("UP yojana?", hits, llm=llm, allow_mock=True)
        assert result.refused is True
        assert result.refusal_reason == "empty_generation"
        assert result.sentences == []
        assert result.refusal_message is not None


def test_answer_llm_not_configured_refusal_reason() -> None:
    from bharosa.rag.llm import LLMNotConfiguredError
    hits = mock_scheme_hits()
    class NoKeyLLM:
        def complete(self, prompt: str) -> str:
            raise LLMNotConfiguredError("Missing key")
    result = answer("UP yojana?", hits, llm=NoKeyLLM(), allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "llm_not_configured"


def test_answer_llm_unavailable_refusal_reason() -> None:
    from bharosa.rag.llm import LLMProviderError
    hits = mock_scheme_hits()
    class BrokenLLM:
        def complete(self, prompt: str) -> str:
            raise LLMProviderError("Network down")
    result = answer("UP yojana?", hits, llm=BrokenLLM(), allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "llm_unavailable"

    class TimeoutLLM:
        def complete(self, prompt: str) -> str:
            raise TimeoutError("Request timed out")
    result_to = answer("UP yojana?", hits, llm=TimeoutLLM(), allow_mock=True)
    assert result_to.refused is True
    assert result_to.refusal_reason == "llm_unavailable"


def test_parse_accepts_doc_id_cite() -> None:
    hits = mock_scheme_hits()
    labels = {f"Z{i}": f"{hit.doc_id}::{hit.zone}" for i, hit in enumerate(hits, start=1)}
    raw = json.dumps(
        {"claims": [{"text": "Income below two lakh may be eligible.", "cite_ids": [hits[0].doc_id]}]}
    )
    sentences = parse_llm_answer(raw, labels)
    assert sentences[0].cite_ids == ["Z1"]


def test_answer_default_refuses_mock_evidence() -> None:
    hits = mock_scheme_hits()
    assert any(h.source == "mock_fixture" for h in hits)
    llm = FakeLLM(response='{"claims":[{"text":"Income below two lakh in UP may be eligible.","cite_ids":["Z1"]}]}')
    res = answer("UP yojana?", hits, llm=llm)
    assert res.refused is True
    assert res.refusal_reason == "mock_evidence"



