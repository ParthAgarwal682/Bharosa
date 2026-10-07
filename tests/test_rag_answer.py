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
    from tests.test_rag_fixtures import FakeLLM, mock_scheme_hits, weak_hits
except ImportError:
    from test_rag_fixtures import FakeLLM, mock_scheme_hits, weak_hits


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
    result = answer("UP heart yojana?", weak_hits(), llm=llm, allow_mock=True)
    assert result.refused is True
    assert result.refusal_reason == "low_retrieval_score"
    assert llm.call_count == 0
    assert max(h.score for h in weak_hits()) < get_config().refusal_score_threshold


def test_refuse_missing_state_filter() -> None:
    hits = mock_scheme_hits()
    assert should_refuse(hits, required_state="Kerala", allow_mock=True) == "no_matching_filter"
    llm = FakeLLM(response='{"claims": []}')
    result = answer("yojana?", hits, llm=llm, required_state="Kerala", allow_mock=True)
    assert result.refused is True
    assert llm.call_count == 0


def test_state_filter_matching_with_and_without_aliases() -> None:
    hits = mock_scheme_hits()
    assert all(h.state == "UP" for h in hits)
    # Without aliases: exact casefold only -> "Uttar Pradesh" != "UP" -> refuses
    assert should_refuse(hits, required_state="Uttar Pradesh", allow_mock=True) == "no_matching_filter"

    # With aliases: maps "uttar pradesh" to "up" -> allows
    aliases = {"up": {"uttar pradesh"}}
    assert should_refuse(hits, required_state="Uttar Pradesh", allow_mock=True, state_aliases=aliases) is None
    assert should_refuse(hits, required_state="UP", allow_mock=True, state_aliases=aliases) is None
    assert should_refuse(hits, required_state="Maharashtra", allow_mock=True, state_aliases=aliases) == "no_matching_filter"



def test_build_prompt_labels_zones() -> None:
    hits = mock_scheme_hits()
    prompt = build_prompt("papa ke heart operation ke liye yojana UP", hits)
    assert "[Z1]" in prompt
    assert "[Z2]" in prompt
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


def test_answer_default_refuses_mock_evidence() -> None:
    hits = mock_scheme_hits()
    assert any(h.source == "mock_fixture" for h in hits)
    llm = FakeLLM(response='{"claims":[{"text":"Income below two lakh in UP may be eligible.","cite_ids":["Z1"]}]}')
    res = answer("UP yojana?", hits, llm=llm)
    assert res.refused is True
    assert res.refusal_reason == "mock_evidence"



