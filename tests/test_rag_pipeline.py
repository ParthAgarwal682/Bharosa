"""Tests for post-generation citation policy pipeline."""

import pytest
from bharosa.rag.pipeline import answer_checked
from tests.test_rag_fixtures import FakeLLM, mock_scheme_hits


def test_answer_checked_flag_policy_allows_unsupported() -> None:
    hits = mock_scheme_hits()
    # LLM outputs a claim that is unsupported by cited zone
    fake = FakeLLM(response='{"claims":[{"text":"Martians get free spacecraft.","cite_ids":["Z1"]}]}')
    ans = answer_checked("q", hits, llm=fake, policy="flag", allow_mock=True)
    assert ans.refused is False
    assert len(ans.sentences) == 1
    assert ans.refusal_reason is None


def test_answer_checked_withhold_policy_refuses_when_unsupported() -> None:
    hits = mock_scheme_hits()
    fake = FakeLLM(response='{"claims":[{"text":"Martians get free spacecraft.","cite_ids":["Z1"]}]}')
    ans = answer_checked("q", hits, llm=fake, policy="withhold", allow_mock=True)
    assert ans.refused is True
    assert ans.refusal_reason == "citation_validation_failed"
    assert ans.sentences == []


def test_answer_checked_withhold_policy_succeeds_when_supported() -> None:
    hits = mock_scheme_hits()
    fake = FakeLLM(response='{"claims":[{"text":"Income below two lakh rupees in Uttar Pradesh may be eligible.","cite_ids":["Z1"]}]}')
    ans = answer_checked("q", hits, llm=fake, policy="withhold", allow_mock=True)
    assert ans.refused is False
    assert ans.refusal_reason is None
    assert len(ans.sentences) == 1


def test_answer_checked_reads_policy_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = mock_scheme_hits()
    fake = FakeLLM(response='{"claims":[{"text":"Martians get free spacecraft.","cite_ids":["Z1"]}]}')
    monkeypatch.setenv("RAG_CITATION_POLICY", "withhold")
    ans = answer_checked("q", hits, llm=fake, allow_mock=True)
    assert ans.refused is True
    assert ans.refusal_reason == "citation_validation_failed"


def test_answer_checked_default_blocks_mock_hits() -> None:
    hits = mock_scheme_hits()
    fake = FakeLLM(response='{"claims":[{"text":"Income below two lakh in UP may be eligible.","cite_ids":["Z1"]}]}')
    ans = answer_checked("q", hits, llm=fake)
    assert ans.refused is True
    assert ans.refusal_reason == "mock_evidence"

