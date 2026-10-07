"""CLI wiring from scheme retrieval into RAG and the claim checker.

The crawler rows in this file are fixtures for the command boundary.
They are not a live crawl. Test doubles stand in for the LLM only.
Retrieval, citation checks, refusal, and claim comparison stay the
real module code.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from bharosa.app.cli import main
from bharosa.crawl.db import CrawlDB
from bharosa.crawl.dedup import compute_content_hash
from bharosa.index.zones import ZoneHit
from tests.test_rag_fixtures import FakeLLM

_CLI_PATH = Path(__file__).resolve().parents[1] / "bharosa" / "app" / "cli.py"
_FETCHED = "2026-10-06T10:00:00Z"
_ZONE_TEXT = (
    "Families with annual income below two lakh rupees in Uttar Pradesh "
    "may be eligible for Ayushman Bharat cover."
)
_UP_URL = "https://scheme.example/uttar-pradesh"
_BIHAR_URL = "https://scheme.example/bihar"


def _page(body: str, title: str = "Scheme fixture page") -> str:
    return (
        "<!DOCTYPE html><html><head><title>"
        f"{title}</title></head><body>{body}</body></html>"
    )


def _store(db_path: Path, url: str, html: str) -> None:
    raw = db_path.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    body = html.encode("utf-8")
    digest = compute_content_hash(body)
    path = raw / f"{digest[:16]}.html"
    path.write_bytes(body)
    database = CrawlDB(db_path=str(db_path))
    try:
        database.upsert_page(
            url=url,
            domain="scheme.example",
            fetched_at=_FETCHED,
            content_hash=digest,
            g_score=0.75,
            text_path=str(path),
        )
    finally:
        database.close()


def _eligibility_db(tmp_path: Path) -> Path:
    """Two pages so a matching query has a non-zero ltc idf.

    lnc.ltc gives every query term idf 0 when that term is in every
    document. A second, unrelated page keeps the Ayushman terms rare.
    """
    db_path = tmp_path / "bharosa.db"
    _store(
        db_path,
        _UP_URL,
        _page(f"<h2>Eligibility</h2><p>{_ZONE_TEXT}</p>"),
    )
    _store(
        db_path,
        "https://scheme.example/other",
        _page(
            "<h2>Benefits</h2>"
            "<p>Wheat procurement payments follow a separate agriculture notice.</p>"
        ),
    )
    return db_path


def _two_state_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "bharosa.db"
    _store(
        db_path,
        _UP_URL,
        _page(
            "<p>State: Uttar Pradesh</p>"
            f"<h2>Eligibility</h2><p>{_ZONE_TEXT}</p>"
        ),
    )
    _store(
        db_path,
        _BIHAR_URL,
        _page(
            "<p>State: Bihar</p>"
            "<h2>Eligibility</h2>"
            "<p>Families with annual income below two lakh rupees in Bihar "
            "may be eligible for Ayushman Bharat cover.</p>"
        ),
    )
    return db_path


def _clear_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bharosa.rag.llm._load_dotenv", lambda: None)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _clear_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider(monkeypatch)
    for name in (
        "RAG_REFUSAL_THRESHOLD",
        "RAG_CITATION_THRESHOLD",
        "RAG_CITATION_POLICY",
        "RAG_MIN_RETRIEVAL_SCORE",
    ):
        monkeypatch.delenv(name, raising=False)


def _install_llm(monkeypatch: pytest.MonkeyPatch, response: str) -> FakeLLM:
    fake = FakeLLM(response=response)
    answer_module = importlib.import_module("bharosa.rag.answer")
    monkeypatch.setattr(answer_module, "get_llm", lambda: fake)
    return fake


def _cited_response(text: str, cite: str = "Z1") -> str:
    return json.dumps({"claims": [{"text": text, "cite_ids": [cite]}]})


def test_scheme_retrieval_passes_zone_hits_to_checked_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _eligibility_db(tmp_path)
    fake = _install_llm(monkeypatch, _cited_response(_ZONE_TEXT))
    seen: dict[str, object] = {}

    import bharosa.rag.pipeline as pipeline

    real_answer = pipeline.answer_checked

    def wrapped_answer(query: str, hits: list[ZoneHit], **kwargs: object) -> object:
        seen["query"] = query
        seen["hits"] = list(hits)
        seen["kwargs"] = kwargs
        return real_answer(query, hits, **kwargs)

    monkeypatch.setattr(pipeline, "answer_checked", wrapped_answer)

    main(["scheme", _ZONE_TEXT, "--db", str(db_path), "--verbose"])

    hits = seen["hits"]
    assert isinstance(hits, list) and hits
    assert all(type(hit) is ZoneHit for hit in hits)
    assert all(hit.net is not None and hit.bm25_score is None for hit in hits)
    kwargs = seen["kwargs"]
    assert kwargs == {}
    assert seen["query"] == _ZONE_TEXT
    assert fake.call_count == 1

    output = capsys.readouterr().out
    assert f"query: {_ZONE_TEXT}" in output
    assert "filters: {}" in output
    assert "refused: False" in output
    assert _ZONE_TEXT in output
    assert "cite_ids: Z1" in output
    assert _UP_URL in output
    assert "doc_id=" in output
    assert "net=<not scored>" not in output
    assert "bm25_score=<not scored>" in output
    assert "lexical_overlap=" in output
    assert "supported=True" in output
    assert "[tokens]" in output
    assert "citation_policy: flag" in output


def test_scheme_weak_net_refuses_without_calling_the_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("RAG_REFUSAL_THRESHOLD", "1.01")
    db_path = _eligibility_db(tmp_path)

    def reject_llm() -> FakeLLM:
        raise AssertionError("weak evidence must refuse before any LLM call")

    answer_module = importlib.import_module("bharosa.rag.answer")
    monkeypatch.setattr(answer_module, "get_llm", reject_llm)

    main(["scheme", _ZONE_TEXT, "--db", str(db_path)])

    output = capsys.readouterr().out
    assert "refused: True" in output
    assert "refusal_reason: low_retrieval_score" in output
    assert "citation checks" in output
    assert "sentences:" not in output
    error = capsys.readouterr().err
    assert error == ""


def test_scheme_citation_checks_are_invoked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("RAG_CITATION_POLICY", "flag")
    db_path = _eligibility_db(tmp_path)
    _install_llm(
        monkeypatch,
        _cited_response("Martians receive free spacecraft from this scheme."),
    )
    calls: list[object] = []

    import bharosa.rag.pipeline as pipeline

    real_check = pipeline.check_citations

    def wrapped_check(ans: object, hits: list[ZoneHit], threshold: float | None = None):
        calls.append(ans)
        return real_check(ans, hits, threshold=threshold)

    monkeypatch.setattr(pipeline, "check_citations", wrapped_check)

    main(["scheme", _ZONE_TEXT, "--db", str(db_path)])

    assert len(calls) == 1
    output = capsys.readouterr().out
    assert "refused: False" in output
    assert "citation_policy: flag" in output
    assert "supported=False" in output
    assert "flag=unsupported" in output
    assert "lexical_overlap=" in output


def test_scheme_withhold_policy_refuses_when_citation_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("RAG_CITATION_POLICY", "withhold")
    db_path = _eligibility_db(tmp_path)
    _install_llm(
        monkeypatch,
        _cited_response("Martians receive free spacecraft from this scheme."),
    )

    main(["scheme", _ZONE_TEXT, "--db", str(db_path), "--verbose"])

    output = capsys.readouterr().out
    assert "citation_policy: withhold" in output
    assert "refused: True" in output
    assert "refusal_reason: citation_validation_failed" in output
    assert "supported=False" in output
    assert "refusal_score_threshold:" in output


def test_scheme_state_filter_limits_hits_passed_to_rag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _two_state_db(tmp_path)
    _install_llm(monkeypatch, _cited_response(_ZONE_TEXT))
    seen: dict[str, object] = {}

    import bharosa.rag.pipeline as pipeline

    real_answer = pipeline.answer_checked

    def wrapped_answer(query: str, hits: list[ZoneHit], **kwargs: object) -> object:
        seen["hits"] = list(hits)
        seen["kwargs"] = kwargs
        return real_answer(query, hits, **kwargs)

    monkeypatch.setattr(pipeline, "answer_checked", wrapped_answer)

    main(
        [
            "scheme",
            "Ayushman Bharat eligible families",
            "--db",
            str(db_path),
            "--state",
            "Uttar Pradesh",
        ]
    )

    hits = seen["hits"]
    assert isinstance(hits, list) and hits
    assert {hit.url for hit in hits} == {_UP_URL}
    assert "filters" not in seen["kwargs"]
    assert seen["kwargs"] == {}
    output = capsys.readouterr().out
    assert "filters: state=Uttar Pradesh" in output
    assert _BIHAR_URL not in output
    assert _UP_URL in output


def test_scheme_bm25_does_not_convert_score_into_net(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _eligibility_db(tmp_path)

    def reject_llm() -> FakeLLM:
        raise AssertionError("BM25 evidence must refuse before any LLM call")

    answer_module = importlib.import_module("bharosa.rag.answer")
    monkeypatch.setattr(answer_module, "get_llm", reject_llm)

    main(["scheme", _ZONE_TEXT, "--db", str(db_path), "--bm25"])

    output = capsys.readouterr().out
    assert "flags: bm25=True" in output
    assert "net=<not scored>" in output
    assert "bm25_score=<not scored>" not in output
    assert "refused: True" in output
    assert "refusal_reason: unsupported_bm25_evidence" in output
    assert "sentences:" not in output


def test_claim_passes_retrieve_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _two_state_db(tmp_path)
    message = "Pay Rs 500 to activate Ayushman Bharat for eligible families."
    seen: dict[str, object] = {}

    import bharosa.rag.claimcheck as claimcheck

    real_check = claimcheck.check_claim

    def wrapped_check(message: str, retrieve: object, **kwargs: object) -> object:
        assert callable(retrieve)
        hits = list(retrieve(message))
        seen["hits"] = hits
        seen["kwargs"] = kwargs
        return real_check(message, retrieve, **kwargs)

    monkeypatch.setattr(claimcheck, "check_claim", wrapped_check)

    main(
        [
            "claim",
            message,
            "--db",
            str(db_path),
            "--state",
            "Uttar Pradesh",
            "--verbose",
        ]
    )

    hits = seen["hits"]
    assert isinstance(hits, list) and hits
    assert all(type(hit) is ZoneHit for hit in hits)
    assert {hit.url for hit in hits} == {_UP_URL}
    assert all(hit.bm25_score is None for hit in hits)
    assert seen["kwargs"] == {}
    output = capsys.readouterr().out
    assert f"message: {message}" in output
    assert "filters: state=Uttar Pradesh" in output
    assert "[verdict]" in output
    assert "label:" in output
    assert "[retrieval]" in output
    assert f"query: {message}" in output
    assert _BIHAR_URL not in output
    assert _UP_URL in output


def test_claim_without_valid_evidence_returns_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _eligibility_db(tmp_path)
    message = "pay Rs 500 to activate Ayushman Bharat"

    main(["claim", message, "--db", str(db_path), "--verbose"])

    output = capsys.readouterr().out
    assert f"message: {message}" in output
    assert "label: INSUFFICIENT_EVIDENCE" in output
    assert "Official text does not clearly confirm or deny the claim." in output
    assert "label: SUPPORTED" not in output
    assert "label: CONTRADICTED" not in output
    assert "evidence_cite: Z1" in output
    assert _ZONE_TEXT in output
    assert "retrieval_score: <not scored>" not in output
    assert "evidence_source=yes" in output
    assert "https://scheme.example/other" not in output
    assert _UP_URL in output


def test_scheme_missing_llm_configuration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _eligibility_db(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main(["scheme", _ZONE_TEXT, "--db", str(db_path)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "[answer]" not in captured.out
    assert "sentences:" not in captured.out
    assert "cite_ids:" not in captured.out
    assert "RAG is not configured:" in captured.err
    assert "LLM not configured" in captured.err
    assert "LLM_PROVIDER" in captured.err
    assert "LLM_MODEL" in captured.err
    assert "LLM_API_KEY" in captured.err
    assert "Martians" not in captured.out
    assert "mock" not in captured.out.casefold()


def test_missing_crawler_database_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "absent" / "bharosa.db"

    with pytest.raises(SystemExit) as scheme_exit:
        main(["scheme", "ayushman", "--db", str(missing)])
    assert scheme_exit.value.code == 1
    scheme_error = capsys.readouterr().err
    assert "scheme corpus is unavailable:" in scheme_error
    assert "crawler database not found" in scheme_error
    assert "empty" not in scheme_error.casefold() or "not returned" in scheme_error

    with pytest.raises(SystemExit) as claim_exit:
        main(["claim", "pay Rs 500", "--db", str(missing)])
    assert claim_exit.value.code == 1
    claim_error = capsys.readouterr().err
    assert "scheme corpus is unavailable:" in claim_error
    assert "crawler database not found" in claim_error


def test_missing_rag_module_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_rag_env(monkeypatch)
    db_path = _eligibility_db(tmp_path)
    import bharosa.rag.pipeline as pipeline

    monkeypatch.delattr(pipeline, "answer_checked")

    with pytest.raises(SystemExit) as caught:
        main(["scheme", _ZONE_TEXT, "--db", str(db_path)])

    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "answer_checked could not be imported" in error
    assert "No answer was invented." in error


def test_missing_claim_checker_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bharosa.rag.claimcheck as claimcheck

    monkeypatch.delattr(claimcheck, "check_claim")

    with pytest.raises(SystemExit) as caught:
        main(["claim", "pay Rs 500 to activate"])

    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "check_claim could not be imported" in error
    assert "No mock verdict was used." in error


def test_cli_does_not_fabricate_rag_or_claim_results() -> None:
    source = _CLI_PATH.read_text(encoding="utf-8")
    assert "allow_mock" not in source
    assert "FakeLLM" not in source
    assert "mock_scheme_hits" not in source
    assert "answer_checked(query, hits)" in source
    assert "check_claim(args.message, retrieve)" in source
    assert "check_claim(message)" not in source
    assert "check_claim(args.message)" not in source
    assert "net = hit.bm25_score" not in source
    assert "net=hit.bm25_score" not in source
