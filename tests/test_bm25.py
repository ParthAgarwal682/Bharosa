"""Tests for the external BM25 baseline.

Score checks use the published BM25Okapi formula and are skipped when
``rank_bm25`` is not installed. This file does not reimplement that
formula. Structural checks run either way.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import pytest

from bharosa.rank.bm25 import (
    EXTERNAL_CLASS,
    EXTERNAL_LIBRARY,
    LIBRARY_B,
    LIBRARY_EPSILON,
    LIBRARY_K1,
    BM25Baseline,
)


def _source() -> str:
    import bharosa.rank.bm25 as bm25

    assert bm25.__file__ is not None
    return Path(bm25.__file__).read_text(encoding="utf-8")


def test_module_is_labelled_as_an_external_baseline() -> None:
    source = _source()
    folded = source.casefold()
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    assert EXTERNAL_LIBRARY == "rank_bm25"
    assert EXTERNAL_CLASS == "BM25Okapi"
    assert (LIBRARY_K1, LIBRARY_B, LIBRARY_EPSILON) == (1.5, 0.75, 0.25)
    assert "external baseline" in folded
    assert "optimal" not in folded
    assert "optimum" not in folded
    assert "_calc_idf" not in source
    assert "math.log" not in source
    assert "bharosa.rank.tfidf" not in imported
    assert "bharosa.rank.netscore" not in imported
    assert "bharosa.rank.topk" not in imported
    assert "rank_bm25" in imported


def test_empty_corpus_does_not_invent_a_length() -> None:
    baseline = BM25Baseline({})

    assert baseline.document_ids() == ()
    assert baseline.scores("heart") == ()
    assert baseline.rank("heart", 3) == ()
    assert baseline.parameters() == (LIBRARY_K1, LIBRARY_B, LIBRARY_EPSILON)
    assert baseline.uses_library_defaults()
    assert repr(baseline) == "BM25Baseline(documents=0, library=rank_bm25.BM25Okapi)"
    with pytest.raises(ValueError, match="not invented"):
        BM25Baseline({"d1": "..."})
    with pytest.raises(TypeError):
        BM25Baseline(["heart"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        baseline.rank("heart", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        baseline.rank("heart", -1)


def test_missing_library_is_reported() -> None:
    if importlib.util.find_spec("rank_bm25") is not None:
        pytest.skip("rank_bm25 is installed")
    with pytest.raises(ImportError, match="rank_bm25"):
        BM25Baseline({"d1": "heart"})


def test_hand_computed_okapi_score_for_a_single_query_term() -> None:
    pytest.importorskip("rank_bm25")
    # N = 3, dl = avgdl = 2, tf(heart, d1) = 1, df(heart) = 1.
    # idf = ln((3 - 1 + 0.5) / (1 + 0.5)) = ln(5/3).
    # With dl = avgdl the tf component is 1 for tf = 1, so the score is the idf.
    baseline = BM25Baseline(
        {
            "d1": "heart scheme",
            "d2": "cover notes",
            "d3": "other terms",
        }
    )
    expected = math.log(5 / 3)
    scores = baseline.scores("heart")

    assert [hit.doc_id for hit in scores] == ["d1", "d2", "d3"]
    assert scores[0].score == pytest.approx(expected)
    assert scores[1].score == 0.0
    assert scores[2].score == 0.0
    assert baseline.scores("HEART")[0].score == pytest.approx(expected)
    # The library adds one term score per query token, so a repeated
    # query token adds the same document contribution again.
    assert baseline.scores("heart heart")[0].score == pytest.approx(2 * expected)
    ranked = baseline.rank("heart", 5)
    assert [hit.doc_id for hit in ranked] == ["d1"]
    assert ranked[0].score == pytest.approx(expected)
    assert baseline.rank("...", 3) == ()
    assert "external baseline" in baseline.format_scores("heart", k=1)
    assert "rank_bm25.BM25Okapi" in baseline.format_scores("heart", k=1)
    assert "not lnc.ltc and not net score" in baseline.format_scores("heart", k=1)
    assert "published library defaults" in baseline.format_scores("heart", k=1)


def test_tf_saturation_and_tie_break_follow_the_library_score() -> None:
    pytest.importorskip("rank_bm25")
    # N = 5, every document has length 2, avgdl = 2, df(heart) = 2.
    # idf = ln(3.5 / 2.5) = ln(1.4).
    # k1 = 1.5, b = 0.75, dl = avgdl, so the length factor is 1.
    # tf = 2 -> 2 * 2.5 / (2 + 1.5) = 10/7
    # tf = 1 -> 1
    baseline = BM25Baseline(
        {
            "d1": "heart heart",
            "d2": "heart extra",
            "d3": "alpha beta",
            "d4": "gamma delta",
            "d5": "epsilon zeta",
        }
    )
    idf = math.log(1.4)
    scores = baseline.scores("heart")

    assert scores[0].score == pytest.approx(idf * (10 / 7))
    assert scores[1].score == pytest.approx(idf)
    assert scores[2].score == 0.0
    assert [hit.doc_id for hit in baseline.rank("heart", 5)] == ["d1", "d2"]

    tied = BM25Baseline(
        {
            "b": "heart zz",
            "a": "heart yy",
            "c": "aa bb",
            "d": "cc dd",
            "e": "ee ff",
        }
    )
    assert [hit.doc_id for hit in tied.rank("heart", 2)] == ["a", "b"]
    assert tied.rank("heart", 2)[0].score == pytest.approx(tied.rank("heart", 2)[1].score)


def test_negative_idf_floor_is_the_library_epsilon() -> None:
    pytest.importorskip("rank_bm25")
    # Both documents contain the only term, so the library's raw idf is
    # ln(0.2), which is negative. BM25Okapi replaces it with
    # epsilon * average_idf. average_idf is ln(0.2) here.
    from bharosa.index.inverted import InvertedIndex

    index = InvertedIndex.from_documents({"d1": "heart", "d2": "heart"})
    assert index.idf("heart") == 0.0

    baseline = BM25Baseline({"d1": "heart", "d2": "heart"})
    expected = 0.25 * math.log(0.2)
    assert expected < 0
    assert baseline.scores("heart")[0].score == pytest.approx(expected)
    assert baseline.scores("heart")[1].score == pytest.approx(expected)

    doubled = BM25Baseline({"d1": "heart", "d2": "heart"}, epsilon=0.5)
    assert doubled.parameters() == (LIBRARY_K1, LIBRARY_B, 0.5)
    assert doubled.uses_library_defaults() is False
    assert doubled.scores("heart")[0].score == pytest.approx(0.5 * math.log(0.2))
    assert "caller-supplied" in doubled.format_scores("heart", k=1)


def test_stopwords_follow_the_shared_analyser() -> None:
    pytest.importorskip("rank_bm25")
    # "the" occurs in d1, so a query that keeps it is a different BM25
    # sum from a query that drops it.
    documents = {
        "d1": "the heart scheme",
        "d2": "cover notes extra",
        "d3": "alpha beta gamma",
    }
    dropped = BM25Baseline(documents, remove_stopwords=True)
    kept = BM25Baseline(documents)

    assert dropped.scores("the heart")[0].score == pytest.approx(
        dropped.scores("heart")[0].score
    )
    assert kept.scores("the heart")[0].score != pytest.approx(kept.scores("heart")[0].score)
