"""Known-answer tests for lnc.ltc cosine scoring.

The ranking test states the arithmetic for query "heart" on a three
document corpus. Expected weights come from that arithmetic, not from
calling the ranker and copying its output.
"""

from __future__ import annotations

import math

import pytest

from bharosa.index.inverted import InvertedIndex
from bharosa.rank.tfidf import TfidfRanker, cosine, log_tf

# d1 tokens: heart, heart, scheme
# d2 tokens: heart, cover
# d3 tokens: scheme, cover, cover
DOCS = {
    "d1": "heart heart scheme",
    "d2": "heart cover",
    "d3": "scheme cover cover",
}


def _ranker() -> TfidfRanker:
    return TfidfRanker(InvertedIndex.from_documents(DOCS))


def test_log_tf_is_one_plus_log10() -> None:
    assert log_tf(0) == 0.0
    assert log_tf(1) == 1.0
    assert log_tf(2) == pytest.approx(1.0 + math.log10(2))
    with pytest.raises(ValueError):
        log_tf(-1)
    with pytest.raises(TypeError):
        log_tf(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        log_tf(True)  # type: ignore[arg-type]


def test_cosine_dots_normalized_weights_and_zero_vectors_score_zero() -> None:
    # Normalization is a separate step. These coordinates are not unit
    # length, and the product is left as a dot product.
    assert cosine({"a": 2.0}, {"a": 3.0}) == pytest.approx(6.0)
    assert cosine({"a": 0.6, "b": 0.8}, {"a": 0.6, "b": 0.8}) == pytest.approx(1.0)
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0
    assert cosine({}, {"a": 1.0}) == 0.0
    assert cosine({"a": 0.0}, {"a": 1.0}) == 0.0
    assert cosine({"a": 0.0}, {"a": 0.0}) == 0.0


def test_hand_checked_heart_ranking() -> None:
    # Manual check for query "heart". N = 3. heart, scheme, and cover
    # each occur in 2 documents, so idf = log10(3/2) ≈ 0.176091.
    # The query has one term and that idf is not 0, so ltc normalization
    # sets the heart coordinate to 1.
    #
    # Document lnc weights use 1+log10(tf) and no idf.
    #   L = 1+log10(2) ≈ 1.301030
    #   d1: heart tf=2, scheme tf=1
    #     norm = sqrt(L^2 + 1^2) ≈ 1.640938
    #     cosine = L / norm ≈ 0.792857
    #   d2: heart tf=1, cover tf=1
    #     norm = sqrt(2)
    #     cosine = 1/sqrt(2) ≈ 0.707107
    #   d3 has no heart, so the dot product is 0. d3's own vector is
    #   not zero; it still contains scheme and cover.
    #
    # d1 outranks d2. Let L > 1. Compare L/sqrt(L^2+1) with 1/sqrt(2).
    # Both sides are positive, so square them:
    #   L^2 / (L^2+1)  ?  1/2
    #   2 L^2          ?  L^2 + 1
    #   L^2            ?  1
    # which holds because L = 1+log10(2) > 1.
    ranker = _ranker()
    length = 1.0 + math.log10(2)
    d1_norm = math.sqrt(length * length + 1.0)
    d1_expected = length / d1_norm
    d2_expected = 1.0 / math.sqrt(2.0)

    assert length > 1.0
    assert f"{math.log10(3 / 2):.6f}" == "0.176091"
    assert f"{d1_expected:.6f}" == "0.792857"
    assert f"{d2_expected:.6f}" == "0.707107"

    query = ranker.query_vector("heart")
    assert query.kind == "query"
    assert query.doc_id is None
    assert not query.is_zero
    assert [item.term for item in query.weights] == ["heart"]
    heart_q = query.weights[0]
    assert heart_q.tf == 1
    assert heart_q.log_tf == 1.0
    assert heart_q.idf_factor == pytest.approx(math.log10(3 / 2))
    assert heart_q.raw == pytest.approx(math.log10(3 / 2))
    assert query.norm == pytest.approx(math.log10(3 / 2))
    assert heart_q.weight == pytest.approx(1.0)

    d1 = ranker.document_vector("d1")
    assert d1.kind == "document"
    assert d1.doc_id == "d1"
    d1_weights = {item.term: item for item in d1.weights}
    assert set(d1_weights) == {"heart", "scheme"}
    assert d1_weights["heart"].tf == 2
    assert d1_weights["heart"].log_tf == pytest.approx(length)
    assert d1_weights["heart"].idf_factor == 1.0
    assert d1_weights["scheme"].idf_factor == 1.0
    assert d1.norm == pytest.approx(d1_norm)
    assert d1_weights["heart"].weight == pytest.approx(d1_expected)
    assert d1_weights["scheme"].weight == pytest.approx(1.0 / d1_norm)
    assert sum(item.weight**2 for item in d1.weights) == pytest.approx(1.0)

    d2 = ranker.document_vector("d2")
    assert d2.norm == pytest.approx(math.sqrt(2.0))
    assert {item.term: item.weight for item in d2.weights} == pytest.approx(
        {"heart": d2_expected, "cover": d2_expected}
    )

    assert not ranker.document_vector("d3").is_zero
    assert ranker.cosine("heart", "d1") == pytest.approx(d1_expected)
    assert ranker.cosine("heart", "d2") == pytest.approx(d2_expected)
    assert ranker.cosine("heart", "d3") == 0.0
    assert ranker.cosine("HEART", "d1") == pytest.approx(d1_expected)

    assert [item.doc_id for item in ranker.scores("heart")] == ["d1", "d2", "d3"]
    assert ranker.scores("heart")[2].score == 0.0
    ranked = ranker.rank("heart", k=2)
    assert [item.doc_id for item in ranked] == ["d1", "d2"]
    assert ranked[0].score == pytest.approx(d1_expected)
    assert ranked[1].score == pytest.approx(d2_expected)
    assert [item.doc_id for item in ranker.rank("heart", k=10)] == ["d1", "d2"]

    # Query "heart scheme": both terms have tf 1 and the same idf, so
    # each normalized query weight is 1/sqrt(2). d2 contains heart at
    # weight 1/sqrt(2) and does not contain scheme.
    # cosine = (1/sqrt(2)) * (1/sqrt(2)) = 1/2.
    both = ranker.query_vector("heart scheme")
    assert [item.weight for item in both.weights] == pytest.approx(
        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
    )
    assert ranker.cosine("heart scheme", "d2") == pytest.approx(0.5)

    trace = ranker.format_scores("heart", k=2)
    assert "document=lnc" in trace
    assert "query=ltc" in trace
    assert "log_tf=" in trace
    assert "idf_factor=1.000000" in trace
    assert f"{d1_expected:.6f}" in trace
    assert f"{d2_expected:.6f}" in trace
    assert trace.index("  1  d1") < trace.index("  2  d2")
    assert "TfidfRanker(documents=3)" == repr(ranker)


def test_document_weights_do_not_use_idf() -> None:
    # N = 2. common is in both documents, so idf(common) = log10(1) = 0.
    # rare is in one document, so idf(rare) = log10(2).
    # If the document triple were ltc, common's raw weight would be 0
    # and d1 would collapse to a single rare coordinate of weight 1.
    index = InvertedIndex.from_documents(
        {"d1": "common common rare", "d2": "common"}
    )
    ranker = TfidfRanker(index)
    weights = {item.term: item for item in ranker.document_vector("d1").weights}
    length = math.sqrt((1.0 + math.log10(2)) ** 2 + 1.0**2)

    assert index.idf("common") == 0.0
    assert index.idf("rare") == pytest.approx(math.log10(2))
    assert weights["common"].tf == 2
    assert weights["common"].log_tf == pytest.approx(1.0 + math.log10(2))
    assert weights["common"].idf_factor == 1.0
    assert weights["rare"].idf_factor == 1.0
    assert weights["common"].weight == pytest.approx((1.0 + math.log10(2)) / length)
    assert weights["rare"].weight == pytest.approx(1.0 / length)

    assert ranker.query_vector("common").is_zero
    assert ranker.cosine("common", "d1") == 0.0
    assert ranker.cosine("common", "d2") == 0.0
    rare = ranker.query_vector("rare")
    assert rare.weights[0].idf_factor == pytest.approx(math.log10(2))
    assert rare.weights[0].weight == pytest.approx(1.0)
    assert ranker.cosine("rare", "d1") == pytest.approx(1.0 / length)
    assert ranker.cosine("rare", "d2") == 0.0
    assert ranker.rank("common", 2) == ()


def test_repeated_query_term_changes_log_tf_not_the_direction() -> None:
    ranker = _ranker()
    once = ranker.query_vector("heart")
    twice = ranker.query_vector("heart heart")

    assert twice.weights[0].tf == 2
    assert twice.weights[0].log_tf == pytest.approx(1.0 + math.log10(2))
    assert twice.weights[0].raw == pytest.approx((1.0 + math.log10(2)) * math.log10(3 / 2))
    assert twice.norm > once.norm
    assert twice.weights[0].weight == pytest.approx(1.0)
    assert ranker.cosine("heart heart", "d1") == pytest.approx(ranker.cosine("heart", "d1"))


def test_unknown_query_term_has_idf_zero_and_drops_out() -> None:
    ranker = _ranker()
    vector = ranker.query_vector("heart missing")
    weights = {item.term: item for item in vector.weights}

    assert weights["missing"].tf == 1
    assert weights["missing"].log_tf == 1.0
    assert weights["missing"].idf_factor == 0.0
    assert weights["missing"].raw == 0.0
    assert weights["missing"].weight == 0.0
    assert not vector.is_zero
    assert ranker.cosine("heart missing", "d1") == pytest.approx(ranker.cosine("heart", "d1"))


def test_zero_vectors_score_zero() -> None:
    index = InvertedIndex.from_documents({"d1": "heart scheme", "empty": "..."})
    ranker = TfidfRanker(index)

    assert ranker.query_vector("").is_zero
    assert ranker.query_vector("...").is_zero
    assert ranker.query_vector("not-in-the-index").is_zero
    assert ranker.query_vector("not-in-the-index").norm == 0.0
    empty = ranker.document_vector("empty")
    assert empty.is_zero
    assert empty.weights == ()
    assert empty.norm == 0.0
    assert ranker.cosine("", "d1") == 0.0
    assert ranker.cosine("heart", "empty") == 0.0
    assert ranker.cosine("missing", "d1") == 0.0
    assert ranker.rank("missing", 5) == ()
    assert [item.doc_id for item in ranker.rank("heart", 5)] == ["d1"]

    trace = ranker.format_scores("missing", k=2)
    assert "zero_vector=True" in trace
    assert trace.endswith("ranked k=2")


def test_empty_index_query_is_a_zero_vector() -> None:
    ranker = TfidfRanker(InvertedIndex())
    vector = ranker.query_vector("heart")

    assert vector.is_zero
    assert vector.weights[0].term == "heart"
    assert vector.weights[0].idf_factor == 0.0
    assert vector.weights[0].weight == 0.0
    assert ranker.scores("heart") == ()
    assert ranker.rank("heart", 3) == ()
    assert repr(ranker) == "TfidfRanker(documents=0)"


def test_query_uses_the_index_normaliser() -> None:
    # d2 keeps heart's document frequency below N, so the query idf is
    # not log10(1). Indexed tokens of d1 are heart, scheme. The same
    # stop-word list drops "the" from the query.
    index = InvertedIndex.from_documents(
        {"d1": "The HEART scheme.", "d2": "cover"},
        remove_stopwords=True,
    )
    ranker = TfidfRanker(index)
    query = ranker.query_vector("The HEART.")
    document = ranker.document_vector("d1")

    assert [item.term for item in query.weights] == ["heart"]
    assert query.weights[0].weight == pytest.approx(1.0)
    assert [item.term for item in document.weights] == ["heart", "scheme"]
    assert ranker.cosine("The HEART.", "d1") == pytest.approx(1.0 / math.sqrt(2.0))


def test_devanagari_query_uses_indexed_tokens() -> None:
    index = InvertedIndex.from_documents({"d1": "योजना covers", "d2": "covers"})
    ranker = TfidfRanker(index)

    assert ranker.cosine("योजना", "d2") == 0.0
    assert ranker.cosine("योजना", "d1") > 0.0
    assert [item.doc_id for item in ranker.rank("योजना", 1)] == ["d1"]


def test_equal_scores_rank_by_doc_id() -> None:
    # b is indexed first. a and b are each the single term red, so both
    # cosines are 1. c keeps df(red) below N; if red were in every
    # document, idf would be 0 and the query vector would be zero.
    # The smaller id still ranks first.
    ranker = TfidfRanker(
        InvertedIndex.from_documents({"b": "red", "a": "red", "c": "blue"})
    )
    ranked = ranker.rank("red", k=2)

    assert [item.doc_id for item in ranked] == ["a", "b"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx(ranked[0].score)
    assert ranker.cosine("red", "c") == 0.0


def test_format_scores_lists_weights_and_the_ranked_hit() -> None:
    # Two documents, query "heart". df(heart) = 1, N = 2, so
    # idf = log10(2) ≈ 0.301030. One query term normalizes to weight 1.
    # d1's only term is heart, so its lnc weight is 1 and the cosine is 1.
    # d2 does not contain heart, so its contribution is 0.
    ranker = TfidfRanker(InvertedIndex.from_documents({"d1": "heart", "d2": "other"}))
    idf = f"{math.log10(2):.6f}"
    assert idf == "0.301030"

    expected = "\n".join(
        [
            "weighting  document=lnc  query=ltc  log_tf=1+log10(tf)  idf=log10(N/df)",
            f"query 'heart'  norm={idf}  zero_vector=False",
            f"  heart  tf=1  log_tf=1.000000  idf={idf}  raw={idf}  weight=1.000000",
            "document d1  norm=1.000000  zero_vector=False  cosine=1.000000",
            "  heart  tf=1  log_tf=1.000000  idf_factor=1.000000  "
            "raw=1.000000  weight=1.000000  contrib=1.000000",
            "document d2  norm=1.000000  zero_vector=False  cosine=0.000000",
            "  other  tf=1  log_tf=1.000000  idf_factor=1.000000  "
            "raw=1.000000  weight=1.000000  contrib=0.000000",
            "ranked k=1",
            "  1  d1  1.000000",
        ]
    )
    assert ranker.format_scores("heart", k=1) == expected


def test_rejects_bad_inputs() -> None:
    ranker = TfidfRanker(InvertedIndex.from_documents({"d1": "heart"}))

    with pytest.raises(TypeError):
        TfidfRanker([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ranker.query_vector(None)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        ranker.document_vector("missing")
    with pytest.raises(KeyError):
        ranker.cosine("heart", "missing")
    with pytest.raises(TypeError):
        ranker.rank("heart", 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ranker.rank("heart", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ranker.rank("heart", -1)
    with pytest.raises(ValueError):
        ranker.format_scores("heart", k=-1)
