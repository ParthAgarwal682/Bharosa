"""Known-answer tests for the inverted index.

The corpus is three hand-written documents. Expected postings, term
frequencies, and document frequencies are written out below, not computed
by another index.
"""

from __future__ import annotations

import math

import pytest

from bharosa.index.inverted import (
    CollectionStats,
    DocumentStats,
    InvertedIndex,
    Posting,
)

# d1 tokens: the, heart, scheme, covers, heart
# d2 tokens: heart, scheme
# d3 tokens: योजना, covers, covers
DOCS = {
    "d1": "the heart scheme covers heart",
    "d2": "Heart scheme.",
    "d3": "योजना covers covers",
}


def _index() -> InvertedIndex:
    return InvertedIndex.from_documents(DOCS)


def test_vocabulary_and_hand_counted_postings() -> None:
    index = _index()

    assert index.vocabulary() == ("covers", "heart", "scheme", "the", "योजना")
    assert index.dictionary()["heart"] == (
        Posting("d1", 2),
        Posting("d2", 1),
    )
    assert index.postings("heart") == (Posting("d1", 2), Posting("d2", 1))
    assert index.postings("covers") == (Posting("d1", 1), Posting("d3", 2))
    assert index.postings("scheme") == (Posting("d1", 1), Posting("d2", 1))
    assert index.postings("the") == (Posting("d1", 1),)
    assert index.postings("योजना") == (Posting("d3", 1),)

    assert index.document_frequency("heart") == 2
    assert index.document_frequency("covers") == 2
    assert index.document_frequency("scheme") == 2
    assert index.document_frequency("the") == 1
    assert index.document_frequency("योजना") == 1

    # Repeats inside d1 add to cf, not to df.
    assert index.collection_frequency("heart") == 3
    assert index.collection_frequency("covers") == 3
    assert index.collection_frequency("scheme") == 2
    assert index.collection_frequency("the") == 1


def test_term_frequency_is_the_raw_count() -> None:
    index = _index()

    assert index.term_frequency("heart", "d1") == 2
    assert index.term_frequency("heart", "d2") == 1
    assert index.term_frequency("heart", "d3") == 0
    assert index.term_frequency("covers", "d3") == 2
    assert index.term_frequency("scheme", "d1") == 1
    assert index.term_frequency("the", "d2") == 0


def test_lookup_uses_the_shared_normaliser() -> None:
    index = _index()

    assert index.postings("HEART") == index.postings("heart")
    assert index.postings("scheme.") == (Posting("d1", 1), Posting("d2", 1))
    assert index.document_frequency("  Covers ") == 2
    assert index.term_frequency("HEART", "d1") == 2
    assert index.postings("योजना") == (Posting("d3", 1),)


def test_punctuation_splits_into_separate_terms() -> None:
    index = InvertedIndex.from_documents({"z1": "Ayushman Bharat."})

    assert index.vocabulary() == ("ayushman", "bharat")
    assert index.postings("ayushman") == (Posting("z1", 1),)
    assert index.postings("bharat") == (Posting("z1", 1),)
    assert index.postings("Ayushman") == (Posting("z1", 1),)


def test_postings_follow_document_addition_order() -> None:
    index = InvertedIndex()
    index.add_document("z", "scheme")
    index.add_document("a", "scheme scheme")

    assert index.document_ids() == ("z", "a")
    assert index.postings("scheme") == (Posting("z", 1), Posting("a", 2))
    assert index.document_frequency("scheme") == 2
    assert index.collection_frequency("scheme") == 3


def test_document_and_collection_statistics() -> None:
    index = _index()

    assert index.document_ids() == ("d1", "d2", "d3")
    assert index.document_stats("d1") == DocumentStats("d1", length=5, unique_terms=4)
    assert index.document_stats("d2") == DocumentStats("d2", length=2, unique_terms=2)
    assert index.document_stats("d3") == DocumentStats("d3", length=3, unique_terms=2)
    assert index.collection_stats() == CollectionStats(
        num_documents=3,
        vocabulary_size=5,
        total_tokens=10,
        average_document_length=10 / 3,
    )
    # idf is a dictionary statistic: log10(N / df). Not a document score.
    assert index.idf("heart") == pytest.approx(math.log10(3 / 2))
    assert index.idf("the") == pytest.approx(math.log10(3))
    assert index.idf("HEART") == index.idf("heart")
    assert index.idf("missing") == 0.0


def test_empty_document_and_unknown_term() -> None:
    index = _index()
    index.add_document("d4", "   ...   ")

    assert index.document_ids() == ("d1", "d2", "d3", "d4")
    assert index.document_stats("d4") == DocumentStats("d4", length=0, unique_terms=0)
    assert index.vocabulary() == ("covers", "heart", "scheme", "the", "योजना")
    assert index.postings("absent") == ()
    assert index.document_frequency("absent") == 0
    assert index.collection_frequency("absent") == 0
    assert index.term_frequency("absent", "d1") == 0
    stats = index.collection_stats()
    assert stats.num_documents == 4
    assert stats.total_tokens == 10
    assert stats.average_document_length == 2.5


def test_empty_index_statistics() -> None:
    index = InvertedIndex()

    assert index.document_ids() == ()
    assert index.vocabulary() == ()
    assert index.dictionary() == {}
    assert index.postings("heart") == ()
    assert index.idf("heart") == 0.0
    assert index.format_postings() == ""
    assert index.collection_stats() == CollectionStats(
        num_documents=0,
        vocabulary_size=0,
        total_tokens=0,
        average_document_length=0.0,
    )


def test_format_postings_is_inspectable() -> None:
    index = _index()

    assert index.format_postings(["HEART", "covers"]) == "\n".join(
        [
            "heart  df=2  cf=3",
            "  d1  tf=2",
            "  d2  tf=1",
            "covers  df=2  cf=3",
            "  d1  tf=1",
            "  d3  tf=2",
        ]
    )
    assert "योजना  df=1  cf=1" in index.format_postings()
    assert "  d3  tf=1" in index.format_postings()
    assert "InvertedIndex(documents=3, vocabulary=5)" == repr(index)
    with pytest.raises(TypeError):
        index.format_postings("heart")  # type: ignore[arg-type]


def test_stopwords_stay_unless_the_index_opts_in() -> None:
    text = {"s1": "the scheme for papa"}
    kept = InvertedIndex.from_documents(text)
    assert kept.vocabulary() == ("for", "papa", "scheme", "the")
    assert kept.postings("the") == (Posting("s1", 1),)
    assert kept.document_frequency("for") == 1

    dropped = InvertedIndex.from_documents(text, remove_stopwords=True)
    assert dropped.vocabulary() == ("papa", "scheme")
    assert dropped.postings("the") == ()
    assert dropped.document_frequency("the") == 0
    assert dropped.document_stats("s1") == DocumentStats("s1", length=2, unique_terms=2)

    custom = InvertedIndex.from_documents(
        text,
        remove_stopwords=True,
        stopwords=frozenset({"papa"}),
    )
    assert custom.vocabulary() == ("for", "scheme", "the")
    assert custom.postings("papa") == ()
    assert custom.term_frequency("scheme", "s1") == 1


def test_hinglish_tokens_are_indexed() -> None:
    index = InvertedIndex.from_documents(
        {"h1": "Papa के Heart Operation, UP"}
    )

    assert index.vocabulary() == ("heart", "operation", "papa", "up", "के")
    assert index.postings("के") == (Posting("h1", 1),)
    assert index.postings("UP") == (Posting("h1", 1),)
    assert index.document_stats("h1") == DocumentStats("h1", length=5, unique_terms=5)


def test_rejects_bad_inputs() -> None:
    index = _index()

    with pytest.raises(ValueError):
        index.add_document("d1", "again")
    assert index.collection_stats().num_documents == 3

    with pytest.raises(ValueError):
        index.add_document("", "scheme")
    with pytest.raises(TypeError):
        index.add_document(1, "scheme")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.add_document("d9", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.postings(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        index.postings("heart scheme")
    with pytest.raises(KeyError):
        index.term_frequency("heart", "missing")
    with pytest.raises(KeyError):
        index.document_stats("missing")
    with pytest.raises(TypeError):
        InvertedIndex.from_documents(["not", "a", "mapping"])  # type: ignore[arg-type]
