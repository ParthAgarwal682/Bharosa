"""Known-answer tests for the positional index.

Positions below are 0-based token indexes counted by hand from the
shared tokeniser. Phrase hits are the starts of exact consecutive runs.
"""

from __future__ import annotations

import pytest

from bharosa.index.positional import PhraseMatch, PositionalIndex, PositionalPosting
from bharosa.text.normalize import tokenize

# d1 tokens: the, heart, scheme, covers, heart
# d2 tokens: heart, scheme
DOCS = {
    "d1": "the heart scheme covers heart",
    "d2": "Heart scheme.",
}


def test_dictionary_is_term_then_document_then_positions() -> None:
    index = PositionalIndex.from_documents(DOCS)
    tokens = tokenize(DOCS["d1"])

    assert tokens == ["the", "heart", "scheme", "covers", "heart"]
    assert index.vocabulary() == ("covers", "heart", "scheme", "the")
    assert index.dictionary() == {
        "covers": {"d1": (3,)},
        "heart": {"d1": (1, 4), "d2": (0,)},
        "scheme": {"d1": (2,), "d2": (1,)},
        "the": {"d1": (0,)},
    }
    assert index.positions("HEART") == (
        PositionalPosting("d1", (1, 4)),
        PositionalPosting("d2", (0,)),
    )
    assert index.positions("covers") == (PositionalPosting("d1", (3,)),)
    assert index.positions("the") == (PositionalPosting("d1", (0,)),)
    # The first token is position 0, including after leading space.
    spaced = PositionalIndex.from_documents({"s": "  heart"})
    assert spaced.positions("heart") == (PositionalPosting("s", (0,)),)
    assert repr(index) == "PositionalIndex(documents=2, vocabulary=4)"


def test_exact_phrase_requires_consecutive_positions_in_order() -> None:
    index = PositionalIndex.from_documents(DOCS)

    assert index.find_phrase("heart scheme") == (
        PhraseMatch("d1", (1,)),
        PhraseMatch("d2", (0,)),
    )
    assert index.find_phrase("scheme covers") == (PhraseMatch("d1", (2,)),)
    assert index.find_phrase("covers heart") == (PhraseMatch("d1", (3,)),)
    assert index.find_phrase("heart covers") == ()
    assert index.find_phrase("scheme heart") == ()
    assert index.find_phrase("HEART scheme.") == index.find_phrase("heart scheme")
    assert index.find_phrase("heart") == (
        PhraseMatch("d1", (1, 4)),
        PhraseMatch("d2", (0,)),
    )

    repeated = PositionalIndex.from_documents({"d1": "heart scheme heart heart"})
    assert repeated.dictionary()["heart"] == {"d1": (0, 2, 3)}
    assert repeated.find_phrase("heart heart") == (PhraseMatch("d1", (2,)),)

    gapped = PositionalIndex.from_documents({"d1": "ayushman extra bharat"})
    assert gapped.dictionary()["bharat"] == {"d1": (2,)}
    assert gapped.find_phrase("ayushman bharat") == ()
    assert gapped.find_phrase("bharat ayushman") == ()

    names = PositionalIndex.from_documents(
        {
            "z": "prefix Ayushman Bharat scheme",
            "a": "Ayushman, Bharat.",
        }
    )
    assert names.find_phrase("Ayushman Bharat") == (
        PhraseMatch("z", (1,)),
        PhraseMatch("a", (0,)),
    )
    assert names.find_phrase("Ayushman Bharat scheme") == (PhraseMatch("z", (1,)),)


def test_phrase_dump_shows_tokens_and_starts() -> None:
    index = PositionalIndex.from_documents({"d1": "Ayushman Bharat scheme"})

    assert index.format_postings(["Bharat"]) == "\n".join(["bharat", "  d1  1"])
    assert index.format_phrase("Ayushman Bharat") == "\n".join(
        [
            "phrase 'Ayushman Bharat'  tokens=ayushman bharat",
            "  d1  starts=0",
        ]
    )
    assert index.format_phrase("...") == "phrase '...'  tokens=<empty>"
    assert index.find_phrase("...") == ()
    assert index.find_phrase("missing term") == ()


def test_stopwords_change_the_token_positions_only_when_removed() -> None:
    text = {"d1": "ayushman the bharat"}
    kept = PositionalIndex.from_documents(text)
    assert kept.dictionary() == {
        "ayushman": {"d1": (0,)},
        "bharat": {"d1": (2,)},
        "the": {"d1": (1,)},
    }
    assert kept.find_phrase("ayushman bharat") == ()
    assert kept.find_phrase("ayushman the bharat") == (PhraseMatch("d1", (0,)),)

    dropped = PositionalIndex.from_documents(text, remove_stopwords=True)
    assert dropped.dictionary() == {
        "ayushman": {"d1": (0,)},
        "bharat": {"d1": (1,)},
    }
    assert dropped.find_phrase("ayushman bharat") == (PhraseMatch("d1", (0,)),)
    assert dropped.positions("the") == ()


def test_devanagari_phrase_uses_the_same_tokens() -> None:
    index = PositionalIndex.from_documents({"d1": "योजना covers योजना"})

    assert index.positions("योजना") == (PositionalPosting("d1", (0, 2)),)
    assert index.find_phrase("योजना covers") == (PhraseMatch("d1", (0,)),)
    assert index.find_phrase("covers योजना") == (PhraseMatch("d1", (1,)),)


def test_empty_document_and_addition_order() -> None:
    index = PositionalIndex()
    index.add_document("z", "scheme scheme")
    index.add_document("a", "...")
    index.add_document("m", "scheme")

    assert index.document_ids() == ("z", "a", "m")
    assert index.positions("scheme") == (
        PositionalPosting("z", (0, 1)),
        PositionalPosting("m", (0,)),
    )
    assert index.find_phrase("scheme scheme") == (PhraseMatch("z", (0,)),)
    assert index.vocabulary() == ("scheme",)


def test_rejects_bad_inputs() -> None:
    index = PositionalIndex.from_documents(DOCS)

    with pytest.raises(ValueError, match="already indexed"):
        index.add_document("d1", "scheme")
    assert index.document_ids() == ("d1", "d2")
    with pytest.raises(ValueError, match="non-empty"):
        index.add_document("", "scheme")
    with pytest.raises(TypeError):
        index.add_document(1, "scheme")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.add_document("d9", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.positions(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one token"):
        index.positions("heart scheme")
    assert index.positions("...") == ()
    with pytest.raises(TypeError):
        index.format_postings("heart")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PositionalIndex.from_documents(["not", "a", "mapping"])  # type: ignore[arg-type]
