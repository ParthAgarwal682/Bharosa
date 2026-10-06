"""Known-answer tests for the shared index/query normaliser."""

from __future__ import annotations

import pytest

from bharosa.text.normalize import DEFAULT_STOPWORDS, normalize, tokenize


def test_english_casefold_and_tokens() -> None:
    assert normalize("Pantocid Tablet") == "pantocid tablet"
    assert tokenize("Pantocid Tablet") == ["pantocid", "tablet"]
    assert normalize("AYUSHMAN Bharat") == "ayushman bharat"


def test_casefold_is_not_plain_lower() -> None:
    # U+00DF LATIN SMALL LETTER SHARP S casefolds to "ss".
    assert normalize("Straße") == "strasse"
    assert tokenize("STRASSE") == tokenize("Straße")


def test_hinglish_query_tokens() -> None:
    text = "Pantocid 40 Sasta Kya Milega"
    assert tokenize(text) == ["pantocid", "40", "sasta", "kya", "milega"]
    assert normalize(text) == "pantocid 40 sasta kya milega"


def test_hinglish_with_punctuation() -> None:
    assert tokenize("pantocid-40, sasta kya milega?") == [
        "pantocid",
        "40",
        "sasta",
        "kya",
        "milega",
    ]


def test_devanagari_tokens_are_preserved() -> None:
    text = "आयुष्मान भारत योजना"
    assert normalize(text) == "आयुष्मान भारत योजना"
    assert tokenize(text) == ["आयुष्मान", "भारत", "योजना"]
    assert any(ord(ch) > 127 for token in tokenize(text) for ch in token)


def test_devanagari_nukta_nfc_matches() -> None:
    # न + nukta (decomposed) and ऩ (precomposed) are the same letter after NFC.
    decomposed = "\u0928\u093c"
    composed = "\u0929"
    assert normalize(decomposed) == composed
    assert tokenize(f"दवा {decomposed}") == tokenize(f"दवा {composed}")
    assert tokenize(f"दवा {decomposed}") == ["दवा", composed]


def test_devanagari_vowel_sign_stays_inside_the_token() -> None:
    # क + vowel sign ि. The mark is category M and must not be deleted.
    kis = "\u0915\u093f"
    assert tokenize(kis) == [kis]
    assert "\u093f" in tokenize(kis)[0]


def test_devanagari_danda_is_a_boundary() -> None:
    assert tokenize("दवा। सस्ती!") == ["दवा", "सस्ती"]
    assert normalize("योजना।") == "योजना"


def test_mixed_script_text() -> None:
    text = "Papa के Heart Operation, UP"
    assert tokenize(text) == ["papa", "के", "heart", "operation", "up"]
    assert normalize("दवाई sasti hai") == "दवाई sasti hai"


def test_punctuation_does_not_swallow_neighbours() -> None:
    assert tokenize("pay Rs. 500, to activate!") == ["pay", "rs", "500", "to", "activate"]
    assert tokenize("heart-operation") == ["heart", "operation"]
    assert tokenize("hello...world") == ["hello", "world"]
    assert tokenize("Ayushman Bharat.") == ["ayushman", "bharat"]
    assert tokenize("don't") == ["don", "t"]
    assert tokenize("income 2,00,000") == ["income", "2", "00", "000"]
    assert tokenize("₹500") == ["500"]


def test_repeated_whitespace() -> None:
    text = "sasta   dawai\t\tyojana\n\n  UP"
    assert normalize(text) == "sasta dawai yojana up"
    assert tokenize(text) == ["sasta", "dawai", "yojana", "up"]
    assert normalize("  \n\t  ") == ""
    assert tokenize("  \n\t  ") == []


def test_nbsp_and_zero_width_space() -> None:
    assert normalize("sasta\u00a0dawai") == "sasta dawai"
    assert tokenize("sasta\u200bdawai") == ["sasta", "dawai"]
    # Soft hyphen is a format character, not a token boundary.
    assert normalize("medi\u00adcine") == "medicine"


def test_punctuation_only_and_empty() -> None:
    assert normalize("") == ""
    assert tokenize("") == []
    assert tokenize("... !!!") == []
    assert normalize("...") == ""


def test_stopwords_stay_unless_explicitly_removed() -> None:
    text = "the scheme for papa"
    assert tokenize(text) == ["the", "scheme", "for", "papa"]
    assert tokenize(text, remove_stopwords=False) == ["the", "scheme", "for", "papa"]
    # A supplied list must not apply while the flag is off.
    assert tokenize(
        text,
        remove_stopwords=False,
        stopwords=frozenset({"the", "scheme", "for", "papa"}),
    ) == ["the", "scheme", "for", "papa"]


def test_optional_stopword_removal_preserves_order() -> None:
    assert tokenize("the scheme for papa", remove_stopwords=True) == ["scheme", "papa"]
    assert tokenize(
        "sasta dawai yojana",
        remove_stopwords=True,
        stopwords=frozenset({"sasta"}),
    ) == ["dawai", "yojana"]
    # Custom entries go through the same normaliser, and they replace the default set.
    assert tokenize(
        "The Scheme for papa",
        remove_stopwords=True,
        stopwords=["The", "FOR"],
    ) == ["scheme", "papa"]


def test_default_stopwords_are_already_normalised_english() -> None:
    assert "kya" not in DEFAULT_STOPWORDS
    assert "के" not in DEFAULT_STOPWORDS
    for word in DEFAULT_STOPWORDS:
        assert normalize(word) == word
        assert " " not in word


def test_index_and_query_share_one_pipeline() -> None:
    indexed = tokenize("Ayushman Bharat Yojana.")
    queried = tokenize("  ayushman   bharat\tyojana  ")
    assert indexed == queried == ["ayushman", "bharat", "yojana"]


def test_output_is_deterministic_and_idempotent() -> None:
    text = "  Papa के   HEART-operation!!  "
    assert normalize(text) == normalize(text)
    assert tokenize(text) == tokenize(text)
    once = normalize(text)
    assert normalize(once) == once
    assert tokenize(text) == ["papa", "के", "heart", "operation"]


def test_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        normalize(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenize(b"pantocid")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenize("the scheme", remove_stopwords=True, stopwords=["the", 1])  # type: ignore[list-item]
