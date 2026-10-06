"""Unit tests for the Bharosa word shingles and Jaccard similarity module.

Tests identical texts, near-duplicate texts, clearly different texts, short text fallback,
and configurable threshold parameters.
"""

from __future__ import annotations

import pytest

from bharosa.crawl.shingles import (
    get_word_shingles,
    is_near_duplicate,
    jaccard_similarity,
    tokenize_text,
)


def test_tokenize_text() -> None:
    """Test tokenization lowercases and strips punctuation."""
    text = "Ayushman Bharat PM-JAY Scheme, 2026!"
    tokens = tokenize_text(text)
    assert tokens == ["ayushman", "bharat", "pm", "jay", "scheme", "2026"]


def test_shingle_generation_5_words() -> None:
    """Test word 5-shingles generation for standard length text."""
    text = "one two three four five six seven"
    shingles = get_word_shingles(text, k=5)
    assert len(shingles) == 3
    assert ("one", "two", "three", "four", "five") in shingles
    assert ("three", "four", "five", "six", "seven") in shingles


def test_identical_texts() -> None:
    """Test identical long texts have Jaccard similarity of 1.0 and return True for near-duplicate."""
    text1 = (
        "Ayushman Bharat National Health Protection Scheme provides coverage up to five lakh rupees "
        "per family per year for secondary and tertiary care hospitalization across public and empanelled private hospitals."
    )
    text2 = (
        "Ayushman Bharat National Health Protection Scheme provides coverage up to five lakh rupees "
        "per family per year for secondary and tertiary care hospitalization across public and empanelled private hospitals."
    )

    s1 = get_word_shingles(text1)
    s2 = get_word_shingles(text2)

    similarity = jaccard_similarity(s1, s2)
    assert similarity == 1.0
    assert is_near_duplicate(text1, text2, threshold=0.8) is True


def test_near_duplicate_texts() -> None:
    """Test near-duplicate texts (minor word replacements/additions) have high Jaccard similarity."""
    text_original = (
        "Ayushman Bharat National Health Protection Scheme provides coverage up to five lakh rupees "
        "per family per year for secondary and tertiary care hospitalization across all public hospitals."
    )
    text_cloned = (
        "Ayushman Bharat National Health Protection Scheme provides coverage up to five lakh rupees "
        "per family per year for secondary and tertiary care hospitalization across public and private hospitals."
    )

    s1 = get_word_shingles(text_original, k=5)
    s2 = get_word_shingles(text_cloned, k=5)

    similarity = jaccard_similarity(s1, s2)
    assert 0.65 < similarity < 1.0

    # With threshold 0.6, it is marked as near-duplicate
    assert is_near_duplicate(text_original, text_cloned, threshold=0.6) is True
    # With strict threshold 0.99, it is marked as non-duplicate
    assert is_near_duplicate(text_original, text_cloned, threshold=0.99) is False


def test_clearly_different_texts() -> None:
    """Test completely different texts have 0.0 Jaccard similarity."""
    text_scheme = (
        "Government health insurance eligibility depends on annual household income and ration card status."
    )
    text_medicine = (
        "Pantocid 40 mg gastro-resistant tablet active ingredient pantoprazole sodium treats stomach acid."
    )

    s1 = get_word_shingles(text_scheme, k=5)
    s2 = get_word_shingles(text_medicine, k=5)

    similarity = jaccard_similarity(s1, s2)
    assert similarity == 0.0
    assert is_near_duplicate(text_scheme, text_medicine, threshold=0.5) is False


def test_very_short_text_fallback() -> None:
    """Test short text (< 5 words) creates fallback single tuple without crashing."""
    short1 = "Ayushman Bharat"
    short2 = "Ayushman Bharat"
    short3 = "Pantocid Tablet"

    s1 = get_word_shingles(short1, k=5)
    s2 = get_word_shingles(short2, k=5)
    s3 = get_word_shingles(short3, k=5)

    assert s1 == {("ayushman", "bharat")}
    assert s2 == {("ayushman", "bharat")}
    assert s3 == {("pantocid", "tablet")}

    assert jaccard_similarity(s1, s2) == 1.0
    assert jaccard_similarity(s1, s3) == 0.0


def test_empty_text_edge_cases() -> None:
    """Test empty text edge cases."""
    empty1 = ""
    empty2 = "   "
    text = "Valid scheme text content here"

    s_empty1 = get_word_shingles(empty1, k=5)
    s_empty2 = get_word_shingles(empty2, k=5)
    s_text = get_word_shingles(text, k=5)

    assert s_empty1 == set()
    assert s_empty2 == set()

    # Both empty -> 1.0
    assert jaccard_similarity(s_empty1, s_empty2) == 1.0
    # One empty, one valid -> 0.0
    assert jaccard_similarity(s_empty1, s_text) == 0.0
    assert is_near_duplicate(empty1, text, threshold=0.1) is False


def test_configurable_threshold() -> None:
    """Test near duplicate detection respects custom threshold parameter."""
    t1 = (
        "Ayushman Bharat National Health Protection Scheme provides financial coverage "
        "for hospital expenses across all empanelled public hospitals in the state."
    )
    t2 = (
        "Ayushman Bharat National Health Protection Scheme provides financial coverage "
        "for hospital expenses across all empanelled government hospitals in the state."
    )

    # Similarity is approx 0.65
    assert is_near_duplicate(t1, t2, threshold=0.5) is True
    assert is_near_duplicate(t1, t2, threshold=0.85) is False
