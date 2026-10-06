"""Word shingles and Jaccard similarity module for Bharosa crawler.

Provides word k-shingle extraction (default k=5), Jaccard set similarity calculation,
and configurable threshold-based near-duplicate detection.

Edge-Case Behavior:
1. Short Text (words < k): For texts with 1 to k-1 words, the module creates a single
   fallback shingle tuple containing all available words. This prevents very short texts
   from returning an empty shingle set while allowing direct token comparison.
2. Empty Text (words == 0): Returns an empty set set(). If both shingle sets are empty,
   jaccard_similarity returns 1.0. If one set is empty and the other non-empty,
   jaccard_similarity returns 0.0.
3. Configurable Threshold: The threshold parameter in is_near_duplicate is fully
   configurable (default 0.8). No single threshold is globally optimal; application
   callers should tune threshold based on precision/recall requirements for clone detection.
"""

from __future__ import annotations

import re

DEFAULT_SHINGLE_SIZE = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.8


def tokenize_text(text: str) -> list[str]:
    """Basic lowercased word tokenization stripping punctuation and whitespace."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


def get_word_shingles(
    text: str, k: int = DEFAULT_SHINGLE_SIZE
) -> set[tuple[str, ...]]:
    """Extract a set of contiguous word k-shingles (tuples of length k) from text.

    Args:
        text: Input string content.
        k: Shingle length in words (default 5).

    Returns:
        Set of word tuple shingles.

    Edge-Case Behavior:
        - If text is empty (0 words): returns set().
        - If text has 1 to k-1 words (short text): returns a set with a single
          tuple containing all available words (e.g., ('ayushman', 'bharat')).
    """
    tokens = tokenize_text(text)
    n = len(tokens)

    if n == 0:
        return set()

    if n < k:
        # Fallback for short texts with fewer than k words
        return {tuple(tokens)}

    shingles: set[tuple[str, ...]] = set()
    for i in range(n - k + 1):
        shingle = tuple(tokens[i : i + k])
        shingles.add(shingle)

    return shingles


def jaccard_similarity(
    set_a: set[tuple[str, ...]], set_b: set[tuple[str, ...]]
) -> float:
    """Compute the Jaccard similarity score between two shingle sets.

    Formula: |A ∩ B| / |A ∪ B|

    Edge-Case Behavior:
        - If both sets are empty: returns 1.0 (identical empty sets).
        - If one set is empty and the other non-empty: returns 0.0.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    return len(intersection) / len(union)


def is_near_duplicate(
    text1: str,
    text2: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    k: int = DEFAULT_SHINGLE_SIZE,
) -> bool:
    """Check if two texts are near-duplicates using Jaccard similarity over word k-shingles.

    Args:
        text1: First text sample.
        text2: Second text sample.
        threshold: Minimum Jaccard similarity score to consider near-duplicate (0.0 to 1.0).
                   Configurable per application needs; no fixed threshold is universally optimal.
        k: Shingle size in words (default 5).

    Returns:
        True if Jaccard similarity >= threshold, else False.
    """
    shingles1 = get_word_shingles(text1, k=k)
    shingles2 = get_word_shingles(text2, k=k)

    score = jaccard_similarity(shingles1, shingles2)
    return score >= threshold
