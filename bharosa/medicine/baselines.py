"""Baseline medicine matching algorithms: Exact Brand and Phonetic/Soundex.

IR concept: Baseline retrieval models for brand name matching.
In benchmark evaluations, advanced models (character n-grams, BM25) are compared
against two essential syllabus baselines:
1. Exact Match: Boolean lookup over normalized brand tokens (precision-oriented).
2. Phonetic Match (Soundex): Equivalence class partitioning over pronunciation
   codes (recall-oriented baseline for misspellings).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bharosa.medicine.loader import MedicineRecord
from bharosa.medicine.phonetic import soundex_brand
from bharosa.text.normalize import normalize as normalize_text


@dataclass(frozen=True)
class BaselineHit:
    """A deterministic match hit from a baseline matcher."""

    record: MedicineRecord
    query: str
    match_type: str  # "exact" or "phonetic"


def match_exact(
    query: str,
    records: Sequence[MedicineRecord],
) -> list[MedicineRecord]:
    """Return records where normalized brand matches the query.

    Matches if:
    - Entire normalized brand matches normalized query (e.g. 'dolo 650' == 'dolo 650'), OR
    - Query matches the primary brand token (e.g. 'calpol' matches 'Calpol 650 Tablet').

    Edge cases handled:
    - Empty or whitespace query returns empty list [].
    - Missing/empty record brands are safely skipped.
    - Duplicate identical records in the corpus are deduplicated in output.
    - Combination medicines remain intact with is_combination=True.
    - No sorting by price or parametric filtering is performed.
    """
    if not query or not isinstance(query, str):
        return []

    norm_query = normalize_text(query)
    if not norm_query:
        return []

    query_tokens = norm_query.split()
    results: list[MedicineRecord] = []
    seen_records: set[MedicineRecord] = set()

    for rec in records:
        if not rec.brand:
            continue

        norm_brand = normalize_text(rec.brand)
        if not norm_brand:
            continue

        brand_tokens = norm_brand.split()

        # Check full string equality or primary brand token match
        is_match = False
        if norm_query == norm_brand:
            is_match = True
        elif len(query_tokens) == 1 and brand_tokens and query_tokens[0] == brand_tokens[0]:
            is_match = True

        if is_match and rec not in seen_records:
            seen_records.add(rec)
            results.append(rec)

    return results


def match_phonetic(
    query: str,
    records: Sequence[MedicineRecord],
) -> list[MedicineRecord]:
    """Return records where Soundex phonetic representation matches the query.

    Compares the Soundex code of the query's primary brand token against the
    Soundex code of the record's primary brand token.

    Edge cases handled:
    - Empty or whitespace query returns empty list [].
    - Missing/empty record brands are safely skipped.
    - Duplicate identical records in the corpus are deduplicated in output.
    - Combination medicines remain intact with is_combination=True.
    - No sorting by price or parametric filtering is performed.
    """
    if not query or not isinstance(query, str):
        return []

    query_soundex = soundex_brand(query)
    if not query_soundex:
        return []

    results: list[MedicineRecord] = []
    seen_records: set[MedicineRecord] = set()

    for rec in records:
        if not rec.brand:
            continue

        brand_soundex = soundex_brand(rec.brand)
        if not brand_soundex:
            continue

        if query_soundex == brand_soundex and rec not in seen_records:
            seen_records.add(rec)
            results.append(rec)

    return results


class ExactBrandMatcher:
    """Class wrapper for exact normalized brand matching."""

    def __init__(self, records: Sequence[MedicineRecord]) -> None:
        self._records = list(records)

    def match(self, query: str) -> list[MedicineRecord]:
        """Perform exact normalized brand matching."""
        return match_exact(query, self._records)


class PhoneticBrandMatcher:
    """Class wrapper for Soundex phonetic brand matching."""

    def __init__(self, records: Sequence[MedicineRecord]) -> None:
        self._records = list(records)

    def match(self, query: str) -> list[MedicineRecord]:
        """Perform Soundex phonetic brand matching."""
        return match_phonetic(query, self._records)
