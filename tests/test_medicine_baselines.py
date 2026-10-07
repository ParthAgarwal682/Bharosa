"""Unit tests for exact and Soundex baseline medicine matchers.

IR concept: Baseline retrieval tests using tiny synthetic fixtures with
known ground-truth answers. Verifies that exact matching enforces precision,
Soundex matches phonetic equivalents, and combination records remain intact.
"""

from __future__ import annotations

import pytest

from bharosa.medicine.baselines import (
    ExactBrandMatcher,
    PhoneticBrandMatcher,
    match_exact,
    match_phonetic,
)
from bharosa.medicine.loader import MedicineRecord
from bharosa.medicine.phonetic import soundex, soundex_brand, soundex_brand_tokens


def make_record(
    brand: str,
    salt: str = "Paracetamol",
    strength_value: float | None = 650.0,
    strength_unit: str | None = "mg",
    form: str = "tablet",
    mrp: float = 30.0,
    generic_price: float | None = None,
    manufacturer: str = "Test Labs",
    is_combination: bool = False,
) -> MedicineRecord:
    """Helper to instantiate test synthetic MedicineRecord fixtures."""
    return MedicineRecord(
        brand=brand,
        salt=salt,
        strength_value=strength_value,
        strength_unit=strength_unit,
        form=form,
        mrp=mrp,
        generic_price=generic_price,
        manufacturer=manufacturer,
        is_combination=is_combination,
    )


def test_soundex_algorithm_known_cases() -> None:
    """Test standard Soundex encoding rules against textbook cases."""
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"
    assert soundex("Ashcraft") == "A261"
    assert soundex("Tymczak") == "T522"

    # Indian brand variants: Pantocid vs Pantosid vs Pantocide
    assert soundex("Pantocid") == "P532"
    assert soundex("Pantosid") == "P532"
    assert soundex("Pantocide") == "P532"

    assert soundex("Dolo") == "D400"
    assert soundex("Calpol") == "C414"


def test_soundex_edge_cases() -> None:
    """Test empty, whitespace, and non-alphabetic inputs."""
    assert soundex("") == ""
    assert soundex("   ") == ""
    assert soundex("123") == ""
    assert soundex("!@#$%") == ""


def test_soundex_brand_extraction() -> None:
    """Test extracting and encoding leading brand tokens from composite names."""
    assert soundex_brand("Pantocid 40 Tablet") == "P532"
    assert soundex_brand("  Calpol 650 mg  ") == "C414"
    assert soundex_brand("Augmentin 625 Duo") == "A255"
    assert soundex_brand("") == ""
    assert soundex_brand("123 456") == ""
    assert soundex_brand_tokens("Pantocid Tablet") == ("P532", "T143")


def test_exact_normalized_brand_match() -> None:
    """Test that exact matcher finds records with identical brand tokens."""
    rec1 = make_record(brand="Dolo 650", salt="Paracetamol")
    rec2 = make_record(brand="Calpol 500", salt="Paracetamol")
    records = [rec1, rec2]

    # Exact full match
    matches = match_exact("Dolo 650", records)
    assert len(matches) == 1
    assert matches[0] == rec1

    # Exact primary token match
    matches_token = match_exact("Dolo", records)
    assert len(matches_token) == 1
    assert matches_token[0] == rec1


def test_case_and_whitespace_normalization() -> None:
    """Exact matching must be robust to casing, excess spaces, and punctuation."""
    rec = make_record(brand="Calpol 650 Tablet", salt="Paracetamol")
    records = [rec]

    assert match_exact("calpol 650 tablet", records) == [rec]
    assert match_exact("   CALPOL   650   TABLET   ", records) == [rec]
    assert match_exact("calpol-650 tablet", records) == [rec]
    assert match_exact("calpol", records) == [rec]


def test_soundex_phonetic_matching() -> None:
    """Soundex matches misspelled variants that fail exact matching."""
    rec = make_record(brand="Pantocid 40 Tablet", salt="Pantoprazole", strength_value=40.0)
    records = [rec]

    # Exact matcher fails on spelling mistake
    assert match_exact("Pantosid", records) == []
    assert match_exact("Pantocide", records) == []

    # Phonetic matcher succeeds on spelling variants (both map to P532)
    matches_variant1 = match_phonetic("Pantosid", records)
    assert len(matches_variant1) == 1
    assert matches_variant1[0] == rec

    matches_variant2 = match_phonetic("Pantocide", records)
    assert len(matches_variant2) == 1
    assert matches_variant2[0] == rec


def test_no_match_behavior() -> None:
    """Queries for non-existent brands must return empty lists."""
    records = [
        make_record(brand="Dolo 650", salt="Paracetamol"),
        make_record(brand="Calpol 500", salt="Paracetamol"),
    ]

    assert match_exact("Crocin", records) == []
    assert match_phonetic("Azithral", records) == []


def test_empty_query_behavior() -> None:
    """Empty or blank queries must safely return an empty result list."""
    records = [make_record(brand="Dolo 650")]

    assert match_exact("", records) == []
    assert match_exact("   ", records) == []
    assert match_exact("???", records) == []

    assert match_phonetic("", records) == []
    assert match_phonetic("   ", records) == []
    assert match_phonetic("123", records) == []


def test_missing_or_empty_brand_record_safely_handled() -> None:
    """Records with missing or empty brand names should not cause crashes."""
    records = [
        make_record(brand=""),
        make_record(brand="Dolo 650"),
    ]

    assert match_exact("Dolo 650", records) == [records[1]]
    assert match_phonetic("Dolo", records) == [records[1]]


def test_duplicate_and_distinct_formulation_handling() -> None:
    """Duplicate records in input should not produce duplicate results, while distinct formulations are kept."""
    rec_dolo_500 = make_record(brand="Dolo 500", strength_value=500.0)
    rec_dolo_650 = make_record(brand="Dolo 650", strength_value=650.0)
    # Identical record copy
    rec_dolo_650_copy = make_record(brand="Dolo 650", strength_value=650.0)

    records = [rec_dolo_500, rec_dolo_650, rec_dolo_650_copy]

    matches = match_exact("Dolo", records)
    # Both distinct formulations are returned, but the identical duplicate is deduplicated
    assert len(matches) == 2
    assert rec_dolo_500 in matches
    assert rec_dolo_650 in matches


def test_combination_medicine_remains_distinct() -> None:
    """Combination medicines must remain distinct records and not be converted to single-salt."""
    rec_combo = make_record(
        brand="Augmentin 625 Duo Tablet",
        salt="Amoxycillin + Clavulanic Acid",
        strength_value=None,
        strength_unit=None,
        form="tablet",
        mrp=223.42,
        is_combination=True,
    )
    records = [rec_combo]

    exact_matches = match_exact("Augmentin", records)
    assert len(exact_matches) == 1
    res_exact = exact_matches[0]
    assert res_exact.is_combination is True
    assert res_exact.salt == "Amoxycillin + Clavulanic Acid"
    assert res_exact.strength_value is None
    assert res_exact.strength_unit is None

    phonetic_matches = match_phonetic("Augmentin", records)
    assert len(phonetic_matches) == 1
    res_phonetic = phonetic_matches[0]
    assert res_phonetic.is_combination is True
    assert res_phonetic.salt == "Amoxycillin + Clavulanic Acid"


def test_matcher_class_interfaces() -> None:
    """Test object-oriented wrapper interfaces."""
    records = [
        make_record(brand="Pantocid 40"),
        make_record(brand="Calpol 650"),
    ]

    exact_matcher = ExactBrandMatcher(records)
    phonetic_matcher = PhoneticBrandMatcher(records)

    assert len(exact_matcher.match("Pantocid 40")) == 1
    assert len(phonetic_matcher.match("Pantosid")) == 1
    assert len(exact_matcher.match("UnknownBrand")) == 0
