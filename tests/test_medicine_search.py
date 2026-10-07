"""Known-answer tests for medicine n-gram search.

The rows in this file are test fixtures. They are not production
medicine records and they are not loader output. Names such as
``testbranda`` and ``fixturesalta`` are labels for the filter, not
products.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bharosa.medicine.ngram import SCHEME, ngram_similarity
from bharosa.medicine.parametric import read_medicine
from bharosa.medicine.search import (
    CANDIDATE_LABEL,
    DISCLAIMER,
    MedicineSearchResult,
    search_medicine,
)


def _row(brand: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "brand": brand,
        "salt": "fixturesalta",
        "strength_value": 40,
        "strength_unit": "mg",
        "form": "tablet",
    }
    row.update(overrides)
    return row


def _assert_disclosure(result: MedicineSearchResult) -> None:
    assert result.label == CANDIDATE_LABEL == "matching search candidates"
    assert "doctor" in result.disclaimer.lower()
    assert "pharmacist" in result.disclaimer.lower()
    assert result.disclaimer == DISCLAIMER
    text = result.format_result().lower()
    assert "substitute" not in text
    assert "safe" not in result.disclaimer.lower()
    assert result.format_result().splitlines()[0] == "matching search candidates"
    assert result.format_result().splitlines()[1] == DISCLAIMER


def _search(query: str, rows: list[object], k: int = 5) -> MedicineSearchResult:
    result = search_medicine(query, k=k, records=rows)
    _assert_disclosure(result)
    return result


def test_character_grams_use_lengths_2_and_3() -> None:
    exact = ngram_similarity("ab", "ab")
    assert exact.cosine == pytest.approx(1)
    assert exact.shared_ngrams == ("ab",)
    assert exact.scheme == SCHEME
    assert exact.query_norm > 0
    assert exact.cosine == pytest.approx(exact.dot / (exact.query_norm * exact.brand_norm))

    both = ngram_similarity("abc", "abc")
    assert "abc" in both.shared_ngrams
    assert any(len(gram) == 2 for gram in both.shared_ngrams)

    assert ngram_similarity("testbranda", "qqqqqqqq").cosine == 0.0
    assert ngram_similarity("a", "a").cosine == 0.0


def test_exact_brand_returns_same_recorded_attributes() -> None:
    rows = [
        _row("testbranda", strength_unit="MG", form="Tablet", mrp=12.5),
        _row("otherbrandq"),
        _row("xylophoneq", salt="fixturesaltb"),
    ]

    result = _search("testbranda", rows)

    assert [candidate.brand for candidate in result.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    first, second = result.candidates
    assert first.rank == 1
    assert first.similarity.cosine == pytest.approx(1)
    assert first.similarity.scheme == SCHEME
    assert any(len(gram) == 2 for gram in first.similarity.shared_ngrams)
    assert any(len(gram) == 3 for gram in first.similarity.shared_ngrams)
    assert first.similarity.cosine > second.similarity.cosine
    assert first.salt == second.salt == "fixturesalta"
    assert first.strength_value == second.strength_value == 40
    assert first.strength_unit == second.strength_unit == "mg"
    assert first.form == second.form == "tablet"
    assert first.release is None and second.release is None
    assert first.salts is None and second.salts is None
    assert first.mrp == 12.5
    assert second.mrp is None
    assert first.generic_price is None
    assert first.manufacturer is None
    assert "xylophoneq" not in [candidate.brand for candidate in result.candidates]

    limited = _search("testbranda", rows, k=1)
    assert [candidate.brand for candidate in limited.candidates] == ["testbranda"]


def test_misspelling_ranks_the_closer_brand_first() -> None:
    rows = [
        _row("testbranda"),
        _row("otherbrandq"),
        _row("xylophoneq", salt="fixturesaltb"),
    ]

    result = _search("testbrenda", rows)

    assert result.candidates[0].brand == "testbranda"
    assert result.candidates[0].salt == "fixturesalta"
    assert result.candidates[0].strength_value == 40
    assert result.minimum_brand_cosine <= result.candidates[0].similarity.cosine < 1
    assert "xylophoneq" not in [candidate.brand for candidate in result.candidates]
    assert {
        (candidate.salt, candidate.strength_value, candidate.strength_unit, candidate.form)
        for candidate in result.candidates
    } == {("fixturesalta", 40, "mg", "tablet")}
    if len(result.candidates) > 1:
        assert result.candidates[0].similarity.cosine > result.candidates[1].similarity.cosine


def test_hinglish_wrapper_words_leave_brand_and_strength() -> None:
    rows = [
        _row("testbranda", strength_value=40),
        _row("testbranda", strength_value=20),
    ]

    result = _search("testbranda 40 sasta kya milega", rows)

    assert result.parsed_brand == "testbranda"
    assert result.parsed_strength_value == 40
    assert result.parsed_strength_unit is None
    assert result.parsed_form is None
    assert [(candidate.brand, candidate.strength_value) for candidate in result.candidates] == [
        ("testbranda", 40),
    ]


def test_wrong_strength_is_excluded() -> None:
    result = _search("testbranda 20", [_row("testbranda", strength_value=40)])

    assert result.candidates == ()
    assert result.parsed_strength_value == 20
    assert result.best_brand_cosine == pytest.approx(1)


def test_wrong_form_is_excluded() -> None:
    result = _search("testbranda syrup", [_row("testbranda", form="tablet")])

    assert result.candidates == ()
    assert result.parsed_form == "syrup"
    assert result.best_brand_cosine == pytest.approx(1)


def test_wrong_unit_is_excluded_and_units_are_not_converted() -> None:
    recorded = [_row("testbranda", strength_value=1000, strength_unit="mg")]

    wrong_unit = _search("testbranda 1000 mcg", recorded)
    converted = _search("testbranda 1 g", recorded)
    same = _search("testbranda 1000 mg", recorded)

    assert wrong_unit.candidates == ()
    assert wrong_unit.parsed_strength_unit == "mcg"
    assert wrong_unit.best_brand_cosine == pytest.approx(1)
    assert converted.candidates == ()
    assert converted.parsed_strength_value == 1
    assert converted.parsed_strength_unit == "g"
    assert [candidate.strength_value for candidate in same.candidates] == [1000]
    assert same.candidates[0].strength_unit == "mg"
    assert same.parsed_brand == "testbranda"


def test_no_match_returns_no_candidates() -> None:
    result = _search("qqqqqqqq", [_row("testbranda")])

    assert result.candidates == ()
    assert result.best_brand_cosine == 0.0
    assert result.skipped_incomplete_brands == ()
    assert result.parsed_brand == "qqqqqqqq"


def test_missing_fields_are_not_inferred() -> None:
    no_salt = _row("testbranda")
    del no_salt["salt"]
    other = _row("xylophoneq")
    missing_salt = _search("testbranda", [no_salt, other])

    assert missing_salt.candidates == ()
    assert missing_salt.best_brand_cosine == pytest.approx(1)
    assert missing_salt.skipped_incomplete_brands == ("testbranda",)
    assert read_medicine(no_salt).salt is None
    assert read_medicine(no_salt).recorded_key() is None

    incomplete = _row("otherbrandq")
    del incomplete["salt"]
    kept = _search("testbranda", [_row("testbranda"), incomplete])

    assert [candidate.brand for candidate in kept.candidates] == ["testbranda"]
    assert read_medicine(incomplete).salt is None
    assert "otherbrandq" not in [candidate.brand for candidate in kept.candidates]

    no_unit = _row("testbranda")
    del no_unit["strength_unit"]
    missing_unit = _search("testbranda 40 mg", [no_unit])

    assert missing_unit.candidates == ()
    assert missing_unit.skipped_incomplete_brands == ("testbranda",)
    assert read_medicine(no_unit).strength_unit is None


def test_combination_salt_string_is_not_split() -> None:
    rows = [
        _row("testbranda", salt="fixturesalta + fixturesaltb"),
        _row("otherbrandq", salt="fixturesalta + fixturesaltb"),
        _row("qqsingleqq", salt="fixturesalta"),
    ]

    result = _search("testbranda", rows)

    assert [candidate.brand for candidate in result.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    assert result.candidates[0].salt == result.candidates[1].salt
    assert result.candidates[0].salt != "fixturesalta"
    assert "fixturesaltb" in result.candidates[0].salt
    assert all(candidate.salts is None for candidate in result.candidates)


def test_exposed_salts_list_must_match_in_full() -> None:
    rows = [
        _row("testbranda", salts=["fixturesalta", "fixturesaltb"]),
        _row("otherbrandq", salts=["fixturesalta", "fixturesaltb"]),
        _row("qqsingleqq", salts=["fixturesalta"]),
        _row("qqswappedqq", salts=["fixturesaltb", "fixturesalta"]),
        _row("qqunlisted", salt="fixturesalta"),
    ]

    result = _search("testbranda", rows)

    assert [candidate.brand for candidate in result.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    assert result.candidates[0].salts == ("fixturesalta", "fixturesaltb")
    assert result.candidates[1].salts == ("fixturesalta", "fixturesaltb")


def test_release_and_form_distinctions_follow_the_record() -> None:
    rows = [
        _row("testbranda", release="extended release"),
        _row("otherbrandq", release="extended release"),
        _row("qqimmediate", release="immediate release"),
        _row("qqplainqq"),
        _row("qqformbrand", form="extended release tablet"),
    ]

    extended = _search("testbranda", rows)
    form_only = _search("qqformbrand", rows)

    assert [candidate.brand for candidate in extended.candidates] == [
        "testbranda",
        "otherbrandq",
    ]
    assert {candidate.release for candidate in extended.candidates} == {"extended release"}
    assert {candidate.form for candidate in extended.candidates} == {"tablet"}
    assert [candidate.brand for candidate in form_only.candidates] == ["qqformbrand"]
    assert form_only.candidates[0].form == "extended release tablet"
    assert form_only.candidates[0].release is None


def test_dataclass_fixture_row_is_read_without_added_fields() -> None:
    @dataclass
    class FixtureRow:
        brand: str
        salt: str
        strength_value: float
        strength_unit: str
        form: str

    row = FixtureRow("testbranda", "fixturesalta", 40, "mg", "tablet")
    result = _search("testbranda", [row])

    assert [candidate.brand for candidate in result.candidates] == ["testbranda"]
    assert result.candidates[0].release is None
    assert result.candidates[0].salts is None
    assert result.candidates[0].mrp is None


def test_rejects_bad_k_and_a_non_record() -> None:
    with pytest.raises(ValueError):
        search_medicine("testbranda", k=-1, records=[])
    with pytest.raises(TypeError):
        search_medicine("testbranda", k=True, records=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        search_medicine("testbranda", records=["not-a-row"])  # type: ignore[list-item]


def test_normalize_corpus_brand_removes_explicit_constraints() -> None:
    from bharosa.medicine.parametric import RecordedConstraints
    from bharosa.medicine.search import normalize_corpus_brand

    c_strength = RecordedConstraints(strength_value=650.0)
    assert normalize_corpus_brand("Dolo 650 Tablet", c_strength) == "dolo tablet"

    c_both = RecordedConstraints(strength_value=650.0, form="tablet")
    assert normalize_corpus_brand("Dolo 650 Tablet", c_both) == "dolo"
    assert normalize_corpus_brand("Dolo 650mg Tablet", c_both) == "dolo"


def test_search_medicine_dolo_650_sasta_alternative_matches() -> None:
    rows = [
        _row("Dolo 650 Tablet", strength_value=650, form="tablet", salt="paracetamol"),
        _row("Dolopar 650 Tablet", strength_value=650, form="tablet", salt="paracetamol"),
        _row("Crocin 500 Tablet", strength_value=500, form="tablet", salt="paracetamol"),
    ]
    result = _search("dolo 650 sasta alternative", rows)
    assert len(result.candidates) >= 1
    brands = [c.brand for c in result.candidates]
    assert "dolo 650 tablet" in brands
    assert result.candidates[0].brand == "dolo 650 tablet"
    assert result.candidates[0].salt == "paracetamol"
    assert result.candidates[0].strength_value == 650


def test_search_medicine_azithral_500_tablet_matches() -> None:
    rows = [
        _row("Azithral 500 Tablet", strength_value=500, form="tablet", salt="azithromycin"),
        _row("Azithral 250 Tablet", strength_value=250, form="tablet", salt="azithromycin"),
    ]
    result = _search("Azithral 500 Tablet", rows)
    assert len(result.candidates) == 1
    assert result.candidates[0].brand == "azithral 500 tablet"
    assert result.candidates[0].strength_value == 500

