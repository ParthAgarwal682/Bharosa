"""Unit tests for the medicine loader, normalization, and validation rules.

IR concept: Known-answer test fixtures and schema contract verification.
Tests use tiny synthetic fixtures to verify deterministic behaviour,
safety constraints, and accounting accuracy without guessing or depending
on external production state.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from bharosa.medicine.loader import (
    ActiveIngredient,
    LoadStats,
    MedicineRecord,
    load_medicines,
    normalize_dosage_form,
    parse_active_ingredients,
    parse_strength,
)


CSV_HEADER = [
    "product_id",
    "brand_name",
    "manufacturer",
    "price_inr",
    "is_discontinued",
    "dosage_form",
    "pack_size",
    "pack_unit",
    "num_active_ingredients",
    "primary_ingredient",
    "primary_strength",
    "active_ingredients",
    "therapeutic_class",
    "packaging_raw",
    "manufacturer_raw",
]


def create_synthetic_csv(tmp_path: Path, rows: list[list[str]]) -> Path:
    """Helper to write synthetic rows to a temporary CSV file."""
    csv_file = tmp_path / "medicines_fixture.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return csv_file


def test_parse_strength_standards() -> None:
    """Test strength parsing across whitespace, casing, and compound units."""
    assert parse_strength("650mg") == (650.0, "mg")
    assert parse_strength("650 mg") == (650.0, "mg")
    assert parse_strength("650 MG") == (650.0, "mg")
    assert parse_strength("  650   Mg  ") == (650.0, "mg")
    assert parse_strength("0.5ml") == (0.5, "ml")
    assert parse_strength("0.5 ML") == (0.5, "ml")
    assert parse_strength("100mcg") == (100.0, "mcg")
    assert parse_strength("30mg/5ml") == (30.0, "mg/5ml")
    assert parse_strength("2%") == (2.0, "%")
    assert parse_strength("50000 IU") == (50000.0, "iu")


def test_parse_strength_unparseable_or_invalid() -> None:
    """Test that invalid or zero strengths return None."""
    assert parse_strength("") is None
    assert parse_strength("   ") is None
    assert parse_strength("none") is None
    assert parse_strength("mg") is None
    assert parse_strength("0mg") is None
    assert parse_strength("-10mg") is None


def test_normalize_dosage_form() -> None:
    """Test dosage form canonicalization."""
    assert normalize_dosage_form("tab") == "tablet"
    assert normalize_dosage_form("TABS") == "tablet"
    assert normalize_dosage_form("Tablet") == "tablet"
    assert normalize_dosage_form("cap") == "capsule"
    assert normalize_dosage_form("Capsules") == "capsule"
    assert normalize_dosage_form("syp") == "syrup"
    assert normalize_dosage_form("Syrup") == "syrup"
    assert normalize_dosage_form("inj") == "injection"
    assert normalize_dosage_form("Drops") == "drops"
    assert normalize_dosage_form("") == ""


def test_parse_active_ingredients() -> None:
    """Test structured parsing of the active_ingredients JSON/literal string."""
    raw = "[{'name': 'Amoxycillin', 'strength': '500mg', 'full_description': 'Amoxycillin (500mg)'}, {'name': 'Clavulanic Acid', 'strength': '125mg', 'full_description': 'Clavulanic Acid (125mg)'}]"
    ingredients = parse_active_ingredients(raw)
    assert len(ingredients) == 2
    assert ingredients[0] == ActiveIngredient(
        name="Amoxycillin",
        strength_value=500.0,
        strength_unit="mg",
    )
    assert ingredients[1] == ActiveIngredient(
        name="Clavulanic Acid",
        strength_value=125.0,
        strength_unit="mg",
    )

    # Empty or malformed inputs return empty list
    assert parse_active_ingredients("") == []
    assert parse_active_ingredients("not a list") == []


def test_load_single_ingredient_valid(tmp_path: Path) -> None:
    """Test valid single-ingredient row mapping and field preservation."""
    row = [
        "101",
        "Calpol 650 Tablet",
        "GlaxoSmithKline",
        "30.50",
        "False",
        "Tab",
        "15.0",
        "strip",
        "1",
        "Paracetamol",
        "650 MG",
        "[{'name': 'Paracetamol', 'strength': '650mg', 'full_description': 'Paracetamol (650mg)'}]",
        "analgesic",
        "strip of 15 tablets",
        "GlaxoSmithKline",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row])

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 1
    assert stats.accepted_rows == 1
    assert stats.dropped_rows == 0
    assert len(records) == 1

    rec = records[0]
    assert rec.brand == "Calpol 650 Tablet"
    assert rec.salt == "Paracetamol"
    assert rec.strength_value == 650.0
    assert rec.strength_unit == "mg"
    assert rec.form == "tablet"
    assert rec.mrp == 30.50
    assert rec.generic_price is None  # Not provided in source; never fabricated
    assert rec.manufacturer == "GlaxoSmithKline"
    assert rec.is_combination is False
    assert rec.pack_size == 15.0
    assert rec.pack_unit == "strip"
    assert len(rec.ingredients) == 1
    assert rec.ingredients[0] == ActiveIngredient(
        name="Paracetamol",
        strength_value=650.0,
        strength_unit="mg",
    )


def test_load_combination_ingredient_valid(tmp_path: Path) -> None:
    """Test valid combination medicine: sorted joined salts, individual strengths preserved, no false scalar strength."""
    row = [
        "102",
        "Augmentin 625 Duo Tablet",
        "GlaxoSmithKline",
        "223.42",
        "False",
        "tablet",
        "10.0",
        "strip",
        "2",
        "Amoxycillin",
        "500mg",
        "[{'name': 'Clavulanic Acid', 'strength': '125mg'}, {'name': 'Amoxycillin', 'strength': '500mg'}]",
        "antibiotic",
        "strip of 10",
        "GlaxoSmithKline",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row])

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 1
    assert stats.accepted_rows == 1
    assert stats.dropped_rows == 0
    assert len(records) == 1

    rec = records[0]
    assert rec.brand == "Augmentin 625 Duo Tablet"
    # Active ingredients must be sorted case-insensitively and joined with ' + '
    assert rec.salt == "Amoxycillin + Clavulanic Acid"
    assert rec.is_combination is True
    # Do NOT invent a combined numeric strength
    assert rec.strength_value is None
    assert rec.strength_unit is None
    assert rec.mrp == 223.42
    assert rec.generic_price is None
    assert rec.pack_size == 10.0
    assert rec.pack_unit == "strip"
    assert len(rec.ingredients) == 2
    ing_map = {ing.name: (ing.strength_value, ing.strength_unit) for ing in rec.ingredients}
    assert ing_map["Amoxycillin"] == (500.0, "mg")
    assert ing_map["Clavulanic Acid"] == (125.0, "mg")


def test_discontinued_medicine_dropped(tmp_path: Path) -> None:
    """Discontinued medicines must be excluded from accepted production records and tracked."""
    row = [
        "103",
        "Old Brand 500",
        "Pharma Corp",
        "50.0",
        "True",  # Discontinued
        "tablet",
        "10.0",
        "strip",
        "1",
        "Paracetamol",
        "500mg",
        "[{'name': 'Paracetamol', 'strength': '500mg'}]",
        "analgesic",
        "strip of 10",
        "Pharma Corp",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row])

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 1
    assert stats.accepted_rows == 0
    assert stats.dropped_rows == 1
    assert stats.reason_counts.get("discontinued") == 1
    assert len(records) == 0


def test_missing_or_invalid_fields_dropped(tmp_path: Path) -> None:
    """Test dropping rows missing required fields with explicit reason counts."""
    rows = [
        # Missing brand
        [
            "201", "", "Pharma A", "20.0", "False", "tablet", "10", "strip", "1",
            "Paracetamol", "500mg", "[{'name': 'Paracetamol'}]", "class", "raw", "mfg"
        ],
        # Missing dosage form
        [
            "202", "Brand B", "Pharma B", "20.0", "False", "", "10", "strip", "1",
            "Paracetamol", "500mg", "[{'name': 'Paracetamol'}]", "class", "raw", "mfg"
        ],
        # Missing manufacturer
        [
            "203", "Brand C", "", "20.0", "False", "tablet", "10", "strip", "1",
            "Paracetamol", "500mg", "[{'name': 'Paracetamol'}]", "class", "raw", "mfg"
        ],
        # Invalid price (<= 0)
        [
            "204", "Brand D", "Pharma D", "0.0", "False", "tablet", "10", "strip", "1",
            "Paracetamol", "500mg", "[{'name': 'Paracetamol'}]", "class", "raw", "mfg"
        ],
        # Non-numeric price
        [
            "205", "Brand E", "Pharma E", "not_a_price", "False", "tablet", "10", "strip", "1",
            "Paracetamol", "500mg", "[{'name': 'Paracetamol'}]", "class", "raw", "mfg"
        ],
        # Missing salt
        [
            "206", "Brand F", "Pharma F", "20.0", "False", "tablet", "10", "strip", "1",
            "", "500mg", "[]", "class", "raw", "mfg"
        ],
        # Unparseable strength for single-ingredient
        [
            "207", "Brand G", "Pharma G", "20.0", "False", "tablet", "10", "strip", "1",
            "Paracetamol", "various", "[{'name': 'Paracetamol', 'strength': 'various'}]", "class", "raw", "mfg"
        ],
    ]
    csv_file = create_synthetic_csv(tmp_path, rows)

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 7
    assert stats.accepted_rows == 0
    assert stats.dropped_rows == 7
    assert len(records) == 0

    assert stats.reason_counts.get("missing_brand") == 1
    assert stats.reason_counts.get("missing_dosage_form") == 1
    assert stats.reason_counts.get("missing_manufacturer") == 1
    assert stats.reason_counts.get("invalid_price") == 2
    assert stats.reason_counts.get("missing_salt") == 1
    assert stats.reason_counts.get("unparseable_strength") == 1


def test_deduplication_handling(tmp_path: Path) -> None:
    """Duplicate rows must keep the first occurrence and record duplicate_record."""
    row1 = [
        "301",
        "Dolo 650 Tablet",
        "Micro Labs",
        "30.0",
        "False",
        "tablet",
        "15.0",
        "strip",
        "1",
        "Paracetamol",
        "650mg",
        "[{'name': 'Paracetamol', 'strength': '650mg'}]",
        "analgesic",
        "strip",
        "Micro Labs",
    ]
    # Identical record with a different product_id
    row2 = [
        "302",
        "Dolo 650 Tablet",
        "Micro Labs",
        "30.0",
        "False",
        "tablet",
        "15.0",
        "strip",
        "1",
        "Paracetamol",
        "650mg",
        "[{'name': 'Paracetamol', 'strength': '650mg'}]",
        "analgesic",
        "strip",
        "Micro Labs",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row1, row2])

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 2
    assert stats.accepted_rows == 1
    assert stats.dropped_rows == 1
    assert stats.duplicate_counts == 1
    assert stats.reason_counts.get("duplicate_record") == 1
    assert len(records) == 1
    assert records[0].brand == "Dolo 650 Tablet"


def test_distinct_pack_size_or_pack_unit_not_duplicate(tmp_path: Path) -> None:
    """Medicines with identical identity but different pack_size or pack_unit are NOT duplicates."""
    base_row = [
        "301",
        "Dolo 650 Tablet",
        "Micro Labs",
        "30.0",
        "False",
        "tablet",
        "10.0",
        "strip",
        "1",
        "Paracetamol",
        "650mg",
        "[{'name': 'Paracetamol', 'strength': '650mg'}]",
        "analgesic",
        "strip of 10",
        "Micro Labs",
    ]
    # Same medicine identity, but different pack_size (15.0 vs 10.0)
    diff_pack_size_row = [
        "302",
        "Dolo 650 Tablet",
        "Micro Labs",
        "45.0",
        "False",
        "tablet",
        "15.0",
        "strip",
        "1",
        "Paracetamol",
        "650mg",
        "[{'name': 'Paracetamol', 'strength': '650mg'}]",
        "analgesic",
        "strip of 15",
        "Micro Labs",
    ]
    # Same medicine identity and pack_size, but different pack_unit ("bottle" vs "strip")
    diff_pack_unit_row = [
        "303",
        "Dolo 650 Tablet",
        "Micro Labs",
        "30.0",
        "False",
        "tablet",
        "10.0",
        "bottle",
        "1",
        "Paracetamol",
        "650mg",
        "[{'name': 'Paracetamol', 'strength': '650mg'}]",
        "analgesic",
        "bottle of 10",
        "Micro Labs",
    ]
    csv_file = create_synthetic_csv(
        tmp_path, [base_row, diff_pack_size_row, diff_pack_unit_row]
    )

    records, stats = load_medicines(csv_file)

    assert stats.total_rows == 3
    assert stats.accepted_rows == 3
    assert stats.dropped_rows == 0
    assert stats.duplicate_counts == 0
    assert len(records) == 3

    pack_configs = {(r.pack_size, r.pack_unit) for r in records}
    assert pack_configs == {(10.0, "strip"), (15.0, "strip"), (10.0, "bottle")}


def test_schema_mismatch_raises_error(tmp_path: Path) -> None:
    """Missing expected columns in CSV must raise ValueError instead of guessing."""
    bad_csv = tmp_path / "bad_columns.csv"
    with bad_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "some_random_column"])
        writer.writerow(["1", "value"])

    with pytest.raises(ValueError, match="Missing required columns"):
        load_medicines(bad_csv)


def test_file_not_found_raises(tmp_path: Path) -> None:
    """Non-existent file path must raise FileNotFoundError."""
    missing = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_medicines(missing)


def test_release_is_not_inferred(tmp_path: Path) -> None:
    """Release profile (IR/SR/ER/CR) is not available in source and must not be inferred."""
    row = [
        "104",
        "Pantocid DSR Capsule",
        "Sun Pharma",
        "150.0",
        "False",
        "capsule",
        "10.0",
        "strip",
        "1",
        "Pantoprazole",
        "40mg",
        "[{'name': 'Pantoprazole', 'strength': '40mg'}]",
        "gastro",
        "strip of 10",
        "Sun Pharma",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row])
    records, stats = load_medicines(csv_file)
    assert len(records) == 1
    rec = records[0]
    # Verify that release profile is not fabricated or exposed as an inferred attribute
    assert not hasattr(rec, "release_type")
    assert not hasattr(rec, "release_profile")


def test_loader_api_requires_filepath() -> None:
    """Loader contract strictly requires file_path argument and cannot be called without it."""
    with pytest.raises(TypeError):
        load_medicines()  # type: ignore[call-arg]


def test_unparseable_individual_ingredient_strength_preserved_as_none(tmp_path: Path) -> None:
    """An unparseable individual ingredient strength is preserved as None and is NOT evidence of strength equivalence."""
    row = [
        "105",
        "Combo Tonic",
        "Apex Pharma",
        "85.0",
        "False",
        "syrup",
        "100.0",
        "bottle",
        "2",
        "Herb A",
        "",
        "[{'name': 'Herb A', 'strength': 'unparseable_xyz'}, {'name': 'Vitamin B', 'strength': '10mg'}]",
        "tonic",
        "bottle of 100ml",
        "Apex Pharma",
    ]
    csv_file = create_synthetic_csv(tmp_path, [row])
    records, stats = load_medicines(csv_file)
    assert stats.accepted_rows == 1
    assert len(records) == 1
    rec = records[0]
    assert rec.is_combination is True
    assert len(rec.ingredients) == 2

    herb = next(ing for ing in rec.ingredients if ing.name == "Herb A")
    vit = next(ing for ing in rec.ingredients if ing.name == "Vitamin B")

    # Strength is preserved as None, not invented, not zero
    assert herb.strength_value is None
    assert herb.strength_unit is None
    assert vit.strength_value == 10.0
    assert vit.strength_unit == "mg"

    # Explicit check: None strength must NOT be treated as equal or equivalent to zero or other strengths
    assert herb.strength_value is not 0.0
    assert herb.strength_value is not 10.0
    assert (herb.strength_value == 0.0) is False
    assert (herb.strength_value == vit.strength_value) is False
