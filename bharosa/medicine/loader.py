"""Pharmaceutical dataset loader, schema validator, and normaliser.

IR concept: Document ingestion, schema mapping, and data normalisation.
A retrieval system is only as reliable as the underlying index collection.
This module ingests raw pharmaceutical data, enforces strict medical
contracts, normalises textual fields (casing, whitespace, strengths, forms)
without inventing missing facts, and tracks audit statistics.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bharosa.text.normalize import normalize as normalize_text


# Strength parser regex: captures positive float/int and unit (e.g. 650mg, 650 mg, 650 MG, 0.5ml, 30mg/5ml, 2%)
_STRENGTH_PATTERN = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z%]+(?:/[0-9]*[a-zA-Z]+)?)\s*$"
)

# Standard dosage form normalization mappings
_DOSAGE_FORM_MAPPINGS: dict[str, str] = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablet": "tablet",
    "tablets": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsule": "capsule",
    "capsules": "capsule",
    "syp": "syrup",
    "syrup": "syrup",
    "inj": "injection",
    "injection": "injection",
    "injections": "injection",
    "ointment": "ointment",
    "cream": "cream",
    "gel": "gel",
    "drops": "drops",
    "drop": "drops",
    "suspension": "suspension",
    "solution": "solution",
    "inhaler": "inhaler",
    "respules": "respules",
    "lotion": "lotion",
    "powder": "powder",
}

# Required columns in the source CSV
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "brand_name",
        "manufacturer",
        "price_inr",
        "is_discontinued",
        "dosage_form",
        "num_active_ingredients",
        "primary_ingredient",
        "primary_strength",
        "active_ingredients",
    }
)


@dataclass(frozen=True)
class ActiveIngredient:
    """Individual chemical component for auditability."""

    name: str
    strength: str = ""
    full_description: str = ""


@dataclass(frozen=True)
class MedicineRecord:
    """Canonical medicine record mapped from pharmaceutical data.

    Adheres strictly to the canonical Kushagra contract:
    - brand: trade / brand name
    - salt: active ingredient (or sorted joined active ingredients for combos)
    - strength_value: numeric potency (None for multi-ingredient combos)
    - strength_unit: unit of potency (None for multi-ingredient combos)
    - form: normalized dosage form
    - mrp: source full-pack price
    - generic_price: None (not provided in source data; never fabricated)
    - manufacturer: pharmaceutical company name
    - is_combination: True if product contains >1 active ingredient
    """

    brand: str
    salt: str
    strength_value: float | None
    strength_unit: str | None
    form: str
    mrp: float
    generic_price: float | None
    manufacturer: str
    is_combination: bool


@dataclass
class LoadStats:
    """Audit and ingestion metrics tracked during dataset loading."""

    total_rows: int = 0
    accepted_rows: int = 0
    dropped_rows: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    duplicate_counts: int = 0

    def record_drop(self, reason: str) -> None:
        """Record a dropped row with its explicit justification."""
        self.dropped_rows += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def record_duplicate(self) -> None:
        """Record a dropped duplicate row."""
        self.duplicate_counts += 1
        self.record_drop("duplicate_record")

    def record_accepted(self) -> None:
        """Record an accepted valid production row."""
        self.accepted_rows += 1


def parse_strength(strength_str: str) -> tuple[float, str] | None:
    """Parse strength strings into a numeric value and canonical unit.

    Supports forms such as:
    - '650mg', '650 mg', '650 MG' -> (650.0, 'mg')
    - '0.5ml', '0.5 ML' -> (0.5, 'ml')
    - '100mcg', '100 mcg' -> (100.0, 'mcg')
    - '30mg/5ml' -> (30.0, 'mg/5ml')
    - '2%' -> (2.0, '%')

    Returns None if the string is empty or cannot be reliably parsed.
    """
    if not strength_str or not isinstance(strength_str, str):
        return None

    cleaned = strength_str.strip()
    match = _STRENGTH_PATTERN.match(cleaned)
    if not match:
        return None

    val_str, unit_str = match.groups()
    try:
        val = float(val_str)
        if val <= 0.0:
            return None
        unit = unit_str.strip().lower()
        return val, unit
    except (ValueError, TypeError):
        return None


def normalize_dosage_form(form_str: str) -> str:
    """Normalize dosage form string using whitespace and alias mapping."""
    if not form_str or not isinstance(form_str, str):
        return ""
    cleaned = form_str.strip().lower()
    return _DOSAGE_FORM_MAPPINGS.get(cleaned, cleaned)


def parse_active_ingredients(raw_ingredients: str) -> list[ActiveIngredient]:
    """Parse raw active_ingredients representation into structured objects."""
    if not raw_ingredients or not isinstance(raw_ingredients, str):
        return []

    text = raw_ingredients.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []

    parsed_list: list[Any] = []
    # Try ast.literal_eval first (handles single-quoted Python representations)
    try:
        parsed_list = ast.literal_eval(text)
    except Exception:
        try:
            # Fallback to json if formatted as JSON
            parsed_list = json.loads(text.replace("'", '"'))
        except Exception:
            return []

    if not isinstance(parsed_list, list):
        return []

    ingredients: list[ActiveIngredient] = []
    for item in parsed_list:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            strength = str(item.get("strength", "")).strip()
            desc = str(item.get("full_description", "")).strip()
            if name:
                ingredients.append(
                    ActiveIngredient(
                        name=name,
                        strength=strength,
                        full_description=desc,
                    )
                )
    return ingredients


def load_medicines(file_path: Path | str) -> tuple[list[MedicineRecord], LoadStats]:
    """Load, validate, and normalise pharmaceutical records from a CSV file.

    IR concept: Ingestion pipeline and data hygiene.

    Enforces agreed contracts:
    - Never invents missing brand, salt, strength, form, price, or manufacturer.
    - Sets generic_price = None (not provided in source data).
    - Preserves source price_inr as full-pack mrp.
    - Represents combination medicines using sorted, joined active ingredient names
      and sets is_combination=True, with scalar strength set to None.
    - Drops discontinued products (is_discontinued=True) with reason 'discontinued'.
    - Deduplicates records, preserving the first and dropping duplicates under
      'duplicate_record'.
    - Tracks comprehensive statistics in LoadStats.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required dataset columns are missing from the CSV.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Medicine dataset not found: {path}")

    stats = LoadStats()
    accepted_records: list[MedicineRecord] = []
    seen_keys: set[tuple[Any, ...]] = set()

    with path.open(mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or corrupted CSV file: {path}")

        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"Dataset columns do not support agreed schema. Missing required columns: {sorted(missing_cols)}"
            )

        for row in reader:
            stats.total_rows += 1

            # 1. Check discontinued status (must not enter accepted production records)
            is_disc_raw = str(row.get("is_discontinued", "")).strip().lower()
            if is_disc_raw in ("true", "1", "yes"):
                stats.record_drop("discontinued")
                continue

            # 2. Check and clean brand_name
            brand_raw = str(row.get("brand_name", "")).strip()
            if not brand_raw:
                stats.record_drop("missing_brand")
                continue
            brand_clean = " ".join(brand_raw.split())

            # 3. Check and clean dosage_form
            form_raw = str(row.get("dosage_form", "")).strip()
            if not form_raw:
                stats.record_drop("missing_dosage_form")
                continue
            form_clean = normalize_dosage_form(form_raw)
            if not form_clean:
                stats.record_drop("missing_dosage_form")
                continue

            # 4. Check and clean manufacturer
            mfg_raw = str(row.get("manufacturer", "")).strip()
            if not mfg_raw:
                stats.record_drop("missing_manufacturer")
                continue
            mfg_clean = " ".join(mfg_raw.split())

            # 5. Check and parse price_inr (mrp)
            price_raw = str(row.get("price_inr", "")).strip()
            if not price_raw:
                stats.record_drop("invalid_price")
                continue
            try:
                mrp_val = float(price_raw)
                if mrp_val <= 0.0:
                    stats.record_drop("invalid_price")
                    continue
            except (ValueError, TypeError):
                stats.record_drop("invalid_price")
                continue

            # 6. Parse ingredients and handle combination vs single-ingredient
            num_act_raw = str(row.get("num_active_ingredients", "")).strip()
            try:
                num_act = int(float(num_act_raw)) if num_act_raw else 1
            except (ValueError, TypeError):
                num_act = 1

            parsed_ingredients = parse_active_ingredients(
                str(row.get("active_ingredients", ""))
            )
            is_combination = num_act > 1

            if is_combination:
                # Combination medicine: represent salt using sorted, joined active ingredient names
                if parsed_ingredients:
                    ingredient_names = [ing.name.strip() for ing in parsed_ingredients if ing.name.strip()]
                else:
                    primary = str(row.get("primary_ingredient", "")).strip()
                    ingredient_names = [primary] if primary else []

                if not ingredient_names:
                    stats.record_drop("missing_salt")
                    continue

                # Sort ingredient names case-insensitively and join
                sorted_names = sorted(ingredient_names, key=lambda s: s.casefold())
                salt_clean = " + ".join(sorted_names)

                # Do NOT invent a combined numeric strength
                strength_val: float | None = None
                strength_unit: str | None = None
            else:
                # Single-ingredient medicine: use primary_ingredient and parse primary_strength
                primary_ing = str(row.get("primary_ingredient", "")).strip()
                if not primary_ing:
                    if parsed_ingredients:
                        primary_ing = parsed_ingredients[0].name.strip()
                    else:
                        stats.record_drop("missing_salt")
                        continue

                salt_clean = " ".join(primary_ing.split())

                primary_str = str(row.get("primary_strength", "")).strip()
                if not primary_str and parsed_ingredients:
                    primary_str = parsed_ingredients[0].strength.strip()

                parsed_str = parse_strength(primary_str)
                if parsed_str is None:
                    stats.record_drop("unparseable_strength")
                    continue
                strength_val, strength_unit = parsed_str

            # 7. Deduplicate records
            dup_key = (
                brand_clean.casefold(),
                salt_clean.casefold(),
                strength_val,
                strength_unit,
                form_clean.casefold(),
                mfg_clean.casefold(),
            )

            if dup_key in seen_keys:
                stats.record_duplicate()
                continue
            seen_keys.add(dup_key)

            record = MedicineRecord(
                brand=brand_clean,
                salt=salt_clean,
                strength_value=strength_val,
                strength_unit=strength_unit,
                form=form_clean,
                mrp=mrp_val,
                generic_price=None,
                manufacturer=mfg_clean,
                is_combination=is_combination,
            )

            accepted_records.append(record)
            stats.record_accepted()

    return accepted_records, stats
