"""Parametric identity for recorded medicine attributes.

IR concept: a parametric filter. Two rows are in the same identity only
when the recorded salt, strength value, strength unit, and dosage form
all agree. Comparison uses those stored fields after the shared
normaliser. A missing field stays missing. Nothing is copied from the
brand text, from another row, or from a unit-conversion table.

Combination products and release are part of the identity only when the
row itself exposes them. ``salts`` is a sequence of component salts.
``release`` is a single string. If either row records one of those and
the other does not, the identities differ. The salt string is not split
on ``+`` or spaces to invent a component list. Dosage-form text is not
parsed into a release.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bharosa.text.normalize import normalize

# README medicine record. ``release`` and ``salts`` are optional
# distinction fields. They are read only when a row actually has them.
MEDICINE_CONTRACT_FIELDS: frozenset[str] = frozenset(
    {
        "brand",
        "salt",
        "strength_value",
        "strength_unit",
        "form",
        "mrp",
        "generic_price",
        "manufacturer",
    }
)

DISTINCTION_FIELDS: tuple[str, ...] = ("release", "salts")

# Whole-field spelling fold for a closed set of dosage-form words.
# A longer form string is not rewritten, so "extended release tablet"
# stays that string and does not become "tablet".
_FORM_CANON: dict[str, str] = {
    "tablet": "tablet",
    "tablets": "tablet",
    "capsule": "capsule",
    "capsules": "capsule",
    "syrup": "syrup",
    "syrups": "syrup",
    "injection": "injection",
    "injections": "injection",
    "suspension": "suspension",
    "drop": "drop",
    "drops": "drop",
    "cream": "cream",
    "ointment": "ointment",
    "gel": "gel",
    "inhaler": "inhaler",
    "powder": "powder",
    "solution": "solution",
}

FORM_TOKENS: frozenset[str] = frozenset(_FORM_CANON)


@dataclass(frozen=True)
class AttributeKey:
    """Recorded identity used by the parametric filter.

    IR concept: the structured key a Boolean filter tests for equality.
    ``release`` is ``None`` when the row did not record a release.
    ``salts`` is ``None`` when the row did not record a component list.
    Neither ``None`` is a default release or an empty combination. It
    means the field was not in the data. Two ``None`` values agree.
    ``None`` does not agree with a recorded value.
    """

    salt: str
    strength_value: float
    strength_unit: str
    form: str
    release: str | None = None
    salts: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RecordedConstraints:
    """Constraints the query actually stated.

    IR concept: a parametric query. ``None`` means the user did not
    constrain that field. It does not mean "match every value" by
    filling one in, and it does not mean "match rows that left the
    field blank".
    """

    strength_value: float | None = None
    strength_unit: str | None = None
    form: str | None = None
    release: str | None = None


@dataclass(frozen=True)
class MedicineRecord:
    """One medicine row after field reading.

    IR concept: the document stored for parametric lookup. Strings are
    the shared normaliser's form of what the row contained. ``None``
    marks a field the row did not provide. Prices and the manufacturer
    are carried for display. They are not part of the identity.
    """

    brand: str | None
    salt: str | None
    strength_value: float | None
    strength_unit: str | None
    form: str | None
    release: str | None
    salts: tuple[str, ...] | None
    mrp: float | None
    generic_price: float | None
    manufacturer: str | None

    def recorded_key(self) -> AttributeKey | None:
        """Identity of this row, or ``None`` when a required field is absent.

        IR concept: posting a document under its parametric key. Salt,
        strength value, strength unit, and dosage form are all required.
        A blank brand cannot be shown as a candidate, so it also yields
        ``None``. Release and the component list are included only at the
        values this row recorded.
        """
        if (
            self.brand is None
            or self.salt is None
            or self.strength_value is None
            or self.strength_unit is None
            or self.form is None
        ):
            return None
        return AttributeKey(
            salt=self.salt,
            strength_value=self.strength_value,
            strength_unit=self.strength_unit,
            form=self.form,
            release=self.release,
            salts=self.salts,
        )


def canonical_form(value: str) -> str:
    """Normalise a dosage-form string without splitting it.

    IR concept: field normalisation before a parametric comparison.
    Only an exact closed-list word is folded (``tablets`` to ``tablet``).
    Any other string, including one that mentions a release, is returned
    as the shared normaliser left it.
    """
    if not isinstance(value, str):
        raise TypeError(f"form must be str, got {type(value).__name__}")
    normalised = normalize(value)
    if not normalised:
        return ""
    return _FORM_CANON.get(normalised, normalised)


def read_medicine(record: Mapping[str, object]) -> MedicineRecord:
    """Read one medicine row. Do not fill fields the row omits.

    IR concept: building a parametric document from source fields. The
    contract fields and the two distinction fields are the only ones
    read. Any other key is ignored, not mapped onto salt, strength,
    unit, form, release, or combination components.
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"medicine record must be a mapping, got {type(record).__name__}")
    return MedicineRecord(
        brand=_optional_text(record, "brand"),
        salt=_optional_text(record, "salt"),
        strength_value=_optional_strength(record),
        strength_unit=_optional_text(record, "strength_unit"),
        form=_optional_form(record),
        release=_optional_text(record, "release"),
        salts=_optional_salts(record),
        mrp=_optional_number(record, "mrp"),
        generic_price=_optional_number(record, "generic_price"),
        manufacturer=_optional_text(record, "manufacturer"),
    )


def passes_query_constraints(
    record: MedicineRecord,
    constraints: RecordedConstraints,
) -> bool:
    """True when the row's recorded key agrees with every stated constraint.

    IR concept: Boolean parametric selection. The row must already have
    salt, strength value, strength unit, and dosage form. A constraint
    left as ``None`` does not test that field. A stated release matches
    only a row that recorded the same release string.
    """
    key = record.recorded_key()
    if key is None:
        return False
    if constraints.strength_value is not None and key.strength_value != constraints.strength_value:
        return False
    if constraints.strength_unit is not None and key.strength_unit != constraints.strength_unit:
        return False
    if constraints.form is not None and key.form != constraints.form:
        return False
    if constraints.release is not None and key.release != constraints.release:
        return False
    return True


def recorded_conflict(record: MedicineRecord, constraints: RecordedConstraints) -> bool:
    """True when a field the row recorded disagrees with the query.

    IR concept: a failed parametric test on a value that is present.
    A field the row left out is not a conflict. It is absent, and the
    caller must not treat absence as a match or as the query's value.
    """
    if (
        constraints.strength_value is not None
        and record.strength_value is not None
        and record.strength_value != constraints.strength_value
    ):
        return True
    if (
        constraints.strength_unit is not None
        and record.strength_unit is not None
        and record.strength_unit != constraints.strength_unit
    ):
        return True
    if constraints.form is not None and record.form is not None and record.form != constraints.form:
        return True
    if (
        constraints.release is not None
        and record.release is not None
        and record.release != constraints.release
    ):
        return True
    return False


def _optional_form(record: Mapping[str, object]) -> str | None:
    text = _optional_text(record, "form")
    if text is None:
        return None
    canon = _FORM_CANON.get(text, text)
    return canon or None


def _optional_strength(record: Mapping[str, object]) -> float | None:
    if "strength_value" not in record or record["strength_value"] is None:
        return None
    return _canonical_strength(record["strength_value"])


def _optional_salts(record: Mapping[str, object]) -> tuple[str, ...] | None:
    """Component list only when the row has a ``salts`` sequence.

    A string is rejected so the salt field's characters are not read as
    components. Order is the row's order. The string in ``salt`` is not
    split to produce this list.
    """
    if "salts" not in record or record["salts"] is None:
        return None
    value = record["salts"]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"salts must be a sequence of str, got {type(value).__name__}")
    components: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"salts item must be str, got {type(item).__name__}")
        normalised = normalize(item)
        if not normalised:
            raise ValueError("salts item is empty after normalisation")
        components.append(normalised)
    return tuple(components)


def _optional_text(record: Mapping[str, object], key: str) -> str | None:
    if key not in record or record[key] is None:
        return None
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str, got {type(value).__name__}")
    normalised = normalize(value)
    if not normalised:
        return None
    return normalised


def _optional_number(record: Mapping[str, object], key: str) -> float | None:
    if key not in record or record[key] is None:
        return None
    return _finite_number(record[key], key)


def _canonical_strength(value: object) -> float:
    """Round a recorded strength to 6 decimal places.

    IR concept: numeric field canonicalisation. ``40`` and ``40.0`` meet.
    Units are not converted, so 1000 mg does not become 1 g.
    """
    number = _finite_number(value, "strength_value")
    return round(number, 6)


def _finite_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number
