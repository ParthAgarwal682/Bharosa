"""Known-answer tests for the parametric index.

Filters use the zone contract's structured fields only. Expected
document ids are written out from those fields, not copied from a
second index.
"""

from __future__ import annotations

import pytest

from bharosa.index.params import (
    PARAMETRIC_FIELDS,
    ZONE_CONTRACT_FIELDS,
    ZONES,
    ParametricIndex,
    ZoneMeta,
)

CONTRACT_DOC = {
    "doc_id": "z1",
    "url": "https://example.test/scheme",
    "domain": "example.test",
    "title": "Heart scheme",
    "zone": "eligibility",
    "text": "Bihar residents with a heart condition",
    "state": "UP",
    "conditions": ["Heart"],
    "g_score": 0.9,
    "crawled_at": "2026-01-02T00:00:00",
    "last_changed_at": "2026-01-01T00:00:00",
    "content_hash": "abc",
}


def test_contract_fields_are_the_only_metadata() -> None:
    assert "state" in ZONE_CONTRACT_FIELDS
    assert "conditions" in ZONE_CONTRACT_FIELDS
    assert "zone" in ZONE_CONTRACT_FIELDS
    assert "condition" not in ZONE_CONTRACT_FIELDS
    assert "income" not in ZONE_CONTRACT_FIELDS
    assert "salt" not in ZONE_CONTRACT_FIELDS
    assert PARAMETRIC_FIELDS == ("state", "conditions", "zone")
    assert ZONES == (
        "eligibility",
        "benefits",
        "documents",
        "how_to_apply",
        "other",
    )


def test_postings_use_normalised_contract_fields() -> None:
    index = ParametricIndex()
    index.add_document(CONTRACT_DOC)
    index.add_document({"doc_id": "z2"})

    assert index.document_ids() == ("z1", "z2")
    assert index.metadata("z1") == ZoneMeta(
        "z1",
        state="up",
        conditions=("heart",),
        zone="eligibility",
    )
    # z2 did not provide state, conditions, or zone. Nothing is invented,
    # including from the fact that z1's text mentions Bihar.
    assert index.metadata("z2") == ZoneMeta("z2", None, (), None)
    assert index.postings() == {
        "state": {"up": ("z1",), "<missing>": ("z2",)},
        "conditions": {"heart": ("z1",)},
        "zone": {"eligibility": ("z1",), "<missing>": ("z2",)},
    }
    assert index.format_index() == "\n".join(
        [
            "parametric fields: state, conditions, zone",
            "state",
            "  <missing>  z2",
            "  up  z1",
            "conditions",
            "  heart  z1",
            "zone",
            "  <missing>  z2",
            "  eligibility  z1",
        ]
    )
    assert repr(index) == "ParametricIndex(documents=2)"


def test_text_does_not_fill_state_or_conditions() -> None:
    index = ParametricIndex()
    index.add_document(
        {
            "doc_id": "z1",
            "text": "Bihar residents with heart disease",
            "title": "UP heart scheme",
            "url": "https://up.example.test/heart",
            "domain": "up.example.test",
        }
    )

    assert index.metadata("z1").state is None
    assert index.metadata("z1").conditions == ()
    assert index.matching_ids({"state": "Bihar"}) == ()
    assert index.matching_ids({"state": "UP"}) == ()
    assert index.matching_ids({"conditions": ["heart"]}) == ()


def test_filters_intersect_state_conditions_and_zone() -> None:
    index = ParametricIndex()
    index.add_documents(
        [
            {
                "doc_id": "z",
                "state": "Uttar Pradesh",
                "conditions": ["heart", "diabetes"],
                "zone": "eligibility",
            },
            {
                "doc_id": "a",
                "state": "uttar   pradesh",
                "conditions": ["Heart"],
                "zone": "Eligibility",
            },
            {
                "doc_id": "b",
                "state": "Bihar",
                "conditions": ["heart", "diabetes"],
                "zone": "benefits",
            },
        ]
    )

    assert index.matching_ids(None) == ("z", "a", "b")
    assert index.matching_ids({}) == ("z", "a", "b")
    # Insertion order, not alphabetical order.
    assert index.matching_ids({"state": "UTTAR PRADESH"}) == ("z", "a")
    assert index.matching_ids({"conditions": ["diabetes"]}) == ("z", "b")
    assert index.matching_ids({"conditions": ["heart", "diabetes"]}) == ("z", "b")
    assert index.matching_ids(
        {"state": "Bihar", "conditions": ["heart"], "zone": "benefits"}
    ) == ("b",)
    assert index.matching_ids(
        {"state": "Uttar Pradesh", "conditions": ["heart", "cancer"]}
    ) == ()
    assert index.matching_ids({"zone": "documents"}) == ()


def test_condition_match_is_the_whole_normalised_value() -> None:
    index = ParametricIndex()
    index.add_document({"doc_id": "z1", "conditions": ["heart-disease"]})

    assert index.metadata("z1").conditions == ("heart disease",)
    assert index.matching_ids({"conditions": ["heart-disease"]}) == ("z1",)
    assert index.matching_ids({"conditions": ["heart"]}) == ()


def test_state_punctuation_is_not_given_an_alias() -> None:
    index = ParametricIndex()
    index.add_document({"doc_id": "z1", "state": "UP"})
    index.add_document({"doc_id": "z2", "state": "U.P."})

    assert index.metadata("z1").state == "up"
    assert index.metadata("z2").state == "u p"
    assert index.matching_ids({"state": "UP"}) == ("z1",)
    assert index.matching_ids({"state": "U.P."}) == ("z2",)


def test_null_state_and_zone_are_selectable_and_not_defaulted() -> None:
    index = ParametricIndex()
    index.add_document({"doc_id": "z1", "state": None, "zone": None, "conditions": []})
    index.add_document(
        {"doc_id": "z2", "state": "Goa", "zone": "other", "conditions": ["injury"]}
    )

    assert index.matching_ids({"state": None}) == ("z1",)
    assert index.matching_ids({"zone": None}) == ("z1",)
    assert index.metadata("z1").zone is None
    assert index.matching_ids({"conditions": []}) == ("z1", "z2")
    assert index.matching_ids({"zone": "other"}) == ("z2",)
    assert index.matching_ids({"zone": " how_to_apply "}) == ()


def test_closed_zone_vocabulary_is_not_rewritten() -> None:
    index = ParametricIndex()
    index.add_document({"doc_id": "z1", "zone": " how_to_apply "})

    assert index.metadata("z1").zone == "how_to_apply"
    assert index.matching_ids({"zone": "HOW_TO_APPLY"}) == ("z1",)
    with pytest.raises(ValueError, match="how_to_apply"):
        index.add_document({"doc_id": "z2", "zone": "how-to-apply"})
    with pytest.raises(ValueError, match="how_to_apply"):
        index.matching_ids({"zone": "elsewhere"})
    assert index.document_ids() == ("z1",)


def test_non_contract_and_non_parametric_keys_are_rejected() -> None:
    index = ParametricIndex()

    with pytest.raises(ValueError, match="not in the zone contract: income"):
        index.add_document({"doc_id": "z1", "income": "2 lakh"})
    with pytest.raises(ValueError, match="not in the zone contract: condition"):
        index.add_document({"doc_id": "z1", "condition": "heart"})
    with pytest.raises(ValueError, match="not in the zone contract: salt"):
        index.add_document({"doc_id": "z1", "salt": "paracetamol"})

    index.add_document({"doc_id": "z1", "g_score": 0.0, "domain": "gov.in"})
    with pytest.raises(ValueError, match="not a parametric field"):
        index.matching_ids({"g_score": 0.0})
    with pytest.raises(ValueError, match="not a parametric field"):
        index.matching_ids({"domain": "gov.in"})
    with pytest.raises(ValueError, match="not in the zone contract"):
        index.matching_ids({"income": "2 lakh"})
    with pytest.raises(ValueError, match="not in the zone contract"):
        index.matching_ids({"condition": "heart"})
    assert index.document_ids() == ("z1",)


def test_empty_index_and_unknown_value() -> None:
    index = ParametricIndex()

    assert index.document_ids() == ()
    assert index.matching_ids({"state": "Goa"}) == ()
    assert index.matching_ids({"conditions": ["heart"]}) == ()
    assert "conditions" in index.format_index()
    assert "  <empty>" in index.format_index()
    with pytest.raises(KeyError):
        index.metadata("missing")


def test_rejects_bad_inputs() -> None:
    index = ParametricIndex()
    index.add_document({"doc_id": "z1", "state": "Goa"})

    with pytest.raises(ValueError, match="already indexed"):
        index.add_document({"doc_id": "z1"})
    assert index.document_ids() == ("z1",)
    with pytest.raises(ValueError, match="non-empty"):
        index.add_document({"doc_id": ""})
    with pytest.raises(ValueError, match="doc_id is required"):
        index.add_document({"state": "Goa"})
    with pytest.raises(TypeError):
        index.add_document({"doc_id": 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        index.add_document(["z1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.add_documents({"doc_id": "z2", "state": "Goa"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.matching_ids({"conditions": "heart"})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        index.matching_ids({"state": 12})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="empty after normalisation"):
        index.add_document({"doc_id": "z2", "state": "..."})
    with pytest.raises(TypeError):
        index.matching_ids(["state"])  # type: ignore[arg-type]
    assert index.matching_ids({"state": "Goa"}) == ("z1",)
