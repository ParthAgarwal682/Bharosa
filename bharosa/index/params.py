"""Parametric index over zone-document metadata.

IR concept: a parametric index. Each structured field value maps to the
documents that carry it, and a query keeps only the documents in the
intersection of those lists. This module does not score text.

The zone contract (README section 10) is the only metadata this index
reads. Filterable fields are the structured ones in that contract:
``state``, ``conditions``, and ``zone``. ``condition`` singular, income,
age, salt, strength, and form are not fields of a zone document, so they
are rejected rather than stored. ``g_score``, timestamps, ``url``,
``domain``, ``title``, and ``text`` are contract fields and may be
present on a document, but they are not parametric filters. Nothing in
``text`` is copied into ``state`` or ``conditions``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from bharosa.text.normalize import normalize

# Keys of the zone document agreed in the project README. Do not extend
# this set here. A new field belongs in the contract first.
ZONE_CONTRACT_FIELDS: frozenset[str] = frozenset(
    {
        "doc_id",
        "url",
        "domain",
        "title",
        "zone",
        "text",
        "state",
        "conditions",
        "g_score",
        "crawled_at",
        "last_changed_at",
        "content_hash",
    }
)

# Structured fields a parametric query is allowed to constrain.
PARAMETRIC_FIELDS: tuple[str, ...] = ("state", "conditions", "zone")

# Closed zone vocabulary from the contract. Order is the contract's order.
ZONES: tuple[str, ...] = (
    "eligibility",
    "benefits",
    "documents",
    "how_to_apply",
    "other",
)

_ZONE_SET = frozenset(ZONES)
_MISSING = "<missing>"
_PARAMETRIC_LIST = "state, conditions, zone"


@dataclass(frozen=True)
class ZoneMeta:
    """Parametric fields stored for one zone document.

    IR concept: the document's structured attributes, after analysis.
    ``state`` and ``zone`` are ``None`` only when the document did not
    provide them. An empty ``conditions`` tuple means the document listed
    no conditions. Neither is filled from other fields.
    """

    doc_id: str
    state: str | None
    conditions: tuple[str, ...]
    zone: str | None


class ParametricIndex:
    """Field-value postings for zone metadata.

    IR concept: parametric postings. ``state`` and ``zone`` each map a
    normalised value to document ids. ``conditions`` is multi-valued:
    a document is listed under every condition it actually contains.
    Document ids stay in the order they were added.
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._docs: dict[str, ZoneMeta] = {}
        self._by_state: dict[str, list[str]] = {}
        self._missing_state: list[str] = []
        self._by_condition: dict[str, list[str]] = {}
        self._by_zone: dict[str, list[str]] = {}
        self._missing_zone: list[str] = []

    def add_document(self, document: Mapping[str, object]) -> None:
        """Index one zone document's contract metadata.

        IR concept: building a parametric posting. Only ``doc_id``,
        ``state``, ``conditions``, and ``zone`` are stored. Other
        contract keys are accepted and ignored. A key outside the zone
        contract raises ``ValueError``.
        """
        meta = _read_document(document)
        if meta.doc_id in self._docs:
            raise ValueError(f"document already indexed: {meta.doc_id}")
        self._order.append(meta.doc_id)
        self._docs[meta.doc_id] = meta
        if meta.state is None:
            self._missing_state.append(meta.doc_id)
        else:
            self._by_state.setdefault(meta.state, []).append(meta.doc_id)
        for condition in meta.conditions:
            bucket = self._by_condition.setdefault(condition, [])
            if not bucket or bucket[-1] != meta.doc_id:
                bucket.append(meta.doc_id)
        if meta.zone is None:
            self._missing_zone.append(meta.doc_id)
        else:
            self._by_zone.setdefault(meta.zone, []).append(meta.doc_id)

    def add_documents(self, documents: Iterable[Mapping[str, object]]) -> None:
        """Index zone documents in iteration order.

        A single mapping is rejected so a zone document's keys are not
        iterated as if they were documents.
        """
        if isinstance(documents, (str, Mapping)):
            raise TypeError(
                "documents must be a sequence of zone documents, "
                f"got {type(documents).__name__}"
            )
        for document in documents:
            self.add_document(document)

    def document_ids(self) -> tuple[str, ...]:
        """Document ids in the order they were added."""
        return tuple(self._order)

    def metadata(self, doc_id: str) -> ZoneMeta:
        """Stored parametric fields for one indexed document."""
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
        try:
            return self._docs[doc_id]
        except KeyError:
            raise KeyError(doc_id) from None

    def postings(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Field, then value, then document ids.

        IR concept: the parametric dictionary. ``<missing>`` is the
        bucket for a null ``state`` or ``zone``. Conditions have no
        missing bucket: a document with none is simply absent there.
        """
        return {
            "state": _buckets(self._by_state, self._missing_state),
            "conditions": _buckets(self._by_condition, ()),
            "zone": _buckets(self._by_zone, self._missing_zone),
        }

    def matching_ids(self, filters: Mapping[str, object] | None = None) -> tuple[str, ...]:
        """Document ids that satisfy every supplied constraint.

        IR concept: Boolean intersection of parametric postings. Fields
        are combined with AND. Every requested condition must be present
        (also AND). A missing filter key does not constrain that field.
        ``state=None`` or ``zone=None`` selects documents whose field is
        null; it does not mean "skip this field". An empty ``conditions``
        sequence does not constrain conditions.

        Results follow index order. An unknown state or condition matches
        nobody. An unknown zone string raises, because zone is a closed
        vocabulary in the contract. Keys that are not parametric fields
        raise, including contract fields such as ``g_score`` and keys
        that are not in the contract at all.
        """
        selected = list(self._order)
        if filters is None:
            return tuple(selected)
        _validate_filters(filters)
        if "state" in filters:
            selected = _keep(selected, self._state_ids(filters["state"]))
        if "zone" in filters:
            selected = _keep(selected, self._zone_ids(filters["zone"]))
        if "conditions" in filters:
            wanted = _condition_list(filters["conditions"])
            for condition in wanted:
                selected = _keep(selected, set(self._by_condition.get(condition, ())))
        return tuple(selected)

    def format_index(self) -> str:
        """Return an inspectable dump of the parametric postings.

        IR concept: a verbose dictionary. Values are the normalised forms
        used for matching, so a demo can see why ``UP`` meets ``up``.
        """
        lines = [f"parametric fields: {_PARAMETRIC_LIST}"]
        posted = self.postings()
        for field in PARAMETRIC_FIELDS:
            lines.append(field)
            buckets = posted[field]
            if not buckets:
                lines.append("  <empty>")
                continue
            for value in sorted(buckets):
                lines.append(f"  {value}  {' '.join(buckets[value])}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ParametricIndex(documents={len(self._order)})"

    def _state_ids(self, value: object) -> set[str]:
        if value is None:
            return set(self._missing_state)
        key = _normalize_field(value, "state")
        return set(self._by_state.get(key, ()))

    def _zone_ids(self, value: object) -> set[str]:
        if value is None:
            return set(self._missing_zone)
        key = _normalize_zone(value)
        return set(self._by_zone.get(key, ()))


def _read_document(document: Mapping[str, object]) -> ZoneMeta:
    if not isinstance(document, Mapping):
        raise TypeError(
            f"document must be a mapping, got {type(document).__name__}"
        )
    unknown = [key for key in document if key not in ZONE_CONTRACT_FIELDS]
    if unknown:
        names = ", ".join(str(key) for key in sorted(unknown, key=str))
        raise ValueError(f"not in the zone contract: {names}")
    if "doc_id" not in document:
        raise ValueError("doc_id is required")
    doc_id = document["doc_id"]
    if not isinstance(doc_id, str):
        raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
    if doc_id == "":
        raise ValueError("doc_id must be non-empty")

    state = None
    if "state" in document and document["state"] is not None:
        state = _normalize_field(document["state"], "state")
    conditions: tuple[str, ...] = ()
    if "conditions" in document and document["conditions"] is not None:
        conditions = _condition_list(document["conditions"])
    zone = None
    if "zone" in document and document["zone"] is not None:
        zone = _normalize_zone(document["zone"])
    return ZoneMeta(doc_id=doc_id, state=state, conditions=conditions, zone=zone)


def _validate_filters(filters: Mapping[str, object]) -> None:
    if not isinstance(filters, Mapping):
        raise TypeError(f"filters must be a mapping, got {type(filters).__name__}")
    for key in filters:
        if not isinstance(key, str):
            raise TypeError(f"filter key must be str, got {type(key).__name__}")
        if key in PARAMETRIC_FIELDS:
            continue
        if key in ZONE_CONTRACT_FIELDS:
            raise ValueError(
                f"{key} is zone-contract metadata but is not a parametric field; "
                f"parametric fields are {_PARAMETRIC_LIST}"
            )
        raise ValueError(
            f"{key} is not in the zone contract; parametric fields are {_PARAMETRIC_LIST}"
        )


def _condition_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "conditions must be a list of strings, "
            f"got {type(value).__name__}"
        )
    seen: list[str] = []
    for item in value:
        normalised = _normalize_field(item, "conditions")
        if normalised not in seen:
            seen.append(normalised)
    return tuple(seen)


def _normalize_field(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    normalised = normalize(value)
    if not normalised:
        raise ValueError(f"{field} is empty after normalisation")
    return normalised


def _normalize_zone(value: object) -> str:
    """Case-fold a zone token without the text tokeniser.

    IR concept: a closed parametric vocabulary. The contract spells
    ``how_to_apply`` with underscores. The shared text normaliser would
    turn that underscore into a token boundary, so zone values are only
    NFC-case-folded and stripped. Unknown strings are not rewritten to
    ``other``.
    """
    if not isinstance(value, str):
        raise TypeError(f"zone must be str, got {type(value).__name__}")
    folded = unicodedata.normalize("NFC", value).casefold()
    folded = unicodedata.normalize("NFC", folded).strip()
    if folded not in _ZONE_SET:
        allowed = ", ".join(ZONES)
        raise ValueError(f"zone must be one of {allowed}; got {value!r}")
    return folded


def _buckets(
    by_value: Mapping[str, list[str]],
    missing: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    posted = {value: tuple(doc_ids) for value, doc_ids in by_value.items()}
    missing_ids = tuple(missing)
    if missing_ids:
        posted[_MISSING] = missing_ids
    return posted


def _keep(selected: list[str], allowed: set[str]) -> list[str]:
    return [doc_id for doc_id in selected if doc_id in allowed]
