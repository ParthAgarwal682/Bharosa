"""Brand query parsing and ranked medicine-candidate search.

IR concept: a two-stage retrieval. Character 2-gram and 3-gram cosine
finds recorded brands that resemble the typed brand. A parametric filter
then keeps a row only when its recorded salt, strength value, strength
unit, and dosage form agree with the matched identity and with every
constraint the query actually stated. Release and combination components
take part only when the rows record them.

The returned list is matching search candidates. The score on each
candidate is the n-gram cosine of its own brand against the parsed
brand. No language model is called. No attribute that a row left blank
is filled in.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bharosa.medicine.ngram import SCHEME, NgramSimilarity, ngram_similarity
from bharosa.medicine.parametric import (
    FORM_TOKENS,
    MedicineRecord,
    RecordedConstraints,
    canonical_form,
    passes_query_constraints,
    read_medicine,
    recorded_conflict,
)
from bharosa.rank.topk import top_k
from bharosa.text.normalize import tokenize

CANDIDATE_LABEL: str = "matching search candidates"
DISCLAIMER: str = "Verify these matching search candidates with a doctor or pharmacist."

# A direct brand match below this cosine is not used as an identity.
# One changed character in a fixture-length brand stays above it.
# A string with no shared slice stays at 0.
MIN_BRAND_COSINE: float = 0.5

# Query scaffolding. Exact token match after the shared normaliser.
# This is not a medicine lexicon and it is not expanded into drug facts.
_QUERY_WRAPPERS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "alternative",
        "alternatives",
        "aur",
        "bata",
        "batao",
        "chahie",
        "chahiye",
        "cheap",
        "cheaper",
        "dawa",
        "dawai",
        "for",
        "generic",
        "hai",
        "hain",
        "in",
        "ka",
        "ke",
        "ki",
        "kitna",
        "ko",
        "koi",
        "kya",
        "liye",
        "me",
        "medicine",
        "mein",
        "mere",
        "milega",
        "milegi",
        "milta",
        "milti",
        "mujhe",
        "of",
        "on",
        "option",
        "options",
        "please",
        "sasta",
        "saste",
        "sasti",
        "se",
        "substitute",
        "substitutes",
        "the",
        "to",
        "wala",
        "wale",
        "wali",
        "with",
        "ya",
    }
)

# Units the parser will attach to a strength token. They are not
# converted into one another. "ug" stays "ug" and does not become "mcg".
_UNITS: frozenset[str] = frozenset({"g", "iu", "mcg", "mg", "ml", "ug"})

# Single tokens that constrain the recorded ``release`` field. The
# constraint is the token itself. It is not rewritten to a phrase, and
# it is not read out of the brand or the dosage form.
_RELEASE_TOKENS: frozenset[str] = frozenset({"cr", "er", "ir", "mr", "sr", "xr"})

_RELEASE_BIGRAMS: dict[tuple[str, str], str] = {
    ("controlled", "release"): "controlled release",
    ("extended", "release"): "extended release",
    ("immediate", "release"): "immediate release",
    ("modified", "release"): "modified release",
    ("sustained", "release"): "sustained release",
}

_STRENGTH_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)(mcg|mg|ml|iu|ug|g)?$")


@dataclass(frozen=True)
class ParsedMedicineQuery:
    """Brand text and the constraints a query stated in so many words.

    IR concept: query understanding before retrieval. Wrapper tokens are
    dropped. A strength, unit, dosage form, or release is kept only when
    a token matched that closed pattern. ``ambiguous`` is set when two
    stated values disagree; the search then returns no candidates rather
    than picking one.
    """

    brand: str
    constraints: RecordedConstraints
    ambiguous: bool

    @property
    def strength_value(self) -> float | None:
        return self.constraints.strength_value

    @property
    def strength_unit(self) -> str | None:
        return self.constraints.strength_unit

    @property
    def form(self) -> str | None:
        return self.constraints.form

    @property
    def release(self) -> str | None:
        return self.constraints.release


@dataclass(frozen=True)
class SearchCandidate:
    """One ranked row that passed the brand cutoff and the recorded filter.

    IR concept: a retrieved document plus its similarity. ``similarity``
    is the character n-gram cosine of this row's brand against the
    parsed query brand. ``rank`` is 1 for the highest cosine. Salt,
    strength, unit, form, release, and component salts are the row's
    own values.
    """

    brand: str
    salt: str
    strength_value: float
    strength_unit: str
    form: str
    release: str | None
    salts: tuple[str, ...] | None
    mrp: float | None
    generic_price: float | None
    manufacturer: str | None
    similarity: NgramSimilarity
    rank: int


@dataclass(frozen=True)
class MedicineSearchResult:
    """Matching search candidates, the score cutoff, and the disclaimer.

    IR concept: a retrieval result set. ``candidates`` is empty when no
    row passed both stages. The label and the disclaimer are still
    present in that case. ``best_brand_cosine`` is the highest brand
    cosine among rows that had a brand, including rows the filter
    removed, so a strength miss can be told apart from an unknown brand.
    """

    label: str
    disclaimer: str
    candidates: tuple[SearchCandidate, ...]
    parsed_brand: str
    parsed_strength_value: float | None
    parsed_strength_unit: str | None
    parsed_form: str | None
    parsed_release: str | None
    query_ambiguous: bool
    minimum_brand_cosine: float
    best_brand_cosine: float | None
    skipped_incomplete_brands: tuple[str, ...]

    def __iter__(self) -> object:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def format_result(self) -> str:
        """Return an inspectable dump of the parse, the cutoff, and the rows.

        IR concept: a verbose retrieval trace. Each candidate line shows
        the cosine and the recorded attributes the filter used. A field
        the row did not have is printed as ``<not recorded>``.
        """
        lines = [
            self.label,
            self.disclaimer,
            f"scheme: {SCHEME}",
            f"minimum_brand_cosine: {self.minimum_brand_cosine:.6f}",
            f"parsed_brand: {self.parsed_brand}",
            f"parsed_strength_value: {_plain(self.parsed_strength_value)}",
            f"parsed_strength_unit: {_plain(self.parsed_strength_unit)}",
            f"parsed_form: {_plain(self.parsed_form)}",
            f"parsed_release: {_plain(self.parsed_release)}",
            f"query_ambiguous: {self.query_ambiguous}",
            f"best_brand_cosine: {_plain_float(self.best_brand_cosine)}",
        ]
        if self.skipped_incomplete_brands:
            lines.append(
                "skipped_incomplete_brands: " + " ".join(self.skipped_incomplete_brands)
            )
        else:
            lines.append("skipped_incomplete_brands:")
        if not self.candidates:
            lines.append("candidates:")
            return "\n".join(lines)
        lines.append("candidates:")
        for candidate in self.candidates:
            lines.append(_format_candidate(candidate))
        return "\n".join(lines)


def parse_medicine_query(query: str) -> ParsedMedicineQuery:
    """Split a query into brand text and stated strength, unit, and form.

    IR concept: query segmentation. Tokens come from the shared
    tokeniser. A number, optionally followed by a unit, is a strength.
    A closed dosage-form word is a form constraint. Wrapper words are
    left out of the brand. Two conflicting strengths, units, forms, or
    releases mark the query ambiguous.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")

    tokens = tokenize(query)
    brand: list[str] = []
    strengths: list[tuple[float, str | None]] = []
    forms: list[str] = []
    releases: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index + 1 < len(tokens):
            release = _RELEASE_BIGRAMS.get((token, tokens[index + 1]))
            if release is not None:
                releases.append(release)
                index += 2
                continue
        if token in _RELEASE_TOKENS:
            releases.append(token)
            index += 1
            continue
        strength = _strength_token(token)
        if strength is not None:
            value, unit = strength
            if unit is None and index + 1 < len(tokens) and tokens[index + 1] in _UNITS:
                unit = tokens[index + 1]
                index += 2
            else:
                index += 1
            strengths.append((value, unit))
            continue
        if token in FORM_TOKENS:
            forms.append(canonical_form(token))
            index += 1
            continue
        if token in _QUERY_WRAPPERS or token in _UNITS:
            index += 1
            continue
        brand.append(token)
        index += 1

    strength_value, strength_unit, strength_ambiguous = _merge_strengths(strengths)
    form, form_ambiguous = _merge_text(forms)
    release, release_ambiguous = _merge_text(releases)
    return ParsedMedicineQuery(
        brand=" ".join(brand),
        constraints=RecordedConstraints(
            strength_value=strength_value,
            strength_unit=strength_unit,
            form=form,
            release=release,
        ),
        ambiguous=strength_ambiguous or form_ambiguous or release_ambiguous,
    )


def search_medicine(
    query: str,
    k: int = 5,
    records: Sequence[Mapping[str, object]] | None = None,
) -> MedicineSearchResult:
    """Return up to ``k`` matching search candidates for ``query``.

    IR concept: ranked retrieval with a parametric filter. When
    ``records`` is omitted, rows are read from ``bharosa.medicine.loader``
    (Kushagra's module) and are not rewritten. Passed-in rows are used
    as given.

    Brand cosine selects the identity. That identity is one recorded
    key: salt, strength value, strength unit, dosage form, plus release
    and ``salts`` when the winning rows recorded them. If the rows above
    the cutoff share that key, every complete row with the key is a
    candidate, including a different brand, and the heap top-k orders
    them by cosine. If those rows disagree on the key, only the rows
    above the cutoff are returned, and no third brand is attached.
    """
    _check_k(k)
    parsed = parse_medicine_query(query)
    rows = _resolve_records(records)
    indexed = [(index, read_medicine(_coerce_row(row))) for index, row in enumerate(rows)]
    return _search_indexed(parsed, indexed, k)


def normalize_corpus_brand(
    brand_text: str | None,
    constraints: RecordedConstraints,
) -> str:
    """Normalize recorded brand for character n-gram similarity ONLY.

    IR concept: token-level constraint alignment. When a query explicitly
    segments out strength tokens (e.g., 650, 650mg) and/or dosage form
    tokens (e.g., tablet), removing those corresponding explicit constraint
    tokens from the recorded brand prevents length-inflation from diluting
    the character n-gram cosine of the core brand name.
    """
    if not brand_text:
        return ""
    tokens = tokenize(brand_text)
    kept: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        st = _strength_token(t)
        if st is not None and constraints.strength_value is not None:
            val, unit = st
            if abs(val - constraints.strength_value) < 1e-6:
                if unit is None and i + 1 < len(tokens) and tokens[i + 1] in _UNITS:
                    i += 2
                    continue
                i += 1
                continue

        if (
            constraints.form is not None
            and t in FORM_TOKENS
            and canonical_form(t) == constraints.form
        ):
            i += 1
            continue

        kept.append(t)
        i += 1

    return " ".join(kept) if kept else brand_text


def _search_indexed(
    parsed: ParsedMedicineQuery,
    indexed: list[tuple[int, MedicineRecord]],
    k: int,
) -> MedicineSearchResult:
    if parsed.ambiguous or parsed.brand == "":
        return _result(parsed, (), best=None, skipped=())

    scored: list[tuple[int, MedicineRecord, NgramSimilarity]] = []
    for index, record in indexed:
        if record.brand is None:
            continue
        sim_brand = normalize_corpus_brand(record.brand, parsed.constraints)
        scored.append((index, record, ngram_similarity(parsed.brand, sim_brand)))

    best = max((sim.cosine for _, _, sim in scored), default=None)
    skipped = tuple(
        record.brand
        for _, record, sim in scored
        if record.brand is not None
        and sim.cosine >= MIN_BRAND_COSINE
        and record.recorded_key() is None
        and not recorded_conflict(record, parsed.constraints)
    )
    eligible = [
        (index, record, sim)
        for index, record, sim in scored
        if passes_query_constraints(record, parsed.constraints)
        and sim.cosine >= MIN_BRAND_COSINE
    ]
    if not eligible:
        return _result(parsed, (), best=best, skipped=skipped)

    best_eligible = max(sim.cosine for _, _, sim in eligible)
    leaders = [
        (index, record, sim)
        for index, record, sim in eligible
        if abs(sim.cosine - best_eligible) <= 1e-12
    ]
    leader_keys = {record.recorded_key() for _, record, _ in leaders}
    if len(leader_keys) == 1:
        identity = next(iter(leader_keys))
        pool = [
            (
                index,
                record,
                _similarity_for(
                    scored, index, record, parsed.brand, parsed.constraints
                ),
            )
            for index, record in indexed
            if record.recorded_key() == identity
            and passes_query_constraints(record, parsed.constraints)
        ]
    else:
        pool = leaders

    winners = top_k(((f"{index:06d}", sim.cosine) for index, _, sim in pool), k)
    by_index = {index: (record, sim) for index, record, sim in pool}
    candidates: list[SearchCandidate] = []
    for rank, (doc_id, _) in enumerate(winners, start=1):
        record, sim = by_index[int(doc_id)]
        candidates.append(_to_candidate(record, sim, rank))
    return _result(parsed, tuple(candidates), best=best, skipped=skipped)


def _similarity_for(
    scored: list[tuple[int, MedicineRecord, NgramSimilarity]],
    index: int,
    record: MedicineRecord,
    brand: str,
    constraints: RecordedConstraints | None = None,
) -> NgramSimilarity:
    """Reuse a cosine already computed for this row index."""
    for scored_index, _, sim in scored:
        if scored_index == index:
            return sim
    if record.brand is None:
        return ngram_similarity(brand, "")
    sim_brand = (
        normalize_corpus_brand(record.brand, constraints)
        if constraints is not None
        else record.brand
    )
    return ngram_similarity(brand, sim_brand)


def _to_candidate(record: MedicineRecord, sim: NgramSimilarity, rank: int) -> SearchCandidate:
    key = record.recorded_key()
    if key is None or record.brand is None:
        raise RuntimeError("a candidate requires a complete recorded identity")
    return SearchCandidate(
        brand=record.brand,
        salt=key.salt,
        strength_value=key.strength_value,
        strength_unit=key.strength_unit,
        form=key.form,
        release=key.release,
        salts=key.salts,
        mrp=record.mrp,
        generic_price=record.generic_price,
        manufacturer=record.manufacturer,
        similarity=sim,
        rank=rank,
    )


def _result(
    parsed: ParsedMedicineQuery,
    candidates: tuple[SearchCandidate, ...],
    *,
    best: float | None,
    skipped: tuple[str, ...],
) -> MedicineSearchResult:
    return MedicineSearchResult(
        label=CANDIDATE_LABEL,
        disclaimer=DISCLAIMER,
        candidates=candidates,
        parsed_brand=parsed.brand,
        parsed_strength_value=parsed.strength_value,
        parsed_strength_unit=parsed.strength_unit,
        parsed_form=parsed.form,
        parsed_release=parsed.release,
        query_ambiguous=parsed.ambiguous,
        minimum_brand_cosine=MIN_BRAND_COSINE,
        best_brand_cosine=best,
        skipped_incomplete_brands=skipped,
    )


def _resolve_records(
    records: Sequence[Mapping[str, object]] | None,
) -> list[object]:
    """Use the caller's rows, or the loader's rows when none were passed."""
    if records is None:
        return _records_from_loader()
    if isinstance(records, (str, Mapping)):
        raise TypeError(
            "records must be a sequence of medicine rows, "
            f"got {type(records).__name__}"
        )
    return list(records)


def _records_from_loader() -> list[object]:
    """Call Kushagra's loader. Do not invent rows if it is absent.

    The first callable among ``load_medicines``, ``load_records``,
    ``load_medicine_records``, and ``load`` is used, with no arguments.
    """
    try:
        from bharosa.medicine import loader as loader_module
    except ImportError as exc:
        raise RuntimeError(
            "medicine records were not provided and bharosa.medicine.loader "
            "could not be imported"
        ) from exc

    for name in ("load_medicines", "load_records", "load_medicine_records", "load"):
        fn = getattr(loader_module, name, None)
        if not callable(fn):
            continue
        loaded = fn()
        if isinstance(loaded, (str, Mapping)):
            raise TypeError(
                f"loader.{name}() must return a sequence of medicine rows, "
                f"got {type(loaded).__name__}"
            )
        return list(loaded)
    raise RuntimeError(
        "bharosa.medicine.loader has no load_medicines(), load_records(), "
        "load_medicine_records(), or load()"
    )


def _coerce_row(row: object) -> Mapping[str, object]:
    """View a loader row as a mapping without adding fields."""
    if isinstance(row, Mapping):
        return row
    fields = getattr(type(row), "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return {name: getattr(row, name) for name in fields}
    raw = getattr(row, "__dict__", None)
    if isinstance(raw, dict):
        return {
            key: value
            for key, value in raw.items()
            if isinstance(key, str) and not key.startswith("_")
        }
    raise TypeError(f"medicine record must be a mapping, got {type(row).__name__}")


def _strength_token(token: str) -> tuple[float, str | None] | None:
    """Parse a single strength token. ``40`` and ``40mg`` both match."""
    match = _STRENGTH_TOKEN.fullmatch(token)
    if match is None:
        return None
    value = round(float(match.group(1)), 6)
    unit = match.group(2)
    return value, unit


def _merge_strengths(
    strengths: list[tuple[float, str | None]],
) -> tuple[float | None, str | None, bool]:
    """One strength and one unit, or ambiguous when the statements disagree.

    A later bare number does not overwrite a unit already seen for the
    same value. ``40`` followed by ``40 mg`` is 40 mg. ``40 mg`` followed
    by ``20`` is ambiguous.
    """
    if not strengths:
        return None, None, False
    value = strengths[0][0]
    unit = strengths[0][1]
    for next_value, next_unit in strengths[1:]:
        if next_value != value:
            return None, None, True
        if next_unit is None:
            continue
        if unit is None:
            unit = next_unit
            continue
        if next_unit != unit:
            return None, None, True
    return value, unit, False


def _merge_text(values: list[str]) -> tuple[str | None, bool]:
    """Keep one stated string. Two different strings are ambiguous."""
    if not values:
        return None, False
    first = values[0]
    for value in values[1:]:
        if value != first:
            return None, True
    return first, False


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 0:
        raise ValueError("k must be non-negative")


def _format_candidate(candidate: SearchCandidate) -> str:
    strength = _format_number(candidate.strength_value)
    salts = "<not recorded>" if candidate.salts is None else " ".join(candidate.salts)
    release = "<not recorded>" if candidate.release is None else candidate.release
    return (
        f"{candidate.rank}  brand={candidate.brand}  "
        f"cosine={candidate.similarity.cosine:.6f}  "
        f"salt={candidate.salt}  "
        f"strength={strength} {candidate.strength_unit}  "
        f"form={candidate.form}  "
        f"release={release}  "
        f"salts={salts}"
    )


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value)


def _plain(value: object) -> str:
    if value is None:
        return "<not recorded>"
    return str(value)


def _plain_float(value: float | None) -> str:
    if value is None:
        return "<not recorded>"
    return f"{value:.6f}"
