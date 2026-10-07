"""Zone documents and scheme retrieval over the existing IR stack.

IR concept: a zone index. Each zone is its own document. This module
does not keep a second postings list, a second tf-idf formula, or a
second net-score formula. It stores the zone contract, builds the
existing inverted, positional, and parametric indexes, and ranks with
``TfidfRanker`` and ``NetScorer``. A quoted query adds an exact-phrase
constraint from ``PositionalIndex`` and still requires a positive
lnc.ltc cosine. The ``bm25`` flag selects ``BM25Baseline`` instead.
That library score is stored separately and is not passed into
``NetScorer``.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from bharosa.index.inverted import InvertedIndex
from bharosa.index.params import ZONE_CONTRACT_FIELDS, ZONES, ParametricIndex
from bharosa.index.positional import PositionalIndex
from bharosa.rank.bm25 import BM25Baseline, BM25Hit
from bharosa.rank.netscore import (
    DocumentSignals,
    NetScore,
    NetScoreConfig,
    NetScoreWeights,
    NetScorer,
)
from bharosa.rank.tfidf import TfidfRanker
from bharosa.text.hinglish import HinglishLexicon
from bharosa.text.normalize import tokenize

# Coefficients used only when the caller does not pass NetScoreWeights.
# One leaves an enabled feature unchanged. These numbers are not a fit
# and they are not a recommended setting.
_UNWEIGHTED = NetScoreWeights(1.0, 1.0, 1.0)

_INTEGRATION_FLAGS = frozenset({"hinglish", "bm25"})
_NET_FLAGS = frozenset({"g_score", "freshness", "zone_weights"})
_REQUIRED_TEXT = ("doc_id", "url", "domain", "title", "zone", "text", "content_hash")

LexiconInput = HinglishLexicon | Mapping[str, str | Sequence[str]]


@dataclass(frozen=True)
class ZoneDoc:
    """One zone of a crawled page, in the README zone-document contract.

    IR concept: the retrieval unit for scheme search. ``zone`` is one of
    ``eligibility``, ``benefits``, ``documents``, ``how_to_apply``, and
    ``other``. ``g_score``, ``crawled_at``, and ``last_changed_at`` may
    be ``None``. A missing quality score stays missing, and a missing
    timestamp is not invented or copied from the other timestamp.
    ``conditions`` is stored as a tuple. :meth:`as_contract` returns
    that sequence as a list, which is the external zone contract.
    Crawler columns such as ``fetched_at``, ``detected_at``, and
    ``text_path`` are not fields of this record.
    """

    doc_id: str
    url: str
    domain: str
    title: str
    zone: str
    text: str
    state: str | None
    conditions: tuple[str, ...]
    g_score: float | None
    crawled_at: str | None
    last_changed_at: str | None
    content_hash: str

    def __post_init__(self) -> None:
        doc_id = _require_text("doc_id", self.doc_id)
        if doc_id == "":
            raise ValueError("doc_id must be non-empty")
        object.__setattr__(self, "doc_id", doc_id)
        for field in ("url", "domain", "title", "text", "content_hash"):
            object.__setattr__(self, field, _require_text(field, getattr(self, field)))
        object.__setattr__(self, "zone", _canonical_zone(self.zone))
        object.__setattr__(self, "conditions", _as_conditions(self.conditions))
        if self.state is not None and not isinstance(self.state, str):
            raise TypeError(f"state must be str or None, got {type(self.state).__name__}")
        object.__setattr__(self, "g_score", _optional_score(self.g_score))
        _require_timestamp("crawled_at", self.crawled_at)
        _require_timestamp("last_changed_at", self.last_changed_at)
        DocumentSignals.from_zone(self.as_contract(), cosine=0.0)
        ParametricIndex().add_document(self.as_contract())

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> ZoneDoc:
        """Build a zone document from the contract mapping.

        IR concept: the boundary between a stored record and the index.
        Keys outside the zone contract raise, including crawler columns
        ``fetched_at``, ``detected_at``, and ``text_path``. Omitted
        ``g_score``, ``state``, ``crawled_at``, and ``last_changed_at``
        stay missing. ``None`` is not replaced with ``0`` or with the
        other timestamp.
        """
        if not isinstance(document, Mapping):
            raise TypeError(
                f"document must be a mapping, got {type(document).__name__}"
            )
        raw = dict(document)
        unknown = [key for key in raw if key not in ZONE_CONTRACT_FIELDS]
        if unknown:
            names = ", ".join(str(key) for key in sorted(unknown, key=str))
            raise ValueError(f"not in the zone contract: {names}")
        for name in _REQUIRED_TEXT:
            if name not in raw:
                raise ValueError(f"{name} is required")
        conditions: object = ()
        if "conditions" in raw and raw["conditions"] is not None:
            conditions = raw["conditions"]
        return cls(
            doc_id=raw["doc_id"],  # type: ignore[arg-type]
            url=raw["url"],  # type: ignore[arg-type]
            domain=raw["domain"],  # type: ignore[arg-type]
            title=raw["title"],  # type: ignore[arg-type]
            zone=raw["zone"],  # type: ignore[arg-type]
            text=raw["text"],  # type: ignore[arg-type]
            state=raw["state"] if "state" in raw else None,  # type: ignore[arg-type]
            conditions=conditions,  # type: ignore[arg-type]
            g_score=raw["g_score"] if "g_score" in raw else None,  # type: ignore[arg-type]
            crawled_at=raw["crawled_at"] if "crawled_at" in raw else None,  # type: ignore[arg-type]
            last_changed_at=(
                raw["last_changed_at"] if "last_changed_at" in raw else None
            ),  # type: ignore[arg-type]
            content_hash=raw["content_hash"],  # type: ignore[arg-type]
        )

    def as_contract(self) -> dict[str, object]:
        """Return the README zone-document mapping for this record."""
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "zone": self.zone,
            "text": self.text,
            "state": self.state,
            "conditions": list(self.conditions),
            "g_score": self.g_score,
            "crawled_at": self.crawled_at,
            "last_changed_at": self.last_changed_at,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ZoneHit:
    """One ranked zone for the RAG layer.

    IR concept: a retrieval result. ``text``, ``url``, and ``zone`` are
    what a citation needs. ``last_changed_at`` is the source zone's
    change time, or ``None`` when that zone did not have one.
    ``cosine`` is the lnc.ltc score from ``TfidfRanker``. ``g_component``
    is NetScore's weighted g(d) term when that switch is on, and 0 when
    the switch is off. ``freshness`` is the freshness value when that
    switch is on, and 0 when it is off. ``zone_weight`` is the caller's
    multiplier when that switch is on, and 1 when the sum is not
    multiplied. ``net`` is ``NetScorer``'s score, or ``None`` when this
    hit was ranked by the BM25 baseline. ``bm25_score`` is that baseline
    score, or ``None`` on the net-score path. ``rank`` starts at 1.
    """

    doc_id: str
    text: str
    url: str
    zone: str
    last_changed_at: str | None
    cosine: float
    g_component: float
    freshness: float
    zone_weight: float
    net: float | None
    bm25_score: float | None
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(self.doc_id).__name__}")
        if self.doc_id == "":
            raise ValueError("doc_id must be non-empty")
        object.__setattr__(self, "text", _require_text("text", self.text))
        object.__setattr__(self, "url", _require_text("url", self.url))
        object.__setattr__(self, "zone", _canonical_zone(self.zone))
        _require_timestamp("last_changed_at", self.last_changed_at)
        for field in ("cosine", "g_component", "freshness", "zone_weight"):
            object.__setattr__(self, field, _finite(field, getattr(self, field)))
        object.__setattr__(self, "net", _optional_finite("net", self.net))
        object.__setattr__(
            self, "bm25_score", _optional_finite("bm25_score", self.bm25_score)
        )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(f"rank must be int, got {type(self.rank).__name__}")
        if self.rank < 1:
            raise ValueError("rank must be 1 or greater")


@dataclass(frozen=True)
class _PreparedQuery:
    """Analysed query text, plus the raw phrase when the user quoted it."""

    scoring_text: str
    phrase_text: str | None


class ZoneIndex:
    """Inverted, positional, and parametric indexes over zone documents.

    IR concept: the scheme-side zone index. Postings, phrase positions,
    and structured filters are the existing indexes. This object only
    chooses which documents they see.
    """

    def __init__(self, documents: Iterable[ZoneDoc | Mapping[str, object]]) -> None:
        if isinstance(documents, (str, Mapping, ZoneDoc)):
            raise TypeError(
                "documents must be a sequence of zone documents, "
                f"got {type(documents).__name__}"
            )
        docs: list[ZoneDoc] = []
        seen: set[str] = set()
        for document in documents:
            zone_doc = (
                document if isinstance(document, ZoneDoc) else ZoneDoc.from_mapping(document)
            )
            if zone_doc.doc_id in seen:
                raise ValueError(f"document already indexed: {zone_doc.doc_id}")
            seen.add(zone_doc.doc_id)
            docs.append(zone_doc)
        self._docs = tuple(docs)
        self._by_id = {doc.doc_id: doc for doc in docs}
        texts = {doc.doc_id: doc.text for doc in docs}
        self._inverted = InvertedIndex.from_documents(texts)
        self._positional = PositionalIndex.from_documents(texts)
        self._parametric = ParametricIndex()
        self._parametric.add_documents(doc.as_contract() for doc in docs)
        self._tfidf = TfidfRanker(self._inverted)

    def search(
        self,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 5,
        flags: Mapping[str, object] | None = None,
        *,
        weights: NetScoreWeights | None = None,
        zone_weights: Mapping[str, float] | None = None,
        freshness_scale_days: float | None = None,
        as_of: datetime | str | None = None,
        lexicon: LexiconInput | None = None,
    ) -> list[ZoneHit]:
        """Return at most ``k`` zones for ``query``.

        IR concept: retrieve, then score. The query is analysed with
        ``tokenize``. When ``flags["hinglish"]`` is true, that analysis
        is ``HinglishLexicon.expand`` and nothing else. ``filters`` is
        passed to the parametric index (``state``, ``conditions``,
        ``zone``). Candidates on the default path are documents with a
        positive lnc.ltc cosine. A query wrapped in double quotes must
        also be an exact phrase in the positional index. Final order on
        that path is ``NetScorer`` over those cosines.

        ``flags["bm25"]`` selects ``BM25Baseline`` for the candidate
        list and the order. The library score is ``bm25_score``.
        ``net`` stays ``None`` on that path. Net-score switches cannot
        be combined with it.

        No row is added when nothing matches. ``k == 0`` returns an
        empty list. Ties on the net-score path follow ``NetScorer``:
        higher score first, then the smaller ``doc_id``.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        _check_k(k)
        if weights is not None and not isinstance(weights, NetScoreWeights):
            raise TypeError(
                f"weights must be NetScoreWeights, got {type(weights).__name__}"
            )
        integration, net_flags = _split_flags(flags)
        _reject_bm25_mix(integration, net_flags)
        if weights is None and _net_features_on(net_flags) and not integration["bm25"]:
            raise ValueError(
                "NetScoreWeights are required when g_score, freshness, or "
                "zone_weights is on; coefficients are not invented"
            )
        prepared = _prepare_query(
            query, hinglish=integration["hinglish"], lexicon=lexicon
        )
        allowed = set(self._parametric.matching_ids(filters))
        if integration["bm25"]:
            if k == 0 or prepared is None:
                return []
            return self._bm25_hits(prepared, allowed, k)

        scorer = NetScorer(
            _net_config(
                net_flags,
                weights if weights is not None else _UNWEIGHTED,
                zone_weights=zone_weights,
                freshness_scale_days=freshness_scale_days,
            )
        )
        candidates = (
            {} if prepared is None else self._net_candidates(prepared, allowed)
        )
        signals = [
            DocumentSignals.from_zone(
                self._by_id[doc_id].as_contract(), cosine=candidates[doc_id]
            )
            for doc_id in candidates
        ]
        ranking = scorer.rank(signals, k, as_of=as_of)
        hits: list[ZoneHit] = []
        for rank, scored in enumerate(ranking.ranked, start=1):
            doc = self._by_id[scored.doc_id]
            hits.append(_hit_from_net(doc, candidates[scored.doc_id], scored, rank))
        return hits

    def __repr__(self) -> str:
        return f"ZoneIndex(documents={len(self._docs)})"

    def _net_candidates(
        self, prepared: _PreparedQuery, allowed: set[str]
    ) -> dict[str, float]:
        """Positive lnc.ltc scores, optionally restricted to an exact phrase.

        IR concept: vector-space retrieval. A cosine of 0 is the
        ranker's miss and is not kept. When the raw query was quoted,
        the positional index also drops documents that contain the words
        out of order or with a gap. A phrase match with cosine 0 is
        still a miss.
        """
        if prepared.phrase_text is not None:
            found: dict[str, float] = {}
            for match in self._positional.find_phrase(prepared.phrase_text):
                if match.doc_id not in allowed:
                    continue
                cosine = self._tfidf.cosine(prepared.scoring_text, match.doc_id)
                if cosine > 0.0:
                    found[match.doc_id] = cosine
            return found
        found = {}
        for scored in self._tfidf.scores(prepared.scoring_text):
            if scored.doc_id in allowed and scored.score > 0.0:
                found[scored.doc_id] = scored.score
        return found

    def _bm25_hits(
        self,
        prepared: _PreparedQuery,
        allowed: set[str],
        k: int,
    ) -> list[ZoneHit]:
        """Rank with the external BM25 baseline after the same filters.

        IR concept: a baseline run with the same query analysis and the
        same parametric constraints. ``bm25_score`` is the library score.
        ``net`` is ``None`` because this path does not call ``NetScorer``.
        ``cosine`` stays the lnc.ltc cosine. A quoted phrase still has
        to have a positive cosine.
        """
        baseline = BM25Baseline({doc.doc_id: doc.text for doc in self._docs})
        phrase_ids: set[str] | None = None
        if prepared.phrase_text is not None:
            phrase_ids = {
                match.doc_id
                for match in self._positional.find_phrase(prepared.phrase_text)
            }
        chosen: list[tuple[BM25Hit, float]] = []
        for hit in baseline.rank(prepared.scoring_text, len(self._docs)):
            if hit.doc_id not in allowed:
                continue
            if phrase_ids is not None and hit.doc_id not in phrase_ids:
                continue
            cosine = self._tfidf.cosine(prepared.scoring_text, hit.doc_id)
            if phrase_ids is not None and cosine <= 0.0:
                continue
            chosen.append((hit, cosine))
            if len(chosen) == k:
                break
        results: list[ZoneHit] = []
        for rank, (hit, cosine) in enumerate(chosen, start=1):
            doc = self._by_id[hit.doc_id]
            results.append(
                ZoneHit(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    url=doc.url,
                    zone=doc.zone,
                    last_changed_at=doc.last_changed_at,
                    cosine=cosine,
                    g_component=0.0,
                    freshness=0.0,
                    zone_weight=1.0,
                    net=None,
                    bm25_score=hit.score,
                    rank=rank,
                )
            )
        return results


def search_schemes(
    query: str,
    filters: Mapping[str, object] | None = None,
    k: int = 5,
    flags: Mapping[str, object] | None = None,
    *,
    documents: Sequence[ZoneDoc | Mapping[str, object]] | None = None,
    weights: NetScoreWeights | None = None,
    zone_weights: Mapping[str, float] | None = None,
    freshness_scale_days: float | None = None,
    as_of: datetime | str | None = None,
    lexicon: LexiconInput | None = None,
) -> list[ZoneHit]:
    """Return up to ``k`` ranked zones for ``query``.

    IR concept: the Module C search signature. ``filters`` and ``flags``
    match the contract. ``documents`` is the zone collection to index.
    Omitting it raises, because there is no loader here and an empty
    result would look like a real miss. An empty list is an empty
    corpus and returns no hits.

    ``weights``, ``zone_weights``, ``freshness_scale_days``, ``as_of``,
    and ``lexicon`` are keyword-only. They are not added to
    ``NetScorer``'s flag set. ``flags`` may contain ``hinglish`` and
    ``bm25`` as well as ``g_score``, ``freshness``, and ``zone_weights``.
    Only the last three are forwarded to ``NetScoreConfig.from_flags``.
    When any of those three is on, ``weights`` is required. When all
    three are off, an omitted ``weights`` uses coefficient 1 so the
    score is the cosine. That coefficient is not a fit. A missing
    zone-weight table, freshness scale, or ``as_of`` is still an error
    when that switch is on. This function does not read the clock.

    ``flags["hinglish"]`` with no ``lexicon`` loads the reviewed CSV
    through ``HinglishLexicon.load``. A missing file raises
    ``FileNotFoundError`` from that loader.
    """
    if documents is None:
        raise ValueError(
            "a zone document corpus is required; pass documents. "
            "An empty list means the corpus contains no documents."
        )
    return ZoneIndex(documents).search(
        query,
        filters,
        k,
        flags,
        weights=weights,
        zone_weights=zone_weights,
        freshness_scale_days=freshness_scale_days,
        as_of=as_of,
        lexicon=lexicon,
    )


def _hit_from_net(doc: ZoneDoc, cosine: float, scored: NetScore, rank: int) -> ZoneHit:
    """Copy one complete net score onto a zone hit.

    A disabled feature contributes nothing. It is reported as 0, or as
    a zone multiplier of 1, rather than as a fill-in for a missing
    document value. Incomplete scores are not passed here.
    """
    parts = {part.component: part for part in scored.contributions}
    g_part = parts["g_score"]
    fresh_part = parts["freshness"]
    zone_part = parts["zone_weight"]
    if scored.score is None:
        raise RuntimeError("complete net score is missing a number")
    if g_part.status == "used" and g_part.product is not None:
        g_component = g_part.product
    else:
        g_component = 0.0
    if fresh_part.status == "used" and fresh_part.value is not None:
        freshness = fresh_part.value
    else:
        freshness = 0.0
    if zone_part.status == "used" and zone_part.weight is not None:
        zone_weight = zone_part.weight
    else:
        zone_weight = 1.0
    return ZoneHit(
        doc_id=doc.doc_id,
        text=doc.text,
        url=doc.url,
        zone=doc.zone,
        cosine=cosine,
        g_component=g_component,
        freshness=freshness,
        zone_weight=zone_weight,
        last_changed_at=doc.last_changed_at,
        net=scored.score,
        bm25_score=None,
        rank=rank,
    )


def _net_config(
    net_flags: Mapping[str, bool] | None,
    weights: NetScoreWeights,
    *,
    zone_weights: Mapping[str, float] | None,
    freshness_scale_days: float | None,
) -> NetScoreConfig:
    """Build a NetScore config without handing it integration flags.

    Zone weights and the freshness scale are passed only when that
    switch is on. ``NetScoreConfig`` rejects them when the switch is off.
    """
    use_zone = bool(net_flags and net_flags.get("zone_weights"))
    use_fresh = bool(net_flags and net_flags.get("freshness"))
    return NetScoreConfig.from_flags(
        weights,
        net_flags,
        zone_weights=zone_weights if use_zone else None,
        freshness_scale_days=freshness_scale_days if use_fresh else None,
    )


def _prepare_query(
    query: str,
    *,
    hinglish: bool,
    lexicon: LexiconInput | None,
) -> _PreparedQuery | None:
    """Normalise ``query`` and, when asked, append reviewed lexicon tokens.

    IR concept: query analysis shared with the index. Expansion goes
    through ``HinglishLexicon.expand`` once. The returned scoring text
    is those tokens joined, so the ranker tokenises the same sequence
    and does not take a second hop. A raw query wrapped in double quotes
    is also an exact phrase; expansion does not rewrite that phrase.
    An analysis with no tokens matches nothing.
    """
    phrase = _quoted_phrase(query)
    source = phrase if phrase is not None else query
    if hinglish:
        tokens = _reviewed_lexicon(lexicon).expand(source).query_tokens
    else:
        tokens = tuple(tokenize(source))
    if not tokens:
        return None
    return _PreparedQuery(" ".join(tokens), phrase)


def _reviewed_lexicon(lexicon: LexiconInput | None) -> HinglishLexicon:
    """Return the caller's lexicon, or load the reviewed CSV.

    IR concept: controlled query expansion. When the caller does not
    pass a lexicon, ``HinglishLexicon.load`` reads the reviewed file.
    A missing file raises ``FileNotFoundError``. This function does not
    invent entries and does not replace that error.
    """
    if lexicon is None:
        return HinglishLexicon.load()
    if isinstance(lexicon, HinglishLexicon):
        return lexicon
    return HinglishLexicon(lexicon)


def _quoted_phrase(query: str) -> str | None:
    """Return the inside of a query wrapped in double quotes.

    IR concept: a phrase query. Quotes are detected on the raw string
    because the shared normaliser turns them into token boundaries.
    Any other query is bag-of-words.
    """
    stripped = query.strip()
    if len(stripped) < 2 or stripped[0] != '"' or stripped[-1] != '"':
        return None
    inner = stripped[1:-1].strip()
    if inner == "":
        return None
    return inner


def _split_flags(
    flags: Mapping[str, object] | None,
) -> tuple[dict[str, bool], dict[str, bool] | None]:
    """Separate integration switches from net-score switches.

    IR concept: ablation flags. ``hinglish`` and ``bm25`` stay here.
    ``g_score``, ``freshness``, and ``zone_weights`` are returned for
    ``NetScoreConfig.from_flags``. An unknown name raises.
    """
    if flags is None:
        return {"hinglish": False, "bm25": False}, None
    if isinstance(flags, str) or not isinstance(flags, Mapping):
        raise TypeError(f"flags must be a mapping, got {type(flags).__name__}")
    integration = {"hinglish": False, "bm25": False}
    net: dict[str, bool] = {}
    for key, value in flags.items():
        if not isinstance(key, str):
            raise TypeError(f"flag name must be str, got {type(key).__name__}")
        if key in _INTEGRATION_FLAGS:
            if type(value) is not bool:
                raise TypeError(
                    f"flag {key} must be bool, got {type(value).__name__}"
                )
            integration[key] = value
        elif key in _NET_FLAGS:
            net[key] = value  # type: ignore[assignment]
        else:
            raise ValueError(
                f"unknown search flag {key!r}; "
                "flags are zone_weights, g_score, freshness, hinglish, bm25"
            )
    return integration, net or None


def _net_features_on(net_flags: Mapping[str, bool] | None) -> bool:
    """True when a net-score switch is on and therefore needs weights."""
    if not net_flags:
        return False
    return any(bool(net_flags.get(name, False)) for name in _NET_FLAGS)


def _reject_bm25_mix(
    integration: Mapping[str, bool],
    net_flags: Mapping[str, bool] | None,
) -> None:
    """Refuse to add net-score switches onto the BM25 baseline."""
    if not net_flags:
        return
    for key, value in net_flags.items():
        if type(value) is not bool:
            raise TypeError(f"flag {key} must be bool, got {type(value).__name__}")
    if integration["bm25"] and any(net_flags.values()):
        raise ValueError(
            "bm25 selects the external baseline in bharosa.rank.bm25; "
            "g_score, freshness, and zone_weights are net-score flags and "
            "are not applied to a BM25 score"
        )


def _canonical_zone(value: object) -> str:
    """Case-fold a zone and reject anything outside the contract list."""
    if not isinstance(value, str):
        raise TypeError(f"zone must be str, got {type(value).__name__}")
    folded = unicodedata.normalize("NFC", value).casefold()
    folded = unicodedata.normalize("NFC", folded).strip()
    if folded not in ZONES:
        allowed = ", ".join(ZONES)
        raise ValueError(f"zone must be one of {allowed}; got {value!r}")
    return folded


def _as_conditions(value: object) -> tuple[str, ...]:
    """Copy a condition list into a tuple. Do not read conditions from text."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "conditions must be a list of strings, "
            f"got {type(value).__name__}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"conditions must be str, got {type(item).__name__}")
        items.append(item)
    return tuple(items)


def _optional_score(value: object) -> float | None:
    """Return a finite g(d), or ``None`` when the document did not have one."""
    if value is None:
        return None
    return _finite("g_score", value)


def _require_timestamp(field: str, value: object) -> None:
    """Reject a timestamp NetScore would not accept as ``last_changed_at``.

    The check goes through ``DocumentSignals.from_zone`` so a date with
    no time is not filled in here either. ``None`` is a missing value.
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str or None, got {type(value).__name__}")
    try:
        DocumentSignals.from_zone(
            {"doc_id": "timestamp-check", "last_changed_at": value},
            cosine=0.0,
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("last_changed_at", field)) from exc


def _require_text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    return value


def _optional_finite(field: str, value: object) -> float | None:
    """Return ``None`` or a finite score. Do not replace ``None`` with 0."""
    if value is None:
        return None
    return _finite(field, value)


def _finite(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 0:
        raise ValueError("k must be non-negative")
