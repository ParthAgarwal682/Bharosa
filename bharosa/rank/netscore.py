"""Linear net score over cosine, static quality, freshness, and zone weight.

IR concept: a linear combination of retrieval features. The project
formula is ``w_cosine * cosine + w_g_score * g(d) + w_freshness *
freshness``, and a zone weight multiplies that sum when the caller turns
it on. Each coefficient is an argument. This module does not search for
coefficients and does not store a recommended setting.

``cosine`` is the caller's lnc.ltc score from ``bharosa.rank.tfidf``.
``g_score`` and ``zone`` are zone-contract fields. Freshness is computed
only from ``last_changed_at`` and an ``as_of`` time the caller supplies.
``crawled_at`` is not a substitute. A missing enabled feature excludes
the document and is reported; it is not replaced with 0 or 1. Turning a
feature off is an ablation: that term is left out of the sum, and the
remaining weights are not rescaled.

BM25 is not a term in this formula. The external baseline lives in
``bharosa.rank.bm25``.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from bharosa.rank.topk import top_k

ComponentStatus = Literal["used", "disabled", "unavailable"]

# Closed zone vocabulary from the zone contract.
ZONES: tuple[str, ...] = (
    "eligibility",
    "benefits",
    "documents",
    "how_to_apply",
    "other",
)
_ZONE_SET = frozenset(ZONES)

_SCORE_FLAGS = frozenset({"zone_weights", "g_score", "freshness"})
_REJECTED_FLAGS = {
    "bm25": (
        "bm25 is an external baseline in bharosa.rank.bm25, "
        "not a net-score component"
    ),
    "hinglish": (
        "hinglish expansion is bharosa.text.hinglish.expand, "
        "applied to the query before scoring"
    ),
}

_FORMULA_LINES = (
    "formula  net = w_cosine*cosine + w_g_score*g_score + w_freshness*freshness",
    "formula  freshness = 1/(1+age_days/freshness_scale_days) from last_changed_at",
    "formula  zone_weight multiplies that sum when zone_weights is on",
    "note  every weight is caller configuration and is not fitted here",
)


@dataclass(frozen=True)
class NetScoreWeights:
    """Caller-chosen coefficients for the three linear features.

    IR concept: the weights in a linear net score. ``cosine``,
    ``g_score``, and ``freshness`` are w_cosine, w_g_score, and
    w_freshness. There is no default: a caller writes the numbers at
    the call site. Any finite value is accepted, including 0 and
    negative numbers. Nothing here was fit to relevance labels.
    """

    cosine: float
    g_score: float
    freshness: float

    def __post_init__(self) -> None:
        _require_finite("w_cosine", self.cosine)
        _require_finite("w_g_score", self.g_score)
        _require_finite("w_freshness", self.freshness)


@dataclass(frozen=True)
class NetScoreConfig:
    """Which features are on, and the weights used when they are.

    IR concept: an ablation setup. A feature that is off contributes
    nothing. A feature that is on still needs a real value. Omitted
    flags leave ``g_score``, freshness, and zone weights off; they are
    not switched on by a hidden coefficient.

    ``zone_weights`` is required only when ``use_zone_weights`` is true,
    and it must list only contract zones the caller chose to weight.
    Zones left out of the mapping are not given a fill-in weight.
    ``freshness_scale_days`` is required only when freshness is on.
    It is the age, in days, at which freshness is one half.
    """

    weights: NetScoreWeights
    use_g_score: bool = False
    use_freshness: bool = False
    use_zone_weights: bool = False
    zone_weights: Mapping[str, float] | None = None
    freshness_scale_days: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weights, NetScoreWeights):
            raise TypeError(
                "weights must be NetScoreWeights, "
                f"got {type(self.weights).__name__}"
            )
        for name, flag in (
            ("use_g_score", self.use_g_score),
            ("use_freshness", self.use_freshness),
            ("use_zone_weights", self.use_zone_weights),
        ):
            if type(flag) is not bool:
                raise TypeError(f"{name} must be bool, got {type(flag).__name__}")
        object.__setattr__(self, "zone_weights", _freeze_zone_weights(self))
        object.__setattr__(
            self,
            "freshness_scale_days",
            _freeze_scale(self.use_freshness, self.freshness_scale_days),
        )

    @classmethod
    def from_flags(
        cls,
        weights: NetScoreWeights,
        flags: Mapping[str, bool] | None = None,
        *,
        zone_weights: Mapping[str, float] | None = None,
        freshness_scale_days: float | None = None,
    ) -> NetScoreConfig:
        """Build a config from the contract's flag names.

        IR concept: ablation flags. Recognised flags are ``g_score``,
        ``freshness``, and ``zone_weights``. A flag that is absent or a
        config built with ``flags=None`` leaves that feature off.
        ``bm25`` and ``hinglish`` are rejected because they are not
        net-score features.
        """
        parsed = _parse_flags(flags)
        return cls(
            weights=weights,
            use_g_score=parsed["g_score"],
            use_freshness=parsed["freshness"],
            use_zone_weights=parsed["zone_weights"],
            zone_weights=zone_weights,
            freshness_scale_days=freshness_scale_days,
        )


@dataclass(frozen=True)
class DocumentSignals:
    """Inputs for one document's net score.

    IR concept: the feature vector before weighting. ``cosine`` is a
    retrieval score supplied by the caller. ``g_score`` and ``zone``
    come from the zone contract when they were provided. ``None`` means
    the value was not provided. ``has_crawled_at`` records only whether
    that contract field was present; the timestamp itself is not stored,
    so it cannot be used as ``last_changed_at``.
    """

    doc_id: str
    cosine: float | None
    g_score: float | None = None
    zone: str | None = None
    last_changed_at: datetime | None = None
    has_crawled_at: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(self.doc_id).__name__}")
        if self.doc_id == "":
            raise ValueError("doc_id must be non-empty")
        if self.cosine is not None:
            _require_finite("cosine", self.cosine)
        if self.g_score is not None:
            _require_finite("g_score", self.g_score)
        if self.zone is not None:
            object.__setattr__(self, "zone", _normalize_zone(self.zone))
        if self.last_changed_at is not None and not isinstance(
            self.last_changed_at, datetime
        ):
            raise TypeError(
                "last_changed_at must be datetime or None, "
                f"got {type(self.last_changed_at).__name__}"
            )
        if type(self.has_crawled_at) is not bool:
            raise TypeError(
                "has_crawled_at must be bool, "
                f"got {type(self.has_crawled_at).__name__}"
            )

    @classmethod
    def from_zone(
        cls,
        document: Mapping[str, object],
        cosine: float | None,
    ) -> DocumentSignals:
        """Read scoring fields from a zone document.

        IR concept: feature extraction from the agreed record. The
        fields read are ``doc_id``, ``g_score``, ``zone``, and
        ``last_changed_at``. ``crawled_at`` is noticed only so a missing
        change time can say it was not used. Other contract keys are
        allowed and ignored. A key outside the zone contract raises.
        """
        _reject_unknown_zone_keys(document)
        if "doc_id" not in document:
            raise ValueError("doc_id is required")
        doc_id = document["doc_id"]
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
        g_score = None
        if "g_score" in document and document["g_score"] is not None:
            g_score = _number("g_score", document["g_score"])
        zone = None
        if "zone" in document and document["zone"] is not None:
            zone = document["zone"]
        last_changed_at = None
        if "last_changed_at" in document and document["last_changed_at"] is not None:
            last_changed_at = _parse_datetime(
                "last_changed_at", document["last_changed_at"]
            )
        has_crawled_at = "crawled_at" in document and document["crawled_at"] is not None
        return cls(
            doc_id=doc_id,
            cosine=cosine,
            g_score=g_score,
            zone=zone if isinstance(zone, str) or zone is None else _reject_zone_type(zone),
            last_changed_at=last_changed_at,
            has_crawled_at=has_crawled_at,
        )


@dataclass(frozen=True)
class Contribution:
    """One feature's status in a single document's net score.

    IR concept: a term in a linear combination, or the zone multiplier.
    ``status`` is ``used``, ``disabled``, or ``unavailable``. ``product``
    is ``weight * value`` for a used linear feature, and the multiplied
    total for a used zone weight. A disabled or unavailable feature has
    ``product is None`` and is not part of a reported net score.
    """

    component: str
    status: ComponentStatus
    weight: float | None = None
    value: float | None = None
    product: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class NetScore:
    """The net score of one document, or an explicit exclusion.

    IR concept: a combined retrieval score. ``score`` is set only when
    every enabled feature had a real value. Otherwise ``score`` is
    ``None`` and ``reason`` names the missing features. A missing
    feature is not reported as 0.
    """

    doc_id: str
    score: float | None
    complete: bool
    contributions: tuple[Contribution, ...]
    reason: str | None = None


@dataclass(frozen=True)
class NetScoreRanking:
    """Ranked complete scores, plus documents that could not be scored.

    IR concept: top-K over the net score. ``ranked`` is best first and
    contains only complete scores. ``excluded`` keeps input order and
    contains the documents whose enabled features were unavailable.
    """

    ranked: tuple[NetScore, ...]
    excluded: tuple[NetScore, ...]


def freshness_from_age(age_days: float, freshness_scale_days: float) -> float:
    """Return ``1 / (1 + age_days / freshness_scale_days)``.

    IR concept: a freshness feature in ``(0, 1]``. It is 1 at age 0 and
    one half when ``age_days`` equals ``freshness_scale_days``. The
    scale is a caller-chosen unit. A negative age raises, because this
    function does not clamp it up to 0.
    """
    age = _require_finite("age_days", age_days)
    scale = _require_finite("freshness_scale_days", freshness_scale_days)
    if age < 0:
        raise ValueError("age_days must be non-negative; a negative age is not clamped")
    if scale <= 0:
        raise ValueError("freshness_scale_days must be positive")
    return 1.0 / (1.0 + age / scale)


def linear_net(
    *,
    cosine: float,
    w_cosine: float,
    g_score: float | None = None,
    w_g_score: float | None = None,
    freshness: float | None = None,
    w_freshness: float | None = None,
    zone_weight: float | None = None,
) -> float:
    """Combine features that are actually present.

    IR concept: the net-score sum. A weight of ``None`` leaves that
    feature out. A weight that is set while the feature value is
    ``None`` raises. The function does not replace a missing value.
    When ``zone_weight`` is set, it multiplies the sum. Remaining
    weights are not rescaled to make up for a feature that was left out.
    """
    total = _require_finite("w_cosine", w_cosine) * _require_finite("cosine", cosine)
    if w_g_score is not None:
        if g_score is None:
            raise ValueError("g_score is unavailable")
        total += _require_finite("w_g_score", w_g_score) * _require_finite(
            "g_score", g_score
        )
    if w_freshness is not None:
        if freshness is None:
            raise ValueError("freshness is unavailable")
        total += _require_finite("w_freshness", w_freshness) * _require_finite(
            "freshness", freshness
        )
    if zone_weight is not None:
        total *= _require_finite("zone_weight", zone_weight)
    return total


class NetScorer:
    """Apply one :class:`NetScoreConfig` to document signals.

    IR concept: net-score ranking. The scorer does not retrieve
    documents and does not compute cosine. It weights the signals it
    is given and keeps the K highest complete scores with the heap in
    ``bharosa.rank.topk``.
    """

    def __init__(self, config: NetScoreConfig) -> None:
        if not isinstance(config, NetScoreConfig):
            raise TypeError(
                f"config must be NetScoreConfig, got {type(config).__name__}"
            )
        self._config = config

    @property
    def config(self) -> NetScoreConfig:
        """The weights and switches used for every document."""
        return self._config

    def score_document(
        self,
        signals: DocumentSignals,
        *,
        as_of: datetime | str | None = None,
    ) -> NetScore:
        """Score one document, or exclude it when an enabled feature is missing.

        IR concept: evaluating the linear formula on one document.
        ``as_of`` is required when freshness is on. The current clock
        is not read. A disabled feature is recorded as disabled even
        when the document happens to carry a value for it.
        """
        if not isinstance(signals, DocumentSignals):
            raise TypeError(
                f"signals must be DocumentSignals, got {type(signals).__name__}"
            )
        moment = self._as_of(as_of)
        contributions, score, reason = self._evaluate(signals, moment)
        return NetScore(
            doc_id=signals.doc_id,
            score=score,
            complete=reason is None,
            contributions=contributions,
            reason=reason,
        )

    def rank(
        self,
        documents: Sequence[DocumentSignals],
        k: int,
        *,
        as_of: datetime | str | None = None,
    ) -> NetScoreRanking:
        """Return the top-K complete net scores and the excluded documents.

        IR concept: top-K over a linear net score. Incomplete documents
        are not given a stand-in score and are not placed on the heap.
        Ties follow ``top_k``: higher score first, then the smaller
        ``doc_id``.
        """
        if isinstance(documents, str) or not isinstance(documents, Sequence):
            raise TypeError(
                "documents must be a sequence of DocumentSignals, "
                f"got {type(documents).__name__}"
            )
        moment = self._as_of(as_of)
        seen: set[str] = set()
        complete: list[NetScore] = []
        excluded: list[NetScore] = []
        for signals in documents:
            if not isinstance(signals, DocumentSignals):
                raise TypeError(
                    "documents must contain DocumentSignals, "
                    f"got {type(signals).__name__}"
                )
            if signals.doc_id in seen:
                raise ValueError(f"duplicate doc_id: {signals.doc_id}")
            seen.add(signals.doc_id)
            contributions, score, reason = self._evaluate(signals, moment)
            scored = NetScore(
                doc_id=signals.doc_id,
                score=score,
                complete=reason is None,
                contributions=contributions,
                reason=reason,
            )
            if scored.complete:
                complete.append(scored)
            else:
                excluded.append(scored)

        winners = top_k(
            tuple((item.doc_id, _required_score(item)) for item in complete),
            k,
        )
        by_id = {item.doc_id: item for item in complete}
        ranked = tuple(by_id[doc_id] for doc_id, _score in winners)
        return NetScoreRanking(ranked=ranked, excluded=tuple(excluded))

    def format_ranking(
        self,
        documents: Sequence[DocumentSignals],
        k: int,
        *,
        as_of: datetime | str | None = None,
    ) -> str:
        """Return an inspectable dump of the formula, weights, and scores.

        IR concept: a verbose net-score trace. Disabled features are
        labelled disabled and do not show a fill-in value. Excluded
        documents show ``score=unavailable`` rather than a number.
        """
        ranking = self.rank(documents, k, as_of=as_of)
        breakdown = tuple(
            self.score_document(signals, as_of=as_of) for signals in documents
        )
        return "\n".join(_format_lines(self._config, ranking, breakdown, k, as_of))

    def __repr__(self) -> str:
        flags = self._config
        return (
            "NetScorer("
            f"g_score={flags.use_g_score}, "
            f"freshness={flags.use_freshness}, "
            f"zone_weights={flags.use_zone_weights})"
        )

    def _as_of(self, as_of: datetime | str | None) -> datetime | None:
        if not self._config.use_freshness:
            return None
        if as_of is None:
            raise ValueError(
                "freshness is on but as_of is missing; the current time is not assumed"
            )
        return _parse_datetime("as_of", as_of)

    def _evaluate(
        self,
        signals: DocumentSignals,
        as_of: datetime | None,
    ) -> tuple[tuple[Contribution, ...], float | None, str | None]:
        config = self._config
        weights = config.weights
        cosine, g_part, fresh_part, zone_part = _parts(config, signals, as_of)
        contributions = (cosine, g_part, fresh_part, zone_part)
        missing = [item.component for item in contributions if item.status == "unavailable"]
        if missing:
            reason = "unavailable: " + ", ".join(missing)
            return contributions, None, reason
        base = linear_net(
            cosine=_required_value(cosine),
            w_cosine=_required_weight(cosine),
            g_score=g_part.value if g_part.status == "used" else None,
            w_g_score=g_part.weight if g_part.status == "used" else None,
            freshness=fresh_part.value if fresh_part.status == "used" else None,
            w_freshness=fresh_part.weight if fresh_part.status == "used" else None,
        )
        if zone_part.status != "used":
            return contributions, base, None
        zone_weight = _required_weight(zone_part)
        score = base * zone_weight
        zone_part = Contribution(
            "zone_weight",
            "used",
            weight=zone_weight,
            value=base,
            product=score,
            detail=zone_part.detail,
        )
        return (cosine, g_part, fresh_part, zone_part), score, None


def _parts(
    config: NetScoreConfig,
    signals: DocumentSignals,
    as_of: datetime | None,
) -> tuple[Contribution, Contribution, Contribution, Contribution]:
    weights = config.weights
    if signals.cosine is None:
        cosine = Contribution(
            "cosine",
            "unavailable",
            weight=weights.cosine,
            detail="cosine is absent",
        )
    else:
        product = weights.cosine * signals.cosine
        cosine = Contribution(
            "cosine",
            "used",
            weight=weights.cosine,
            value=signals.cosine,
            product=product,
        )
    g_part = _linear_part(
        "g_score",
        enabled=config.use_g_score,
        weight=weights.g_score,
        value=signals.g_score,
        missing_detail="g_score is absent",
    )
    fresh_part = _freshness_part(config, signals, as_of)
    zone_part = _zone_part(config, signals)
    return cosine, g_part, fresh_part, zone_part


def _linear_part(
    component: str,
    *,
    enabled: bool,
    weight: float,
    value: float | None,
    missing_detail: str,
) -> Contribution:
    if not enabled:
        return Contribution(
            component,
            "disabled",
            weight=weight,
            value=value,
            detail="switch off; value is not added",
        )
    if value is None:
        return Contribution(component, "unavailable", weight=weight, detail=missing_detail)
    return Contribution(
        component,
        "used",
        weight=weight,
        value=value,
        product=weight * value,
    )


def _freshness_part(
    config: NetScoreConfig,
    signals: DocumentSignals,
    as_of: datetime | None,
) -> Contribution:
    weight = config.weights.freshness
    if not config.use_freshness:
        return Contribution(
            "freshness",
            "disabled",
            weight=weight,
            detail="switch off; value is not added",
        )
    scale = config.freshness_scale_days
    if scale is None:
        raise ValueError("freshness is on but freshness_scale_days is missing")
    if signals.last_changed_at is None:
        detail = "last_changed_at is absent"
        if signals.has_crawled_at:
            detail += "; crawled_at is not a freshness input"
        return Contribution("freshness", "unavailable", weight=weight, detail=detail)
    changed = signals.last_changed_at
    if as_of is None:
        raise ValueError("freshness is on but as_of is missing")
    if (changed.tzinfo is None) != (as_of.tzinfo is None):
        return Contribution(
            "freshness",
            "unavailable",
            weight=weight,
            detail="as_of and last_changed_at must both be naive or both be aware",
        )
    age_days = (as_of - changed).total_seconds() / 86400.0
    if age_days < 0:
        return Contribution(
            "freshness",
            "unavailable",
            weight=weight,
            detail="last_changed_at is after as_of; age is not clamped to zero",
        )
    value = freshness_from_age(age_days, scale)
    return Contribution(
        "freshness",
        "used",
        weight=weight,
        value=value,
        product=weight * value,
        detail=f"age_days={_fmt(age_days)} scale_days={_fmt(scale)}",
    )


def _zone_part(config: NetScoreConfig, signals: DocumentSignals) -> Contribution:
    if not config.use_zone_weights:
        return Contribution(
            "zone_weight",
            "disabled",
            detail="switch off; the sum is not multiplied",
        )
    weights = config.zone_weights
    if weights is None:
        raise ValueError("zone_weights is on but no zone weights were provided")
    if signals.zone is None:
        return Contribution(
            "zone_weight",
            "unavailable",
            detail="zone is absent; no fill-in weight is applied",
        )
    if signals.zone not in weights:
        return Contribution(
            "zone_weight",
            "unavailable",
            detail=(
                f"no caller weight for zone {signals.zone}; "
                "no fill-in weight is applied"
            ),
        )
    return Contribution(
        "zone_weight",
        "used",
        weight=weights[signals.zone],
        detail=f"zone={signals.zone}; multiplies the sum",
    )


def _format_lines(
    config: NetScoreConfig,
    ranking: NetScoreRanking,
    breakdown: Sequence[NetScore],
    k: int,
    as_of: datetime | str | None,
) -> list[str]:
    weights = config.weights
    lines = list(_FORMULA_LINES)
    lines.append(
        "weights  "
        f"w_cosine={_fmt(weights.cosine)}  "
        f"w_g_score={_fmt(weights.g_score)}  "
        f"w_freshness={_fmt(weights.freshness)}"
    )
    lines.append(
        "switches  "
        f"g_score={'on' if config.use_g_score else 'off'}  "
        f"freshness={'on' if config.use_freshness else 'off'}  "
        f"zone_weights={'on' if config.use_zone_weights else 'off'}"
    )
    if config.use_freshness:
        if config.freshness_scale_days is None:
            raise ValueError("freshness is on but freshness_scale_days is missing")
        lines.append(f"freshness_scale_days={_fmt(config.freshness_scale_days)}")
        if isinstance(as_of, datetime):
            lines.append(f"as_of={as_of.isoformat()}")
        elif isinstance(as_of, str):
            lines.append(f"as_of={as_of}")
    if config.use_zone_weights and config.zone_weights is not None:
        rendered = " ".join(
            f"{zone}={_fmt(config.zone_weights[zone])}"
            for zone in ZONES
            if zone in config.zone_weights
        )
        lines.append(f"zone_weights  {rendered}")

    for item in breakdown:
        if item.complete:
            if item.score is None:
                raise ValueError("complete net score is missing a number")
            lines.append(
                f"document {item.doc_id}  score={_fmt(item.score)}  complete=True"
            )
        else:
            lines.append(
                f"document {item.doc_id}  score=unavailable  complete=False  "
                f"reason={item.reason}"
            )
        for part in item.contributions:
            lines.append(_contribution_line(part))
    lines.append(f"ranked k={k}")
    for place, item in enumerate(ranking.ranked, start=1):
        if item.score is None:
            raise ValueError("complete net score is missing a number")
        lines.append(f"  {place}  {item.doc_id}  {_fmt(item.score)}")
    lines.append(f"excluded  {len(ranking.excluded)}")
    for item in ranking.excluded:
        lines.append(f"  {item.doc_id}  {item.reason}")
    return lines


def _contribution_line(part: Contribution) -> str:
    line = f"  {part.component}  status={part.status}"
    if part.weight is not None:
        line += f"  weight={_fmt(part.weight)}"
    if part.status == "disabled":
        if part.value is not None:
            line += f"  value={_fmt(part.value)}  product=not-applied"
        if part.detail:
            line += f"  detail={part.detail}"
        return line
    if part.status == "unavailable":
        if part.detail:
            line += f"  detail={part.detail}"
        return line
    if part.value is not None:
        line += f"  value={_fmt(part.value)}"
    if part.product is not None:
        line += f"  product={_fmt(part.product)}"
    if part.detail:
        line += f"  detail={part.detail}"
    return line


def _freeze_zone_weights(config: NetScoreConfig) -> Mapping[str, float] | None:
    provided = config.zone_weights
    if not config.use_zone_weights:
        if provided is not None:
            raise ValueError(
                "zone weights were provided while zone_weights is off; "
                "turn the switch on or omit the weights"
            )
        return None
    if provided is None or not isinstance(provided, Mapping) or isinstance(provided, str):
        raise TypeError(
            "zone_weights must be a mapping of contract zones to weights "
            "when zone_weights is on"
        )
    if len(provided) == 0:
        raise ValueError(
            "zone_weights is on but no zone weights were provided; "
            "no fill-in weights are created"
        )
    frozen: dict[str, float] = {}
    for key, value in provided.items():
        zone = _normalize_zone(key)
        if zone in frozen:
            raise ValueError(f"duplicate zone weight: {zone}")
        frozen[zone] = _require_finite(f"zone weight {zone}", value)
    return MappingProxyType(frozen)


def _freeze_scale(enabled: bool, scale: float | None) -> float | None:
    if not enabled:
        if scale is not None:
            raise ValueError(
                "freshness_scale_days was provided while freshness is off; "
                "turn the switch on or omit the scale"
            )
        return None
    if scale is None:
        raise ValueError(
            "freshness is on but freshness_scale_days is missing; "
            "no scale is assumed"
        )
    number = _require_finite("freshness_scale_days", scale)
    if number <= 0:
        raise ValueError("freshness_scale_days must be positive")
    return number


def _parse_flags(flags: Mapping[str, bool] | None) -> dict[str, bool]:
    if flags is None:
        return {name: False for name in _SCORE_FLAGS}
    if isinstance(flags, str) or not isinstance(flags, Mapping):
        raise TypeError(f"flags must be a mapping, got {type(flags).__name__}")
    for key in flags:
        if not isinstance(key, str):
            raise TypeError(f"flag name must be str, got {type(key).__name__}")
        if key in _REJECTED_FLAGS:
            raise ValueError(_REJECTED_FLAGS[key])
        if key not in _SCORE_FLAGS:
            raise ValueError(
                f"unknown net-score flag {key!r}; "
                "flags are g_score, freshness, zone_weights"
            )
        if type(flags[key]) is not bool:
            raise TypeError(
                f"flag {key} must be bool, got {type(flags[key]).__name__}"
            )
    return {name: bool(flags[name]) if name in flags else False for name in _SCORE_FLAGS}


def _reject_unknown_zone_keys(document: Mapping[str, object]) -> None:
    if not isinstance(document, Mapping):
        raise TypeError(f"document must be a mapping, got {type(document).__name__}")
    allowed = {
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
    unknown = [key for key in document if key not in allowed]
    if unknown:
        names = ", ".join(str(key) for key in sorted(unknown, key=str))
        raise ValueError(f"not in the zone contract: {names}")


def _reject_zone_type(zone: object) -> str:
    raise TypeError(f"zone must be str, got {type(zone).__name__}")


def _normalize_zone(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"zone must be str, got {type(value).__name__}")
    folded = unicodedata.normalize("NFC", value).casefold()
    folded = unicodedata.normalize("NFC", folded).strip()
    if folded not in _ZONE_SET:
        allowed = ", ".join(ZONES)
        raise ValueError(f"zone must be one of {allowed}; got {value!r}")
    return folded


def _parse_datetime(field: str, value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO datetime string or datetime")
    if "T" not in value:
        raise ValueError(
            f"{field} must be an ISO datetime; a date with no time is not filled in"
        )
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not an ISO datetime: {value!r}") from exc


def _number(field: str, value: object) -> float:
    return _require_finite(field, value)


def _require_finite(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _required_score(item: NetScore) -> float:
    if item.score is None:
        raise ValueError("complete net score is missing a number")
    return item.score


def _required_weight(part: Contribution) -> float:
    if part.weight is None:
        raise ValueError(f"{part.component} is missing a weight")
    return part.weight


def _required_value(part: Contribution) -> float:
    if part.value is None:
        raise ValueError(f"{part.component} is missing a value")
    return part.value


def _fmt(value: float) -> str:
    return f"{value:.6f}"
