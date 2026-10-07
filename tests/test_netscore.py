"""Known-answer tests for the linear net score.

Each expected number is the written formula on caller-chosen weights.
Missing quality or freshness stays missing.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bharosa.rank.netscore import (
    Contribution,
    DocumentSignals,
    NetScoreConfig,
    NetScoreWeights,
    NetScorer,
    freshness_from_age,
    linear_net,
)


def _weights(cosine: float = 1.0, g_score: float = 1.0, freshness: float = 1.0) -> NetScoreWeights:
    return NetScoreWeights(cosine, g_score, freshness)


def _scorer(**kwargs: object) -> NetScorer:
    config = NetScoreConfig(
        weights=kwargs.get("weights", _weights()),  # type: ignore[arg-type]
        use_g_score=kwargs.get("use_g_score", False),  # type: ignore[arg-type]
        use_freshness=kwargs.get("use_freshness", False),  # type: ignore[arg-type]
        use_zone_weights=kwargs.get("use_zone_weights", False),  # type: ignore[arg-type]
        zone_weights=kwargs.get("zone_weights"),  # type: ignore[arg-type]
        freshness_scale_days=kwargs.get("freshness_scale_days"),  # type: ignore[arg-type]
    )
    return NetScorer(config)


def test_formula_uses_the_weights_that_were_passed() -> None:
    # 2 * 0.5 + 4 * 0.25 = 2. Zone weight 3 multiplies the sum: 6.
    assert linear_net(cosine=0.5, w_cosine=2, g_score=0.25, w_g_score=4) == pytest.approx(2.0)
    assert linear_net(
        cosine=0.5,
        w_cosine=2,
        g_score=0.25,
        w_g_score=4,
        zone_weight=3,
    ) == pytest.approx(6.0)
    assert freshness_from_age(0, 10) == 1.0
    assert freshness_from_age(10, 10) == 0.5
    assert freshness_from_age(3, 1) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="unavailable"):
        linear_net(cosine=1, w_cosine=1, w_g_score=1, g_score=None)
    with pytest.raises(ValueError, match="not clamped"):
        freshness_from_age(-1, 1)
    # A real zero is a value. Leaving the weight unset leaves the feature out.
    assert linear_net(cosine=0.5, w_cosine=2, g_score=0.0, w_g_score=4) == pytest.approx(1.0)
    assert linear_net(cosine=0.5, w_cosine=2) == pytest.approx(1.0)


def test_disabled_weights_stay_out_of_the_sum() -> None:
    # w_g_score is 100 and the document's g_score is 0.5. The switch is
    # off, so the score is 2 * 0.25.
    scorer = _scorer(weights=_weights(2, 100, 7))
    scored = scorer.score_document(DocumentSignals("d1", cosine=0.25, g_score=0.5))

    assert scored.complete
    assert scored.score == pytest.approx(0.5)
    g_part = scored.contributions[1]
    assert g_part == Contribution(
        "g_score",
        "disabled",
        weight=100,
        value=0.5,
        product=None,
        detail="switch off; value is not added",
    )
    assert scored.contributions[2].status == "disabled"
    assert scored.contributions[2].value is None
    assert scored.contributions[3].status == "disabled"
    assert "not fitted" in scorer.format_ranking([scored_signals()], 1)


def scored_signals() -> DocumentSignals:
    return DocumentSignals("d1", cosine=0.25, g_score=0.5)


def test_format_labels_a_disabled_feature_instead_of_a_fill_in() -> None:
    scorer = _scorer(weights=_weights(1, 100, 7))
    expected = "\n".join(
        [
            "formula  net = w_cosine*cosine + w_g_score*g_score + w_freshness*freshness",
            "formula  freshness = 1/(1+age_days/freshness_scale_days) from last_changed_at",
            "formula  zone_weight multiplies that sum when zone_weights is on",
            "note  every weight is caller configuration and is not fitted here",
            "weights  w_cosine=1.000000  w_g_score=100.000000  w_freshness=7.000000",
            "switches  g_score=off  freshness=off  zone_weights=off",
            "document d1  score=0.250000  complete=True",
            "  cosine  status=used  weight=1.000000  value=0.250000  product=0.250000",
            "  g_score  status=disabled  weight=100.000000  value=0.500000  "
            "product=not-applied  detail=switch off; value is not added",
            "  freshness  status=disabled  weight=7.000000  "
            "detail=switch off; value is not added",
            "  zone_weight  status=disabled  detail=switch off; the sum is not multiplied",
            "ranked k=1",
            "  1  d1  0.250000",
            "excluded  0",
        ]
    )
    assert scorer.format_ranking([DocumentSignals("d1", cosine=0.25, g_score=0.5)], 1) == expected


def test_omitted_flags_do_not_turn_features_on() -> None:
    config = NetScoreConfig.from_flags(_weights(1, 1, 1))
    scorer = NetScorer(config)
    scored = scorer.score_document(DocumentSignals("d1", cosine=0.25, g_score=0.9))

    assert config.use_g_score is False
    assert config.use_freshness is False
    assert config.use_zone_weights is False
    assert scored.score == pytest.approx(0.25)


def test_missing_quality_excludes_the_document() -> None:
    scorer = _scorer(weights=_weights(1, 1, 1), use_g_score=True)
    missing = scorer.score_document(
        DocumentSignals("high", cosine=0.99, g_score=None)
    )
    present_zero = scorer.score_document(
        DocumentSignals("low", cosine=0.1, g_score=0.0)
    )
    official_domain = DocumentSignals.from_zone(
        {
            "doc_id": "gov",
            "domain": "gov.in",
            "text": "official scheme",
            "g_score": None,
        },
        cosine=0.99,
    )
    from_domain = scorer.score_document(official_domain)

    assert missing.score is None
    assert missing.complete is False
    assert missing.reason == "unavailable: g_score"
    assert missing.contributions[1].detail == "g_score is absent"
    assert present_zero.complete
    assert present_zero.score == pytest.approx(0.1)
    assert from_domain.score is None
    assert from_domain.contributions[1].detail == "g_score is absent"

    ranking = scorer.rank(
        [
            DocumentSignals("high", cosine=0.99),
            DocumentSignals("low", cosine=0.1, g_score=0.0),
            DocumentSignals("real", cosine=0.1, g_score=1.0),
        ],
        k=5,
    )
    assert [item.doc_id for item in ranking.ranked] == ["real", "low"]
    assert [item.score for item in ranking.ranked] == pytest.approx([1.1, 0.1])
    assert [item.doc_id for item in ranking.excluded] == ["high"]


def test_missing_freshness_is_not_replaced() -> None:
    as_of = datetime(2026, 1, 11)
    scorer = _scorer(weights=_weights(1, 1, 2), use_freshness=True, freshness_scale_days=10)
    changed = scorer.score_document(
        DocumentSignals("d1", cosine=0.0, last_changed_at=datetime(2026, 1, 1)),
        as_of=as_of,
    )
    # age is 10 days and the scale is 10, so freshness is 1/2.
    # contribution is 2 * 0.5 = 1. cosine is 0.
    assert changed.complete
    assert changed.score == pytest.approx(1.0)
    freshness = changed.contributions[2]
    assert freshness.value == pytest.approx(0.5)
    assert freshness.product == pytest.approx(1.0)
    assert freshness.detail == "age_days=10.000000 scale_days=10.000000"

    crawled_only = DocumentSignals.from_zone(
        {
            "doc_id": "crawled",
            "crawled_at": "2026-01-11T00:00:00",
            "zone": "other",
        },
        cosine=0.0,
    )
    excluded = scorer.score_document(crawled_only, as_of=as_of)
    assert crawled_only.has_crawled_at is True
    assert crawled_only.last_changed_at is None
    assert excluded.score is None
    assert "crawled_at is not a freshness input" in excluded.contributions[2].detail

    future = scorer.score_document(
        DocumentSignals("future", cosine=1.0, last_changed_at=datetime(2026, 1, 12)),
        as_of=as_of,
    )
    assert future.score is None
    assert "not clamped" in (future.contributions[2].detail)

    mismatch = scorer.score_document(
        DocumentSignals("mixed", cosine=1.0, last_changed_at=datetime(2026, 1, 1)),
        as_of=datetime(2026, 1, 11, tzinfo=timezone.utc),
    )
    assert mismatch.score is None
    assert "both be naive or both be aware" in mismatch.contributions[2].detail

    aware = scorer.score_document(
        DocumentSignals(
            "aware",
            cosine=0.0,
            last_changed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        as_of=datetime(2026, 1, 11, tzinfo=timezone.utc),
    )
    assert aware.score == pytest.approx(1.0)

    with pytest.raises(ValueError, match="current time is not assumed"):
        scorer.score_document(DocumentSignals("d1", cosine=1.0))
    with pytest.raises(ValueError, match="no time"):
        DocumentSignals.from_zone(
            {"doc_id": "d1", "last_changed_at": "2026-01-01"},
            cosine=1.0,
        )


def test_zone_weight_multiplies_only_when_the_caller_supplied_it() -> None:
    # Base is 0.5 + 0.5 = 1. Caller weight 2 multiplies that sum.
    scorer = _scorer(
        use_g_score=True,
        use_zone_weights=True,
        zone_weights={"eligibility": 2.0},
    )
    scored = scorer.score_document(
        DocumentSignals("d1", cosine=0.5, g_score=0.5, zone="Eligibility")
    )

    assert scored.score == pytest.approx(2.0)
    zone_part = scored.contributions[3]
    assert zone_part.status == "used"
    assert zone_part.weight == pytest.approx(2.0)
    assert zone_part.value == pytest.approx(1.0)
    assert zone_part.product == pytest.approx(2.0)

    missing_weight = scorer.score_document(
        DocumentSignals("other", cosine=1.0, g_score=1.0, zone="benefits")
    )
    assert missing_weight.score is None
    assert "no fill-in weight" in missing_weight.contributions[3].detail

    absent_zone = scorer.score_document(
        DocumentSignals("blank", cosine=1.0, g_score=1.0, zone=None)
    )
    assert absent_zone.score is None
    assert "zone is absent" in absent_zone.contributions[3].detail

    with pytest.raises(ValueError, match="omit the weights"):
        _scorer(zone_weights={"eligibility": 2.0})
    with pytest.raises(ValueError, match="no fill-in"):
        _scorer(use_zone_weights=True, zone_weights={})
    with pytest.raises(ValueError, match="eligibility"):
        DocumentSignals("bad", cosine=1.0, zone="how-to-apply")


def test_flags_reject_features_that_are_not_part_of_the_formula() -> None:
    weights = _weights()
    with pytest.raises(ValueError, match="external baseline"):
        NetScoreConfig.from_flags(weights, {"bm25": False})
    with pytest.raises(ValueError, match="hinglish"):
        NetScoreConfig.from_flags(weights, {"hinglish": True})
    with pytest.raises(ValueError, match="unknown net-score flag"):
        NetScoreConfig.from_flags(weights, {"page_rank": True})
    with pytest.raises(ValueError, match="no scale is assumed"):
        NetScoreConfig.from_flags(weights, {"freshness": True})
    with pytest.raises(TypeError):
        NetScoreConfig.from_flags(weights, {"g_score": 1})  # type: ignore[dict-item]

    configured = NetScoreConfig.from_flags(
        weights,
        {"g_score": True, "freshness": True, "zone_weights": False},
        freshness_scale_days=4,
    )
    assert configured.use_g_score is True
    assert configured.use_freshness is True
    assert configured.freshness_scale_days == pytest.approx(4.0)


def test_ties_break_by_doc_id_and_incomplete_rows_stay_out() -> None:
    scorer = _scorer()
    ranking = scorer.rank(
        [
            DocumentSignals("b", cosine=1.0),
            DocumentSignals("a", cosine=1.0),
            DocumentSignals("c", cosine=None),
        ],
        k=5,
    )

    assert [item.doc_id for item in ranking.ranked] == ["a", "b"]
    assert ranking.ranked[0].score == pytest.approx(1.0)
    assert [item.doc_id for item in ranking.excluded] == ["c"]
    assert scorer.rank([DocumentSignals("a", cosine=1.0)], 0).ranked == ()
    assert "NetScorer(" in repr(scorer)


def test_module_does_not_treat_weights_as_fitted_or_call_bm25() -> None:
    source = Path(netscore_path()).read_text(encoding="utf-8")
    folded = source.casefold()
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    assert "optimal" not in folded
    assert "optimum" not in folded
    assert "datetime.now" not in source
    assert "date.today" not in source
    assert "bharosa.rank.bm25" not in imported
    assert "rank_bm25" not in imported
    assert not hasattr(pytest.importorskip("bharosa.rank.netscore"), "DEFAULT_WEIGHTS")


def netscore_path() -> Path:
    import bharosa.rank.netscore as netscore

    assert netscore.__file__ is not None
    return Path(netscore.__file__)


def test_from_zone_reads_only_contract_scoring_fields() -> None:
    signals = DocumentSignals.from_zone(
        {
            "doc_id": "z1",
            "url": "https://example.test",
            "domain": "example.test",
            "title": "Scheme",
            "zone": "benefits",
            "text": "not a quality score",
            "state": "UP",
            "conditions": ["heart"],
            "g_score": 0.25,
            "crawled_at": "2026-01-02T00:00:00",
            "last_changed_at": "2026-01-01T00:00:00",
            "content_hash": "abc",
        },
        cosine=0.5,
    )

    assert signals.g_score == pytest.approx(0.25)
    assert signals.zone == "benefits"
    assert signals.last_changed_at == datetime(2026, 1, 1)
    assert signals.has_crawled_at is True
    with pytest.raises(ValueError, match="income"):
        DocumentSignals.from_zone({"doc_id": "z1", "income": 1}, cosine=1.0)


def test_rejects_bad_inputs() -> None:
    scorer = _scorer()
    with pytest.raises(TypeError):
        NetScoreWeights(True, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NetScoreWeights(float("nan"), 1, 1)
    with pytest.raises(TypeError):
        scorer.score_document({"doc_id": "d1", "cosine": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        scorer.rank(DocumentSignals("d1", cosine=1.0), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        scorer.rank(
            [DocumentSignals("d1", cosine=1.0), DocumentSignals("d1", cosine=0.2)],
            1,
        )
    with pytest.raises(ValueError):
        scorer.rank([DocumentSignals("d1", cosine=1.0)], -1)
    with pytest.raises(TypeError):
        DocumentSignals("d1", cosine=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NetScorer({})  # type: ignore[arg-type]
