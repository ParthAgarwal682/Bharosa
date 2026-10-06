"""Known-answer tests for scheme retrieval.

The documents in this file are fixtures. They are not crawled pages and
they are not government results. URLs use the ``fixture`` scheme.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from bharosa.index.inverted import InvertedIndex
from bharosa.index.zones import ZoneDoc, ZoneHit, search_schemes
from bharosa.rank.netscore import NetScoreWeights, freshness_from_age
from bharosa.rank.tfidf import TfidfRanker
from bharosa.text.hinglish import HinglishLexicon


def _zone(
    doc_id: str,
    zone: str,
    text: str,
    *,
    state: str | None = "UP",
    conditions: tuple[str, ...] = ("Heart",),
    g_score: float | None = 0.5,
    crawled_at: str | None = "2026-01-02T00:00:00",
    last_changed_at: str | None = "2026-01-01T00:00:00",
) -> ZoneDoc:
    return ZoneDoc(
        doc_id=doc_id,
        url=f"fixture://zones/{doc_id}",
        domain="fixture.local",
        title=f"{zone} fixture",
        zone=zone,
        text=text,
        state=state,
        conditions=conditions,
        g_score=g_score,
        crawled_at=crawled_at,
        last_changed_at=last_changed_at,
        content_hash=f"hash-{doc_id}",
    )


def _zones() -> list[ZoneDoc]:
    """Four zone fixtures plus one document that keeps shared-term idf above 0.

    A term that occurs in every document has lnc.ltc idf 0. The extra
    document does not contain the shared token ``fixture``.
    """
    return [
        _zone("1-eligibility", "eligibility", "alphaeligibility fixture scheme"),
        _zone("2-benefits", "benefits", "alphabenefits fixture scheme"),
        _zone(
            "3-documents",
            "documents",
            "alphadocuments fixture scheme",
            state="Bihar",
            conditions=("Kidney",),
        ),
        _zone(
            "4-apply",
            "how_to_apply",
            "alphahowtoapply fixture scheme",
            state="Bihar",
            conditions=("Kidney",),
        ),
        _zone(
            "0-filler",
            "other",
            "unrelated filler token",
            state=None,
            conditions=(),
            g_score=0.1,
        ),
    ]


def _contract(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "doc_id": "z",
        "url": "fixture://zones/z",
        "domain": "fixture.local",
        "title": "Fixture",
        "zone": "other",
        "text": "fixture text",
        "state": None,
        "conditions": [],
        "g_score": None,
        "content_hash": "hash-z",
    }
    row.update(overrides)
    return row


def test_zone_doc_stores_the_contract_and_nothing_else() -> None:
    doc = ZoneDoc(
        doc_id="z1",
        url="fixture://zones/z1",
        domain="fixture.local",
        title="Eligibility fixture",
        zone=" Eligibility ",
        text="alphaeligibility fixture scheme",
        state="UP",
        conditions=["Heart", "Heart"],  # type: ignore[arg-type]
        g_score=0.5,
        crawled_at="2026-01-02T00:00:00",
        last_changed_at="2026-01-01T00:00:00",
        content_hash="hash-z1",
    )

    assert doc.zone == "eligibility"
    assert doc.state == "UP"
    assert doc.conditions == ("Heart", "Heart")
    assert doc.as_contract()["conditions"] == ["Heart", "Heart"]
    assert doc.g_score == pytest.approx(0.5)
    assert doc.crawled_at == "2026-01-02T00:00:00"
    assert doc.last_changed_at == "2026-01-01T00:00:00"
    assert ZoneDoc.from_mapping(doc.as_contract()) == doc
    with pytest.raises(FrozenInstanceError):
        doc.text = "rewritten"  # type: ignore[misc]

    missing = _zone(
        "blank",
        "other",
        "some text",
        state=None,
        conditions=(),
        g_score=None,
        crawled_at=None,
        last_changed_at=None,
    )
    assert missing.g_score is None
    assert missing.crawled_at is None
    assert missing.last_changed_at is None

    for field in ("fetched_at", "detected_at", "text_path"):
        with pytest.raises(ValueError, match="not in the zone contract"):
            ZoneDoc.from_mapping(_contract(**{field: "crawler-value"}))
    with pytest.raises(ValueError, match="one of"):
        _zone("bad", "salary", "text")
    with pytest.raises(ValueError, match="ISO"):
        _zone("bad", "other", "text", last_changed_at="2026-01-01")
    with pytest.raises(TypeError, match="list of strings"):
        _zone("bad", "other", "text", conditions="heart")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="income"):
        ZoneDoc.from_mapping(_contract(income=1))


def test_zone_hit_keeps_the_citation_fields() -> None:
    hit = ZoneHit(
        doc_id="z1",
        text="alphaeligibility fixture scheme",
        url="fixture://zones/z1",
        zone="Benefits",
        last_changed_at="2026-01-01T00:00:00",
        cosine=0.25,
        g_component=0.0,
        freshness=0.0,
        zone_weight=1.0,
        net=0.25,
        bm25_score=None,
        rank=1,
    )

    assert hit.zone == "benefits"
    assert hit.text == "alphaeligibility fixture scheme"
    assert hit.url == "fixture://zones/z1"
    assert hit.last_changed_at == "2026-01-01T00:00:00"
    assert hit.cosine == pytest.approx(0.25)
    assert hit.net == pytest.approx(0.25)
    assert hit.bm25_score is None
    assert hit.rank == 1
    with pytest.raises(FrozenInstanceError):
        hit.rank = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="one of"):
        ZoneHit(
            doc_id="z1",
            text="t",
            url="fixture://zones/z1",
            zone="salary",
            last_changed_at=None,
            cosine=0.0,
            g_component=0.0,
            freshness=0.0,
            zone_weight=1.0,
            net=0.0,
            bm25_score=None,
            rank=1,
        )
    with pytest.raises(ValueError, match="rank"):
        ZoneHit(
            doc_id="z1",
            text="t",
            url="fixture://zones/z1",
            zone="other",
            last_changed_at=None,
            cosine=0.0,
            g_component=0.0,
            freshness=0.0,
            zone_weight=1.0,
            net=0.0,
            bm25_score=None,
            rank=0,
        )


def test_basic_retrieval_returns_the_matching_zone() -> None:
    docs = _zones()
    hits = search_schemes("alphaeligibility", documents=docs)

    assert len(hits) == 1
    hit = hits[0]
    expected = TfidfRanker(
        InvertedIndex.from_documents({doc.doc_id: doc.text for doc in docs})
    ).cosine("alphaeligibility", "1-eligibility")
    assert hit.doc_id == "1-eligibility"
    assert hit.text == "alphaeligibility fixture scheme"
    assert hit.url == "fixture://zones/1-eligibility"
    assert hit.zone == "eligibility"
    assert hit.rank == 1
    assert hit.cosine == pytest.approx(expected)
    assert hit.cosine > 0
    assert hit.g_component == pytest.approx(0.0)
    assert hit.freshness == pytest.approx(0.0)
    assert hit.zone_weight == pytest.approx(1.0)
    assert hit.net == pytest.approx(hit.cosine)
    assert hit.bm25_score is None
    assert hit.last_changed_at == "2026-01-01T00:00:00"
    assert "http://" not in hit.url
    assert "https://" not in hit.url
    assert {item.zone for item in docs[:4]} == {
        "eligibility",
        "benefits",
        "documents",
        "how_to_apply",
    }


def test_query_normalisation_matches_the_shared_analyser() -> None:
    docs = [
        _zone("heart-doc", "eligibility", "Heart Operation coverage"),
        _zone("other-doc", "benefits", "renal dialysis only"),
    ]

    folded = search_schemes("  HEART-operation!!  ", documents=docs)
    plain = search_schemes("heart operation", documents=docs)

    assert [hit.doc_id for hit in folded] == ["heart-doc"]
    assert [hit.doc_id for hit in plain] == ["heart-doc"]
    assert folded == plain
    assert search_schemes("!!!", documents=docs) == []


def test_hinglish_expansion_uses_the_reviewed_lexicon_only() -> None:
    lexicon = {"testsurface": "alphabenefits"}
    expanded = search_schemes(
        "testsurface",
        flags={"hinglish": True},
        documents=_zones(),
        lexicon=lexicon,
    )
    suppressed = search_schemes(
        "testsurface",
        flags={"hinglish": False},
        documents=_zones(),
        lexicon=lexicon,
    )
    omitted = search_schemes("testsurface", documents=_zones(), lexicon=lexicon)

    assert [hit.doc_id for hit in expanded] == ["2-benefits"]
    assert expanded[0].text == "alphabenefits fixture scheme"
    assert suppressed == []
    assert omitted == []


def test_hinglish_flag_loads_the_reviewed_lexicon(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="reviewed lexicon file not found"):
        search_schemes(
            "testsurface",
            flags={"hinglish": True},
            documents=_zones(),
        )

    def load(cls: type[HinglishLexicon], path: object = None) -> HinglishLexicon:
        return HinglishLexicon({"testsurface": "alphabenefits"})

    monkeypatch.setattr(HinglishLexicon, "load", classmethod(load))
    loaded = search_schemes(
        "testsurface",
        flags={"hinglish": True},
        documents=_zones(),
    )
    assert [hit.doc_id for hit in loaded] == ["2-benefits"]


def test_state_filter_uses_the_parametric_index() -> None:
    docs = _zones()

    assert docs[0].state == "UP"
    hits = search_schemes("fixture", {"state": "up"}, documents=docs)

    assert [hit.doc_id for hit in hits] == ["1-eligibility", "2-benefits"]
    assert [hit.doc_id for hit in search_schemes("fixture", {"state": "BIHAR"}, documents=docs)] == [
        "3-documents",
        "4-apply",
    ]
    bihar = search_schemes("fixture", {"state": "bihar"}, documents=docs)
    assert [hit.doc_id for hit in bihar] == ["3-documents", "4-apply"]


def test_conditions_filter_requires_every_listed_condition() -> None:
    docs = _zones()

    assert docs[0].conditions == ("Heart",)
    heart = search_schemes("fixture", {"conditions": ["heart"]}, documents=docs)
    assert [hit.doc_id for hit in heart] == ["1-eligibility", "2-benefits"]
    kidney = search_schemes("fixture", {"conditions": ["Kidney"]}, documents=docs)
    assert [hit.doc_id for hit in kidney] == ["3-documents", "4-apply"]
    assert search_schemes(
        "fixture",
        {"conditions": ["heart", "kidney"]},
        documents=docs,
    ) == []


def test_zone_filter_uses_the_closed_zone_vocabulary() -> None:
    docs = _zones()

    documents = search_schemes("fixture", {"zone": "Documents"}, documents=docs)
    assert [hit.doc_id for hit in documents] == ["3-documents"]
    assert documents[0].zone == "documents"
    assert documents[0].text == "alphadocuments fixture scheme"
    assert documents[0].url == "fixture://zones/3-documents"

    apply = search_schemes("fixture", {"zone": "how_to_apply"}, documents=docs)
    assert [hit.doc_id for hit in apply] == ["4-apply"]
    assert apply[0].zone == "how_to_apply"

    with pytest.raises(ValueError, match="one of"):
        search_schemes("fixture", {"zone": "salary"}, documents=docs)


def test_top_k_stops_at_k_and_does_not_pad() -> None:
    docs = _zones()

    top_two = search_schemes("fixture", k=2, documents=docs)
    assert [hit.doc_id for hit in top_two] == ["1-eligibility", "2-benefits"]
    assert [hit.rank for hit in top_two] == [1, 2]

    assert [hit.doc_id for hit in search_schemes("fixture", k=1, documents=docs)] == [
        "1-eligibility"
    ]
    wide = search_schemes("fixture", k=10, documents=docs)
    assert [hit.doc_id for hit in wide] == [
        "1-eligibility",
        "2-benefits",
        "3-documents",
        "4-apply",
    ]
    assert [hit.rank for hit in wide] == [1, 2, 3, 4]
    assert "0-filler" not in [hit.doc_id for hit in wide]
    assert search_schemes("fixture", k=0, documents=docs) == []
    with pytest.raises(ValueError):
        search_schemes("fixture", k=-1, documents=docs)
    with pytest.raises(TypeError):
        search_schemes("fixture", k=True, documents=docs)  # type: ignore[arg-type]


def test_ranking_is_deterministic_and_breaks_ties_by_doc_id() -> None:
    docs = [
        _zone("b-tie", "benefits", "hearttie cover"),
        _zone("a-tie", "eligibility", "hearttie cover"),
        _zone(
            "z-filler",
            "other",
            "unrelated filler token",
            state=None,
            conditions=(),
        ),
    ]

    first = search_schemes("hearttie", documents=docs)
    second = search_schemes("hearttie", documents=docs)

    assert first == second
    assert [hit.doc_id for hit in first] == ["a-tie", "b-tie"]
    assert first[0].net == pytest.approx(first[1].net)
    assert [hit.rank for hit in first] == [1, 2]


def test_no_match_returns_an_empty_list() -> None:
    docs = _zones()

    assert search_schemes("zzzznotoken", documents=docs) == []
    assert search_schemes("fixture", {"state": "goa"}, documents=docs) == []
    assert search_schemes("fixture", documents=[]) == []
    assert search_schemes("...", documents=docs) == []
    with pytest.raises(ValueError, match="corpus is required"):
        search_schemes("fixture")
    with pytest.raises(ValueError, match="income"):
        search_schemes("fixture", {"income": "1"}, documents=docs)
    with pytest.raises(ValueError, match="condition"):
        search_schemes("fixture", {"condition": "heart"}, documents=docs)
    with pytest.raises(ValueError, match="salt"):
        search_schemes("fixture", {"salt": "x"}, documents=docs)
    with pytest.raises(TypeError):
        search_schemes("fixture", documents=docs[0])  # type: ignore[arg-type]


def test_zone_weight_changes_order_only_when_the_caller_supplies_it() -> None:
    docs = [
        _zone("a-eligibility", "eligibility", "heart extra"),
        _zone("b-benefits", "benefits", "heart"),
        _zone("c-documents", "documents", "heart papers"),
        _zone(
            "z-filler",
            "other",
            "unrelated filler token",
            state=None,
            conditions=(),
        ),
    ]
    table = {"eligibility": 2.0, "benefits": 1.0}

    plain = search_schemes("heart", documents=docs, zone_weights=table)
    weighted = search_schemes(
        "heart",
        flags={"zone_weights": True},
        zone_weights=table,
        weights=NetScoreWeights(1.0, 1.0, 1.0),
        documents=docs,
    )

    assert [hit.doc_id for hit in plain] == [
        "b-benefits",
        "a-eligibility",
        "c-documents",
    ]
    assert [hit.doc_id for hit in weighted] == ["a-eligibility", "b-benefits"]
    assert weighted[0].cosine < weighted[1].cosine
    assert weighted[0].zone_weight == pytest.approx(2.0)
    assert weighted[1].zone_weight == pytest.approx(1.0)
    assert weighted[0].net == pytest.approx(weighted[0].cosine * 2.0)
    assert weighted[1].net == pytest.approx(weighted[1].cosine)
    assert weighted[0].net > weighted[1].net
    assert "c-documents" not in [hit.doc_id for hit in weighted]
    with pytest.raises(TypeError, match="zone_weights"):
        search_schemes(
            "heart",
            flags={"zone_weights": True},
            weights=NetScoreWeights(1.0, 1.0, 1.0),
            documents=docs,
        )


def test_missing_g_score_excludes_the_document() -> None:
    docs = [
        _zone("missing", "eligibility", "heart uniquemissing", g_score=None),
        _zone("present", "benefits", "heart", g_score=0.0),
        _zone(
            "filler",
            "other",
            "unrelated filler token",
            g_score=1.0,
            state=None,
            conditions=(),
        ),
    ]

    weights = NetScoreWeights(1.0, 1.0, 1.0)
    hits = search_schemes(
        "heart",
        flags={"g_score": True},
        weights=weights,
        documents=docs,
    )

    assert [hit.doc_id for hit in hits] == ["present"]
    assert hits[0].g_component == pytest.approx(0.0)
    assert hits[0].net == pytest.approx(hits[0].cosine + hits[0].g_component)
    assert search_schemes(
        "uniquemissing",
        flags={"g_score": True},
        weights=weights,
        documents=docs,
    ) == []


def test_missing_freshness_is_not_taken_from_crawled_at() -> None:
    as_of = datetime(2026, 1, 11)
    crawled = ZoneDoc.from_mapping(
        {
            "doc_id": "crawled",
            "url": "fixture://zones/crawled",
            "domain": "fixture.local",
            "title": "Crawled fixture",
            "zone": "eligibility",
            "text": "heartfresh",
            "state": "up",
            "conditions": ["heart"],
            "g_score": 0.4,
            "crawled_at": "2026-01-11T00:00:00",
            "content_hash": "hash-crawled",
        }
    )
    changed = _zone(
        "changed",
        "benefits",
        "heartfresh extra",
        g_score=0.4,
        crawled_at="2026-01-02T00:00:00",
        last_changed_at="2026-01-01T00:00:00",
    )
    filler = _zone(
        "filler",
        "other",
        "unrelated filler token",
        state=None,
        conditions=(),
        g_score=0.4,
    )
    docs = [crawled, changed, filler]
    assert crawled.crawled_at == "2026-01-11T00:00:00"
    assert crawled.last_changed_at is None

    weights = NetScoreWeights(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="current time"):
        search_schemes(
            "heartfresh",
            flags={"freshness": True},
            weights=weights,
            freshness_scale_days=10,
            documents=docs,
        )
    hits = search_schemes(
        "heartfresh",
        flags={"freshness": True},
        weights=weights,
        freshness_scale_days=10,
        as_of=as_of,
        documents=docs,
    )

    assert [hit.doc_id for hit in hits] == ["changed"]
    assert hits[0].freshness == pytest.approx(freshness_from_age(10, 10))
    assert hits[0].freshness == pytest.approx(0.5)
    assert hits[0].net == pytest.approx(hits[0].cosine + hits[0].freshness)
    assert hits[0].last_changed_at == "2026-01-01T00:00:00"
    assert crawled.crawled_at != hits[0].last_changed_at
    assert "crawled" not in [hit.doc_id for hit in hits]


def test_quoted_query_requires_a_positional_phrase() -> None:
    docs = [
        _zone("p-ordered", "eligibility", "alpha beta extra"),
        _zone("p-gap", "benefits", "alpha extra beta"),
        _zone("p-reversed", "documents", "beta alpha extra"),
        _zone(
            "p-filler",
            "other",
            "unrelated filler token",
            state=None,
            conditions=(),
        ),
    ]

    plain = search_schemes("alpha beta", documents=docs)
    quoted = search_schemes('"alpha beta"', documents=docs)

    assert {hit.doc_id for hit in plain} == {"p-ordered", "p-gap", "p-reversed"}
    assert [hit.doc_id for hit in quoted] == ["p-ordered"]
    assert quoted[0].text == "alpha beta extra"
    assert quoted[0].net == pytest.approx(quoted[0].cosine)


def test_quoted_phrase_with_zero_cosine_is_not_a_hit() -> None:
    docs = [
        _zone("p1", "eligibility", "alpha beta"),
        _zone("p2", "benefits", "alpha beta"),
    ]
    cosine = TfidfRanker(
        InvertedIndex.from_documents({doc.doc_id: doc.text for doc in docs})
    ).cosine("alpha beta", "p1")

    assert cosine == pytest.approx(0.0)
    assert search_schemes('"alpha beta"', documents=docs) == []


def test_missing_netscore_weights_are_not_invented() -> None:
    docs = _zones()
    with pytest.raises(ValueError, match="coefficients are not invented"):
        search_schemes("fixture", flags={"g_score": True}, documents=docs)
    with pytest.raises(ValueError, match="coefficients are not invented"):
        search_schemes(
            "fixture",
            flags={"freshness": True},
            freshness_scale_days=10,
            as_of=datetime(2026, 1, 11),
            documents=docs,
        )
    with pytest.raises(ValueError, match="coefficients are not invented"):
        search_schemes(
            "fixture",
            flags={"zone_weights": True},
            zone_weights={"eligibility": 2.0},
            documents=docs,
        )


def test_last_changed_at_survives_onto_the_hit() -> None:
    docs = [
        _zone(
            "known",
            "eligibility",
            "uniquetoken",
            last_changed_at="2026-04-01T00:00:00",
        ),
        _zone(
            "blank",
            "benefits",
            "uniquetoken extra",
            crawled_at="2026-01-02T00:00:00",
            last_changed_at=None,
        ),
        _zone(
            "filler",
            "other",
            "unrelated filler token",
            state=None,
            conditions=(),
        ),
    ]

    hits = {hit.doc_id: hit for hit in search_schemes("uniquetoken", documents=docs)}

    assert hits["known"].last_changed_at == "2026-04-01T00:00:00"
    assert hits["blank"].last_changed_at is None
    assert docs[1].crawled_at == "2026-01-02T00:00:00"


def test_bm25_flag_is_not_combined_with_net_score_flags() -> None:
    with pytest.raises(ValueError, match="baseline"):
        search_schemes(
            "heart",
            flags={"bm25": True, "g_score": True},
            documents=_zones(),
        )


def test_bm25_path_ranks_with_the_external_baseline() -> None:
    pytest.importorskip("rank_bm25")
    from bharosa.rank.bm25 import BM25Baseline

    docs = [
        _zone("d1", "eligibility", "heart scheme", g_score=None),
        _zone("d2", "benefits", "cover notes"),
        _zone("d3", "documents", "other terms"),
        _zone("d4", "how_to_apply", "heart cover", state="Bihar"),
    ]
    baseline = BM25Baseline({doc.doc_id: doc.text for doc in docs})
    expected = baseline.rank("heart", len(docs))
    hits = search_schemes("heart", flags={"bm25": True}, documents=docs)

    assert [hit.doc_id for hit in hits] == [hit.doc_id for hit in expected]
    assert [hit.bm25_score for hit in hits] == pytest.approx(
        [hit.score for hit in expected]
    )
    assert all(hit.net is None for hit in hits)
    assert "d1" in [hit.doc_id for hit in hits]
    cosine = TfidfRanker(
        InvertedIndex.from_documents({doc.doc_id: doc.text for doc in docs})
    ).cosine("heart", hits[0].doc_id)
    assert hits[0].cosine == pytest.approx(cosine)
    assert hits[0].g_component == pytest.approx(0.0)
    assert hits[0].zone_weight == pytest.approx(1.0)

    filtered = search_schemes(
        "heart",
        {"state": "up"},
        flags={"bm25": True},
        documents=docs,
    )
    assert [hit.doc_id for hit in filtered] == ["d1"]
    assert "d4" not in [hit.doc_id for hit in filtered]


def test_module_does_not_read_the_clock_or_reimplement_rankers() -> None:
    import bharosa.index.zones as zones

    assert zones.__file__ is not None
    source = Path(zones.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    assert "datetime.now" not in source
    assert "date.today" not in source
    assert "time.time" not in source
    assert "idf" not in defined
    assert "log_tf" not in defined
    assert "freshness_from_age" not in defined
    assert "bharosa.rank.tfidf" in imported
    assert "bharosa.rank.netscore" in imported
    assert "bharosa.rank.bm25" in imported
    assert "bharosa.index.inverted" in imported
    assert "bharosa.index.positional" in imported
    assert "bharosa.index.params" in imported
    assert "bharosa.text.hinglish" in imported
    assert "bharosa.text.normalize" in imported
    assert "rank_bm25" not in imported
