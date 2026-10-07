"""Evaluation harness tests.

The scheme qrels in this file live only inside the test. They are not
written to ``eval/``. The repository's own label files are read to check
that unreviewed and missing judgements stay out of the scored metrics.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("bharosa_run_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bharosa_run_eval"] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(rows: list[dict[str, str]], method: str, relevance: str, column: str) -> str:
    matches = [
        row for row in rows if row["method"] == method and row["relevance"] == relevance
    ]
    assert len(matches) == 1
    return matches[0][column]


def test_precision_and_recall_at_k() -> None:
    flags = [True, False, True, False, False, True]
    assert run_eval.precision_at_k(flags, 5) == pytest.approx(2 / 5)
    assert run_eval.recall_at_k(flags, 5, 4) == pytest.approx(2 / 4)
    assert run_eval.top1_hit(flags) == 1.0
    assert run_eval.precision_at_k([], 5) == 0.0
    assert run_eval.recall_at_k([], 5, 3) == 0.0
    assert run_eval.set_precision([]) == 0.0
    assert run_eval.recall_at_k([True], 5, 0) is None
    assert run_eval.macro_average([]) is None
    assert run_eval.macro_average([1.0, 0.0]) == pytest.approx(0.5)


def test_ranked_doc_metrics_use_the_cutoff_denominator() -> None:
    metrics = run_eval.ranked_doc_metrics(["a", "b", "a"], {"a", "c"}, ks=(5, 10))
    assert metrics["P@5"] == pytest.approx(1 / 5)
    assert metrics["recall@5"] == pytest.approx(1 / 2)
    assert metrics["n_returned"] == 2


def test_real_medicine_queries_are_partitioned() -> None:
    rows = run_eval.load_csv(ROOT / "eval" / "medicine_queries.csv")
    counts: dict[str, int] = {}
    for row in rows:
        status = row["label_status"].strip()
        counts[status] = counts.get(status, 0) + 1
    assert len(rows) == 29
    assert counts["VERIFIED_FROM_DATASET"] == 15
    assert counts["NEEDS_HUMAN_REVIEW"] == 14
    scored_hard = [
        row
        for row in rows
        if row["label_status"].strip() == "VERIFIED_FROM_DATASET"
        and run_eval.notes_are_misspelling_or_hinglish(row.get("notes") or "")
    ]
    assert scored_hard == []


def test_real_claims_are_not_scored() -> None:
    rows = run_eval.load_csv(ROOT / "eval" / "claims.csv")
    result = run_eval.score_claims(rows, retrieve=None, allow_mock=False)
    assert len(rows) == 5
    assert result["n_human_reviewed_yes"] == 0
    assert result["n_ai_drafted"] == 5
    assert result["n_scored"] == 0
    assert result["precision"]["value"] == "N/A"
    assert result["recall"]["value"] == "N/A"
    assert result["keyword_baseline"]["value"] == "N/A"
    assert all(row["used_for_precision_recall"] is False for row in result["audit_rows"])
    assert all(row["human_reviewed"] == "NEEDS_HUMAN_REVIEW" for row in result["audit_rows"])


def test_real_scheme_rag_and_ablation_are_na() -> None:
    paths = run_eval.RunPaths.defaults()
    scheme = run_eval.evaluate_scheme(paths)
    assert scheme["ran"] is False
    assert scheme["n_qrel_pairs"] == 0
    assert scheme["n_zones"] == 0
    p_at_5 = [
        row
        for row in scheme["metric_rows"]
        if row["system"] == "plain_tfidf" and row["metric"] == "P@5"
    ]
    assert len(p_at_5) == 1
    assert p_at_5[0]["value"] == "N/A"
    assert "scheme_qrels.csv" in p_at_5[0]["reason"]
    assert all(row["value"] == "N/A" for row in scheme["ablation_rows"])
    assert {row["component"] for row in scheme["ablation_rows"]} == set(
        run_eval.ABLATION_FLAGS
    )
    hinglish = next(row for row in scheme["ablation_rows"] if row["component"] == "hinglish")
    assert hinglish["implemented"] is True
    assert hinglish["backed_by_labelled_data"] is False
    assert "hinglish_lexicon.csv" in hinglish["reason"]
    rag = run_eval.evaluate_rag(paths)
    assert all(row["value"] == "N/A" for row in rag["metric_rows"])
    assert rag["thresholds"]["tuned_on_this_run"] is False
    assert rag["thresholds"]["used_to_compute_metrics"] is False
    freshness = run_eval.evaluate_freshness(paths)
    assert all(row["value"] == "N/A" for row in freshness["metric_rows"])
    assert not (ROOT / "eval" / "scheme_qrels.csv").exists()
    assert not (ROOT / "eval" / "scheme_queries.csv").exists()


def test_reviewed_claim_is_scored_and_draft_row_is_not() -> None:
    try:
        from tests.test_rag_fixtures import mock_scheme_hits
    except ImportError:
        from test_rag_fixtures import mock_scheme_hits

    rows = [
        {
            "message": "pay Rs 500 to activate free health card",
            "label": "CONTRADICTED",
            "origin": "HUMAN",
            "human_reviewed": "yes",
            "source_note": "test fixture, not a repository gold file",
        },
        {
            "message": "hello how are you today",
            "label": "CONTRADICTED",
            "origin": "AI_DRAFTED",
            "human_reviewed": "NEEDS_HUMAN_REVIEW",
            "source_note": "must stay out of the counts",
        },
    ]
    result = run_eval.score_claims(
        rows,
        retrieve=lambda _query: mock_scheme_hits(),
        allow_mock=True,
    )
    assert result["n_scored"] == 1
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"]["value"] == pytest.approx(1.0)
    assert result["recall"]["value"] == pytest.approx(1.0)
    draft = result["audit_rows"][1]
    assert draft["used_for_precision_recall"] is False


def _zone(doc_id: str, text: str, g_score: float, zone: str = "eligibility") -> dict:
    return {
        "doc_id": doc_id,
        "url": f"fixture://zones/{doc_id}",
        "domain": "fixture.local",
        "title": doc_id,
        "zone": zone,
        "text": text,
        "state": "UP",
        "conditions": ["Heart"],
        "g_score": g_score,
        "crawled_at": "2026-01-02T00:00:00",
        "last_changed_at": "2026-01-01T00:00:00",
        "content_hash": f"hash-{doc_id}",
    }


def _write_scheme_fixture(tmp: Path) -> run_eval.RunPaths:
    eval_dir = tmp / "eval"
    data_dir = tmp / "data"
    config_dir = tmp / "config"
    eval_dir.mkdir()
    data_dir.mkdir()
    config_dir.mkdir()
    zones = [
        _zone("a-relevant", "targetword", 0.0),
        _zone("b1", "targetword", 1.0),
        _zone("b2", "targetword", 1.0),
        _zone("b3", "targetword", 1.0),
        _zone("b4", "targetword", 1.0),
        _zone("b5", "targetword", 1.0),
        _zone("z-filler", "unrelatedtoken", 0.1, zone="other"),
    ]
    zone_path = eval_dir / "scheme_zones.jsonl"
    zone_path.write_text(
        "\n".join(json.dumps(zone) for zone in zones) + "\n",
        encoding="utf-8",
    )
    query_path = eval_dir / "scheme_queries.csv"
    with query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "query": "targetword"})
    qrel_path = eval_dir / "scheme_qrels.csv"
    with qrel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "doc_id", "relevance"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "doc_id": "a-relevant", "relevance": "1"})
    from bharosa.index.params import ZONES

    weights = {
        "cosine": 1.0,
        "g_score": 1.0,
        "freshness": 1.0,
        "zone_weights": {zone: 1.0 for zone in ZONES},
        "freshness_scale_days": 30.0,
        "as_of": "2026-02-01T00:00:00",
    }
    (config_dir / "net_weights.json").write_text(json.dumps(weights), encoding="utf-8")
    (config_dir / "hinglish_lexicon.csv").write_text(
        "source,canonical\nsasta,cheap\n",
        encoding="utf-8",
    )
    return run_eval.RunPaths(
        root=tmp,
        output=tmp / "out",
        medicine_csv=tmp / "missing-medicines.csv",
        medicine_queries=tmp / "missing-queries.csv",
        claims_csv=tmp / "missing-claims.csv",
        eval_dir=eval_dir,
        data_dir=data_dir,
        config_dir=config_dir,
        lexicon_csv=config_dir / "hinglish_lexicon.csv",
    )


def _scheme_value(result: dict, system: str, metric: str) -> object:
    matches = [
        row
        for row in result["metric_rows"]
        if row["system"] == system and row["metric"] == metric
    ]
    assert len(matches) == 1
    return matches[0]["value"]


def test_scheme_metrics_when_labels_and_weights_are_supplied(tmp_path: Path) -> None:
    paths = _write_scheme_fixture(tmp_path)
    result = run_eval.evaluate_scheme(paths)
    assert result["ran"] is True
    assert result["n_qrel_pairs"] == 1
    assert result["n_zones"] == 7
    assert _scheme_value(result, "plain_tfidf", "P@5") == pytest.approx(0.2)
    assert _scheme_value(result, "plain_tfidf", "recall@5") == pytest.approx(1.0)
    assert _scheme_value(result, "bm25", "P@5") == pytest.approx(0.2)
    assert _scheme_value(result, "bm25", "recall@5") == pytest.approx(1.0)
    assert _scheme_value(result, "enhanced", "P@5") == pytest.approx(0.0)
    assert _scheme_value(result, "enhanced", "recall@5") == pytest.approx(0.0)
    assert _scheme_value(result, "enhanced", "recall@10") == pytest.approx(1.0)
    deltas = {row["component"]: row["value"] for row in result["ablation_rows"]}
    assert deltas["g_score"] == pytest.approx(0.2)
    assert deltas["freshness"] == pytest.approx(0.0)
    assert deltas["zone_weights"] == pytest.approx(0.0)
    assert deltas["hinglish"] == pytest.approx(0.0)
    assert result["bm25_parameters"]["fit_on_labels"] is False
    assert not (ROOT / "eval" / "scheme_qrels.csv").exists()


def _write_toy_medicines(path: Path) -> None:
    fieldnames = [
        "brand_name",
        "manufacturer",
        "price_inr",
        "is_discontinued",
        "dosage_form",
        "num_active_ingredients",
        "primary_ingredient",
        "primary_strength",
        "active_ingredients",
    ]
    rows = [
        ["Dolo 650 Tablet", "Maker A", "30", "False", "tablet", "1", "Paracetamol", "650mg", ""],
        ["Dolo 650 Tablet", "Maker B", "20", "False", "tablet", "1", "Paracetamol", "650mg", ""],
        ["Calpol 650 Tablet", "Maker C", "25", "False", "tablet", "1", "Paracetamol", "650mg", ""],
        ["Crocin", "Maker E", "15", "False", "tablet", "1", "Paracetamol", "500mg", ""],
        ["Azithral 500 Tablet", "Maker D", "100", "False", "tablet", "1", "Azithromycin", "500mg", ""],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _write_toy_queries(path: Path) -> None:
    fieldnames = [
        "query",
        "gold_brand",
        "gold_salt",
        "gold_strength",
        "gold_form",
        "label_status",
        "notes",
    ]
    rows = [
        [
            "Dolo 650 Tablet",
            "Dolo 650 Tablet",
            "Paracetamol",
            "650.0 mg",
            "tablet",
            "VERIFIED_FROM_DATASET",
            "Exact brand query",
        ],
        [
            "Dolo 650",
            "Dolo 650 Tablet",
            "Paracetamol",
            "650.0 mg",
            "tablet",
            "VERIFIED_FROM_DATASET",
            "Brand and strength query",
        ],
        [
            "Crocin",
            "Crocin",
            "Paracetamol",
            "500.0 mg",
            "tablet",
            "VERIFIED_FROM_DATASET",
            "Exact short brand",
        ],
        [
            "Nope Tablet",
            "Nope Tablet",
            "Paracetamol",
            "10.0 mg",
            "tablet",
            "VERIFIED_FROM_DATASET",
            "Absent from this toy corpus",
        ],
        [
            "Doloo 650",
            "Dolo 650 Tablet",
            "Paracetamol",
            "650.0 mg",
            "tablet",
            "NEEDS_HUMAN_REVIEW",
            "misspelling candidate",
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _write_toy_claims(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["message", "label", "source_note", "origin", "human_reviewed"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "message": "pay Rs 500 to activate free health card",
                "label": "CONTRADICTED",
                "source_note": "toy audit row",
                "origin": "AI_DRAFTED",
                "human_reviewed": "NEEDS_HUMAN_REVIEW",
            }
        )


def test_end_to_end_toy_run(tmp_path: Path) -> None:
    medicine_csv = tmp_path / "medicines.csv"
    queries = tmp_path / "queries.csv"
    claims = tmp_path / "claims.csv"
    _write_toy_medicines(medicine_csv)
    _write_toy_queries(queries)
    _write_toy_claims(claims)
    output = tmp_path / "results"
    paths = run_eval.RunPaths(
        root=tmp_path,
        output=output,
        medicine_csv=medicine_csv,
        medicine_queries=queries,
        claims_csv=claims,
        eval_dir=tmp_path / "eval",
        data_dir=tmp_path / "data",
        config_dir=tmp_path / "config",
        lexicon_csv=tmp_path / "config" / "hinglish_lexicon.csv",
    )
    paths.eval_dir.mkdir()
    summary = run_eval.execute(paths, progress=False)

    assert summary["tuned_on_this_run"] is False
    assert summary["medicine_min_brand_cosine"] == run_eval.MIN_BRAND_COSINE
    medicine = summary["medicine"]
    assert medicine["n_queries"] == 5
    assert medicine["n_verified"] == 4
    assert medicine["n_needs_human_review"] == 1
    assert medicine["n_scored_brand"] == 3
    assert medicine["n_excluded_brand_not_in_corpus"] == 1
    assert medicine["n_misspelling_or_hinglish_scored"] == 0
    assert medicine["accepted_records"] == 5
    assert any("too small" in line for line in summary["limitations"])

    metrics = _read_csv(output / "medicine_metrics.csv")
    assert _metric(metrics, "exact", "gold_brand", "n_queries_scored") == "3"
    assert float(_metric(metrics, "exact", "gold_brand", "top1_accuracy")) == pytest.approx(2 / 3)
    assert float(_metric(metrics, "soundex", "gold_brand", "top1_accuracy")) == pytest.approx(1.0)
    # "dolo" against "Dolo 650 Tablet" is under MIN_BRAND_COSINE, so those
    # two queries miss. "Crocin" matches the recorded brand and hits.
    assert float(_metric(metrics, "ours_ngram_parametric", "gold_brand", "top1_accuracy")) == pytest.approx(1 / 3)
    assert float(_metric(metrics, "ours_ngram_parametric", "gold_brand", "P@5")) == pytest.approx(
        0.2 / 3, abs=1e-5
    )
    assert _metric(metrics, "brand_token_tfidf", "N/A", "P@5") == "N/A"
    assert _metric(metrics, "brand_bm25", "N/A", "P@5") == "N/A"

    per_query = _read_csv(output / "medicine_per_query.csv")
    short = [
        row
        for row in per_query
        if row["query"] == "Dolo 650" and row["method"] == "exact"
    ]
    assert short[0]["used_in_scored_metrics"] == "true"
    assert float(short[0]["brand_top1"]) == pytest.approx(0.0)
    ours_short = [
        row
        for row in per_query
        if row["query"] == "Dolo 650" and row["method"] == "ours_ngram_parametric"
    ]
    assert float(ours_short[0]["brand_top1"]) == pytest.approx(0.0)
    ours_crocin = [
        row
        for row in per_query
        if row["query"] == "Crocin" and row["method"] == "ours_ngram_parametric"
    ]
    assert float(ours_crocin[0]["brand_top1"]) == pytest.approx(1.0)
    failures = _read_csv(output / "medicine_failures.csv")
    assert {row["query"] for row in failures} == {"Dolo 650 Tablet", "Dolo 650"}
    review = [row for row in per_query if row["query"] == "Doloo 650"]
    assert review
    assert all(row["used_in_scored_metrics"] == "false" for row in review)
    assert all(row["brand_P@5"] == "N/A" for row in review)

    unreviewed = _read_csv(output / "medicine_unreviewed_predictions.csv")
    assert "P@5" not in unreviewed[0]
    assert {row["query"] for row in unreviewed} == {"Doloo 650"}
    assert all(row["human_review"] == "required" for row in unreviewed)
    assert all(row["label_status"] == "NEEDS_HUMAN_REVIEW" for row in unreviewed)

    exclusions = _read_csv(output / "medicine_exclusions.csv")
    assert [row["query"] for row in exclusions] == ["Nope Tablet"]

    scheme = _read_csv(output / "scheme_metrics.csv")
    assert all(row["value"] == "N/A" for row in scheme)
    claims_metrics = _read_csv(output / "claims_metrics.csv")
    precision = next(row for row in claims_metrics if row["metric"] == "precision_contradicted")
    assert precision["value"] == "N/A"
    audit = _read_csv(output / "claims_label_audit.csv")
    assert audit[0]["used_for_precision_recall"] == "false"
    assert audit[0]["origin"] == "AI_DRAFTED"
    assert not (output / "scheme_qrels.csv").exists()
    assert (output / "summary.json").is_file()
    assert not (ROOT / "eval" / "scheme_qrels.csv").exists()


def test_search_flags_match_the_ablation_list() -> None:
    from bharosa.index.zones import search_schemes

    for name in (*run_eval.ABLATION_FLAGS, "bm25"):
        hits = search_schemes("targetword", documents=[], k=1, flags={name: False})
        assert hits == []
    with pytest.raises(ValueError, match="unknown search flag"):
        search_schemes("targetword", documents=[], k=1, flags={"invented": False})
