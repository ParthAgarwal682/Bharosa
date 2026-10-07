"""RAG and claim-check evaluation harness (Person D owned).

Writes tables under ``eval/results/rag_fixture_*`` and ``eval/results/claims_fixture_*``
only from executable runs. Fixture mode is labelled MOCK_FIXTURE and marked
as NOT FINAL.

Usage:
  python eval/rag_eval.py --fixtures
  python eval/rag_eval.py --claims-only --fixtures
  python eval/rag_eval.py --rag-only --fixtures

Setup requirements for non-fixture runs:
  - The team retriever wired by the caller (not implemented here)
  - Human-reviewed labels in eval/claims.csv (human_reviewed=yes)
  - LLM credentials for live answer generation

This script does NOT edit or replace eval/run_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Allow running as ``python eval/rag_eval.py`` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bharosa.rag.answer import answer
from bharosa.rag.citations import check_citations
from bharosa.rag.claimcheck import check_claim
from bharosa.rag.config import get_config
from bharosa.rag.types import ZoneHit

RESULTS = _ROOT / "eval" / "results"
CLAIMS_CSV = _ROOT / "eval" / "claims.csv"


class _EvalFakeLLM:
    """Local stand-in for FakeLLM in evaluation fixtures mode."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []
        self.call_count: int = 0

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        return self.response


def _eval_mock_zones() -> list[ZoneHit]:
    """Local mock fixture zones used only for --fixtures dry-run evaluation."""
    return [
        ZoneHit(
            doc_id="fixture-ayushman-eligibility",
            url="https://example.invalid/ayushman/eligibility",
            domain="example.invalid",
            title="Ayushman Bharat eligibility (fixture)",
            zone="eligibility",
            text=(
                "Families with annual income below two lakh rupees in "
                "Uttar Pradesh may be eligible for Ayushman Bharat cover "
                "subject to the official beneficiary list."
            ),
            state="UP",
            conditions=["heart", "hospitalization"],
            g_score=0.9,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-elig",
            score=0.82,
            rank=1,
            source="mock_fixture",
        ),
        ZoneHit(
            doc_id="fixture-ayushman-benefits",
            url="https://example.invalid/ayushman/benefits",
            domain="example.invalid",
            title="Ayushman Bharat benefits (fixture)",
            zone="benefits",
            text=(
                "The scheme provides health cover up to five lakh rupees "
                "per family per year for listed secondary and tertiary care. "
                "There is no registration fee to activate the card."
            ),
            state="UP",
            conditions=["hospitalization"],
            g_score=0.9,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-ben",
            score=0.71,
            rank=2,
            source="mock_fixture",
        ),
        ZoneHit(
            doc_id="fixture-ayushman-documents",
            url="https://example.invalid/ayushman/documents",
            domain="example.invalid",
            title="Ayushman Bharat documents (fixture)",
            zone="documents",
            text=(
                "Carry a government photo identity proof and the beneficiary "
                "card at the empanelled hospital. Do not share Aadhaar OTP "
                "with unknown callers."
            ),
            state="UP",
            conditions=[],
            g_score=0.85,
            crawled_at="2026-01-01T00:00:00+00:00",
            last_changed_at="2026-01-01T00:00:00+00:00",
            content_hash="fixture-hash-docs",
            score=0.55,
            rank=3,
            source="mock_fixture",
        ),
    ]


def _eval_weak_zones() -> list[ZoneHit]:
    """Local weak mock fixture zones for out-of-scope refusal check."""
    base = _eval_mock_zones()[0]
    return [
        ZoneHit(
            doc_id="fixture-weak",
            url=base.url,
            domain=base.domain,
            title=base.title,
            zone=base.zone,
            text=base.text,
            state=base.state,
            conditions=list(base.conditions),
            g_score=base.g_score,
            crawled_at=base.crawled_at,
            last_changed_at=base.last_changed_at,
            content_hash=base.content_hash,
            score=0.02,
            rank=1,
            source="mock_fixture",
        )
    ]


def _ensure_results_dir() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def _get_run_config(*, model: str | None = None, query_count: int = 0) -> dict:
    """Capture configuration used for this evaluation run."""
    cfg = get_config()
    return {
        "refusal_score_threshold": cfg.refusal_score_threshold,
        "citation_threshold": cfg.citation_threshold,
        "min_retrieval_score": cfg.min_retrieval_score,
        "model": model,
        "query_count": query_count,
        "run_date": datetime.now(timezone.utc).isoformat(),
    }


def run_rag_fixture_eval() -> dict:
    """Run refusal + citation checks on mock fixtures with FakeLLM.

    Numbers are reproducible over fixtures only. Marked as NOT FINAL.
    """
    print("NOT FINAL — fixture data")
    hits = _eval_mock_zones()
    llm = _EvalFakeLLM(
        response=json.dumps(
            {
                "claims": [
                    {
                        "text": "Families with income below two lakh in UP may be eligible.",
                        "cite_ids": ["Z1"],
                    },
                    {
                        "text": "Cover is up to five lakh rupees per family per year.",
                        "cite_ids": ["Z2"],
                    },
                    {
                        "text": "Martians get free care everywhere.",
                        "cite_ids": ["Z2"],
                    },
                ]
            }
        )
    )

    in_scope_queries = [
        "papa ke heart operation ke liye yojana, UP, income 2 lakh",
    ]
    for q in in_scope_queries:
        in_scope = answer(q, hits, llm=llm)

    checks = check_citations(in_scope, hits)
    supported = sum(1 for c in checks if c.supported)
    total = len(checks)

    oos_queries = [
        "scheme for building a private rocket pad in Antarctica",
        "eligibility for free unicorn surgery in Goa",
    ]
    refusals = 0
    for q in oos_queries:
        r = answer(q, _eval_weak_zones(), llm=_EvalFakeLLM(response="should not run"))
        if r.refused:
            refusals += 1
    empty = answer("anything", [], llm=_EvalFakeLLM(response="no"))
    if empty.refused:
        refusals += 1
    refusal_denom = len(oos_queries) + 1

    total_queries_executed = len(in_scope_queries) + len(oos_queries) + 1

    return {
        "final": False,
        "mode": "MOCK_FIXTURE",
        "status": "NOT FINAL — fixture data",
        "config": _get_run_config(model="_EvalFakeLLM", query_count=total_queries_executed),
        "warning": (
            "Results use hand-written mock_fixture zones and FakeLLM. "
            "Do not cite these as live crawl / production evaluation."
        ),
        "sentences_total": total,
        "sentences_supported": supported,
        "pct_sentences_supported": (100.0 * supported / total) if total else 0.0,
        "refusal_count": refusals,
        "refusal_denom": refusal_denom,
        "refusal_rate_pct": (100.0 * refusals / refusal_denom) if refusal_denom else 0.0,
        "citation_flags": Counter(c.flag or "ok" for c in checks),
    }


def run_claims_eval(*, use_fixtures: bool) -> dict:
    """Score claim checker against eval/claims.csv.

    Only rows with human_reviewed=yes count toward precision/recall.
    If none exist, prints 'INSUFFICIENT INFORMATION — no human-reviewed labels'
    and computes no precision/recall metrics.
    For unreviewed rows, records predictions only.
    """
    if not CLAIMS_CSV.is_file():
        return {
            "error": "INSUFFICIENT INFORMATION — DO NOT GUESS. Missing eval/claims.csv"
        }

    zones = _eval_mock_zones() if use_fixtures else None
    rows: list[dict[str, str]] = []
    with CLAIMS_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    reviewed = [r for r in rows if (r.get("human_reviewed") or "").strip().lower() == "yes"]
    unreviewed = [r for r in rows if r not in reviewed]

    reviewed_metrics = None
    if reviewed and zones is not None:
        tp = fp = fn = tn = 0
        details: list[dict] = []
        for row in reviewed:
            message = row["message"]
            gold = (row.get("label") or "").strip().upper()
            pred = check_claim(message, lambda q: zones).label
            details.append(
                {
                    "message": message,
                    "gold": gold,
                    "pred": pred,
                    "origin": row.get("origin", ""),
                }
            )
            gold_pos = gold == "CONTRADICTED"
            pred_pos = pred == "CONTRADICTED"
            if gold_pos and pred_pos:
                tp += 1
            elif not gold_pos and pred_pos:
                fp += 1
            elif gold_pos and not pred_pos:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        reviewed_metrics = {
            "n": len(reviewed),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision_contradicts": precision,
            "recall_contradicts": recall,
            "details": details,
        }
    elif not reviewed:
        print("INSUFFICIENT INFORMATION — no human-reviewed labels")

    # For other (unreviewed) rows, record predictions only without computing precision/recall
    unreviewed_predictions: list[dict] = []
    if zones is not None:
        for row in unreviewed:
            message = row["message"]
            pred = check_claim(message, lambda q: zones).label
            unreviewed_predictions.append(
                {
                    "mode": "MOCK_FIXTURE",
                    "message": message,
                    "draft_label": (row.get("label") or "").strip(),
                    "pred": pred,
                    "origin": row.get("origin", ""),
                    "human_reviewed": (row.get("human_reviewed") or "").strip(),
                }
            )

    return {
        "final": False,
        "mode": "MOCK_FIXTURE" if use_fixtures else "NO_ZONES",
        "status": "NOT FINAL — fixture data",
        "config": _get_run_config(query_count=len(rows)),
        "warning": (
            "Draft rows (human_reviewed!=yes) are not gold labels. "
            "AI-generated or unreviewed labels must not be reported as ground truth."
        ),
        "reviewed_metrics": reviewed_metrics,
        "reviewed_note": "INSUFFICIENT INFORMATION — no human-reviewed labels" if not reviewed else None,
        "unreviewed_count": len(unreviewed),
        "total_rows_processed": len(rows),
        "unreviewed_predictions": unreviewed_predictions,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry for RAG/claim evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Use mock_fixture zones + FakeLLM (labelled MOCK_FIXTURE in output)",
    )
    parser.add_argument(
        "--claims-only",
        action="store_true",
        help="Only run claim-check evaluation",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Only run RAG fixture evaluation",
    )
    args = parser.parse_args(argv)

    if not args.fixtures:
        print(
            "INSUFFICIENT INFORMATION — DO NOT GUESS.\n"
            "Cannot run in production mode. Missing requirements:\n"
            "1. Real retriever: the team retriever is not integrated.\n"
            "2. Real labels: No human-reviewed ground truth (human_reviewed=yes) in eval/claims.csv.",
            file=sys.stderr,
        )
        return 2

    out_dir = _ensure_results_dir()
    wrote: list[str] = []

    if args.fixtures and not args.claims_only:
        rag = run_rag_fixture_eval()
        rag["citation_flags"] = dict(rag.get("citation_flags", {}))
        path = out_dir / "rag_fixture_metrics.json"
        path.write_text(json.dumps(rag, indent=2), encoding="utf-8")
        wrote.append(str(path))
        print(json.dumps(rag, indent=2))

    if args.fixtures and not args.rag_only:
        claims = run_claims_eval(use_fixtures=True)
        # Pop unreviewed_predictions before dumping metrics JSON
        preds = claims.pop("unreviewed_predictions", [])
        path = out_dir / "claims_fixture_metrics.json"
        path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
        wrote.append(str(path))

        # Write predictions CSV with mode column
        csv_path = out_dir / "claims_fixture_predictions.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "mode",
                    "message",
                    "draft_label",
                    "pred",
                    "origin",
                    "human_reviewed",
                ],
            )
            writer.writeheader()
            for row in preds:
                writer.writerow(row)
        wrote.append(str(csv_path))
        print(json.dumps(claims, indent=2))

    print("Wrote:", *wrote, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
