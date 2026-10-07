"""Reproduce Bharosa evaluation from one command.

    python eval/run_eval.py

Numbers written under ``eval/results/`` are computed in this process.
A metric is ``N/A`` when the labelled data or the component it needs is
absent. This script does not create relevance judgements, does not search
cutoffs, and does not read mock-fixture result files.

Medicine relevance uses ``eval/medicine_queries.csv``.

- ``VERIFIED_FROM_DATASET`` rows are the scored set.
- Any other ``label_status``, including ``NEEDS_HUMAN_REVIEW``, is saved
  for review and left out of the averages.
- A retrieved row is brand-relevant when its normalised brand equals the
  verified ``gold_brand``. The relevant-set size is the number of accepted
  corpus rows with that brand.
- A retrieved row is identity-relevant when salt, strength, unit, and
  dosage form match the verified gold fields. An empty gold strength
  matches rows that have no scalar strength. The relevant set is counted
  from the accepted corpus. No qrel file is written.

``P@k`` divides by ``k``. Ranks the system did not return count as not
relevant. Exact match and Soundex have no score; their ``P@k`` uses
corpus order and ``set_recall`` uses the full returned list. The n-gram
matcher is called at ``k=10`` and ranked by its own cosine.

Scheme ``P@k``, BM25, enhanced retrieval, and ablations run only when
queries, qrels, and a zone corpus are already on disk. Enhanced
retrieval and ablations also require the reviewed Hinglish lexicon and a
declared weight file. Missing pieces stay ``N/A``.

Claim precision and recall use the positive class ``CONTRADICTED`` and
only rows with ``human_reviewed=yes``. RAG citation support, refusal,
and answer correctness require a human judgement file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bharosa.index.params import ZONES
from bharosa.index.zones import ZoneDoc, search_schemes
from bharosa.medicine.baselines import match_exact, match_phonetic
from bharosa.medicine.loader import LoadStats, MedicineRecord, load_medicines
from bharosa.medicine.parametric import canonical_form
from bharosa.medicine.search import MIN_BRAND_COSINE, search_medicine
from bharosa.rag.claimcheck import check_claim
from bharosa.rag.config import get_config
from bharosa.rank.bm25 import LIBRARY_B, LIBRARY_EPSILON, LIBRARY_K1
from bharosa.rank.netscore import NetScoreWeights
from bharosa.text.hinglish import DEFAULT_LEXICON_PATH, HinglishLexicon
from bharosa.text.normalize import normalize

NA = "N/A"
VERIFIED_STATUS = "VERIFIED_FROM_DATASET"
REVIEW_STATUS = "NEEDS_HUMAN_REVIEW"
CLAIM_POSITIVE = "CONTRADICTED"
OURS_K = 10
KS = (5, 10)
SMALL_QUERY_SET = 30
ABLATION_FLAGS = ("zone_weights", "g_score", "freshness", "hinglish")
SCHEME_SYSTEMS = ("plain_tfidf", "bm25", "enhanced")
RAG_JUDGEMENT_NAMES = (
    "rag_judgements.csv",
    "rag_labels.csv",
    "oos_queries.csv",
    "answer_judgements.csv",
)
WEIGHTS_NAME = "net_weights.json"
ZONE_CORPUS_NAME = "scheme_zones.jsonl"

_STRENGTH = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z%]+(?:/[0-9]*[A-Za-z]+)?)$"
)


@dataclass(frozen=True)
class RunPaths:
    """Input and output locations for one evaluation run."""

    root: Path
    output: Path
    medicine_csv: Path
    medicine_queries: Path
    claims_csv: Path
    eval_dir: Path
    data_dir: Path
    config_dir: Path
    lexicon_csv: Path

    @classmethod
    def defaults(cls) -> RunPaths:
        """Paths for ``python eval/run_eval.py`` from a checkout of this repo."""
        return cls(
            root=_ROOT,
            output=_ROOT / "eval" / "results",
            medicine_csv=_ROOT / "data" / "medicines" / "indian_pharmaceutical_products_clean.csv",
            medicine_queries=_ROOT / "eval" / "medicine_queries.csv",
            claims_csv=_ROOT / "eval" / "claims.csv",
            eval_dir=_ROOT / "eval",
            data_dir=_ROOT / "data",
            config_dir=_ROOT / "config",
            lexicon_csv=DEFAULT_LEXICON_PATH,
        )


@dataclass(frozen=True)
class Hit:
    """One retrieved medicine row, in the order the matcher returned it."""

    brand: str
    salt: str | None
    strength_value: float | None
    strength_unit: str | None
    form: str | None
    score: float | None


@dataclass(frozen=True)
class DeclaredWeights:
    """Caller-declared net-score coefficients read from a file.

    IR concept: a frozen ranking configuration. The values are whatever
    the file contains. This module does not search them.
    """

    weights: NetScoreWeights
    zone_weights: dict[str, float]
    freshness_scale_days: float
    as_of: str
    source: str


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Fraction of the first ``k`` hits that are relevant.

    IR concept: precision at cutoff. The denominator is ``k``. A list
    shorter than ``k`` does not shrink the denominator.
    """
    _check_k(k)
    return sum(1 for flag in list(flags)[:k] if flag) / k


def recall_at_k(flags: Sequence[bool], k: int, n_relevant: int) -> float | None:
    """Relevant hits inside the first ``k``, divided by the relevant-set size.

    IR concept: recall at cutoff. ``None`` when the relevant set is empty,
    because the ratio is then undefined.
    """
    _check_k(k)
    if n_relevant <= 0:
        return None
    return sum(1 for flag in list(flags)[:k] if flag) / n_relevant


def top1_hit(flags: Sequence[bool]) -> float:
    """1 when the first hit is relevant, otherwise 0."""
    return 1.0 if flags and flags[0] else 0.0


def set_precision(flags: Sequence[bool]) -> float:
    """Relevant rows divided by returned rows.

    An empty return is 0. The matcher found nothing it could count as a hit.
    """
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def set_recall(flags: Sequence[bool], n_relevant: int) -> float | None:
    """Relevant rows in the whole returned list, divided by the relevant set."""
    if n_relevant <= 0:
        return None
    return sum(1 for flag in flags if flag) / n_relevant


def macro_average(values: Sequence[float]) -> float | None:
    """Unweighted mean across queries. ``None`` when there are no queries."""
    if not values:
        return None
    return sum(values) / len(values)


def ranked_doc_metrics(
    ranked_ids: Sequence[str],
    relevant: set[str],
    ks: Sequence[int] = KS,
) -> dict[str, float | None]:
    """P@k and recall@k for one ranked list of document ids.

    A document id is counted once even if a ranker repeated it. Recall is
    undefined when ``relevant`` is empty.
    """
    flags: list[bool] = []
    seen: set[str] = set()
    for doc_id in ranked_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        flags.append(doc_id in relevant)
    out: dict[str, float | None] = {
        "top1": top1_hit(flags),
        "set_precision": set_precision(flags),
        "set_recall": set_recall(flags, len(relevant)),
        "n_returned": float(len(flags)),
        "n_relevant": float(len(relevant)),
    }
    for k in ks:
        out[f"P@{k}"] = precision_at_k(flags, k)
        out[f"recall@{k}"] = recall_at_k(flags, k, len(relevant))
    return out


def load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as strings. A missing file returns an empty list."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: list[dict[str, str]], columns: Sequence[str], path: Path) -> None:
    """Raise when a present header is missing a required column."""
    if not rows and not path.is_file():
        return
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
    missing = [name for name in columns if name not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")


def parse_gold_strength(raw: str) -> tuple[float | None, str | None] | None:
    """Parse a gold strength cell.

    An empty cell means the label has no scalar strength. A cell that
    does not match the number-plus-unit pattern returns ``None`` so the
    identity is left unscored rather than guessed.
    """
    text = (raw or "").strip()
    if text == "":
        return (None, None)
    match = _STRENGTH.fullmatch(text)
    if match is None:
        return None
    return (round(float(match.group(1)), 6), normalize(match.group(2)))


def identity_key(
    salt: str | None,
    strength_value: float | None,
    strength_unit: str | None,
    form: str | None,
) -> tuple[str, float | None, str | None, str]:
    """Normalised salt, strength, unit, and form used as a relevance key."""
    salt_key = normalize(salt) if salt else ""
    if strength_value is None:
        value: float | None = None
        unit: str | None = None
    else:
        value = round(float(strength_value), 6)
        unit = normalize(strength_unit) if strength_unit else None
    form_key = canonical_form(form) if form else ""
    return (salt_key, value, unit, form_key)


def brand_key(brand: str | None) -> str:
    """Normalised brand string. An empty brand stays empty."""
    if not brand:
        return ""
    return normalize(brand)


def notes_are_misspelling_or_hinglish(notes: str) -> bool:
    """True when the query note itself describes a misspelling or Hinglish."""
    text = (notes or "").casefold()
    return "misspell" in text or "hinglish" in text


def index_medicines(
    records: Sequence[MedicineRecord],
) -> tuple[Counter[str], Counter[tuple[str, float | None, str | None, str]]]:
    """Count accepted rows by brand and by salt/strength/form identity."""
    brands: Counter[str] = Counter()
    identities: Counter[tuple[str, float | None, str | None, str]] = Counter()
    for record in records:
        brands[brand_key(record.brand)] += 1
        identities[
            identity_key(
                record.salt,
                record.strength_value,
                record.strength_unit,
                record.form,
            )
        ] += 1
    return brands, identities


def _hit_from_record(record: MedicineRecord) -> Hit:
    return Hit(
        brand=record.brand,
        salt=record.salt,
        strength_value=record.strength_value,
        strength_unit=record.strength_unit,
        form=record.form,
        score=None,
    )


def retrieve_exact(query: str, records: Sequence[MedicineRecord]) -> list[Hit]:
    """Exact brand baseline. Order is corpus order. There is no score."""
    return [_hit_from_record(record) for record in match_exact(query, records)]


def retrieve_soundex(query: str, records: Sequence[MedicineRecord]) -> list[Hit]:
    """Soundex baseline. Order is corpus order. There is no score."""
    return [_hit_from_record(record) for record in match_phonetic(query, records)]


def retrieve_ours(query: str, records: Sequence[MedicineRecord], k: int) -> list[Hit]:
    """N-gram plus parametric matcher, ranked by its cosine, cutoff ``k``."""
    result = search_medicine(query, k=k, records=records)
    hits: list[Hit] = []
    for candidate in result.candidates:
        hits.append(
            Hit(
                brand=candidate.brand,
                salt=candidate.salt,
                strength_value=candidate.strength_value,
                strength_unit=candidate.strength_unit,
                form=candidate.form,
                score=candidate.similarity.cosine,
            )
        )
    return hits


def _flags_for(hits: Sequence[Hit], relevant) -> list[bool]:
    return [relevant(hit) for hit in hits]


def _query_metrics(
    hits: Sequence[Hit],
    n_relevant: int,
    relevant,
) -> dict[str, float | None]:
    flags = _flags_for(hits, relevant)
    metrics: dict[str, float | None] = {
        "top1": top1_hit(flags),
        "set_precision": set_precision(flags),
        "set_recall": set_recall(flags, n_relevant),
        "n_returned": float(len(hits)),
        "n_relevant": float(n_relevant) if n_relevant > 0 else 0.0,
    }
    for k in KS:
        metrics[f"P@{k}"] = precision_at_k(flags, k)
        metrics[f"recall@{k}"] = recall_at_k(flags, k, n_relevant)
        metrics[f"P@{k}_hits"] = float(sum(1 for flag in flags[:k] if flag))
    return metrics


def medicine_method_specs() -> list[dict[str, object]]:
    """Matchers this checkout can actually call, plus the ones it cannot."""
    specs: list[dict[str, object]] = [
        {
            "method": "exact",
            "available": True,
            "rank_basis": "unranked_corpus_order",
            "list_policy": "full_unranked_list",
            "retrieved_cutoff": "",
        },
        {
            "method": "soundex",
            "available": True,
            "rank_basis": "unranked_corpus_order",
            "list_policy": "full_unranked_list",
            "retrieved_cutoff": "",
        },
        {
            "method": "ours_ngram_parametric",
            "available": True,
            "rank_basis": "char_2gram_3gram_cosine",
            "list_policy": f"top_{OURS_K}_by_cosine",
            "retrieved_cutoff": OURS_K,
        },
    ]
    import bharosa.medicine.baselines as baselines

    if not hasattr(baselines, "match_tfidf"):
        specs.append(
            {
                "method": "brand_token_tfidf",
                "available": False,
                "rank_basis": "",
                "list_policy": "",
                "retrieved_cutoff": "",
                "reason": (
                    "bharosa.medicine.baselines has no brand-token tf-idf matcher. "
                    "P@k and recall are undefined for that system."
                ),
            }
        )
    if not hasattr(baselines, "match_bm25"):
        specs.append(
            {
                "method": "brand_bm25",
                "available": False,
                "rank_basis": "",
                "list_policy": "",
                "retrieved_cutoff": "",
                "reason": (
                    "bharosa.medicine.baselines has no brand BM25 matcher. "
                    "P@k and recall are undefined for that system."
                ),
            }
        )
    return specs


def evaluate_medicine(
    records: Sequence[MedicineRecord],
    query_rows: Sequence[Mapping[str, str]],
    *,
    progress: bool = False,
) -> dict[str, object]:
    """Score verified medicine queries and keep unreviewed rows unscored."""
    brands, identities = index_medicines(records)
    specs = medicine_method_specs()
    available = [spec for spec in specs if spec["available"]]
    per_query: list[dict[str, object]] = []
    hits_out: list[dict[str, object]] = []
    unreviewed: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    buckets: dict[tuple[str, str], list[dict[str, float | None]]] = defaultdict(list)

    n_verified = 0
    n_review = 0
    n_other = 0
    n_scored_brand = 0
    n_misspelling_scored = 0
    status_counts: Counter[str] = Counter()

    total = len(query_rows)
    started = time.perf_counter()
    for row_id, row in enumerate(query_rows, start=1):
        query = (row.get("query") or "").strip()
        status = (row.get("label_status") or "").strip()
        notes = (row.get("notes") or "").strip()
        gold_brand = (row.get("gold_brand") or "").strip()
        gold_salt = (row.get("gold_salt") or "").strip()
        gold_form = (row.get("gold_form") or "").strip()
        status_counts[status or "(blank)"] += 1
        verified = status == VERIFIED_STATUS
        if verified:
            n_verified += 1
        elif status == REVIEW_STATUS:
            n_review += 1
        else:
            n_other += 1

        gold_brand_norm = brand_key(gold_brand)
        n_brand = brands[gold_brand_norm] if gold_brand_norm else 0
        parsed_strength = parse_gold_strength(row.get("gold_strength") or "")
        identity = None
        n_identity = 0
        if parsed_strength is not None:
            strength_value, strength_unit = parsed_strength
            identity = identity_key(gold_salt, strength_value, strength_unit, gold_form)
            n_identity = identities[identity]

        if verified and n_brand > 0 and gold_brand_norm:
            scored = True
            exclude_reason = ""
            n_scored_brand += 1
            if notes_are_misspelling_or_hinglish(notes):
                n_misspelling_scored += 1
        elif not verified:
            scored = False
            exclude_reason = (
                f"label_status={status or '(blank)'}. "
                "This row is not a verified label, so precision and recall are not computed."
            )
        elif not gold_brand_norm:
            scored = False
            exclude_reason = "gold_brand is empty, so the relevant set is undefined."
        else:
            scored = False
            exclude_reason = (
                "gold_brand is not in the accepted medicine corpus, "
                "so this verified row is excluded rather than scored as a miss."
            )
            exclusions.append(
                {
                    "row_id": row_id,
                    "query": query,
                    "gold_brand": gold_brand,
                    "label_status": status,
                    "reason": exclude_reason,
                }
            )

        if progress:
            print(
                f"medicine {row_id}/{total} scored={scored} {query}",
                flush=True,
            )

        for spec in available:
            method = str(spec["method"])
            try:
                if method == "exact":
                    found = retrieve_exact(query, records)
                elif method == "soundex":
                    found = retrieve_soundex(query, records)
                elif method == "ours_ngram_parametric":
                    found = retrieve_ours(query, records, OURS_K)
                else:
                    raise RuntimeError(f"no retriever for {method}")
                error = ""
            except Exception as exc:  # noqa: BLE001 - one query must not drop the run
                found = []
                error = f"{type(exc).__name__}: {exc}"

            brand_metrics: dict[str, float | None] | None = None
            identity_metrics: dict[str, float | None] | None = None
            if scored and not error:
                brand_metrics = _query_metrics(
                    found,
                    n_brand,
                    lambda hit, key=gold_brand_norm: brand_key(hit.brand) == key,
                )
                buckets[(method, "gold_brand")].append(brand_metrics)
                if identity is not None and n_identity > 0:
                    identity_metrics = _query_metrics(
                        found,
                        n_identity,
                        lambda hit, key=identity: identity_key(
                            hit.salt,
                            hit.strength_value,
                            hit.strength_unit,
                            hit.form,
                        )
                        == key,
                    )
                    buckets[(method, "gold_identity")].append(identity_metrics)

            per_query.append(
                {
                    "row_id": row_id,
                    "query": query,
                    "label_status": status,
                    "notes": notes,
                    "used_in_scored_metrics": scored and not error,
                    "exclude_reason": error or exclude_reason,
                    "method": method,
                    "rank_basis": spec["rank_basis"],
                    "list_policy": spec["list_policy"],
                    "n_returned": len(found),
                    "error": error,
                    "n_relevant_brand": n_brand if scored else NA,
                    "brand_top1": _pick(brand_metrics, "top1"),
                    "brand_P@5": _pick(brand_metrics, "P@5"),
                    "brand_P@5_hits": _pick(brand_metrics, "P@5_hits"),
                    "brand_recall@5": _pick(brand_metrics, "recall@5"),
                    "brand_P@10": _pick(brand_metrics, "P@10"),
                    "brand_recall@10": _pick(brand_metrics, "recall@10"),
                    "brand_set_precision": _pick(brand_metrics, "set_precision"),
                    "brand_set_recall": _pick(brand_metrics, "set_recall"),
                    "n_relevant_identity": n_identity if scored and identity is not None else NA,
                    "identity_top1": _pick(identity_metrics, "top1"),
                    "identity_P@5": _pick(identity_metrics, "P@5"),
                    "identity_recall@5": _pick(identity_metrics, "recall@5"),
                    "identity_P@10": _pick(identity_metrics, "P@10"),
                    "identity_recall@10": _pick(identity_metrics, "recall@10"),
                    "identity_set_precision": _pick(identity_metrics, "set_precision"),
                    "identity_set_recall": _pick(identity_metrics, "set_recall"),
                    "top1_brand": found[0].brand if found else "",
                    "top1_salt": found[0].salt or "" if found else "",
                    "top1_strength": _strength_cell(found[0]) if found else "",
                    "top1_form": found[0].form or "" if found else "",
                    "top1_score": found[0].score if found and found[0].score is not None else "",
                }
            )

            preview = found[:OURS_K]
            if not preview:
                hits_out.append(
                    _hit_row(row_id, query, status, scored, spec, None, 0, gold_brand_norm, identity)
                )
            for rank, hit in enumerate(preview, start=1):
                hits_out.append(
                    _hit_row(
                        row_id,
                        query,
                        status,
                        scored and not error,
                        spec,
                        hit,
                        rank,
                        gold_brand_norm,
                        identity if scored else None,
                    )
                )

            if status == REVIEW_STATUS:
                draft_rows = preview or [None]
                for rank, hit in enumerate(draft_rows, start=1):
                    unreviewed.append(
                        {
                            "row_id": row_id,
                            "query": query,
                            "label_status": status,
                            "human_review": "required",
                            "excluded_from_metrics": "yes",
                            "draft_gold_brand": gold_brand,
                            "draft_gold_salt": gold_salt,
                            "draft_gold_strength": (row.get("gold_strength") or "").strip(),
                            "draft_gold_form": gold_form,
                            "notes": notes,
                            "method": method,
                            "rank_basis": spec["rank_basis"],
                            "rank": "" if hit is None else rank,
                            "returned_brand": "" if hit is None else hit.brand,
                            "returned_salt": "" if hit is None else (hit.salt or ""),
                            "returned_strength": "" if hit is None else _strength_cell(hit),
                            "returned_form": "" if hit is None else (hit.form or ""),
                            "score": "" if hit is None or hit.score is None else hit.score,
                        }
                    )

    metric_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for spec in specs:
        method = str(spec["method"])
        if not spec["available"]:
            metric_rows.append(
                {
                    "method": method,
                    "available": False,
                    "relevance": NA,
                    "rank_basis": "",
                    "list_policy": "",
                    "n_queries_scored": 0,
                    "top1_accuracy": NA,
                    "P@5": NA,
                    "P@10": NA,
                    "recall@5": NA,
                    "recall@10": NA,
                    "set_precision": NA,
                    "set_recall": NA,
                    "reason": spec.get("reason", ""),
                }
            )
            continue
        for relevance in ("gold_brand", "gold_identity"):
            group = buckets[(method, relevance)]
            metric_rows.append(
                _aggregate_row(method, relevance, spec, group)
            )

    for row in per_query:
        if (
            row["method"] == "ours_ngram_parametric"
            and row["used_in_scored_metrics"]
            and row["brand_top1"] == 0.0
        ):
            failures.append(
                {
                    "row_id": row["row_id"],
                    "query": row["query"],
                    "gold_brand_relevant_at_rank_1": 0,
                    "top1_brand": row["top1_brand"],
                    "top1_strength": row["top1_strength"],
                    "top1_form": row["top1_form"],
                    "top1_score": row["top1_score"],
                    "notes": row["notes"],
                }
            )

    identity_ns = {
        str(row["method"]): row["n_queries_scored"]
        for row in metric_rows
        if row["relevance"] == "gold_identity" and row["available"]
    }
    limitations = medicine_limitations(
        n_scored=n_scored_brand,
        n_misspelling_or_hinglish=n_misspelling_scored,
        n_review=n_review,
    )
    elapsed = time.perf_counter() - started
    return {
        "n_queries": total,
        "n_verified": n_verified,
        "n_needs_human_review": n_review,
        "n_other_status": n_other,
        "n_scored_brand": n_scored_brand,
        "n_scored_identity_by_method": identity_ns,
        "n_excluded_brand_not_in_corpus": len(exclusions),
        "n_misspelling_or_hinglish_scored": n_misspelling_scored,
        "label_status_counts": dict(status_counts),
        "accepted_records": len(records),
        "min_brand_cosine": MIN_BRAND_COSINE,
        "ours_k": OURS_K,
        "ks": list(KS),
        "tuned_on_this_run": False,
        "threshold_source": "bharosa.medicine.search.MIN_BRAND_COSINE",
        "elapsed_seconds": elapsed,
        "relevance": {
            "gold_brand": (
                "A hit is relevant when its normalised brand equals the verified "
                "gold_brand. The denominator of recall is the number of accepted "
                "corpus rows with that brand."
            ),
            "gold_identity": (
                "A hit is relevant when normalised salt, strength value, strength "
                "unit, and dosage form equal the verified gold fields. An empty "
                "gold strength matches rows with no scalar strength. The "
                "denominator is the number of accepted rows with that identity."
            ),
        },
        "metric_rows": metric_rows,
        "per_query": per_query,
        "hits": hits_out,
        "unreviewed": unreviewed,
        "exclusions": exclusions,
        "failures": failures,
        "limitations": limitations,
    }


def medicine_limitations(
    *,
    n_scored: int,
    n_misspelling_or_hinglish: int,
    n_review: int,
) -> list[str]:
    """State when the scored medicine set cannot support a strong claim."""
    lines: list[str] = []
    if n_scored < SMALL_QUERY_SET:
        lines.append(
            f"Scored medicine queries: {n_scored}. "
            f"A stable estimate needs on the order of {SMALL_QUERY_SET} judged queries. "
            "This set is too small."
        )
    if n_misspelling_or_hinglish == 0:
        lines.append(
            "Scored queries whose notes mention a misspelling or Hinglish: 0. "
            f"{n_review} rows marked {REVIEW_STATUS} are stored for human review "
            "and are left out of the averages."
        )
    return lines


def _aggregate_row(
    method: str,
    relevance: str,
    spec: Mapping[str, object],
    group: Sequence[Mapping[str, float | None]],
) -> dict[str, object]:
    def col(name: str) -> float | None:
        values = [row[name] for row in group if row.get(name) is not None]
        return macro_average([float(value) for value in values])

    if not group:
        reason = (
            f"No scored queries have a non-empty {relevance} relevant set, "
            "so this average is undefined."
        )
        return {
            "method": method,
            "available": True,
            "relevance": relevance,
            "rank_basis": spec["rank_basis"],
            "list_policy": spec["list_policy"],
            "n_queries_scored": 0,
            "top1_accuracy": NA,
            "P@5": NA,
            "P@10": NA,
            "recall@5": NA,
            "recall@10": NA,
            "set_precision": NA,
            "set_recall": NA,
            "reason": reason,
        }
    return {
        "method": method,
        "available": True,
        "relevance": relevance,
        "rank_basis": spec["rank_basis"],
        "list_policy": spec["list_policy"],
        "n_queries_scored": len(group),
        "top1_accuracy": col("top1"),
        "P@5": col("P@5"),
        "P@10": col("P@10"),
        "recall@5": col("recall@5"),
        "recall@10": col("recall@10"),
        "set_precision": col("set_precision"),
        "set_recall": col("set_recall"),
        "reason": (
            "Macro average over scored queries. "
            f"Rank basis: {spec['rank_basis']}. List: {spec['list_policy']}."
        ),
    }


def _pick(metrics: Mapping[str, float | None] | None, name: str) -> float | str:
    if metrics is None:
        return NA
    value = metrics.get(name)
    if value is None:
        return NA
    return value


def _strength_cell(hit: Hit) -> str:
    if hit.strength_value is None:
        return ""
    unit = hit.strength_unit or ""
    return f"{hit.strength_value:g} {unit}".strip()


def _hit_row(
    row_id: int,
    query: str,
    status: str,
    scored: bool,
    spec: Mapping[str, object],
    hit: Hit | None,
    rank: int,
    gold_brand_norm: str,
    identity: tuple[str, float | None, str | None, str] | None,
) -> dict[str, object]:
    if hit is None:
        brand_rel: object = NA
        ident_rel: object = NA
    elif not scored:
        brand_rel = NA
        ident_rel = NA
    else:
        brand_rel = int(brand_key(hit.brand) == gold_brand_norm)
        if identity is None:
            ident_rel = NA
        else:
            ident_rel = int(
                identity_key(hit.salt, hit.strength_value, hit.strength_unit, hit.form)
                == identity
            )
    return {
        "row_id": row_id,
        "query": query,
        "label_status": status,
        "used_in_scored_metrics": bool(scored) if hit is not None or scored else False,
        "method": spec["method"],
        "rank_basis": spec["rank_basis"],
        "rank": "" if hit is None else rank,
        "brand": "" if hit is None else hit.brand,
        "salt": "" if hit is None else (hit.salt or ""),
        "strength": "" if hit is None else _strength_cell(hit),
        "form": "" if hit is None else (hit.form or ""),
        "score": "" if hit is None or hit.score is None else hit.score,
        "brand_relevant": brand_rel,
        "identity_relevant": ident_rel,
    }


def _sqlite_counts(db_path: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {"pages": None, "versions": None, "changes": None}
    if not db_path.is_file():
        return counts
    uri = db_path.resolve().as_posix()
    connection = sqlite3.connect(f"file:{uri}?mode=ro", uri=True)
    try:
        for table in ("pages", "versions", "changes"):
            try:
                counts[table] = int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
            except sqlite3.Error:
                counts[table] = None
    finally:
        connection.close()
    return counts


def collect_inventory(paths: RunPaths) -> dict[str, object]:
    """Count files and rows that exist. Do not treat missing files as zero labels."""
    eval_csvs = sorted(path.name for path in paths.eval_dir.glob("*.csv")) if paths.eval_dir.is_dir() else []
    raw_dir = paths.data_dir / "raw"
    snapshot_dir = paths.data_dir / "snapshots"
    html_count = len(list(raw_dir.glob("*.html"))) if raw_dir.is_dir() else 0
    snapshot_files = (
        len([path for path in snapshot_dir.rglob("*") if path.is_file()])
        if snapshot_dir.is_dir()
        else 0
    )
    db_counts = _sqlite_counts(paths.data_dir / "bharosa.db")
    zone_paths = _zone_corpus_candidates(paths)
    judgement_hits = [name for name in RAG_JUDGEMENT_NAMES if (paths.eval_dir / name).is_file()]
    return {
        "eval_csv_files": eval_csvs,
        "medicine_queries_present": paths.medicine_queries.is_file(),
        "claims_present": paths.claims_csv.is_file(),
        "scheme_queries_present": (paths.eval_dir / "scheme_queries.csv").is_file(),
        "scheme_qrels_present": (paths.eval_dir / "scheme_qrels.csv").is_file(),
        "zone_corpus_present": any(path.is_file() for path in zone_paths),
        "zone_corpus_candidates": [str(path) for path in zone_paths],
        "net_weights_present": (paths.config_dir / WEIGHTS_NAME).is_file(),
        "lexicon_present": paths.lexicon_csv.is_file(),
        "lexicon_path": str(paths.lexicon_csv),
        "rag_judgement_files": judgement_hits,
        "crawled_html_files": html_count,
        "crawled_pages": db_counts["pages"],
        "crawled_versions": db_counts["versions"],
        "crawled_changes": db_counts["changes"],
        "snapshot_files": snapshot_files,
        "snapshots_dir_present": snapshot_dir.is_dir(),
    }


def _zone_corpus_candidates(paths: RunPaths) -> list[Path]:
    data_zones = paths.data_dir / "zones.jsonl"
    return [paths.eval_dir / ZONE_CORPUS_NAME, data_zones]


def load_zone_corpus(paths: RunPaths) -> tuple[list[ZoneDoc] | None, str]:
    """Load a zone JSONL file when one exists. Do not build one from HTML."""
    found = [path for path in _zone_corpus_candidates(paths) if path.is_file()]
    if not found:
        looked = ", ".join(str(path) for path in _zone_corpus_candidates(paths))
        return None, f"No zone-document file at {looked}."
    path = found[0]
    documents: list[ZoneDoc] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            documents.append(ZoneDoc.from_mapping(payload))
    if not documents:
        return None, f"{path} has no zone documents."
    return documents, str(path)


def load_declared_weights(paths: RunPaths) -> tuple[DeclaredWeights | None, str]:
    """Read ``config/net_weights.json`` when it exists. Do not fill coefficients."""
    path = paths.config_dir / WEIGHTS_NAME
    if not path.is_file():
        return None, (
            f"{path} is absent. Net-score coefficients are not chosen in this run."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None, f"{path} is not a JSON object."
    try:
        weights = NetScoreWeights(
            float(payload["cosine"]),
            float(payload["g_score"]),
            float(payload["freshness"]),
        )
        raw_zones = payload["zone_weights"]
        scale = float(payload["freshness_scale_days"])
        as_of = str(payload["as_of"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"{path} is incomplete ({exc}). Missing coefficients are not filled in."
    if not isinstance(raw_zones, dict):
        return None, f"{path} zone_weights is not an object."
    if scale <= 0:
        return None, f"{path} freshness_scale_days must be positive."
    if as_of.strip() == "":
        return None, f"{path} as_of is empty."
    zone_weights: dict[str, float] = {}
    for zone in ZONES:
        if zone not in raw_zones:
            return None, f"{path} has no weight for zone {zone}. That weight is not invented."
        zone_weights[zone] = float(raw_zones[zone])
    extra = [key for key in raw_zones if key not in ZONES]
    if extra:
        return None, f"{path} has unknown zone weights: {', '.join(map(str, extra))}."
    return (
        DeclaredWeights(weights, zone_weights, scale, as_of, str(path)),
        str(path),
    )


def _load_qrels(path: Path) -> tuple[dict[str, set[str]], int]:
    relevant: dict[str, set[str]] = defaultdict(set)
    pairs = 0
    for row in load_csv(path):
        query_id = (row.get("query_id") or "").strip()
        doc_id = (row.get("doc_id") or "").strip()
        raw = (row.get("relevance") or "").strip()
        if not query_id or not doc_id or raw == "":
            continue
        grade = int(raw)
        pairs += 1
        if grade > 0:
            relevant[query_id].add(doc_id)
    return relevant, pairs


def _na_metric_row(system: str, metric: str, reason: str, n: int = 0) -> dict[str, object]:
    return {
        "system": system,
        "metric": metric,
        "value": NA,
        "n_queries": n,
        "reason": reason,
    }


def _scheme_metric_names() -> list[str]:
    names = ["top1", "P@5", "P@10", "recall@5", "recall@10", "set_precision", "set_recall"]
    return names


def evaluate_scheme(paths: RunPaths) -> dict[str, object]:
    """Score scheme runs when queries, qrels, and zones are already files."""
    inventory = collect_inventory(paths)
    query_path = paths.eval_dir / "scheme_queries.csv"
    qrel_path = paths.eval_dir / "scheme_qrels.csv"
    reasons: list[str] = []
    queries = load_csv(query_path) if query_path.is_file() else []
    qrels: dict[str, set[str]] = {}
    qrel_pairs = 0
    if query_path.is_file():
        require_columns(queries, ("query_id", "query"), query_path)
    else:
        reasons.append(f"{query_path} is absent.")
    if qrel_path.is_file():
        require_columns([], ("query_id", "doc_id", "relevance"), qrel_path)
        qrels, qrel_pairs = _load_qrels(qrel_path)
        if qrel_pairs == 0:
            reasons.append(f"{qrel_path} has no relevance rows.")
    else:
        reasons.append(f"{qrel_path} is absent.")
    zones, zone_note = load_zone_corpus(paths)
    if zones is None:
        reasons.append(zone_note)
    pages = inventory["crawled_pages"]
    if zones is None and isinstance(pages, int) and pages > 0:
        reasons.append(
            f"The crawl database has {pages} pages. "
            "Those pages are not a zone corpus and have no relevance labels."
        )

    rank_reason = " ".join(reasons) if reasons else ""
    rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    ran = False
    if not reasons and zones is not None:
        ran = True
        rows.extend(_run_scheme_systems(queries, qrels, zones, paths))
        ablation_rows.extend(_run_ablations(queries, qrels, zones, paths))
    else:
        for system in SCHEME_SYSTEMS:
            system_reason = rank_reason
            if system == "enhanced":
                system_reason = _enhanced_blockers(paths, rank_reason)
            for metric in _scheme_metric_names():
                rows.append(_na_metric_row(system, metric, system_reason, len(queries)))
        ablation_reason = _enhanced_blockers(paths, rank_reason)
        for flag in ABLATION_FLAGS:
            ablation_rows.append(
                {
                    "component": flag,
                    "implemented": True,
                    "backed_by_labelled_data": False,
                    "metric": "delta_P@5",
                    "value": NA,
                    "reason": (
                        f"Removing {flag} is implemented as a search flag. "
                        f"{ablation_reason} "
                        "delta P@5 is undefined. "
                        "BM25 stays a separate system because net-score flags are not applied to it."
                    ),
                }
            )
    return {
        "ran": ran,
        "n_scheme_queries": len(queries),
        "n_qrel_pairs": qrel_pairs,
        "n_zones": 0 if zones is None else len(zones),
        "metric_rows": rows,
        "ablation_rows": ablation_rows,
        "bm25_parameters": {
            "k1": LIBRARY_K1,
            "b": LIBRARY_B,
            "epsilon": LIBRARY_EPSILON,
            "source": "rank_bm25.BM25Okapi constructor defaults",
            "fit_on_labels": False,
            "used_in_this_run": ran,
        },
    }


def _enhanced_blockers(paths: RunPaths, rank_reason: str) -> str:
    parts = [rank_reason] if rank_reason else []
    if not paths.lexicon_csv.is_file():
        parts.append(
            f"The reviewed Hinglish lexicon is absent at {paths.lexicon_csv}."
        )
    weights, weight_note = load_declared_weights(paths)
    if weights is None:
        parts.append(weight_note)
    return " ".join(part for part in parts if part)


def _query_pairs(
    queries: Sequence[Mapping[str, str]],
    qrels: Mapping[str, set[str]],
) -> list[tuple[str, str, set[str]]]:
    pairs: list[tuple[str, str, set[str]]] = []
    for row in queries:
        query_id = (row.get("query_id") or "").strip()
        text = (row.get("query") or "").strip()
        relevant = set(qrels.get(query_id, set()))
        if query_id and text and relevant:
            pairs.append((query_id, text, relevant))
    return pairs


def _average_metrics(
    per_query: Sequence[Mapping[str, float | None]],
) -> dict[str, float | None]:
    averaged: dict[str, float | None] = {}
    if not per_query:
        for name in _scheme_metric_names():
            averaged[name] = None
        return averaged
    for name in _scheme_metric_names():
        values = [float(row[name]) for row in per_query if row.get(name) is not None]
        averaged[name] = macro_average(values)
    return averaged


def _search_ids(
    query: str,
    documents: Sequence[ZoneDoc],
    *,
    mode: str,
    flags: dict[str, bool] | None = None,
    weights: DeclaredWeights | None = None,
    lexicon: HinglishLexicon | None = None,
) -> list[str]:
    if mode == "plain_tfidf":
        hits = search_schemes(query, documents=documents, k=max(KS))
    elif mode == "bm25":
        hits = search_schemes(
            query,
            documents=documents,
            k=max(KS),
            flags={"bm25": True, "hinglish": False},
        )
    elif mode == "enhanced":
        if weights is None:
            raise RuntimeError("enhanced retrieval requires declared weights")
        hits = search_schemes(
            query,
            documents=documents,
            k=max(KS),
            flags=flags,
            weights=weights.weights,
            zone_weights=weights.zone_weights,
            freshness_scale_days=weights.freshness_scale_days,
            as_of=weights.as_of,
            lexicon=lexicon,
        )
    else:
        raise RuntimeError(f"unknown scheme mode {mode}")
    return [hit.doc_id for hit in hits]


def _run_scheme_systems(
    queries: Sequence[Mapping[str, str]],
    qrels: Mapping[str, set[str]],
    zones: Sequence[ZoneDoc],
    paths: RunPaths,
) -> list[dict[str, object]]:
    pairs = _query_pairs(queries, qrels)
    rows: list[dict[str, object]] = []
    plain_detail = [_metrics_for_query(text, relevant, zones, mode="plain_tfidf") for _, text, relevant in pairs]
    bm25_detail = [_metrics_for_query(text, relevant, zones, mode="bm25") for _, text, relevant in pairs]
    rows.extend(_system_rows("plain_tfidf", plain_detail, "lnc.ltc cosine with net-score flags off."))
    rows.extend(
        _system_rows(
            "bm25",
            bm25_detail,
            (
                f"rank_bm25.BM25Okapi defaults k1={LIBRARY_K1}, b={LIBRARY_B}, "
                f"epsilon={LIBRARY_EPSILON}. These defaults are not fit on labels."
            ),
        )
    )
    weights, weight_note = load_declared_weights(paths)
    lexicon_ok = paths.lexicon_csv.is_file()
    if weights is None or not lexicon_ok:
        reason = _enhanced_blockers(paths, "")
        if not reason:
            reason = weight_note
        for metric in _scheme_metric_names():
            rows.append(_na_metric_row("enhanced", metric, reason, len(pairs)))
        return rows
    lexicon = HinglishLexicon.load(paths.lexicon_csv)
    flags = _enhanced_flags()
    enhanced_detail = [
        _metrics_for_query(
            text,
            relevant,
            zones,
            mode="enhanced",
            flags=flags,
            weights=weights,
            lexicon=lexicon,
        )
        for _, text, relevant in pairs
    ]
    rows.extend(
        _system_rows(
            "enhanced",
            enhanced_detail,
            (
                f"Flags {flags}. Weights from {weights.source}. "
                "Coefficients are the declared file, not a search on this set."
            ),
        )
    )
    return rows


def _metrics_for_query(
    query: str,
    relevant: set[str],
    zones: Sequence[ZoneDoc],
    *,
    mode: str,
    flags: dict[str, bool] | None = None,
    weights: DeclaredWeights | None = None,
    lexicon: HinglishLexicon | None = None,
) -> dict[str, float | None]:
    ranked = _search_ids(
        query,
        zones,
        mode=mode,
        flags=flags,
        weights=weights,
        lexicon=lexicon,
    )
    return ranked_doc_metrics(ranked, relevant, KS)


def _system_rows(
    system: str,
    detail: Sequence[Mapping[str, float | None]],
    note: str,
) -> list[dict[str, object]]:
    averaged = _average_metrics(detail)
    rows: list[dict[str, object]] = []
    for metric in _scheme_metric_names():
        value = averaged[metric]
        rows.append(
            {
                "system": system,
                "metric": metric,
                "value": NA if value is None else value,
                "n_queries": len(detail),
                "reason": note if value is not None else f"{note} The average is undefined.",
            }
        )
    return rows


def _enhanced_flags() -> dict[str, bool]:
    return {
        "zone_weights": True,
        "g_score": True,
        "freshness": True,
        "hinglish": True,
        "bm25": False,
    }


def _run_ablations(
    queries: Sequence[Mapping[str, str]],
    qrels: Mapping[str, set[str]],
    zones: Sequence[ZoneDoc],
    paths: RunPaths,
) -> list[dict[str, object]]:
    weights, weight_note = load_declared_weights(paths)
    if weights is None or not paths.lexicon_csv.is_file():
        reason = _enhanced_blockers(paths, "") or weight_note
        return [
            {
                "component": flag,
                "implemented": True,
                "backed_by_labelled_data": False,
                "metric": "delta_P@5",
                "value": NA,
                "reason": (
                    f"Removing {flag} is implemented as a search flag. {reason} "
                    "delta P@5 is undefined."
                ),
            }
            for flag in ABLATION_FLAGS
        ]
    lexicon = HinglishLexicon.load(paths.lexicon_csv)
    pairs = _query_pairs(queries, qrels)
    full_flags = _enhanced_flags()
    full = [
        _metrics_for_query(
            text,
            relevant,
            zones,
            mode="enhanced",
            flags=full_flags,
            weights=weights,
            lexicon=lexicon,
        )
        for _, text, relevant in pairs
    ]
    full_p5 = _average_metrics(full)["P@5"]
    rows: list[dict[str, object]] = []
    for flag in ABLATION_FLAGS:
        flags = dict(full_flags)
        flags[flag] = False
        ablated = [
            _metrics_for_query(
                text,
                relevant,
                zones,
                mode="enhanced",
                flags=flags,
                weights=weights,
                lexicon=lexicon if flags["hinglish"] else None,
            )
            for _, text, relevant in pairs
        ]
        ablated_p5 = _average_metrics(ablated)["P@5"]
        if full_p5 is None or ablated_p5 is None:
            value: object = NA
            reason = "P@5 is undefined for the full system or the ablated system."
        else:
            value = ablated_p5 - full_p5
            reason = (
                "delta P@5 = P@5(component off) - P@5(all declared components on). "
                f"Weights from {weights.source}. Coefficients were not searched."
            )
        rows.append(
            {
                "component": flag,
                "implemented": True,
                "backed_by_labelled_data": True,
                "metric": "delta_P@5",
                "value": value,
                "reason": reason,
            }
        )
    return rows


def evaluate_rag(paths: RunPaths) -> dict[str, object]:
    """RAG metrics only where a human judgement file exists."""
    present = [name for name in RAG_JUDGEMENT_NAMES if (paths.eval_dir / name).is_file()]
    cfg = get_config()
    reason = (
        "Human judgements of citation support, refusal, and answer correctness "
        "are absent. Looked for "
        + ", ".join(str(paths.eval_dir / name) for name in RAG_JUDGEMENT_NAMES)
        + "."
    )
    if present:
        reason = (
            "Judgement files were found ("
            + ", ".join(present)
            + ") but this run has no reader for their columns, so the metrics stay N/A "
            "rather than guessing a schema."
        )
    metrics = (
        "pct_sentences_supported_with_checker",
        "pct_sentences_supported_without_checker",
        "refusal_rate",
        "answer_correctness",
    )
    rows = [
        {"metric": name, "value": NA, "n_human_judgements": 0, "reason": reason}
        for name in metrics
    ]
    return {
        "metric_rows": rows,
        "thresholds": {
            "refusal_score_threshold": cfg.refusal_score_threshold,
            "citation_threshold": cfg.citation_threshold,
            "min_retrieval_score": cfg.min_retrieval_score,
            "used_to_compute_metrics": False,
            "tuned_on_this_run": False,
            "reason": (
                "These are the current configuration values. "
                "This run does not change them and does not apply them, "
                "because there is no human-labelled set."
            ),
        },
    }


def score_claims(
    rows: Sequence[Mapping[str, str]],
    *,
    retrieve=None,
    allow_mock: bool = False,
) -> dict[str, object]:
    """Precision and recall for CONTRADICTED, restricted to reviewed rows.

    ``retrieve`` is used only when at least one row has ``human_reviewed=yes``.
    Unreviewed and AI-drafted rows stay in the audit and out of the counts.
    """
    audit: list[dict[str, object]] = []
    reviewed: list[Mapping[str, str]] = []
    n_ai = 0
    for row in rows:
        origin = (row.get("origin") or "").strip()
        reviewed_cell = (row.get("human_reviewed") or "").strip()
        if origin.casefold() == "ai_drafted":
            n_ai += 1
        is_reviewed = reviewed_cell.casefold() == "yes"
        if is_reviewed:
            reviewed.append(row)
            reason = "human_reviewed=yes, eligible for precision and recall."
            used = True
        else:
            reason = (
                "Marked for human review. "
                "AI-drafted and unreviewed labels are not gold, "
                "so precision and recall skip this row."
                if origin.casefold() == "ai_drafted" or reviewed_cell.casefold() == REVIEW_STATUS.casefold()
                else "human_reviewed is not yes, so this row is not gold."
            )
            used = False
        audit.append(
            {
                "message": (row.get("message") or "").strip(),
                "draft_label": (row.get("label") or "").strip(),
                "origin": origin,
                "human_reviewed": reviewed_cell,
                "used_for_precision_recall": used,
                "reason": reason,
            }
        )

    base = {
        "n_rows": len(rows),
        "n_human_reviewed_yes": len(reviewed),
        "n_ai_drafted": n_ai,
        "n_scored": 0,
        "positive_class": CLAIM_POSITIVE,
        "tp": NA,
        "fp": NA,
        "fn": NA,
        "tn": NA,
        "audit_rows": audit,
    }
    undefined = (
        f"Precision and recall are undefined. "
        f"{len(reviewed)} of {len(rows)} rows have human_reviewed=yes. "
        f"{n_ai} rows have origin AI_DRAFTED. "
        f"Even a fully reviewed set of {len(rows)} messages would be too small "
        f"for a stable estimate."
    )
    if not reviewed:
        base["precision"] = {"value": NA, "reason": undefined}
        base["recall"] = {"value": NA, "reason": undefined}
        base["keyword_baseline"] = {
            "value": NA,
            "reason": "No separate keyword-baseline claim checker is implemented.",
        }
        return base
    if retrieve is None:
        reason = (
            f"{len(reviewed)} rows have human_reviewed=yes. "
            "No official-zone retriever was supplied, and fixture zones are not substituted."
        )
        base["precision"] = {"value": NA, "reason": reason}
        base["recall"] = {"value": NA, "reason": reason}
        base["keyword_baseline"] = {
            "value": NA,
            "reason": "No separate keyword-baseline claim checker is implemented.",
        }
        return base

    tp = fp = fn = tn = 0
    for row in reviewed:
        gold = (row.get("label") or "").strip().upper()
        if gold == "":
            continue
        verdict = check_claim(
            (row.get("message") or "").strip(),
            retrieve,
            allow_mock=allow_mock,
        )
        pred = verdict.label.strip().upper()
        gold_pos = gold == CLAIM_POSITIVE
        pred_pos = pred == CLAIM_POSITIVE
        if gold_pos and pred_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
        base["n_scored"] = int(base["n_scored"]) + 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    base.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    base["precision"] = {
        "value": NA if precision is None else precision,
        "reason": (
            f"Positive class {CLAIM_POSITIVE}. "
            + ("Undefined because there are no positive predictions." if precision is None else "tp/(tp+fp) on human_reviewed=yes rows.")
        ),
    }
    base["recall"] = {
        "value": NA if recall is None else recall,
        "reason": (
            f"Positive class {CLAIM_POSITIVE}. "
            + ("Undefined because the reviewed set has no positive gold labels." if recall is None else "tp/(tp+fn) on human_reviewed=yes rows.")
        ),
    }
    base["keyword_baseline"] = {
        "value": NA,
        "reason": "No separate keyword-baseline claim checker is implemented.",
    }
    if int(base["n_scored"]) < SMALL_QUERY_SET:
        base["limitation"] = (
            f"Scored claim rows: {base['n_scored']}. "
            f"This is below {SMALL_QUERY_SET}, so the estimate is unstable."
        )
    return base


def evaluate_freshness(paths: RunPaths) -> dict[str, object]:
    """Freshness metrics only from archived snapshots, which this repo lacks."""
    snapshot_dir = paths.data_dir / "snapshots"
    present = snapshot_dir.is_dir() and any(snapshot_dir.rglob("*"))
    reason = (
        f"{snapshot_dir} has no archived page snapshots. "
        "Changes caught and mean staleness are undefined. "
        "This run does not simulate page changes."
    )
    if present:
        reason = (
            f"{snapshot_dir} contains files, but this script has no snapshot replay. "
            "Changes caught and mean staleness stay N/A rather than using a synthetic clock."
        )
    return {
        "metric_rows": [
            {"metric": "changes_caught", "value": NA, "reason": reason},
            {"metric": "mean_staleness", "value": NA, "reason": reason},
        ]
    }


def dataset_rows(
    inventory: Mapping[str, object],
    medicine: Mapping[str, object] | None,
    load_stats: LoadStats | None,
    claims: Mapping[str, object],
    scheme: Mapping[str, object],
) -> list[dict[str, object]]:
    """Flat counts printed and saved for the run."""
    rows: list[dict[str, object]] = []

    def add(name: str, value: object) -> None:
        rows.append({"name": name, "value": "" if value is None else value})

    if load_stats is not None:
        add("medicine_source_rows", load_stats.total_rows)
        add("medicine_accepted_records", load_stats.accepted_rows)
        add("medicine_dropped_rows", load_stats.dropped_rows)
        add("medicine_duplicate_rows", load_stats.duplicate_counts)
        for reason, count in sorted(load_stats.reason_counts.items()):
            add(f"medicine_drop_{reason}", count)
    if medicine is not None:
        add("medicine_queries", medicine["n_queries"])
        add("medicine_verified", medicine["n_verified"])
        add("medicine_needs_human_review", medicine["n_needs_human_review"])
        add("medicine_other_status", medicine["n_other_status"])
        add("medicine_scored_brand", medicine["n_scored_brand"])
        add("medicine_excluded_brand_not_in_corpus", medicine["n_excluded_brand_not_in_corpus"])
        add(
            "medicine_misspelling_or_hinglish_scored",
            medicine["n_misspelling_or_hinglish_scored"],
        )
    add("scheme_queries", scheme["n_scheme_queries"])
    add("scheme_qrel_pairs", scheme["n_qrel_pairs"])
    add("scheme_zones", scheme["n_zones"])
    add("crawled_pages", inventory.get("crawled_pages"))
    add("crawled_html_files", inventory.get("crawled_html_files"))
    add("crawled_changes", inventory.get("crawled_changes"))
    add("snapshot_files", inventory.get("snapshot_files"))
    add("hinglish_lexicon_present", inventory.get("lexicon_present"))
    add("net_weights_present", inventory.get("net_weights_present"))
    add("claims_rows", claims["n_rows"])
    add("claims_human_reviewed_yes", claims["n_human_reviewed_yes"])
    add("claims_ai_drafted", claims["n_ai_drafted"])
    add("rag_human_judgements", 0)
    return rows


def execute(paths: RunPaths, *, progress: bool = True) -> dict[str, object]:
    """Run every section and write the result files. Return the summary."""
    paths.output.mkdir(parents=True, exist_ok=True)
    inventory = collect_inventory(paths)
    if progress:
        print("dataset inventory", flush=True)
        print(f"  eval csv files: {inventory['eval_csv_files']}", flush=True)
        print(f"  crawled pages: {inventory['crawled_pages']}", flush=True)
        print(f"  scheme qrels present: {inventory['scheme_qrels_present']}", flush=True)
        print(f"  lexicon present: {inventory['lexicon_present']}", flush=True)

    if not paths.medicine_queries.is_file():
        raise FileNotFoundError(f"Medicine queries are absent: {paths.medicine_queries}")
    require_columns(
        [],
        ("query", "gold_brand", "gold_salt", "gold_strength", "gold_form", "label_status"),
        paths.medicine_queries,
    )
    query_rows = load_csv(paths.medicine_queries)
    if progress:
        print(f"medicine queries in file: {len(query_rows)}", flush=True)

    if not paths.medicine_csv.is_file():
        raise FileNotFoundError(f"Medicine dataset is absent: {paths.medicine_csv}")
    if progress:
        print(f"loading medicines from {paths.medicine_csv}", flush=True)
    records, load_stats = load_medicines(paths.medicine_csv)
    if progress:
        print(
            f"medicine source rows: {load_stats.total_rows}; "
            f"accepted: {load_stats.accepted_rows}; "
            f"dropped: {load_stats.dropped_rows}",
            flush=True,
        )
    medicine = evaluate_medicine(records, query_rows, progress=progress)

    scheme = evaluate_scheme(paths)
    claims_rows = load_csv(paths.claims_csv) if paths.claims_csv.is_file() else []
    if paths.claims_csv.is_file():
        require_columns(
            claims_rows,
            ("message", "label", "origin", "human_reviewed"),
            paths.claims_csv,
        )
    claims = score_claims(claims_rows, retrieve=None, allow_mock=False)
    rag = evaluate_rag(paths)
    freshness = evaluate_freshness(paths)
    counts = dataset_rows(inventory, medicine, load_stats, claims, scheme)

    limitations = list(medicine["limitations"])
    if not scheme["ran"]:
        limitations.append(
            "Scheme P@k, recall, and ablation delta P@5 are undefined. "
            + _enhanced_blockers(paths, " ".join(
                row["reason"]
                for row in scheme["metric_rows"]
                if row["system"] == "plain_tfidf" and row["metric"] == "P@5"
            ))
        )
    claim_precision = claims["precision"]["reason"]
    limitations.append(claim_precision)
    limitations.append(rag["metric_rows"][0]["reason"])
    limitations.append(freshness["metric_rows"][0]["reason"])

    summary: dict[str, object] = {
        "command": "python eval/run_eval.py",
        "tuned_on_this_run": False,
        "medicine_min_brand_cosine": MIN_BRAND_COSINE,
        "medicine_threshold_source": "bharosa.medicine.search.MIN_BRAND_COSINE",
        "inputs": {
            "medicine_csv": str(paths.medicine_csv),
            "medicine_queries": str(paths.medicine_queries),
            "claims_csv": str(paths.claims_csv),
            "eval_dir": str(paths.eval_dir),
            "data_dir": str(paths.data_dir),
        },
        "dataset": counts,
        "limitations": limitations,
        "medicine": {
            "n_queries": medicine["n_queries"],
            "n_verified": medicine["n_verified"],
            "n_needs_human_review": medicine["n_needs_human_review"],
            "n_scored_brand": medicine["n_scored_brand"],
            "n_scored_identity_by_method": medicine["n_scored_identity_by_method"],
            "n_excluded_brand_not_in_corpus": medicine["n_excluded_brand_not_in_corpus"],
            "n_misspelling_or_hinglish_scored": medicine["n_misspelling_or_hinglish_scored"],
            "label_status_counts": medicine["label_status_counts"],
            "accepted_records": medicine["accepted_records"],
            "load_stats": {
                "total_rows": load_stats.total_rows,
                "accepted_rows": load_stats.accepted_rows,
                "dropped_rows": load_stats.dropped_rows,
                "duplicate_counts": load_stats.duplicate_counts,
                "reason_counts": load_stats.reason_counts,
            },
            "relevance": medicine["relevance"],
            "ours_k": OURS_K,
            "elapsed_seconds": medicine["elapsed_seconds"],
            "metric_rows": medicine["metric_rows"],
            "n_failures_ours_brand_top1": len(medicine["failures"]),
        },
        "scheme": {
            "ran": scheme["ran"],
            "n_scheme_queries": scheme["n_scheme_queries"],
            "n_qrel_pairs": scheme["n_qrel_pairs"],
            "n_zones": scheme["n_zones"],
            "bm25_parameters": scheme["bm25_parameters"],
            "metric_rows": scheme["metric_rows"],
        },
        "ablations": scheme["ablation_rows"],
        "rag": rag,
        "claims": {
            "n_rows": claims["n_rows"],
            "n_human_reviewed_yes": claims["n_human_reviewed_yes"],
            "n_ai_drafted": claims["n_ai_drafted"],
            "n_scored": claims["n_scored"],
            "positive_class": claims["positive_class"],
            "precision": claims["precision"],
            "recall": claims["recall"],
            "keyword_baseline": claims["keyword_baseline"],
            "tp": claims["tp"],
            "fp": claims["fp"],
            "fn": claims["fn"],
            "tn": claims["tn"],
        },
        "freshness": freshness,
        "inventory": inventory,
    }

    files = _write_outputs(paths.output, summary, medicine, claims)
    summary["files"] = files
    if progress:
        print(render_report(summary), flush=True)
    return summary


def _write_outputs(
    output: Path,
    summary: Mapping[str, object],
    medicine: Mapping[str, object],
    claims: Mapping[str, object],
) -> list[str]:
    written: list[str] = []

    empty_headers = {
        "medicine_exclusions.csv": [
            "row_id",
            "query",
            "gold_brand",
            "label_status",
            "reason",
        ],
        "medicine_failures.csv": [
            "row_id",
            "query",
            "gold_brand_relevant_at_rank_1",
            "top1_brand",
            "top1_strength",
            "top1_form",
            "top1_score",
            "notes",
        ],
    }

    def dump(name: str, rows: Sequence[Mapping[str, object]]) -> None:
        path = output / name
        _write_csv(path, rows, empty_headers.get(name))
        written.append(str(path))

    dump("dataset_counts.csv", summary["dataset"])  # type: ignore[arg-type]
    dump("medicine_metrics.csv", medicine["metric_rows"])  # type: ignore[arg-type]
    dump("medicine_per_query.csv", medicine["per_query"])  # type: ignore[arg-type]
    dump("medicine_hits.csv", medicine["hits"])  # type: ignore[arg-type]
    dump("medicine_unreviewed_predictions.csv", medicine["unreviewed"])  # type: ignore[arg-type]
    dump("medicine_exclusions.csv", medicine["exclusions"])  # type: ignore[arg-type]
    dump("medicine_failures.csv", medicine["failures"])  # type: ignore[arg-type]
    dump("scheme_metrics.csv", summary["scheme"]["metric_rows"])  # type: ignore[index]
    dump("ablation_metrics.csv", summary["ablations"])  # type: ignore[arg-type]
    dump("rag_metrics.csv", summary["rag"]["metric_rows"])  # type: ignore[index]
    dump("freshness_metrics.csv", summary["freshness"]["metric_rows"])  # type: ignore[index]
    claim_metric_rows = [
        {
            "metric": "precision_contradicted",
            "value": summary["claims"]["precision"]["value"],  # type: ignore[index]
            "reason": summary["claims"]["precision"]["reason"],  # type: ignore[index]
        },
        {
            "metric": "recall_contradicted",
            "value": summary["claims"]["recall"]["value"],  # type: ignore[index]
            "reason": summary["claims"]["recall"]["reason"],  # type: ignore[index]
        },
        {
            "metric": "keyword_baseline_precision",
            "value": summary["claims"]["keyword_baseline"]["value"],  # type: ignore[index]
            "reason": summary["claims"]["keyword_baseline"]["reason"],  # type: ignore[index]
        },
        {"metric": "n_rows", "value": summary["claims"]["n_rows"], "reason": ""},  # type: ignore[index]
        {
            "metric": "n_human_reviewed_yes",
            "value": summary["claims"]["n_human_reviewed_yes"],  # type: ignore[index]
            "reason": "",
        },
        {
            "metric": "n_ai_drafted",
            "value": summary["claims"]["n_ai_drafted"],  # type: ignore[index]
            "reason": "AI-generated rows are marked for human review and are not gold.",
        },
    ]
    dump("claims_metrics.csv", claim_metric_rows)
    dump("claims_label_audit.csv", claims["audit_rows"])  # type: ignore[arg-type]
    summary_path = output / "summary.json"
    summary["files"] = [*written, str(summary_path)]  # type: ignore[index]
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    written.append(str(summary_path))
    return written


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    fieldnames = names
    if not fieldnames:
        fieldnames = ["value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def render_report(summary: Mapping[str, object]) -> str:
    """Plain-text report of the counts and metric tables."""
    lines: list[str] = []
    lines.append("Bharosa evaluation")
    lines.append(f"command: {summary['command']}")
    lines.append("thresholds tuned on this run: false")
    lines.append(
        f"medicine minimum brand cosine: {summary['medicine_min_brand_cosine']} "
        f"({summary['medicine_threshold_source']})"
    )
    lines.append("")
    lines.append("dataset counts")
    for row in summary["dataset"]:  # type: ignore[union-attr]
        lines.append(f"  {row['name']}: {row['value']}")
    lines.append("")
    lines.append("limitations")
    for item in summary["limitations"]:  # type: ignore[union-attr]
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("medicine metrics (macro average)")
    lines.append(
        "method relevance n top1 P@5 recall@5 P@10 recall@10 set_precision set_recall"
    )
    for row in summary["medicine"]["metric_rows"]:  # type: ignore[index]
        lines.append(
            " ".join(
                [
                    str(row["method"]),
                    str(row["relevance"]),
                    str(row["n_queries_scored"]),
                    _show(row["top1_accuracy"]),
                    _show(row["P@5"]),
                    _show(row["recall@5"]),
                    _show(row["P@10"]),
                    _show(row["recall@10"]),
                    _show(row["set_precision"]),
                    _show(row["set_recall"]),
                ]
            )
        )
        if row.get("reason") and row["P@5"] == NA:
            lines.append(f"  reason: {row['reason']}")
    lines.append("")
    lines.append(
        f"ours brand top-1 misses on scored queries: "
        f"{summary['medicine']['n_failures_ours_brand_top1']}"  # type: ignore[index]
    )
    lines.append("")
    lines.append("scheme metrics")
    for row in summary["scheme"]["metric_rows"]:  # type: ignore[index]
        if row["metric"] in {"P@5", "P@10", "recall@5", "recall@10"}:
            lines.append(f"  {row['system']} {row['metric']}: {_show(row['value'])}")
            if row["value"] == NA:
                lines.append(f"    {row['reason']}")
    lines.append("")
    lines.append("ablations (delta P@5 = component off minus full system)")
    for row in summary["ablations"]:  # type: ignore[union-attr]
        lines.append(f"  {row['component']}: {_show(row['value'])}")
        if row["value"] == NA:
            lines.append(f"    {row['reason']}")
    lines.append("")
    lines.append("rag")
    for row in summary["rag"]["metric_rows"]:  # type: ignore[index]
        lines.append(f"  {row['metric']}: {_show(row['value'])}")
        lines.append(f"    {row['reason']}")
    lines.append("")
    lines.append("claims")
    claims = summary["claims"]
    lines.append(
        f"  rows: {claims['n_rows']}; human_reviewed=yes: {claims['n_human_reviewed_yes']}; "  # type: ignore[index]
        f"AI_DRAFTED: {claims['n_ai_drafted']}"  # type: ignore[index]
    )
    lines.append(f"  precision: {_show(claims['precision']['value'])}")  # type: ignore[index]
    lines.append(f"    {claims['precision']['reason']}")  # type: ignore[index]
    lines.append(f"  recall: {_show(claims['recall']['value'])}")  # type: ignore[index]
    lines.append(f"    {claims['recall']['reason']}")  # type: ignore[index]
    lines.append("")
    lines.append("freshness")
    for row in summary["freshness"]["metric_rows"]:  # type: ignore[index]
        lines.append(f"  {row['metric']}: {_show(row['value'])}")
        lines.append(f"    {row['reason']}")
    lines.append("")
    lines.append("files")
    for path in summary.get("files", []):
        lines.append(f"  {path}")
    return "\n".join(lines)


def _show(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_args(argv: list[str] | None = None) -> RunPaths:
    """CLI defaults point at this repository. Tests pass other directories."""
    defaults = RunPaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--medicine-csv", type=Path, default=defaults.medicine_csv)
    parser.add_argument("--medicine-queries", type=Path, default=defaults.medicine_queries)
    parser.add_argument("--claims", type=Path, default=defaults.claims_csv)
    parser.add_argument("--eval-dir", type=Path, default=defaults.eval_dir)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--config-dir", type=Path, default=defaults.config_dir)
    parser.add_argument("--lexicon", type=Path, default=defaults.lexicon_csv)
    args = parser.parse_args(argv)
    return RunPaths(
        root=defaults.root,
        output=args.output,
        medicine_csv=args.medicine_csv,
        medicine_queries=args.medicine_queries,
        claims_csv=args.claims,
        eval_dir=args.eval_dir,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        lexicon_csv=args.lexicon,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 after the result files are written."""
    paths = parse_args(argv)
    execute(paths, progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
