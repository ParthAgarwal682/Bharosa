"""Sentence-level citation checker for RAG answers.

IR concept: post-generation support check. Each answer sentence is
compared to its cited zone text using a lexical-overlap heuristic.
Below-threshold pairs are flagged unsupported.

Lexical overlap is a heuristic only — the fraction of claim tokens found
in the cited zone. It is not cosine similarity and not logical entailment.
Cite IDs are checked against structured labels and against ``doc_id``.
"""

from __future__ import annotations

from collections.abc import Sequence

from bharosa.rag.answer import label_hits
from bharosa.rag.config import get_config
from bharosa.rag.similarity import lexical_overlap_score
from bharosa.rag.types import (
    Answer,
    SentenceCheck,
    ZoneEvidence,
    canonical_cite,
    label_index,
    normalize_zone_label,
)


def _resolve_hit(
    cite: str,
    labels: dict[str, str],
    hits: Sequence[ZoneEvidence],
) -> ZoneEvidence | None:
    """Resolve a cite to a hit via its label or its doc_id.

    A structured label must map to a ``doc_id::zone`` present in ``hits``.
    A raw doc id is accepted when it identifies exactly one hit. Unknown
    labels and unknown doc ids resolve to None.
    """
    by_key = {f"{hit.doc_id}::{hit.zone}": hit for hit in hits}
    canon = canonical_cite(cite, labels)
    if canon is not None:
        if canon not in labels:
            return None
        return by_key.get(labels[canon])
    token = cite.strip()
    if token in by_key:
        return by_key[token]
    matched = [hit for hit in hits if hit.doc_id == token]
    if len(matched) == 1:
        return matched[0]
    return None


def check_citations(
    ans: Answer,
    hits: Sequence[ZoneEvidence],
    *,
    threshold: float | None = None,
) -> list[SentenceCheck]:
    """Flag answer sentences that are missing or weakly supported by cites.

    IR concept: citation validation. Cite IDs must match a structured label
    (``Z1``) whose value is a hit ``doc_id::zone``, or a hit ``doc_id``.
    Support is a lexical-overlap heuristic against the cited zone text, not
    entailment. ``supported`` if the best overlap is at least the threshold.
    Threshold is read from config at call time when None. Refused answers
    yield an empty check list.
    """
    if threshold is None:
        threshold = get_config().citation_threshold

    if ans.refused:
        return []

    labels = label_index(ans.hit_labels or label_hits(hits))
    checks: list[SentenceCheck] = []
    for index, sentence in enumerate(ans.sentences):
        cite_ids = [normalize_zone_label(c) for c in sentence.cite_ids]
        if not cite_ids:
            checks.append(
                SentenceCheck(
                    sentence_index=index,
                    cite_ids=[],
                    lexical_overlap=0.0,
                    supported=False,
                    flag="missing_citation",
                )
            )
            continue

        resolved: list[ZoneEvidence] = []
        unknown = False
        for cite in cite_ids:
            hit = _resolve_hit(cite, labels, hits)
            if hit is None:
                unknown = True
                break
            resolved.append(hit)
        if unknown:
            checks.append(
                SentenceCheck(
                    sentence_index=index,
                    cite_ids=cite_ids,
                    lexical_overlap=0.0,
                    supported=False,
                    flag="unknown_cite_id",
                )
            )
            continue

        scores = [lexical_overlap_score(sentence.text, hit.text) for hit in resolved]
        max_overlap = max(scores) if scores else 0.0
        supported = max_overlap >= threshold
        checks.append(
            SentenceCheck(
                sentence_index=index,
                cite_ids=cite_ids,
                lexical_overlap=max_overlap,
                supported=supported,
                flag=None if supported else "unsupported",
            )
        )
    return checks
