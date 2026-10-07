"""Sentence-level citation checker for RAG answers.

IR concept: post-generation support check. Each answer sentence is
compared to its cited zone text using lexical overlap similarity.
Below-threshold pairs are flagged unsupported.

Lexical similarity is a heuristic only — not logical entailment.
"""

from __future__ import annotations

from bharosa.rag.answer import label_hits
from bharosa.rag.config import get_config
from bharosa.rag.similarity import text_cosine
from bharosa.rag.types import Answer, SentenceCheck, ZoneHit


def check_citations(
    ans: Answer,
    hits: list[ZoneHit],
    *,
    threshold: float | None = None,
) -> list[SentenceCheck]:
    """Flag answer sentences that are missing or weakly supported by cites.

    IR concept: citation validation. Uses lexical similarity against each cited zone;
    ``supported`` if max similarity >= threshold. Threshold is read from config
    dynamically at call time when None. Refused answers yield an empty check list.
    """
    if threshold is None:
        threshold = get_config().citation_threshold

    if ans.refused:
        return []

    labels = ans.hit_labels or label_hits(hits)
    label_to_hit: dict[str, ZoneHit] = {}
    for label, hit in zip(labels, hits, strict=False):
        label_to_hit[label] = hit
    # Rebuild from positional labels if hit_labels keys are Z1..Zk
    if len(label_to_hit) != len(hits):
        label_to_hit = {
            f"Z{i}": hit for i, hit in enumerate(hits, start=1)
        }

    checks: list[SentenceCheck] = []
    for index, sentence in enumerate(ans.sentences):
        cite_ids = [c.upper() for c in sentence.cite_ids]
        if not cite_ids:
            checks.append(
                SentenceCheck(
                    sentence_index=index,
                    cite_ids=[],
                    max_cosine=0.0,
                    supported=False,
                    flag="missing_citation",
                )
            )
            continue

        unknown = [c for c in cite_ids if c not in label_to_hit]
        if unknown:
            checks.append(
                SentenceCheck(
                    sentence_index=index,
                    cite_ids=cite_ids,
                    max_cosine=0.0,
                    supported=False,
                    flag="unknown_cite_id",
                )
            )
            continue

        scores = [
            text_cosine(sentence.text, label_to_hit[c].text) for c in cite_ids
        ]
        max_cos = max(scores) if scores else 0.0
        supported = max_cos >= threshold
        checks.append(
            SentenceCheck(
                sentence_index=index,
                cite_ids=cite_ids,
                max_cosine=max_cos,
                supported=supported,
                flag=None if supported else "unsupported",
            )
        )
    return checks
