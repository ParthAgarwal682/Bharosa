"""Scheme RAG: cited answers from retrieved zones, with refusal.

IR concept: generation only after retrieval. The LLM sees labelled
top-K zones and must return JSON containing claims with valid zone IDs.
If the best ZoneHit.net is below the threshold, refuse without calling
the LLM. BM25 baseline hits (net is None) are not scored on that scale.
Never used for medicine / drug facts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from bharosa.rag.config import get_config
from bharosa.rag.llm import (
    LLMClient,
    LLMNotConfiguredError,
    LLMProviderError,
    get_llm,
)
from bharosa.rag.types import (
    Answer,
    CitedSentence,
    RefusalReason,
    ZoneEvidence,
    canonical_cite,
    evidence_net,
    hits_with_net,
    label_index,
    missing_net_reason,
)

REFUSAL_USER_MESSAGE = "I couldn't find this in official sources"
INVALID_GENERATION_MESSAGE = "I couldn't produce a verified answer. Please try again."
BM25_REFUSAL_MESSAGE = (
    "I couldn't find this in official sources "
    "(BM25 evidence has no NetScore; RAG will not treat bm25_score as net)"
)
MISSING_NET_MESSAGE = (
    "I couldn't find this in official sources "
    "(retrieved evidence has no NetScore; RAG will not invent one)"
)


class LLMOutputValidationError(ValueError):
    """Raised when LLM output is malformed, lacks valid schema, or cites unknown IDs."""


def _refusal_message(reason: RefusalReason) -> str:
    if reason == "unsupported_bm25_evidence":
        return BM25_REFUSAL_MESSAGE
    if reason == "missing_net_score":
        return MISSING_NET_MESSAGE
    return REFUSAL_USER_MESSAGE


def label_hits(hits: Sequence[ZoneEvidence]) -> dict[str, str]:
    """Map ``Z1``…``Zk`` to ``doc_id::zone``.

    IR concept: citation inventory. Labels are positional in the hit list
    passed in — RAG does not re-rank. The value is the evidence identity
    Paridhi's ZoneHit actually exposes.
    """
    labels: dict[str, str] = {}
    for index, hit in enumerate(hits, start=1):
        labels[f"Z{index}"] = f"{hit.doc_id}::{hit.zone}"
    return labels


def _evidence_header(label: str, hit: ZoneEvidence) -> str:
    """Source metadata ZoneHit actually carries. No title, state, or filters."""
    parts = [
        f"[{label}]",
        f"doc_id={hit.doc_id}",
        f"zone={hit.zone}",
        f"url={hit.url}",
    ]
    if hit.last_changed_at:
        parts.append(f"last_changed_at={hit.last_changed_at}")
    return " ".join(parts)


def should_refuse(
    hits: Sequence[ZoneEvidence],
    *,
    score_threshold: float | None = None,
    allow_mock: bool = False,
) -> RefusalReason | None:
    """Return a refusal reason, or None if generation may proceed.

    IR concept: confidence gate before generation. The retrieval score is
    ``ZoneHit.net``. Empty evidence or a top net below threshold means the
    index did not support an answer. BM25-only hits are refused as
    unsupported evidence rather than compared to the NetScore threshold.
    """
    if not hits:
        return "empty_hits"
    if not allow_mock and any(getattr(hit, "source", None) == "mock_fixture" for hit in hits):
        return "mock_evidence"
    unscored = missing_net_reason(hits)
    if unscored is not None:
        return unscored
    net_hits = hits_with_net(hits)
    if all(not (hit.text or "").strip() for hit in net_hits):
        return "empty_evidence"
    if score_threshold is None:
        score_threshold = get_config().refusal_score_threshold
    scores: list[float] = []
    for hit in net_hits:
        score = evidence_net(hit)
        if score is not None:
            scores.append(score)
    top_score = max(scores)
    if top_score < score_threshold:
        return "low_retrieval_score"
    return None


def build_prompt(query: str, hits: Sequence[ZoneEvidence]) -> str:
    """Build the constrained RAG prompt from labelled zones only."""
    labels = label_hits(hits)
    blocks: list[str] = []
    for label, hit in zip(labels, hits, strict=True):
        blocks.append(f"{_evidence_header(label, hit)}\n{hit.text.strip()}")
    evidence = "\n\n".join(blocks)
    return (
        "You answer using ONLY the official evidence zones below.\n"
        "Rules:\n"
        "- Write simple Hinglish or English.\n"
        "- One claim per sentence.\n"
        "- Return valid JSON only, using this exact schema:\n"
        "  {\"claims\": [{\"text\": \"...\", \"cite_ids\": [\"Z1\"]}]}\n"
        "- Do not use inline [Zn] syntax in text.\n"
        "- Every claim MUST cite one or more valid zone IDs (e.g. [\"Z1\"]) from the evidence.\n"
        "- Use only facts present in the provided text.\n"
        "- If the answer is not in the zones, return:\n"
        "  {\"claims\": [{\"text\": \"not found in official sources\", \"cite_ids\": [\"Z1\"]}]}\n"
        "- Do not give medical advice or medicine recommendations.\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer (JSON):"
    )


def parse_llm_answer(raw: str, hit_labels: dict[str, str]) -> list[CitedSentence]:
    """Parse JSON LLM output into cited sentences, validating schema and rejecting unknown IDs.

    IR concept: structured generation output. Validates {"claims": [{"text": "...", "cite_ids": [...]}]}.
    A cite is valid when it is a structured label (``Z1``) or the ``doc_id`` /
    ``doc_id::zone`` of exactly one label. Unknown IDs raise
    ``LLMOutputValidationError``. Stored cite ids stay structured labels.
    Never emits raw LLM text as an answer sentence.
    """
    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise LLMOutputValidationError(f"Malformed JSON in LLM response: {raw!r}") from exc

    if not isinstance(data, dict) or "claims" not in data or not isinstance(data["claims"], list):
        raise LLMOutputValidationError("Invalid schema: 'claims' list is required in output JSON")

    known = sorted(label_index(hit_labels))
    sentences: list[CitedSentence] = []

    for item in data["claims"]:
        if not isinstance(item, dict):
            raise LLMOutputValidationError("Invalid schema: each claim must be a JSON object")
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        raw_cites = item.get("cite_ids")
        if not raw_cites or not isinstance(raw_cites, list):
            raise LLMOutputValidationError(f"Claim is missing cite_ids: {text!r}")

        normalized_cites: list[str] = []
        for c in raw_cites:
            canon = canonical_cite(str(c), hit_labels)
            if canon is None:
                raise LLMOutputValidationError(
                    f"Unknown cite_id '{c}' not in evidence labels: {known}"
                )
            normalized_cites.append(canon)

        sentences.append(CitedSentence(text=text, cite_ids=normalized_cites))

    return sentences


def answer(
    query: str,
    hits: Sequence[ZoneEvidence],
    *,
    llm: LLMClient | None = None,
    score_threshold: float | None = None,
    allow_mock: bool = False,
) -> Answer:
    """Produce a cited answer from retrieved zones, or refuse.

    IR concept: RAG over ranked evidence. Does not retrieve; consumes
    Paridhi's ``ZoneHit`` list. The retrieval score is ``ZoneHit.net``.
    Refuses without an LLM call when the confidence gate fails.
    """
    if score_threshold is None:
        score_threshold = get_config().refusal_score_threshold
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")

    reason = should_refuse(
        hits,
        score_threshold=score_threshold,
        allow_mock=allow_mock,
    )
    prompt_hits: Sequence[ZoneEvidence] = hits_with_net(hits) if reason is None else hits
    labels = label_hits(prompt_hits)
    if reason is not None:
        return Answer(
            query=query,
            refused=True,
            refusal_reason=reason,
            sentences=[],
            hit_labels=labels,
            model=None,
            raw_llm=None,
            refusal_message=_refusal_message(reason),
        )

    client = llm if llm is not None else get_llm()
    prompt = build_prompt(query, prompt_hits)
    try:
        raw = client.complete(prompt)
    except LLMNotConfiguredError:
        return Answer(
            query=query,
            refused=True,
            refusal_reason="llm_not_configured",
            sentences=[],
            hit_labels=labels,
            model=None,
            raw_llm=None,
            refusal_message=(
                "I couldn't find this in official sources "
                "(LLM API not configured — set LLM_PROVIDER, LLM_MODEL, "
                "LLM_API_KEY in .env)"
            ),
        )
    except (LLMProviderError, TimeoutError):
        model_name = getattr(client, "model", None)
        if not isinstance(model_name, str):
            model_name = type(client).__name__
        return Answer(
            query=query,
            refused=True,
            refusal_reason="llm_unavailable",
            sentences=[],
            hit_labels=labels,
            model=model_name,
            raw_llm=None,
            refusal_message="I couldn't reach the answer service. Please try again later.",
        )

    model_name = getattr(client, "model", None)
    if not isinstance(model_name, str):
        model_name = type(client).__name__

    try:
        sentences = parse_llm_answer(raw, labels)
    except LLMOutputValidationError:
        return Answer(
            query=query,
            refused=True,
            refusal_reason="invalid_generation",
            sentences=[],
            hit_labels=labels,
            model=model_name,
            raw_llm=raw,
            refusal_message=INVALID_GENERATION_MESSAGE,
        )

    if not sentences:
        return Answer(
            query=query,
            refused=True,
            refusal_reason="empty_generation",
            sentences=[],
            hit_labels=labels,
            model=model_name,
            raw_llm=raw,
            refusal_message=REFUSAL_USER_MESSAGE,
        )

    return Answer(
        query=query,
        refused=False,
        refusal_reason=None,
        sentences=sentences,
        hit_labels=labels,
        model=model_name,
        raw_llm=raw,
        refusal_message=None,
    )
