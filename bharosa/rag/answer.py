"""Scheme RAG: cited answers from retrieved zones, with refusal.

IR concept: generation only after retrieval. The LLM sees labelled
top-K zones and must return JSON containing claims with valid zone IDs.
If retrieval confidence is low, refuse without calling the LLM.
Never used for medicine / drug facts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from bharosa.rag.config import get_config
from bharosa.rag.llm import LLMClient, LLMNotConfiguredError, get_llm
from bharosa.rag.types import Answer, CitedSentence, RefusalReason, ZoneHit

REFUSAL_USER_MESSAGE = "I couldn't find this in official sources"
INVALID_GENERATION_MESSAGE = "I couldn't produce a verified answer. Please try again."


class LLMOutputValidationError(ValueError):
    """Raised when LLM output is malformed, lacks valid schema, or cites unknown IDs."""


def label_hits(hits: Sequence[ZoneHit]) -> dict[str, str]:
    """Map ``Z1``…``Zk`` to a stable hit key (``doc_id`` + zone).

    IR concept: citation inventory. Labels are positional in the
    ranked hit list Paridhi already returned — RAG does not re-rank.
    """
    labels: dict[str, str] = {}
    for index, hit in enumerate(hits, start=1):
        labels[f"Z{index}"] = f"{hit.doc_id}::{hit.zone}"
    return labels


def should_refuse(
    hits: Sequence[ZoneHit],
    *,
    required_state: str | None = None,
    required_conditions: Sequence[str] | None = None,
    score_threshold: float | None = None,
) -> RefusalReason | None:
    """Return a refusal reason, or None if generation may proceed.

    IR concept: confidence gate before generation. Empty evidence or a
    top score below threshold means the index did not support an answer.
    """
    if score_threshold is None:
        score_threshold = get_config().refusal_score_threshold

    if not hits:
        return "empty_hits"
    if all(not (hit.text or "").strip() for hit in hits):
        return "empty_evidence"
    top_score = max(hit.score for hit in hits)
    if top_score < score_threshold:
        return "low_retrieval_score"
    if required_state:
        state_norm = required_state.strip().casefold()
        if not any(
            (hit.state or "").strip().casefold() == state_norm for hit in hits
        ):
            return "no_matching_filter"
    if required_conditions:
        needed = {c.strip().casefold() for c in required_conditions if c.strip()}
        if needed:
            ok = False
            for hit in hits:
                have = {c.strip().casefold() for c in hit.conditions}
                if needed.issubset(have) or (needed & have):
                    ok = True
                    break
            if not ok:
                return "no_matching_filter"
    return None


def build_prompt(query: str, hits: Sequence[ZoneHit]) -> str:
    """Build the constrained RAG prompt from labelled zones only."""
    labels = label_hits(hits)
    blocks: list[str] = []
    for label, hit in zip(labels, hits, strict=True):
        blocks.append(
            f"[{label}] zone={hit.zone} title={hit.title}\n{hit.text.strip()}"
        )
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
    Raises LLMOutputValidationError if JSON is malformed, schema is violated, cite_ids is empty,
    or any unknown cite ID is encountered. Never emits raw LLM text as an answer sentence.
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

    valid_labels = {k.upper() for k in hit_labels}
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
            c_norm = str(c).strip().upper()
            if c_norm not in valid_labels:
                raise LLMOutputValidationError(
                    f"Unknown cite_id '{c}' not in evidence labels: {sorted(valid_labels)}"
                )
            normalized_cites.append(c_norm)

        sentences.append(CitedSentence(text=text, cite_ids=normalized_cites))

    return sentences


def answer(
    query: str,
    hits: list[ZoneHit],
    *,
    llm: LLMClient | None = None,
    required_state: str | None = None,
    required_conditions: Sequence[str] | None = None,
    score_threshold: float | None = None,
) -> Answer:
    """Produce a cited answer from retrieved zones, or refuse.

    IR concept: RAG over ranked evidence. Does not retrieve; consumes
    Paridhi's ``ZoneHit`` list. Refuses without an LLM call when the
    confidence gate fails.
    """
    if score_threshold is None:
        score_threshold = get_config().refusal_score_threshold
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")

    reason = should_refuse(
        hits,
        required_state=required_state,
        required_conditions=required_conditions,
        score_threshold=score_threshold,
    )
    labels = label_hits(hits)
    if reason is not None:
        return Answer(
            query=query,
            refused=True,
            refusal_reason=reason,
            sentences=[],
            hit_labels=labels,
            model=None,
            raw_llm=None,
            refusal_message=REFUSAL_USER_MESSAGE,
        )

    client = llm if llm is not None else get_llm()
    prompt = build_prompt(query, hits)
    try:
        raw = client.complete(prompt)
    except LLMNotConfiguredError:
        return Answer(
            query=query,
            refused=True,
            refusal_reason="empty_evidence",
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
