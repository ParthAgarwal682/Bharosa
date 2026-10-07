"""Module 3: fee / free / deadline claim checker against official zones.

IR concept: claim verification by (1) pattern extraction, (2) retrieving
official zones via injected callable, (3) comparing across all retrieved
zones. If retrieved zones disagree (one SUPPORTED, another CONTRADICTED),
returns INSUFFICIENT_EVIDENCE with both pieces of evidence. LLM may only
polish an explanation after the verdict; never invent the verdict from the model alone.

RAG does not implement a retriever. The caller supplies a retrieve callable
(e.g., Paridhi's search_schemes) which returns a sequence of ZoneHit objects.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from bharosa.rag.config import get_config
from bharosa.rag.llm import LLMClient
from bharosa.rag.similarity import text_cosine
from bharosa.rag.types import ClaimLabel, Verdict, ZoneHit

_AMOUNT = re.compile(
    r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_FEE_WORD = re.compile(
    r"\b(fee|fees|charge|charges|registration|activate|activation)\b",
    re.IGNORECASE,
)
# "no registration fee" / "no fee" / "free" count as official free signals.
_FREE_WORD = re.compile(
    r"\b(free|without\s+any\s+fee|no\s+(?:registration\s+|activation\s+)?fees?)\b",
    re.IGNORECASE,
)
_DEADLINE = re.compile(
    r"\b(?:before|by|until|deadline|last\s+date)\s+"
    r"([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}|"
    r"[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})\b",
    re.IGNORECASE,
)


def extract_claim_spans(message: str) -> tuple[list[str], list[float], bool, list[str]]:
    """Extract fee/free/deadline spans from a pasted forward.

    Returns ``(spans, amounts, claims_free, deadlines)``.
    IR concept: lightweight query analysis before retrieval.
    """
    if not isinstance(message, str):
        raise TypeError(f"message must be str, got {type(message).__name__}")

    spans: list[str] = []
    amounts: list[float] = []
    for match in _AMOUNT.finditer(message):
        spans.append(match.group(0))
        amounts.append(float(match.group(1).replace(",", "")))

    claims_free = bool(_FREE_WORD.search(message))
    if claims_free:
        spans.append(_FREE_WORD.search(message).group(0))  # type: ignore[union-attr]

    deadlines: list[str] = []
    for match in _DEADLINE.finditer(message):
        spans.append(match.group(0))
        deadlines.append(match.group(1))

    if not spans and _FEE_WORD.search(message):
        spans.append(_FEE_WORD.search(message).group(0))  # type: ignore[union-attr]

    return spans, amounts, claims_free, deadlines


def nearest_zone(
    message: str, zones: Sequence[ZoneHit]
) -> tuple[ZoneHit | None, float, str | None]:
    """Return the zone with highest cosine to ``message``.

    IR concept: ad-hoc similarity lookup over a provided evidence list.
    """
    if not zones:
        return None, 0.0, None
    best: ZoneHit | None = None
    best_score = -1.0
    best_label: str | None = None
    for index, zone in enumerate(zones, start=1):
        score = text_cosine(message, zone.text)
        if score > best_score:
            best = zone
            best_score = score
            best_label = f"Z{index}"
    return best, max(best_score, 0.0), best_label


def _amounts_in_text(text: str) -> list[float]:
    return [
        float(m.group(1).replace(",", ""))
        for m in _AMOUNT.finditer(text)
    ]


def compare_claim(
    *,
    amounts: list[float],
    claims_free: bool,
    deadlines: list[str],
    evidence_text: str,
) -> ClaimLabel:
    """Deterministic compare of extracted claim vs official line.

    Prefers ``INSUFFICIENT_EVIDENCE`` over guessing when the official text is
    silent on the claimed attribute.
    """
    official_amounts = _amounts_in_text(evidence_text)
    official_free = bool(_FREE_WORD.search(evidence_text))
    official_deadlines = [m.group(1) for m in _DEADLINE.finditer(evidence_text)]

    # Fee amount claims
    if amounts:
        if not official_amounts and not official_free:
            return "INSUFFICIENT_EVIDENCE"
        if official_free and any(a > 0 for a in amounts):
            return "CONTRADICTED"
        if official_amounts and any(a in official_amounts for a in amounts):
            return "SUPPORTED"
        if official_amounts and amounts:
            return "CONTRADICTED"
        return "INSUFFICIENT_EVIDENCE"

    if claims_free:
        if official_free:
            return "SUPPORTED"
        if official_amounts and any(a > 0 for a in official_amounts):
            return "CONTRADICTED"
        if _FEE_WORD.search(evidence_text) and not official_free:
            return "INSUFFICIENT_EVIDENCE"
        return "INSUFFICIENT_EVIDENCE"

    if deadlines:
        if not official_deadlines:
            return "INSUFFICIENT_EVIDENCE"
        norm_claim = {d.strip().casefold() for d in deadlines}
        norm_official = {d.strip().casefold() for d in official_deadlines}
        if norm_claim & norm_official:
            return "SUPPORTED"
        return "CONTRADICTED"

    return "NO_CLAIM_FOUND"


def _template_explanation(
    label: ClaimLabel,
    evidence_text: str | None,
) -> str:
    if label == "NO_CLAIM_FOUND":
        return "No fee, free, or deadline claim was detected in the message."
    if label == "INSUFFICIENT_EVIDENCE" or not evidence_text:
        return (
            "INSUFFICIENT INFORMATION — DO NOT GUESS. "
            "Official text does not clearly confirm or deny the claim."
        )
    snippet = evidence_text.strip().replace("\n", " ")
    if len(snippet) > 220:
        snippet = snippet[:217] + "..."
    if label == "CONTRADICTED":
        return f"This claim conflicts with the official line: {snippet}"
    return f"This claim is consistent with the official line: {snippet}"


def check_claim(
    message: str,
    retrieve: Callable[[str], Sequence[ZoneHit]],
    *,
    llm: LLMClient | None = None,
    min_retrieval_score: float | None = None,
) -> Verdict:
    """Check a pasted fee/free/deadline claim against official zones from retrieve.

    IR concept: extract -> retrieve via injected callable -> compare across all hits.
    If official zones disagree (one SUPPORTED, another CONTRADICTED), returns
    INSUFFICIENT_EVIDENCE citing both pieces of evidence.
    Optional ``llm`` only rewrites the explanation after the label is determined.
    """
    if not callable(retrieve):
        raise TypeError(f"retrieve must be a callable, got {type(retrieve).__name__}")

    if min_retrieval_score is None:
        min_retrieval_score = get_config().min_retrieval_score

    spans, amounts, claims_free, deadlines = extract_claim_spans(message)
    if not spans and not amounts and not claims_free and not deadlines:
        return Verdict(
            message=message,
            claim_spans=[],
            label="NO_CLAIM_FOUND",
            evidence_cite=None,
            evidence_text=None,
            explanation=_template_explanation("NO_CLAIM_FOUND", None),
            retrieval_score=None,
            claim_amounts=[],
        )

    zones = list(retrieve(message))
    if not zones:
        return Verdict(
            message=message,
            claim_spans=spans,
            label="INSUFFICIENT_EVIDENCE",
            evidence_cite=None,
            evidence_text=None,
            explanation=(
                "INSUFFICIENT INFORMATION — DO NOT GUESS. "
                "No official zones returned by retriever."
            ),
            retrieval_score=None,
            claim_amounts=amounts,
        )

    # Check against all hits returned by retrieve
    contradiction: tuple[ZoneHit, str, float] | None = None
    support: tuple[ZoneHit, str, float] | None = None

    for index, zone in enumerate(zones, start=1):
        cite = f"Z{index}"
        score = getattr(zone, "score", 0.0)
        zone_label = compare_claim(
            amounts=amounts,
            claims_free=claims_free,
            deadlines=deadlines,
            evidence_text=zone.text,
        )
        if zone_label == "CONTRADICTED" and contradiction is None:
            contradiction = (zone, cite, score)
        elif zone_label == "SUPPORTED" and support is None:
            support = (zone, cite, score)

    # If retrieved zones disagree, return INSUFFICIENT_EVIDENCE with both pieces of evidence
    if contradiction is not None and support is not None:
        c_zone, c_cite, c_score = contradiction
        s_zone, s_cite, s_score = support
        chosen_label: ClaimLabel = "INSUFFICIENT_EVIDENCE"
        chosen_cite = f"{c_cite}, {s_cite}"
        chosen_score = max(c_score, s_score)
        chosen_text = (
            f"[{c_cite} - CONTRADICTS] {c_zone.text.strip()} | "
            f"[{s_cite} - SUPPORTS] {s_zone.text.strip()}"
        )
        explanation = (
            "INSUFFICIENT INFORMATION — DO NOT GUESS. "
            f"Retrieved official zones disagree on this claim: {c_cite} contradicts while {s_cite} supports."
        )
        return Verdict(
            message=message,
            claim_spans=spans,
            label=chosen_label,
            evidence_cite=chosen_cite,
            evidence_text=chosen_text,
            explanation=explanation,
            retrieval_score=chosen_score,
            claim_amounts=amounts,
        )

    if contradiction is not None:
        chosen_zone, chosen_cite, chosen_score = contradiction
        chosen_label = "CONTRADICTED"
        evidence_text = chosen_zone.text
    elif support is not None:
        chosen_zone, chosen_cite, chosen_score = support
        chosen_label = "SUPPORTED"
        evidence_text = chosen_zone.text
    else:
        nearest_z, near_score, near_cite = nearest_zone(message, zones)
        chosen_zone = nearest_z or zones[0]
        chosen_cite = near_cite or "Z1"
        chosen_score = near_score if nearest_z else getattr(zones[0], "score", 0.0)
        chosen_label = "INSUFFICIENT_EVIDENCE"
        evidence_text = chosen_zone.text if chosen_zone else None

    explanation = _template_explanation(chosen_label, evidence_text)

    if llm is not None and chosen_label in {"SUPPORTED", "CONTRADICTED"} and chosen_zone is not None:
        prompt = (
            "Rewrite in one short plain sentence. Do not add facts.\n"
            f"Verdict: {chosen_label}\n"
            f"Official: {chosen_zone.text[:400]}\n"
            f"Draft: {explanation}\n"
        )
        try:
            polished = llm.complete(prompt).strip()
            if polished:
                explanation = polished.split("\n")[0].strip()
        except Exception:
            # Keep template explanation; do not invent success.
            pass

    return Verdict(
        message=message,
        claim_spans=spans,
        label=chosen_label,
        evidence_cite=chosen_cite,
        evidence_text=evidence_text,
        explanation=explanation,
        retrieval_score=chosen_score,
        claim_amounts=amounts,
    )
