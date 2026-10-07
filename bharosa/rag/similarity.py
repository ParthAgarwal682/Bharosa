"""Text similarity helpers for citation and claim checking.

IR concept: lexical overlap / token matching as a heuristic support signal.
Lexical similarity is a heuristic only — NOT proof of logical entailment.

Does not import from Module C (bharosa.rank). An injectable scorer function
is accepted, defaulting to the fraction of claim tokens present in the evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

_TOKEN_RE = re.compile(r"[₹\w]+", re.UNICODE)


def default_tokenize(text: str) -> list[str]:
    """Small local deterministic tokenizer: casefold, regex on word chars and ₹."""
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.casefold())


def token_overlap_scorer(
    claim_tokens: Sequence[str],
    evidence_tokens: Sequence[str],
) -> float:
    """Calculate the fraction of claim tokens found in the evidence text.

    Lexical heuristic only — NOT proof of logical entailment.
    Returns 0.0 if claim has no tokens or disjoint tokens.
    """
    if not claim_tokens:
        return 0.0
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    overlap = claim_set & evidence_set
    return len(overlap) / len(claim_set)


def lexical_overlap_score(
    left: str,
    right: str,
    *,
    scorer: Callable[[Sequence[str], Sequence[str]], float] | None = None,
    tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> float:
    """Return lexical overlap score between two texts after shared tokenization.

    Calculates the fraction of claim (left) tokens found in the evidence (right).
    This is an unweighted lexical overlap heuristic, NOT cosine similarity
    and NOT proof of logical entailment.
    Default scorer is the fraction of left (claim) tokens found in right (evidence).
    """
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("left and right must be str")
    tok = tokenizer if tokenizer is not None else default_tokenize
    left_tokens = tok(left)
    right_tokens = tok(right)
    if scorer is not None:
        return float(scorer(left_tokens, right_tokens))
    return float(token_overlap_scorer(left_tokens, right_tokens))


# Backwards-compatible alias for existing callers
text_cosine = lexical_overlap_score

