"""Text similarity helpers for citation and claim checking.

IR concept: lexical overlap / token matching as a heuristic support signal.
Lexical similarity is a heuristic only — NOT proof of logical entailment.

Does not import from Module C (bharosa.rank). An injectable scorer function
is accepted, defaulting to the fraction of claim tokens present in the evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bharosa.text.normalize import tokenize


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


def text_cosine(
    left: str,
    right: str,
    *,
    scorer: Callable[[Sequence[str], Sequence[str]], float] | None = None,
) -> float:
    """Return lexical similarity score between two texts after shared tokenization.

    IR concept: pairwise citation support / match score in [0, 1].
    Lexical similarity is a heuristic only — NOT proof of entailment.
    Default scorer is the fraction of left (claim) tokens found in right (evidence).
    """
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("left and right must be str")
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if scorer is not None:
        return float(scorer(left_tokens, right_tokens))
    return float(token_overlap_scorer(left_tokens, right_tokens))
