"""Pipeline running answer generation followed by citation validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bharosa.rag.answer import answer
from bharosa.rag.citations import check_citations
from bharosa.rag.config import get_config
from bharosa.rag.llm import LLMClient
from bharosa.rag.types import Answer, SentenceCheck, ZoneHit


class CheckedResult(tuple):
    """Tuple of (Answer, list[SentenceCheck]) delegating Answer attributes."""

    def __new__(cls, ans: Answer, checks: list[SentenceCheck]):
        return super().__new__(cls, (ans, checks))

    @property
    def answer(self) -> Answer:
        return self[0]

    @property
    def checks(self) -> list[SentenceCheck]:
        return self[1]

    @property
    def query(self) -> str:
        return self[0].query

    @property
    def refused(self) -> bool:
        return self[0].refused

    @property
    def refusal_reason(self) -> str | None:
        return self[0].refusal_reason

    @property
    def sentences(self) -> list:
        return self[0].sentences

    @property
    def hit_labels(self) -> dict:
        return self[0].hit_labels

    @property
    def model(self) -> str | None:
        return self[0].model

    @property
    def raw_llm(self) -> str | None:
        return self[0].raw_llm

    @property
    def refusal_message(self) -> str | None:
        return self[0].refusal_message


def answer_checked(
    query: str,
    hits: list[ZoneHit],
    *,
    llm: LLMClient | None = None,
    required_state: str | None = None,
    required_conditions: Sequence[str] | None = None,
    score_threshold: float | None = None,
    citation_threshold: float | None = None,
    policy: str | None = None,
    **kwargs: Any,
) -> CheckedResult:
    """Generate cited answer and validate citations with configured policy."""
    cfg = get_config()
    pol = (policy or cfg.citation_policy).strip().lower()
    if pol not in {"flag", "withhold"}:
        pol = "flag"

    ans = answer(
        query,
        hits,
        llm=llm,
        required_state=required_state,
        required_conditions=required_conditions,
        score_threshold=score_threshold,
        **kwargs,
    )
    if ans.refused:
        return CheckedResult(ans, [])

    checks = check_citations(ans, hits, threshold=citation_threshold)

    if pol == "withhold" and any(not c.supported for c in checks):
        withheld_answer = Answer(
            query=ans.query,
            refused=True,
            refusal_reason="citation_validation_failed",
            sentences=[],
            hit_labels=ans.hit_labels,
            model=ans.model,
            raw_llm=ans.raw_llm,
            refusal_message="I couldn't verify the answer against official sources.",
        )
        return CheckedResult(withheld_answer, checks)

    return CheckedResult(ans, checks)
