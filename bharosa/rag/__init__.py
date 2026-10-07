"""RAG, citation validation, refusal, and claim checking (Person D).

Public entry points match README Module D contracts:

- ``answer(query, hits) -> Answer``
- ``check_citations(ans, hits) -> list[SentenceCheck]``
- ``check_claim(message, retrieve) -> Verdict``

Does not implement retrieval, indexing, or medicine facts.
"""

from __future__ import annotations

from bharosa.rag.answer import (
    INVALID_GENERATION_MESSAGE,
    REFUSAL_USER_MESSAGE,
    LLMOutputValidationError,
    answer,
    build_prompt,
    label_hits,
    parse_llm_answer,
    should_refuse,
)
from bharosa.rag.citations import check_citations
from bharosa.rag.claimcheck import (
    check_claim,
    compare_claim,
    extract_claim_spans,
    nearest_zone,
)
from bharosa.rag.config import (
    DEFAULT_CITATION_THRESHOLD,
    DEFAULT_MIN_RETRIEVAL_SCORE,
    DEFAULT_REFUSAL_THRESHOLD,
    RAGConfig,
    get_config,
)
from bharosa.rag.llm import LLMNotConfiguredError, get_llm
from bharosa.rag.similarity import text_cosine, token_overlap_scorer
from bharosa.rag.types import (
    Answer,
    CitedSentence,
    ClaimLabel,
    RefusalReason,
    SentenceCheck,
    Verdict,
    ZoneHit,
    ZoneName,
)

__all__ = [
    "Answer",
    "CitedSentence",
    "ClaimLabel",
    "DEFAULT_CITATION_THRESHOLD",
    "DEFAULT_MIN_RETRIEVAL_SCORE",
    "DEFAULT_REFUSAL_THRESHOLD",
    "INVALID_GENERATION_MESSAGE",
    "LLMNotConfiguredError",
    "LLMOutputValidationError",
    "RAGConfig",
    "REFUSAL_USER_MESSAGE",
    "RefusalReason",
    "SentenceCheck",
    "Verdict",
    "ZoneHit",
    "ZoneName",
    "answer",
    "build_prompt",
    "check_citations",
    "check_claim",
    "compare_claim",
    "extract_claim_spans",
    "get_config",
    "get_llm",
    "label_hits",
    "nearest_zone",
    "parse_llm_answer",
    "should_refuse",
    "text_cosine",
    "token_overlap_scorer",
]
