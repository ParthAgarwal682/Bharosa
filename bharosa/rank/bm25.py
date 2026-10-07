"""External BM25 baseline.

IR concept: BM25 used as a comparison system. Scores are computed by
``rank_bm25.BM25Okapi``. This module does not implement BM25, and it is
not Bharosa's ranker. Do not add these scores to ``bharosa.rank.netscore``
and do not treat them as the lnc.ltc scores from ``bharosa.rank.tfidf``.

Tokenisation uses ``bharosa.text.normalize.tokenize`` so the baseline
sees the same terms as the index. ``k1``, ``b``, and ``epsilon`` are
passed through to the library. ``LIBRARY_K1``, ``LIBRARY_B``, and
``LIBRARY_EPSILON`` are the defaults published by that library's
constructor, named here so a run can say which values it used. They
were not fit on Bharosa judgments.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from bharosa.text.normalize import tokenize

EXTERNAL_LIBRARY = "rank_bm25"
EXTERNAL_CLASS = "BM25Okapi"

# Constructor defaults of rank_bm25.BM25Okapi. Passed explicitly so the
# baseline does not hide them. Not fit on Bharosa judgments.
LIBRARY_K1 = 1.5
LIBRARY_B = 0.75
LIBRARY_EPSILON = 0.25


@dataclass(frozen=True)
class BM25Hit:
    """One document and the score returned by the external library.

    IR concept: a baseline score. This is not a cosine, not a net score,
    and not an input to ``NetScorer``.
    """

    doc_id: str
    score: float


class BM25Baseline:
    """Rank a fixed corpus with ``rank_bm25.BM25Okapi``.

    IR concept: an external baseline built once on a tokenised corpus.
    The object does not update itself when other indexes change. An
    empty corpus scores nothing and does not call the library, because
    that library divides by the number of documents. A corpus whose
    tokens are all empty is rejected, because average document length
    would be 0 and this module will not invent a length.
    """

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        k1: float = LIBRARY_K1,
        b: float = LIBRARY_B,
        epsilon: float = LIBRARY_EPSILON,
        remove_stopwords: bool = False,
        stopwords: Collection[str] | None = None,
    ) -> None:
        if isinstance(documents, str) or not isinstance(documents, Mapping):
            raise TypeError(
                "documents must be a mapping of doc_id to text, "
                f"got {type(documents).__name__}"
            )
        self._k1 = _parameter("k1", k1, positive=True)
        self._b = _parameter("b", b, unit_interval=True)
        self._epsilon = _parameter("epsilon", epsilon, non_negative=True)
        self._remove_stopwords = remove_stopwords
        self._stopwords = stopwords

        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        seen: set[str] = set()
        for doc_id, text in documents.items():
            if not isinstance(doc_id, str):
                raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
            if doc_id == "":
                raise ValueError("doc_id must be non-empty")
            if doc_id in seen:
                raise ValueError(f"document already indexed: {doc_id}")
            if not isinstance(text, str):
                raise TypeError(f"text must be str, got {type(text).__name__}")
            seen.add(doc_id)
            self._doc_ids.append(doc_id)
            self._tokens.append(self._query_tokens(text))
        self._term_sets = [set(tokens) for tokens in self._tokens]

        self._model: object | None = None
        if not self._doc_ids:
            return
        if sum(len(tokens) for tokens in self._tokens) == 0:
            raise ValueError(
                "BM25Okapi is undefined when every document has zero tokens; "
                "average document length is not invented"
            )
        okapi = _load_bm25okapi()
        self._model = okapi(
            self._tokens,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )

    def document_ids(self) -> tuple[str, ...]:
        """Document ids in the order they were given to the library."""
        return tuple(self._doc_ids)

    def parameters(self) -> tuple[float, float, float]:
        """Return ``(k1, b, epsilon)`` passed to ``BM25Okapi``.

        IR concept: BM25's term-frequency saturation and length
        normalisation. These are the values this baseline was built
        with, not values chosen by a fit on Bharosa labels.
        """
        return (self._k1, self._b, self._epsilon)

    def uses_library_defaults(self) -> bool:
        """True when k1, b, and epsilon are the library constructor defaults."""
        return (
            self._k1 == LIBRARY_K1
            and self._b == LIBRARY_B
            and self._epsilon == LIBRARY_EPSILON
        )

    def scores(self, query: str) -> tuple[BM25Hit, ...]:
        """Library score of every document, in index order.

        IR concept: BM25 scoring. The number is whatever ``get_scores``
        returns, including 0 and any negative value the library emits
        after its idf floor. This method does not replace those numbers.
        An empty corpus returns an empty tuple.
        """
        tokens = self._query_tokens(query)
        if self._model is None:
            return ()
        raw = self._model.get_scores(tokens)  # type: ignore[attr-defined]
        hits: list[BM25Hit] = []
        for doc_id, score in zip(self._doc_ids, raw, strict=True):
            hits.append(BM25Hit(doc_id, float(score)))
        return tuple(hits)

    def rank(self, query: str, k: int) -> tuple[BM25Hit, ...]:
        """Top-K documents that contain a query term, best score first.

        IR concept: ranked BM25 retrieval. A document with no analysed
        query term is omitted. Documents that do contain a query term
        are kept even when the library score is 0 or negative, so a
        negative idf floor is not hidden. Ties break by ``doc_id``.
        ``k == 0`` returns an empty tuple.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or self._model is None:
            return ()
        query_terms = set(self._query_tokens(query))
        if not query_terms:
            return ()
        hits = [
            hit
            for hit, terms in zip(self.scores(query), self._term_sets, strict=True)
            if terms.intersection(query_terms)
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return tuple(hits[:k])

    def format_scores(self, query: str, *, k: int | None = None) -> str:
        """Return an inspectable dump that labels this as the external baseline.

        IR concept: a verbose baseline trace. The library name and the
        ``k1``, ``b``, and ``epsilon`` actually passed in are printed.
        The dump does not describe those parameters as a fit.
        """
        if self.uses_library_defaults():
            parameter_note = "published library defaults; not fit on Bharosa labels"
        else:
            parameter_note = "caller-supplied; not fit on Bharosa labels"
        lines = [
            "role  external baseline",
            f"library  {EXTERNAL_LIBRARY}.{EXTERNAL_CLASS}",
            (
                f"parameters  k1={_fmt(self._k1)}  b={_fmt(self._b)}  "
                f"epsilon={_fmt(self._epsilon)}"
            ),
            f"parameters  {parameter_note}",
            "formula  rank_bm25.BM25Okapi; not lnc.ltc and not net score",
        ]
        scored = self.scores(query)
        for hit in scored:
            lines.append(f"document {hit.doc_id}  score={_fmt(hit.score)}")
        requested = len(self._doc_ids) if k is None else k
        if k is not None and (isinstance(k, bool) or not isinstance(k, int)):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if requested < 0:
            raise ValueError("k must be non-negative")
        ranked = self.rank(query, requested)
        lines.append(f"ranked k={requested}")
        for place, hit in enumerate(ranked, start=1):
            lines.append(f"  {place}  {hit.doc_id}  {_fmt(hit.score)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BM25Baseline(documents={len(self._doc_ids)}, "
            f"library={EXTERNAL_LIBRARY}.{EXTERNAL_CLASS})"
        )

    def _query_tokens(self, text: str) -> list[str]:
        return tokenize(
            text,
            remove_stopwords=self._remove_stopwords,
            stopwords=self._stopwords,
        )


def _load_bm25okapi() -> type:
    """Import the external class. Do not replace it with a local formula."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "BM25Baseline needs the external library rank_bm25 "
            f"({EXTERNAL_CLASS}). It is a baseline only and is not "
            "Bharosa's ranker. Install rank_bm25 to run it."
        ) from exc
    return BM25Okapi


def _parameter(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
    unit_interval: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    if unit_interval and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _fmt(value: float) -> str:
    return f"{value:.6f}"
