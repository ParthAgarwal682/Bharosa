"""lnc.ltc cosine scoring over an inverted index.

IR concept: the vector space model with the SMART scheme lnc.ltc. The
project README names that scheme for this module. The first triple is
the document, the second is the query. Each letter is one component:

* ``l`` — logarithmic term frequency, ``1 + log10(tf)`` when ``tf > 0``.
  An absent term contributes nothing; there is no ``1 + log10(0)``.
* ``n`` — no inverse document frequency. A document coordinate is not
  multiplied by idf.
* ``t`` — idf taken from the inverted index, ``log10(N / df)``. A term
  missing from the vocabulary, or an empty index, has idf 0.
* ``c`` — cosine normalization. Divide every raw coordinate by the L2
  norm of that vector.

The logarithm is base 10 so the ``l`` component uses the same base as
``InvertedIndex.idf``. Raw document weight is ``1 + log10(tf)``. Raw
query weight is ``(1 + log10(tf)) * idf``. Cosine similarity is the dot
product of the two normalized vectors. If either L2 norm is 0, the
similarity is 0.

Term frequency and idf are read from the inverted index. Queries go
through the same analyser options as that index, so a query term meets
a posting only after the shared chain.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from bharosa.index.inverted import InvertedIndex
from bharosa.rank.topk import top_k
from bharosa.text.normalize import tokenize

_WEIGHTING_LINE = (
    "weighting  document=lnc  query=ltc  "
    "log_tf=1+log10(tf)  idf=log10(N/df)"
)


def log_tf(tf: int) -> float:
    """SMART ``l`` component, ``1 + log10(tf)`` for a positive count.

    IR concept: logarithmic term frequency. A term that occurs once has
    weight 1, because ``log10(1)`` is 0. Further copies still add weight,
    but less than a raw count would. ``tf == 0`` is 0, not ``1 + log(0)``.
    """
    if isinstance(tf, bool) or not isinstance(tf, int):
        raise TypeError(f"tf must be int, got {type(tf).__name__}")
    if tf < 0:
        raise ValueError("tf must be non-negative")
    if tf == 0:
        return 0.0
    return 1.0 + math.log10(tf)


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Dot product of two already normalized weight maps.

    IR concept: cosine similarity after the ``c`` step. Normalization
    happens when a vector is built; this function does not divide again.
    A zero vector (no coordinates, or every weight 0) has similarity 0
    with every vector, including itself. Shared terms are the only
    products that can be non-zero.
    """
    if _is_zero(left) or _is_zero(right):
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


@dataclass(frozen=True)
class TermWeight:
    """One coordinate of an lnc document vector or an ltc query vector.

    IR concept: a tf-idf weight, split into the pieces the scheme names.
    ``log_tf`` is the ``l`` component. ``idf_factor`` is the second
    letter: ``1`` for a document (``n``) and ``log10(N/df)`` for a query
    (``t``). ``raw`` is their product, before cosine normalization.
    ``weight`` is ``raw / ||raw||``, or 0 when that norm is 0.
    """

    term: str
    tf: int
    log_tf: float
    idf_factor: float
    raw: float
    weight: float


@dataclass(frozen=True)
class Vector:
    """A cosine-normalized lnc document vector or ltc query vector.

    IR concept: one vector in the vector space model. ``norm`` is the L2
    norm of the raw weights. A zero vector has ``norm == 0``. Cosine
    with a zero vector is defined as 0, not an error.
    """

    kind: Literal["document", "query"]
    doc_id: str | None
    weights: tuple[TermWeight, ...]
    norm: float

    @property
    def is_zero(self) -> bool:
        """True when cosine normalization had nothing non-zero to divide by."""
        return self.norm == 0.0

    def coordinates(self) -> dict[str, float]:
        """Normalized weight of each stored term."""
        return {item.term: item.weight for item in self.weights}


@dataclass(frozen=True)
class ScoredDocument:
    """One document and its lnc.ltc cosine with a query.

    IR concept: a retrieval score. For this weighting the score is in
    ``[0, 1]``, because every coordinate is non-negative and both
    vectors are cosine-normalized. The score is not yet a rank.
    """

    doc_id: str
    score: float


class TfidfRanker:
    """lnc.ltc vectors and cosine scores for one inverted index.

    IR concept: ranking in the vector space model. The ranker does not
    copy postings. Each call reads term frequency and idf from the index,
    so a document added after construction is visible on the next call.
    """

    def __init__(self, index: InvertedIndex) -> None:
        if not isinstance(index, InvertedIndex):
            raise TypeError(
                f"index must be InvertedIndex, got {type(index).__name__}"
            )
        self._index = index

    def document_vector(self, doc_id: str) -> Vector:
        """lnc vector of one indexed document.

        IR concept: a document in the vector space. Coordinates are
        ``1 + log10(tf)``, with no idf, divided by the document's L2
        norm. An empty document is a zero vector.
        """
        self._index.document_stats(doc_id)
        counts: dict[str, int] = {}
        for term in self._index.vocabulary():
            tf = self._index.term_frequency(term, doc_id)
            if tf > 0:
                counts[term] = tf
        return self._vector("document", doc_id, counts, idf_factor=None)

    def query_vector(self, query: str) -> Vector:
        """ltc vector of a query string.

        IR concept: a query in the same space as the documents. Each
        analysed token is counted, weighted by ``(1 + log10(tf)) * idf``,
        and cosine-normalized. idf 0, an empty query, or a query whose
        every term is unknown leaves a zero vector.
        """
        counts = Counter(self._query_tokens(query))
        return self._vector("query", None, counts, idf_factor="index")

    def cosine(self, query: str, doc_id: str) -> float:
        """Cosine of the ltc query vector and the lnc document vector.

        IR concept: cosine similarity. After both ``c`` normalizations
        this is their dot product. A zero query or a zero document
        scores 0. A document that shares no query term also scores 0,
        and that document vector itself need not be zero.
        """
        return cosine(
            self.query_vector(query).coordinates(),
            self.document_vector(doc_id).coordinates(),
        )

    def scores(self, query: str) -> tuple[ScoredDocument, ...]:
        """Cosine of every indexed document, in index order.

        IR concept: scoring the collection. Order here is the order
        documents were added, not rank order. Zero scores are kept so a
        dump can show a miss. ``rank`` is what drops them and sorts.
        """
        query_weights = self.query_vector(query).coordinates()
        scored: list[ScoredDocument] = []
        for doc_id in self._index.document_ids():
            document_weights = self.document_vector(doc_id).coordinates()
            scored.append(ScoredDocument(doc_id, cosine(query_weights, document_weights)))
        return tuple(scored)

    def rank(self, query: str, k: int) -> tuple[ScoredDocument, ...]:
        """Top-K documents with a positive lnc.ltc cosine.

        IR concept: ranked retrieval. Positive scores go through the
        min-heap in ``top_k``. A zero vector, or a query that shares no
        terms with the collection, returns no documents.
        """
        pairs = [
            (item.doc_id, item.score)
            for item in self.scores(query)
            if item.score > 0.0
        ]
        return tuple(
            ScoredDocument(doc_id, score) for doc_id, score in top_k(pairs, k)
        )

    def format_scores(self, query: str, *, k: int | None = None) -> str:
        """Return an inspectable dump of weights and scores.

        IR concept: a verbose ranking trace. Each query and document
        coordinate shows tf, the log-tf component, the idf factor, the
        raw product, and the cosine-normalized weight. Document lines
        also show that coordinate's contribution to the dot product.
        The ranked section is the heap's top-K, best first.
        """
        query_vector = self.query_vector(query)
        query_weights = query_vector.coordinates()
        lines = [
            _WEIGHTING_LINE,
            (
                f"query {query!r}  norm={_fmt(query_vector.norm)}  "
                f"zero_vector={query_vector.is_zero}"
            ),
        ]
        for item in query_vector.weights:
            lines.append(_weight_line(item, query=True))

        for doc_id in self._index.document_ids():
            document = self.document_vector(doc_id)
            score = cosine(query_weights, document.coordinates())
            lines.append(
                f"document {doc_id}  norm={_fmt(document.norm)}  "
                f"zero_vector={document.is_zero}  cosine={_fmt(score)}"
            )
            for item in document.weights:
                contribution = item.weight * query_weights.get(item.term, 0.0)
                lines.append(_weight_line(item, query=False, contrib=contribution))

        requested = len(self._index.document_ids()) if k is None else k
        ranked = self.rank(query, requested)
        lines.append(f"ranked k={requested}")
        for place, item in enumerate(ranked, start=1):
            lines.append(f"  {place}  {item.doc_id}  {_fmt(item.score)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TfidfRanker(documents={len(self._index.document_ids())})"

    def _query_tokens(self, query: str) -> list[str]:
        # The index keeps its analyser flags private. The query has to
        # use those same flags or it will not meet the postings.
        return tokenize(
            query,
            remove_stopwords=self._index._remove_stopwords,
            stopwords=self._index._stopwords,
        )

    def _vector(
        self,
        kind: Literal["document", "query"],
        doc_id: str | None,
        counts: Mapping[str, int],
        *,
        idf_factor: Literal["index"] | None,
    ) -> Vector:
        pieces: list[tuple[str, int, float, float, float]] = []
        raws: list[float] = []
        for term in sorted(counts):
            tf = counts[term]
            if tf <= 0:
                continue
            log_component = log_tf(tf)
            factor = self._index.idf(term) if idf_factor == "index" else 1.0
            raw = log_component * factor
            pieces.append((term, tf, log_component, factor, raw))
            raws.append(raw)

        weights, norm = _l2_normalize(raws)
        term_weights = tuple(
            TermWeight(
                term=term,
                tf=tf,
                log_tf=log_component,
                idf_factor=factor,
                raw=raw,
                weight=weight,
            )
            for (term, tf, log_component, factor, raw), weight in zip(
                pieces, weights, strict=True
            )
        )
        return Vector(kind=kind, doc_id=doc_id, weights=term_weights, norm=norm)


def _l2_normalize(raws: list[float]) -> tuple[list[float], float]:
    """Divide by the L2 norm. A zero vector stays zeros and norm 0."""
    norm = math.sqrt(sum(raw * raw for raw in raws))
    if norm == 0.0:
        return [0.0] * len(raws), 0.0
    return [raw / norm for raw in raws], norm


def _is_zero(weights: Mapping[str, float]) -> bool:
    return all(weight == 0.0 for weight in weights.values())


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _weight_line(item: TermWeight, *, query: bool, contrib: float | None = None) -> str:
    idf_label = "idf" if query else "idf_factor"
    line = (
        f"  {item.term}  tf={item.tf}  log_tf={_fmt(item.log_tf)}  "
        f"{idf_label}={_fmt(item.idf_factor)}  raw={_fmt(item.raw)}  "
        f"weight={_fmt(item.weight)}"
    )
    if contrib is not None:
        line += f"  contrib={_fmt(contrib)}"
    return line
