"""Inverted index: dictionary, postings, and collection statistics.

IR concept: the inverted index. Each vocabulary term maps to a postings
list of ``(doc_id, tf)``. Document frequency is the length of that list.
Document length and collection totals sit beside the dictionary so a
later ranker can read them. This module does not score or rank documents.

Documents and lookup terms both go through
``bharosa.text.normalize.tokenize``. A query term hits a posting only
when it survives that same analysis chain.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from bharosa.text.normalize import tokenize


@dataclass(frozen=True)
class Posting:
    """One document in a term's postings list.

    IR concept: a posting. ``tf`` is the number of times the term occurs
    in ``doc_id`` after tokenisation, not a weight.
    """

    doc_id: str
    tf: int


@dataclass(frozen=True)
class DocumentStats:
    """Statistics stored for one indexed document.

    IR concept: document statistics. ``length`` is |d|, the token count
    after the shared analyser. ``unique_terms`` is the size of d's own
    term set.
    """

    doc_id: str
    length: int
    unique_terms: int


@dataclass(frozen=True)
class CollectionStats:
    """Statistics of the whole indexed collection.

    IR concept: collection statistics. These are inputs to later formulas
    (N, vocabulary size, total tokens, average |d|). They are not scores.
    """

    num_documents: int
    vocabulary_size: int
    total_tokens: int
    average_document_length: float


class InvertedIndex:
    """Dictionary plus postings, built by scanning documents once.

    IR concept: index construction. Tokens are counted per document, then
    one posting is appended for each distinct term. Postings stay in the
    order documents were added. The same ``tokenize`` call used here is
    the call used on a lookup term.
    """

    def __init__(
        self,
        *,
        remove_stopwords: bool = False,
        stopwords: Collection[str] | None = None,
    ) -> None:
        self._remove_stopwords = remove_stopwords
        self._stopwords = stopwords
        self._doc_ids: list[str] = []
        self._length: dict[str, int] = {}
        self._tf: dict[str, dict[str, int]] = {}
        self._postings: dict[str, list[Posting]] = {}

    @classmethod
    def from_documents(
        cls,
        documents: Mapping[str, str],
        *,
        remove_stopwords: bool = False,
        stopwords: Collection[str] | None = None,
    ) -> InvertedIndex:
        """Index ``documents`` in mapping order and return the index."""
        index = cls(remove_stopwords=remove_stopwords, stopwords=stopwords)
        index.add_documents(documents)
        return index

    def add_document(self, doc_id: str, text: str) -> None:
        """Tokenise ``text`` and append one posting per distinct term.

        IR concept: incremental index construction. Term frequency is the
        raw count of the token inside this document. A repeated ``doc_id``
        is rejected so document frequency cannot count one document twice.
        """
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
        if doc_id == "":
            raise ValueError("doc_id must be non-empty")
        if doc_id in self._length:
            raise ValueError(f"document already indexed: {doc_id}")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        tokens = tokenize(
            text,
            remove_stopwords=self._remove_stopwords,
            stopwords=self._stopwords,
        )
        counts = Counter(tokens)
        self._doc_ids.append(doc_id)
        self._length[doc_id] = len(tokens)
        self._tf[doc_id] = dict(counts)
        for term, tf in counts.items():
            self._postings.setdefault(term, []).append(Posting(doc_id, tf))

    def add_documents(self, documents: Mapping[str, str]) -> None:
        """Index each pair in ``documents`` in iteration order."""
        if isinstance(documents, str) or not isinstance(documents, Mapping):
            raise TypeError(
                f"documents must be a mapping of doc_id to text, got {type(documents).__name__}"
            )
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def document_ids(self) -> tuple[str, ...]:
        """Document IDs in the order they were added."""
        return tuple(self._doc_ids)

    def vocabulary(self) -> tuple[str, ...]:
        """Dictionary terms, sorted so a dump is stable."""
        return tuple(sorted(self._postings))

    def dictionary(self) -> dict[str, tuple[Posting, ...]]:
        """Copy of the vocabulary mapped to its postings.

        IR concept: the dictionary. Each key is a term; each value is that
        term's postings list. The copy is safe to print in a demo.
        """
        return {term: tuple(postings) for term, postings in sorted(self._postings.items())}

    def postings(self, term: str) -> tuple[Posting, ...]:
        """Postings for ``term`` after the shared analyser.

        IR concept: postings lookup. The returned list is every document
        that contains the term, in index order, with term frequency.
        An unknown term has an empty list. A string that normalises to
        more than one token is rejected: a postings list is per term.
        """
        key = self._single_term(term)
        if key is None:
            return ()
        return tuple(self._postings.get(key, ()))

    def document_frequency(self, term: str) -> int:
        """Number of documents that contain ``term``.

        IR concept: document frequency, df(t). It is the length of the
        postings list, so a term that repeats inside one document still
        contributes 1.
        """
        return len(self.postings(term))

    def term_frequency(self, term: str, doc_id: str) -> int:
        """Times ``term`` occurs in ``doc_id``.

        IR concept: term frequency, tf(t, d). A known document that does
        not contain the term has tf 0. An unknown document id raises
        ``KeyError``.
        """
        self._require_document(doc_id)
        key = self._single_term(term)
        if key is None:
            return 0
        return self._tf[doc_id].get(key, 0)

    def collection_frequency(self, term: str) -> int:
        """Sum of tf(t, d) over the collection.

        IR concept: collection frequency, cf(t). Unlike df(t), repeats
        inside a document count.
        """
        return sum(posting.tf for posting in self.postings(term))

    def document_stats(self, doc_id: str) -> DocumentStats:
        """Length and unique-term count for one document."""
        self._require_document(doc_id)
        return DocumentStats(
            doc_id=doc_id,
            length=self._length[doc_id],
            unique_terms=len(self._tf[doc_id]),
        )

    def collection_stats(self) -> CollectionStats:
        """N, vocabulary size, total tokens, and average document length."""
        num_documents = len(self._doc_ids)
        total_tokens = sum(self._length.values())
        average = 0.0 if num_documents == 0 else total_tokens / num_documents
        return CollectionStats(
            num_documents=num_documents,
            vocabulary_size=len(self._postings),
            total_tokens=total_tokens,
            average_document_length=average,
        )

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term``.

        IR concept: idf(t) = log10(N / df(t)). This is a collection
        statistic stored with the dictionary, not a document score.
        A term missing from the vocabulary, or an empty index, has idf 0.
        """
        df = self.document_frequency(term)
        num_documents = len(self._doc_ids)
        if df == 0 or num_documents == 0:
            return 0.0
        return math.log10(num_documents / df)

    def format_postings(self, terms: Collection[str] | None = None) -> str:
        """Return a readable dump of postings for the demo.

        IR concept: an inspectable inverted index. With ``terms`` omitted,
        every dictionary term is printed in sorted order. Otherwise each
        requested term is printed in the given order, after analysis.
        """
        if terms is None:
            keys = list(self.vocabulary())
        else:
            if isinstance(terms, str):
                raise TypeError(
                    "terms must be a collection of term strings, not a single str"
                )
            keys = []
            for term in terms:
                key = self._single_term(term)
                keys.append("" if key is None else key)

        lines: list[str] = []
        for key in keys:
            if key == "":
                lines.append("''  df=0  cf=0")
                continue
            df = len(self._postings.get(key, ()))
            cf = sum(posting.tf for posting in self._postings.get(key, ()))
            lines.append(f"{key}  df={df}  cf={cf}")
            for posting in self._postings.get(key, ()):
                lines.append(f"  {posting.doc_id}  tf={posting.tf}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(documents={len(self._doc_ids)}, "
            f"vocabulary={len(self._postings)})"
        )

    def _single_term(self, term: str) -> str | None:
        if not isinstance(term, str):
            raise TypeError(f"term must be str, got {type(term).__name__}")
        tokens = tokenize(
            term,
            remove_stopwords=self._remove_stopwords,
            stopwords=self._stopwords,
        )
        if not tokens:
            return None
        if len(tokens) != 1:
            raise ValueError(f"term must normalise to one token, got {tokens!r}")
        return tokens[0]

    def _require_document(self, doc_id: str) -> None:
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
        if doc_id not in self._length:
            raise KeyError(doc_id)
