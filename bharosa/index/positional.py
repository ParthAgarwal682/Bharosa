"""Positional index and exact phrase matching.

IR concept: a positional inverted index. The dictionary maps a term to
the documents that contain it, and each document maps to the token
positions of that term. A phrase matches only when those positions form
a run of consecutive integers, in query order.

Positions are 0-based indexes into the token list from
``bharosa.text.normalize.tokenize``, using this index's stop-word
settings. They are not character offsets. Exactness is defined on that
token sequence: there is no proximity window and no skipped token.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from bharosa.text.normalize import tokenize


@dataclass(frozen=True)
class PositionalPosting:
    """One document's positions for a single term.

    IR concept: a positional posting. ``positions`` lists every 0-based
    token index where the term occurs, in ascending order.
    """

    doc_id: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class PhraseMatch:
    """One document that contains an exact phrase.

    IR concept: a phrase hit. ``starts`` lists the token indexes where
    the first phrase term occurs and each following term occupies the
    next index. A document that does not contain the phrase is omitted,
    not returned with an empty start list.
    """

    doc_id: str
    starts: tuple[int, ...]


class PositionalIndex:
    """Term to document to positions, built by one scan.

    IR concept: positional index construction. Tokens keep the order
    ``tokenize`` produced. The same call, with the same stop-word
    settings, is used for a phrase, so a query meets a posting only
    after that shared analysis.
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
        self._known: set[str] = set()
        self._positions: dict[str, dict[str, list[int]]] = {}

    @classmethod
    def from_documents(
        cls,
        documents: Mapping[str, str],
        *,
        remove_stopwords: bool = False,
        stopwords: Collection[str] | None = None,
    ) -> PositionalIndex:
        """Index ``documents`` in mapping order and return the index."""
        index = cls(remove_stopwords=remove_stopwords, stopwords=stopwords)
        index.add_documents(documents)
        return index

    def add_document(self, doc_id: str, text: str) -> None:
        """Record a position for every token of ``text``.

        IR concept: positional postings. The ith token is stored at
        position i. A repeated ``doc_id`` is rejected so one document
        cannot contribute two position lists.
        """
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
        if doc_id == "":
            raise ValueError("doc_id must be non-empty")
        if doc_id in self._known:
            raise ValueError(f"document already indexed: {doc_id}")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        tokens = self._tokens(text)
        self._doc_ids.append(doc_id)
        self._known.add(doc_id)
        for position, term in enumerate(tokens):
            self._positions.setdefault(term, {}).setdefault(doc_id, []).append(position)

    def add_documents(self, documents: Mapping[str, str]) -> None:
        """Index each pair in ``documents`` in iteration order."""
        if isinstance(documents, str) or not isinstance(documents, Mapping):
            raise TypeError(
                "documents must be a mapping of doc_id to text, "
                f"got {type(documents).__name__}"
            )
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def document_ids(self) -> tuple[str, ...]:
        """Document ids in the order they were added."""
        return tuple(self._doc_ids)

    def vocabulary(self) -> tuple[str, ...]:
        """Dictionary terms, sorted so a dump is stable."""
        return tuple(sorted(self._positions))

    def dictionary(self) -> dict[str, dict[str, tuple[int, ...]]]:
        """Copy of term, then document, then positions.

        IR concept: the positional dictionary. Terms are sorted.
        Documents inside a term follow index order. The copy is safe
        to print in a demo.
        """
        posted: dict[str, dict[str, tuple[int, ...]]] = {}
        for term in self.vocabulary():
            docs = self._positions[term]
            posted[term] = {
                doc_id: tuple(docs[doc_id])
                for doc_id in self._doc_ids
                if doc_id in docs
            }
        return posted

    def positions(self, term: str) -> tuple[PositionalPosting, ...]:
        """Positional postings for ``term`` after the shared analyser.

        IR concept: a positional lookup. An unknown term, or a string
        that normalises away, has an empty list. A string that
        normalises to more than one token is rejected: positions belong
        to one term. Use ``find_phrase`` for several tokens.
        """
        key = self._single_term(term)
        if key is None:
            return ()
        docs = self._positions.get(key, {})
        return tuple(
            PositionalPosting(doc_id, tuple(docs[doc_id]))
            for doc_id in self._doc_ids
            if doc_id in docs
        )

    def find_phrase(self, phrase: str) -> tuple[PhraseMatch, ...]:
        """Documents where ``phrase`` occurs as consecutive tokens.

        IR concept: exact phrase matching on a positional index. The
        phrase is tokenised with this index's analyser. The match
        requires position p, p+1, p+2, ... in query order. Reordered
        terms do not match, and a gap of one token does not match.
        There is no slop parameter.

        A one-token phrase matches every position of that term. A phrase
        that normalises to no tokens matches nothing. Documents are
        returned in index order, and start positions are ascending.
        """
        tokens = self._tokens(phrase)
        if not tokens:
            return ()
        hits: list[PhraseMatch] = []
        for doc_id in self._doc_ids:
            lists = []
            missing = False
            for term in tokens:
                found = self._positions.get(term, {}).get(doc_id)
                if not found:
                    missing = True
                    break
                lists.append(tuple(found))
            if missing:
                continue
            starts = _exact_starts(lists)
            if starts:
                hits.append(PhraseMatch(doc_id, starts))
        return tuple(hits)

    def format_postings(self, terms: Collection[str] | None = None) -> str:
        """Return a readable dump of term, document, and positions.

        IR concept: an inspectable positional index. With ``terms``
        omitted, every dictionary term is printed in sorted order.
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
                lines.append("''")
                continue
            lines.append(key)
            docs = self._positions.get(key, {})
            for doc_id in self._doc_ids:
                if doc_id not in docs:
                    continue
                rendered = ",".join(str(position) for position in docs[doc_id])
                lines.append(f"  {doc_id}  {rendered}")
        return "\n".join(lines)

    def format_phrase(self, phrase: str) -> str:
        """Return the analysed phrase and each exact start position."""
        tokens = self._tokens(phrase)
        rendered = " ".join(tokens) if tokens else "<empty>"
        lines = [f"phrase {phrase!r}  tokens={rendered}"]
        for hit in self.find_phrase(phrase):
            starts = ",".join(str(position) for position in hit.starts)
            lines.append(f"  {hit.doc_id}  starts={starts}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PositionalIndex(documents={len(self._doc_ids)}, "
            f"vocabulary={len(self._positions)})"
        )

    def _tokens(self, text: str) -> list[str]:
        return tokenize(
            text,
            remove_stopwords=self._remove_stopwords,
            stopwords=self._stopwords,
        )

    def _single_term(self, term: str) -> str | None:
        if not isinstance(term, str):
            raise TypeError(f"term must be str, got {type(term).__name__}")
        tokens = self._tokens(term)
        if not tokens:
            return None
        if len(tokens) != 1:
            raise ValueError(f"term must normalise to one token, got {tokens!r}")
        return tokens[0]


def _exact_starts(lists: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Start positions where each next term sits at the next index.

    IR concept: exact phrase intersection. A start ``p`` matches only
    when the second list contains ``p + 1``, the third contains
    ``p + 2``, and so on. A gap or a reversal produces no start.
    """
    if not lists or any(not positions for positions in lists):
        return ()
    if len(lists) == 1:
        return lists[0]
    followers = [set(positions) for positions in lists[1:]]
    starts = [
        position
        for position in lists[0]
        if all(
            (position + offset) in followers[offset - 1]
            for offset in range(1, len(lists))
        )
    ]
    return tuple(starts)
