"""Heap-based top-K selection.

IR concept: retrieving the K highest-scoring documents without sorting
the whole collection. A min-heap holds at most K winners. Its root is
the worst winner, so a better candidate replaces that root.

Ties use exact numeric equality. The better result is the higher score.
If the scores are equal, the lexicographically smaller ``doc_id`` wins.
If the same id is offered twice with the same score, the earlier input
wins. The same pairs always come back in the same order.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable


class _WorstFirst:
    """One heap entry. A worse candidate compares as smaller.

    The min-heap root is then the item that should be dropped when a
    better document arrives.
    """

    __slots__ = ("score", "doc_id", "order")

    def __init__(self, score: float, doc_id: str, order: int) -> None:
        self.score = score
        self.doc_id = doc_id
        self.order = order

    def __lt__(self, other: _WorstFirst) -> bool:
        if self.score != other.score:
            return self.score < other.score
        if self.doc_id != other.doc_id:
            return self.doc_id > other.doc_id
        return self.order > other.order


def top_k(
    scored: Iterable[tuple[str, float]],
    k: int,
) -> tuple[tuple[str, float], ...]:
    """Return the K best ``(doc_id, score)`` pairs, best first.

    IR concept: top-K retrieval. Scores are not sorted as a full list.
    The heap never grows past K. ``k == 0`` returns an empty tuple.
    Documents the caller did not pass in are not invented, so a list of
    positive scores stays a list of positive scores.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return ()

    heap: list[_WorstFirst] = []
    for order, item in enumerate(scored):
        doc_id, score = _pair(item)
        entry = _WorstFirst(score, doc_id, order)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif heap[0] < entry:
            heapq.heapreplace(heap, entry)

    winners = sorted(heap, key=lambda entry: (-entry.score, entry.doc_id, entry.order))
    return tuple((entry.doc_id, entry.score) for entry in winners)


def _pair(item: object) -> tuple[str, float]:
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(
            "scored items must be (doc_id, score) pairs, "
            f"got {type(item).__name__}"
        )
    doc_id, score = item
    if not isinstance(doc_id, str):
        raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
    if doc_id == "":
        raise ValueError("doc_id must be non-empty")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError(f"score must be a real number, got {type(score).__name__}")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return doc_id, value
