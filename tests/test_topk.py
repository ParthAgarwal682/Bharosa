"""Known-answer tests for heap top-K.

The selection tests feed scores in an order that is not the result
order. A full sort of the input would pass too; keeping an early
low-scoring item in a window of size K would not.
"""

from __future__ import annotations

import pytest

from bharosa.rank.topk import top_k


def test_heap_keeps_the_highest_scores_not_the_first_ones() -> None:
    # First two scores are the worst. A heap of size 2 has to drop them
    # when 0.9 and 0.8 arrive.
    scored = [("a", 0.1), ("e", 0.2), ("c", 0.3), ("d", 0.8), ("b", 0.9)]

    assert top_k(scored, 3) == (("b", 0.9), ("d", 0.8), ("c", 0.3))
    assert top_k(scored, 1) == (("b", 0.9),)
    assert top_k([("c", 0.1), ("b", 0.2), ("a", 0.3)], 2) == (("a", 0.3), ("b", 0.2))


def test_ties_break_by_doc_id_whatever_the_input_order() -> None:
    # Same score. Smaller doc_id ranks first, including when it arrives
    # last and has to replace the current worst winner.
    pairs = [("c", 1.0), ("b", 1.0), ("a", 1.0)]

    assert top_k(pairs, 3) == (("a", 1.0), ("b", 1.0), ("c", 1.0))
    assert top_k(pairs, 2) == (("a", 1.0), ("b", 1.0))
    assert top_k([("b", 1.0), ("a", 1.0)], 1) == (("a", 1.0),)
    assert top_k([("a", 1.0), ("b", 1.0)], 1) == (("a", 1.0),)
    assert top_k([("b", 1.0), ("c", 2.0), ("a", 1.0)], 2) == (("c", 2.0), ("a", 1.0))


def test_boundary_replacement_prefers_the_better_tie() -> None:
    assert top_k([("a", 0.5), ("b", 0.4)], 1) == (("a", 0.5),)
    assert top_k([("a", 0.4), ("b", 0.5)], 1) == (("b", 0.5),)
    assert top_k([("b", 1), ("a", 1)], 1) == (("a", 1.0),)


def test_k_above_the_number_of_scores_returns_every_pair_in_rank_order() -> None:
    assert top_k([("b", 0.2), ("a", 0.4)], 5) == (("a", 0.4), ("b", 0.2))
    assert top_k([], 3) == ()
    assert top_k((pair for pair in [("b", 0.1), ("a", 0.2)]), 1) == (("a", 0.2),)


def test_k_zero_is_empty_and_does_not_reorder_the_input() -> None:
    items = [("b", 0.2), ("a", 0.9)]
    snapshot = list(items)

    assert top_k(items, 0) == ()
    assert top_k(items, 1) == (("a", 0.9),)
    assert items == snapshot


def test_duplicate_doc_ids_are_kept_when_k_allows() -> None:
    assert top_k([("a", 1.0), ("a", 1.0)], 2) == (("a", 1.0), ("a", 1.0))


def test_rejects_bad_k_and_bad_pairs() -> None:
    with pytest.raises(TypeError):
        top_k([("a", 1.0)], 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        top_k([("a", 1.0)], True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        top_k([("a", 1.0)], -1)
    with pytest.raises(TypeError):
        top_k(["a"], 1)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        top_k([(1, 1.0)], 1)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        top_k([("", 1.0)], 1)
    with pytest.raises(TypeError):
        top_k([("a", "high")], 1)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        top_k([("a", True)], 1)
    with pytest.raises(ValueError):
        top_k([("a", float("nan"))], 1)
    with pytest.raises(ValueError):
        top_k([("a", float("inf"))], 1)
