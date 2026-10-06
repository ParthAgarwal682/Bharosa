"""Character 2-gram and 3-gram cosine similarity for brand strings.

IR concept: character n-gram matching. A noisy brand still shares most
of its contiguous character slices with the recorded spelling, so the
two strings can meet when an exact token lookup would miss.

The weight of a slice is ``1 + log10(count)``. Cosine is the dot
product of the two weight maps divided by the product of their L2
norms. Collection idf is not used. A short brand list would make idf
swing with every added row, and a slice that happens to occur in every
row would be wiped out by ``log10(N / df) = 0``. The score of a pair
depends only on that pair.

Strings go through the shared normaliser, then spaces are removed, so
``test brand`` and ``testbrand`` are the same character sequence. Slices
shorter than the string are not padded. A string of fewer than 2
characters has no slices, and its similarity is 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from bharosa.text.normalize import normalize

# Both orders are always counted. A 2-gram and a 3-gram cannot collide:
# they are different strings.
NGRAM_ORDERS: tuple[int, ...] = (2, 3)

SCHEME: str = "char 2-gram+3-gram log-tf cosine"
FORMULA: str = "weight = 1 + log10(tf); cosine = dot / (|query| |brand|)"


@dataclass(frozen=True)
class NgramSimilarity:
    """Cosine of two character n-gram vectors, with the pieces visible.

    IR concept: a transparent similarity. ``cosine`` is the score used
    for ranking. ``dot``, ``query_norm``, and ``brand_norm`` are the
    three numbers that produce it. ``shared_ngrams`` lists the slices
    that contributed to the dot product, in lexicographic order.
    """

    cosine: float
    dot: float
    query_norm: float
    brand_norm: float
    shared_ngrams: tuple[str, ...]
    scheme: str = SCHEME

    def format_score(self) -> str:
        """Return a stable dump of the scheme, the formula, and the parts.

        IR concept: an inspectable similarity. A demo can show why two
        brand strings scored as they did without opening the weight maps.
        """
        shared = " ".join(self.shared_ngrams)
        return "\n".join(
            [
                f"scheme: {self.scheme}",
                f"formula: {FORMULA}",
                f"cosine: {self.cosine:.6f}",
                f"dot: {self.dot:.6f}",
                f"query_norm: {self.query_norm:.6f}",
                f"brand_norm: {self.brand_norm:.6f}",
                f"shared_ngrams: {shared}",
            ]
        )


def gram_text(text: str) -> str:
    """Return the character sequence n-grams are cut from.

    IR concept: the same analysis on the query and the recorded brand.
    The shared normaliser case-folds and turns punctuation into spaces.
    Spaces are then removed so a split typing of one brand still lines
    up with the recorded characters. No character is invented to pad a
    short string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return normalize(text).replace(" ", "")


def character_ngrams(text: str, orders: tuple[int, ...] = NGRAM_ORDERS) -> Counter[str]:
    """Count contiguous character slices of each length in ``orders``.

    IR concept: character n-gram extraction. Counts are raw occurrences.
    A length longer than the character sequence contributes nothing.
    """
    _check_orders(orders)
    sequence = gram_text(text)
    counts: Counter[str] = Counter()
    for order in orders:
        if len(sequence) < order:
            continue
        for start in range(len(sequence) - order + 1):
            counts[sequence[start : start + order]] += 1
    return counts


def ngram_similarity(query_brand: str, record_brand: str) -> NgramSimilarity:
    """Cosine similarity of the character 2-gram and 3-gram vectors.

    IR concept: q-gram similarity between a query brand and one recorded
    brand. The two vectors live in the same slice space. Slices that
    occur on only one side raise that side's norm and do not add to the
    dot product, so a misspelling scores below an exact string and an
    unrelated string with no shared slice scores 0.
    """
    return _similarity_from_counts(
        character_ngrams(query_brand),
        character_ngrams(record_brand),
    )


def _similarity_from_counts(
    query_counts: Counter[str],
    brand_counts: Counter[str],
) -> NgramSimilarity:
    """Cosine of two already extracted count maps."""
    query_weights = _log_tf_weights(query_counts)
    brand_weights = _log_tf_weights(brand_counts)
    dot = _dot(query_weights, brand_weights)
    query_norm = _norm(query_weights)
    brand_norm = _norm(brand_weights)
    if query_norm == 0.0 or brand_norm == 0.0 or dot == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (query_norm * brand_norm)
    shared = tuple(sorted(set(query_weights) & set(brand_weights)))
    return NgramSimilarity(
        cosine=cosine,
        dot=dot,
        query_norm=query_norm,
        brand_norm=brand_norm,
        shared_ngrams=shared,
    )


def _log_tf_weights(counts: Mapping[str, int]) -> dict[str, float]:
    """SMART-style log tf, ``1 + log10(tf)``, for counts that are positive.

    A slice that occurs once has weight 1, because ``log10(1)`` is 0.
    Further copies add less than a raw count would. There is no idf term.
    """
    weights: dict[str, float] = {}
    for gram, tf in counts.items():
        if isinstance(tf, bool) or not isinstance(tf, int):
            raise TypeError(f"n-gram count must be int, got {type(tf).__name__}")
        if tf <= 0:
            continue
        weights[gram] = 1.0 + math.log10(tf)
    return weights


def _dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Dot product over shared slices. A missing slice contributes 0."""
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(gram, 0.0) for gram, weight in left.items())


def _norm(weights: Mapping[str, float]) -> float:
    """L2 norm. An empty weight map has norm 0."""
    if not weights:
        return 0.0
    return math.sqrt(sum(weight * weight for weight in weights.values()))


def _check_orders(orders: tuple[int, ...]) -> None:
    if not isinstance(orders, tuple) or not orders:
        raise TypeError("orders must be a non-empty tuple of ints")
    for order in orders:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError(f"n-gram order must be int, got {type(order).__name__}")
        if order < 1:
            raise ValueError("n-gram order must be positive")
