"""Shared text analysis for Bharosa indexing and querying.

IR concept: normalisation and tokenisation. A query term hits a posting
only when the index and the query run the same analysis chain. Call
``normalize`` and ``tokenize`` on documents and on queries; do not keep a
second, slightly different cleaner on one side.

The chain is Unicode NFC, Unicode case folding, transparent punctuation,
then whitespace collapse. Indic letters and combining marks stay in the
token. Stop-word removal is opt-in so the default chain does not drop
terms before the caller decides they are uninformative.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection

# Closed-class English words only. Hinglish and Indic tokens are absent
# on purpose: in this collection they are often content words
# ("ke", "kya", "mein"), and dropping them would change retrieval.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

# Zero-width space is a word break with no visible glyph. Other format
# characters (ZWJ, ZWNJ, soft hyphen, BOM) are removed in ``_map_char``
# so they cannot glue two tokens together or split one Indic word.
_ZERO_WIDTH_SPACE = "\u200b"


def normalize(text: str) -> str:
    """Return one deterministic normalised string, with every token kept.

    IR concept: normalisation. Equivalent surface forms must share one
    index key. Steps, in order:

    1. Unicode NFC, so decomposed and composed Indic sequences match
       (for example na + nukta and the precomposed nukta letter).
    2. Unicode case folding (``str.casefold``), which is stronger than
       ``str.lower`` and is a no-op for Devanagari.
    3. NFC again, because case folding can change a string's composition.
    4. Transparent punctuation: Unicode punctuation and symbols become
       token boundaries, not deleted neighbours. ``heart-operation`` and
       ``योजना।`` keep the letters on each side.
    5. Whitespace collapse, including repeated spaces, tabs, newlines,
       and non-breaking spaces, to a single ASCII space. Ends are stripped.

    Stop-words are not removed. Use ``tokenize(..., remove_stopwords=True)``
    when a caller explicitly wants that.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    folded = unicodedata.normalize("NFC", text).casefold()
    folded = unicodedata.normalize("NFC", folded)
    mapped = "".join(_map_char(ch) for ch in folded)
    return " ".join(mapped.split())


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = False,
    stopwords: Collection[str] | None = None,
) -> list[str]:
    """Tokenise text for the inverted index or for a query.

    IR concept: tokenisation. The returned list is the term sequence
    stored in postings and the term sequence looked up at query time.
    Both sides must call this function.

    ``remove_stopwords`` defaults to False. Passing a ``stopwords``
    collection does nothing until the flag is True, so information is
    not dropped by accident. When the flag is True and ``stopwords`` is
    None, ``DEFAULT_STOPWORDS`` is used. A custom collection replaces
    that set; each entry is normalised with the same chain as ``text``.
    Token order is the left-to-right order of the input.
    """
    tokens = _split_tokens(normalize(text))
    if not remove_stopwords:
        return tokens

    blocked = DEFAULT_STOPWORDS if stopwords is None else _normalise_stopwords(stopwords)
    return [token for token in tokens if token not in blocked]


def _split_tokens(normalised: str) -> list[str]:
    if not normalised:
        return []
    return normalised.split(" ")


def _normalise_stopwords(stopwords: Collection[str]) -> frozenset[str]:
    terms: list[str] = []
    for word in stopwords:
        if not isinstance(word, str):
            raise TypeError(f"stopword must be str, got {type(word).__name__}")
        normalised = normalize(word)
        if normalised:
            terms.extend(normalised.split(" "))
    return frozenset(terms)


def _map_char(ch: str) -> str:
    """Map one character to itself, a space, or nothing.

    Letters, combining marks, and numbers are kept. Marks have to stay
    so Devanagari vowel signs, nukta, and virama remain inside the token
    instead of being stripped as "non-ASCII". Punctuation and symbols
    become a space (a boundary). Format and control characters disappear,
    except the zero-width space, which is a boundary.
    """
    if ch == _ZERO_WIDTH_SPACE or ch.isspace() or unicodedata.category(ch).startswith("Z"):
        return " "

    category = unicodedata.category(ch)
    if category[0] in {"L", "M", "N"}:
        return ch
    if category[0] in {"P", "S"}:
        return " "
    return ""
