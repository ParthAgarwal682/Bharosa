"""Controlled query expansion from a reviewed lexicon.

IR concept: dictionary query expansion. A reviewer supplies a mapping
from one surface token to one or more canonical tokens the index may
already contain. ``expand`` keeps every original query token, in order,
and appends those canonical tokens. Appending leaves the original
sequence intact, so a later phrase check can still see the user's words
as a prefix.

Expansion is one hop and exact. Lookup uses only tokens from the
analysed query. A canonical token is not looked up again in that call,
another source is not pulled in because it shares a target, and a token
missing from the mapping is unchanged. There is no edit distance, no
morphological guess, and no translation list in this module. Pass a
reviewed mapping, or load a reviewed CSV. An empty mapping adds nothing.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bharosa.text.normalize import tokenize

# Repo layout is ``<root>/bharosa/text/hinglish.py``. The reviewed file
# from the project README is ``<root>/config/hinglish_lexicon.csv``.
# This constant is only a path. Importing the module does not read or
# create that file.
DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "hinglish_lexicon.csv"
)

_SOURCE_HEADERS = frozenset({"source", "hinglish"})
_CANONICAL_HEADERS = frozenset({"canonical"})

LexiconMapping = Mapping[str, str | Sequence[str]]


@dataclass(frozen=True)
class ExpansionHit:
    """One query token that contributed canonical tokens.

    IR concept: an expansion trace. ``source`` is the original token.
    ``added`` is the subset of its reviewed targets that were not already
    in the query and had not already been appended. Empty additions are
    not recorded.
    """

    source: str
    added: tuple[str, ...]


@dataclass(frozen=True)
class ExpansionDebug:
    """Original query tokens and the lexicon tokens appended to them.

    IR concept: expansion transparency. ``original_tokens`` is the
    analysed query. ``expanded_tokens`` is only what the lexicon added.
    ``query_tokens`` is the original sequence followed by those additions,
    which is the token list a ranker should use when expansion is on.
    """

    original_tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    hits: tuple[ExpansionHit, ...] = ()

    def format_debug(self) -> str:
        """Return a stable text dump of the original and expanded tokens.

        IR concept: a verbose expansion trace. The two token lists are
        printed on their own lines so a demo can show that the user's
        terms were kept and which reviewed targets were added.
        """
        lines = [
            _token_line("original_tokens", self.original_tokens),
            _token_line("expanded_tokens", self.expanded_tokens),
            _token_line("query_tokens", self.query_tokens),
        ]
        for hit in self.hits:
            lines.append(f"{hit.source} -> {' '.join(hit.added)}")
        return "\n".join(lines)


class HinglishLexicon:
    """Reviewed surface-to-canonical mapping used at query time.

    IR concept: a controlled vocabulary for query expansion. Entries are
    whatever the caller reviewed. This object does not invent rows, and
    it does not close the mapping under further lookups.
    """

    def __init__(self, mapping: LexiconMapping) -> None:
        """Copy ``mapping`` into a normalised one-token dictionary.

        IR concept: lexicon normalisation. Keys and targets go through
        the same ``tokenize`` chain as the index, so an expansion meets
        a posting only when both sides share that analysis. Each source
        must be one token after that chain. Target order is the order
        of ``mapping``, then the order inside each value. Repeated
        targets for one source are stored once, at first occurrence.
        """
        self._entries = _compile(mapping)

    @classmethod
    def load(cls, path: str | Path | None = None) -> HinglishLexicon:
        """Load a reviewed two-column CSV. Do not create one here.

        IR concept: a lexicon file as a reviewed resource. ``path``
        defaults to ``config/hinglish_lexicon.csv`` under the repository
        root. A missing path raises ``FileNotFoundError``. Rows are
        ``source,canonical`` (header optional; ``hinglish,canonical`` is
        the other accepted header). ``#`` comments and blank lines are
        skipped. Repeated sources append targets in file order.

        The file is only read. No rows are filled in when a source is
        absent, and this method does not write a lexicon.
        """
        lexicon_path = _lexicon_path(path)
        if not lexicon_path.is_file():
            raise FileNotFoundError(f"reviewed lexicon file not found: {lexicon_path}")

        compiled: dict[str, list[str]] = {}
        for line_no, source, canonical in _iter_lexicon_rows(lexicon_path):
            try:
                key = _require_one_token(source, role="lexicon source")
                targets = _require_canonical(canonical)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{lexicon_path}:{line_no}: {exc}") from exc
            bucket = compiled.setdefault(key, [])
            for token in targets:
                if token not in bucket:
                    bucket.append(token)
        return cls({key: tuple(tokens) for key, tokens in compiled.items()})

    def entries(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the normalised mapping in insertion order.

        IR concept: the vocabulary actually used for expansion, after the
        shared analyser, not the raw spellings from the file.
        """
        return tuple(self._entries.items())

    def expand(
        self,
        query: str,
        *,
        remove_stopwords: bool = False,
        stopwords: Collection[str] | None = None,
    ) -> ExpansionDebug:
        """Append reviewed targets for the analysed query tokens.

        IR concept: one-hop dictionary expansion. The query is tokenised
        with the same flags the index would use. Each surviving original
        token is looked up once. Targets are appended in reviewed order.
        A target already present in the original query, or already
        appended, is not added again, so expansion does not raise the
        query count of a term the user already typed and does not emit
        the same added term twice.

        Call this once on the user string. ``query_tokens`` already
        contains the additions; tokenising that sequence again would
        treat the additions as user terms and take another hop.

        ``remove_stopwords`` defaults to False, matching ``tokenize``.
        Lookup runs after that analysis, so a token removed as a
        stop-word is not expanded.
        """
        original = tuple(
            tokenize(query, remove_stopwords=remove_stopwords, stopwords=stopwords)
        )
        seen = set(original)
        added: list[str] = []
        hits: list[ExpansionHit] = []
        for token in original:
            targets = self._entries.get(token)
            if not targets:
                continue
            newly: list[str] = []
            for canonical in targets:
                if canonical in seen:
                    continue
                seen.add(canonical)
                newly.append(canonical)
            if newly:
                added.extend(newly)
                hits.append(ExpansionHit(token, tuple(newly)))

        expanded = tuple(added)
        return ExpansionDebug(
            original_tokens=original,
            expanded_tokens=expanded,
            query_tokens=original + expanded,
            hits=tuple(hits),
        )

    def __repr__(self) -> str:
        return f"HinglishLexicon(entries={len(self._entries)})"


def _compile(mapping: LexiconMapping) -> dict[str, tuple[str, ...]]:
    """Normalise a reviewed mapping. Do not add pairs the mapping lacks.

    IR concept: compiling a controlled vocabulary. Collision of two
    surface strings that analyse to the same token keeps both target
    lists, in iteration order, without duplicating a target.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"lexicon mapping must be a mapping, got {type(mapping).__name__}")

    compiled: dict[str, list[str]] = {}
    for raw_key, raw_value in mapping.items():
        key = _require_one_token(raw_key, role="lexicon source")
        bucket = compiled.setdefault(key, [])
        for token in _require_canonical(raw_value):
            if token not in bucket:
                bucket.append(token)
    return {key: tuple(tokens) for key, tokens in compiled.items()}


def _require_one_token(text: str, *, role: str) -> str:
    """Return the single analysed token a lexicon source is allowed to be.

    IR concept: token-level expansion. A source that analyses to zero
    tokens or to several tokens would match a span this expander does
    not search, so it is rejected instead of being applied halfway.
    """
    if not isinstance(text, str):
        raise TypeError(f"{role} must be str, got {type(text).__name__}")
    tokens = tokenize(text)
    if len(tokens) != 1:
        raise ValueError(
            f"{role} must normalise to exactly one token, got {tokens!r} from {text!r}"
        )
    return tokens[0]


def _require_canonical(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return reviewed target tokens, in the caller's order.

    IR concept: explicit targets only. A string is analysed as text, not
    walked character by character. A sequence of strings is concatenated
    in sequence order. A set is rejected because its order is not part
    of the review. The result must contain at least one token.
    """
    if isinstance(value, str):
        pieces = (value,)
    elif isinstance(value, Sequence):
        pieces_list: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"canonical term must be str, got {type(item).__name__}"
                )
            pieces_list.append(item)
        pieces = tuple(pieces_list)
    else:
        raise TypeError(
            "canonical expansion must be str or a sequence of str, "
            f"got {type(value).__name__}"
        )

    tokens: list[str] = []
    for piece in pieces:
        tokens.extend(tokenize(piece))
    if not tokens:
        raise ValueError(
            f"canonical expansion must normalise to at least one token, got {value!r}"
        )
    return tuple(tokens)


def _lexicon_path(path: str | Path | None) -> Path:
    """Resolve the CSV location without reading it."""
    if path is None:
        return DEFAULT_LEXICON_PATH
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be str or Path, got {type(path).__name__}")
    return Path(path)


def _is_header(source: str, canonical: str) -> bool:
    """True when both cells are the column titles, not a lexicon row."""
    return source.casefold() in _SOURCE_HEADERS and canonical.casefold() in _CANONICAL_HEADERS


def _iter_lexicon_rows(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(line_no, source, canonical)`` from a reviewed CSV.

    IR concept: reading a controlled vocabulary. Comments and a single
    header are not entries. Every data row has two columns, so an extra
    cell cannot be silently dropped.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header_pending = True
        for line_no, row in enumerate(csv.reader(handle), start=1):
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            if cells[0].startswith("#"):
                continue
            if len(cells) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected 2 columns (source, canonical), "
                    f"got {len(cells)}"
                )
            source, canonical = cells
            if header_pending and _is_header(source, canonical):
                header_pending = False
                continue
            header_pending = False
            if source == "" or canonical == "":
                raise ValueError(
                    f"{path}:{line_no}: source and canonical must both be non-empty"
                )
            yield line_no, source, canonical


def _token_line(label: str, tokens: tuple[str, ...]) -> str:
    """Format one debug line. An empty list keeps the label and no space."""
    if not tokens:
        return f"{label}:"
    return f"{label}: " + " ".join(tokens)
