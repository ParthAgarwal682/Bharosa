"""Crawler corpus adapter: persisted pages become zone documents.

IR concept: zone construction. The crawler stores raw HTML and page
metadata. Scheme retrieval indexes zones, not crawler rows. This module
is that boundary. It reads CrawlDB, loads the current ``text_path``, and
emits ``ZoneDoc`` values. It does not rank, and it does not add crawler
columns to the zone contract.

``fetched_at`` is copied to ``crawled_at``. ``last_changed_at`` is copied
only from the page's own change time. A blank change time stays missing.
A quality score is copied when the row has one. Headings choose a zone.
``state`` and ``conditions`` are filled only from an explicit label.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from bharosa.crawl.db import CrawlDB, PageRecord, VersionRecord
from bharosa.crawl.dedup import compute_content_hash
from bharosa.index.params import ZONES
from bharosa.index.zones import ZoneDoc

__all__ = [
    "ZoneCorpusError",
    "load_zone_documents",
    "zone_documents_from_page",
]

_ZONE_SET = frozenset(ZONES)

# A heading matches a zone when the whole heading is one of these phrases,
# or that phrase plus a generic suffix such as "criteria". A heading that
# matches two zones is left as ``other``.
_ZONE_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "eligibility",
        (
            "eligibility",
            "who is eligible",
            "who are eligible",
            "who can apply",
            "पात्रता",
        ),
    ),
    (
        "benefits",
        (
            "benefits",
            "benefit",
            "coverage",
            "what is covered",
            "लाभ",
        ),
    ),
    (
        "documents",
        (
            "documents required",
            "required documents",
            "document checklist",
            "documents",
            "दस्तावेज",
            "दस्तावेज़",
        ),
    ),
    (
        "how_to_apply",
        (
            "how to apply",
            "application process",
            "application procedure",
            "आवेदन प्रक्रिया",
            "आवेदन कैसे करें",
            "आवेदन",
        ),
    ),
)

_HEADING_SUFFIXES = frozenset(
    {
        "criteria",
        "criterion",
        "details",
        "detail",
        "section",
        "information",
        "required",
        "process",
        "procedure",
        "checklist",
        "list",
        "online",
        "offline",
    }
)

_STATE_NAMES: tuple[str, ...] = (
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
)

# Whole-value aliases only. A short form is accepted when it is the entire
# labeled value, not when it appears inside a sentence.
_STATE_ALIASES: tuple[tuple[str, str], ...] = (
    ("up", "Uttar Pradesh"),
    ("u.p.", "Uttar Pradesh"),
    ("mp", "Madhya Pradesh"),
    ("m.p.", "Madhya Pradesh"),
    ("orissa", "Odisha"),
    ("pondicherry", "Puducherry"),
    ("nct of delhi", "Delhi"),
)

_STATE_LABELS = frozenset({"state", "राज्य"})
_CONDITION_LABELS = frozenset(
    {"condition", "conditions", "disease", "ailment"}
)

_LABEL_LINE = re.compile(
    r"^(state|राज्य|condition|conditions|disease|ailment)"
    r"\s*[:\-\u2013\u2014\uff1a]\s*(.+)$",
    re.IGNORECASE,
)

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "li",
        "section",
        "article",
        "blockquote",
        "br",
        "ul",
        "ol",
        "table",
    }
)


class ZoneCorpusError(ValueError):
    """Persisted crawler data cannot be turned into zone documents."""


@dataclass(frozen=True)
class _ExtractedPage:
    """Visible regions of one HTML page, before crawler fields are joined.

    IR concept: a zone split. ``sections`` is ``(zone, text)`` in document
    order. ``state`` and ``conditions`` are page-level only when a label
    said so. They are not inferred from the prose.
    """

    title: str
    state: str | None
    conditions: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]


class _HTMLEvents(HTMLParser):
    """Turn stored HTML into headings, text blocks, and label pairs.

    IR concept: document cleaning before zoning. Script and style text
    are dropped. The parser does not guess a scheme field from a class
    name or from running text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.events: list[tuple[object, ...]] = []
        self._skip = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._heading: str | None = None
        self._heading_parts: list[str] = []
        self._text: list[str] = []
        self._in_dt = False
        self._dt: list[str] = []
        self._in_dd = False
        self._dd: list[str] = []
        self._pending_dt: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._flush_text()
            self._in_title = True
            self._title_parts = []
            return
        if tag in _HEADING_TAGS:
            self._flush_text()
            self._heading = tag
            self._heading_parts = []
            return
        if tag == "tr":
            self._flush_text()
            self._row = []
            return
        if tag in {"td", "th"}:
            self._flush_text()
            self._cell = []
            return
        if tag == "dt":
            self._flush_text()
            self._in_dt = True
            self._dt = []
            return
        if tag == "dd":
            self._flush_text()
            self._in_dd = True
            self._dd = []
            return
        if tag in _BLOCK_TAGS:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip:
            if tag in _SKIP_TAGS:
                self._skip -= 1
            return
        if tag == "title" and self._in_title:
            self.title = _collapse(self._title_parts)
            self._in_title = False
            self._title_parts = []
            return
        if tag in _HEADING_TAGS and self._heading == tag:
            text = _collapse(self._heading_parts)
            self._heading = None
            self._heading_parts = []
            if text:
                self.events.append(("heading", text))
            return
        if tag in {"td", "th"} and self._cell is not None:
            text = _collapse(self._cell)
            self._cell = None
            if self._row is not None:
                self._row.append(text)
            elif text:
                self.events.append(("text", text))
            return
        if tag == "tr" and self._row is not None:
            cells = tuple(cell for cell in self._row if cell)
            self._row = None
            if cells:
                self.events.append(("row", cells))
            return
        if tag == "dt" and self._in_dt:
            self._pending_dt = _collapse(self._dt)
            self._in_dt = False
            self._dt = []
            return
        if tag == "dd" and self._in_dd:
            value = _collapse(self._dd)
            label = self._pending_dt
            self._pending_dt = None
            self._in_dd = False
            self._dd = []
            if label and value:
                self.events.append(("pair", label, value))
            elif value:
                self.events.append(("text", value))
            elif label:
                self.events.append(("text", label))
            return
        if tag in _BLOCK_TAGS:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._skip or data == "":
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._heading is not None:
            self._heading_parts.append(data)
            return
        if self._cell is not None:
            self._cell.append(data)
            return
        if self._in_dt:
            self._dt.append(data)
            return
        if self._in_dd:
            self._dd.append(data)
            return
        self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush_text()
        if self._pending_dt:
            self.events.append(("text", self._pending_dt))
            self._pending_dt = None

    def _flush_text(self) -> None:
        text = _collapse(self._text)
        self._text = []
        if text:
            self.events.append(("text", text))


class _SectionBuilder:
    """Group cleaned blocks into zones and explicit metadata labels."""

    def __init__(self) -> None:
        self.sections: list[tuple[str, str]] = []
        self.state_values: list[str] = []
        self.condition_values: list[str] = []
        self._zone: str | None = None
        self._parts: list[str] = []

    def start(self, zone: str) -> None:
        """Begin a zone. The heading itself is not copied into the text."""
        self._close()
        self._zone = zone
        self._parts = []

    def add(self, text: str) -> None:
        """Append visible text to the current zone, or to ``other``."""
        if text == "":
            return
        if self._zone is None:
            self._zone = "other"
        self._parts.append(text)

    def record_label(self, name: str, value: str) -> None:
        """Keep a field only when the label name is one we recognise."""
        kind = _label_kind(name)
        if kind == "state":
            self.state_values.append(value)
        elif kind == "conditions":
            self.condition_values.extend(_condition_items(value))

    def finish(self) -> None:
        """Close the zone that is still open."""
        self._close()

    def _close(self) -> None:
        if self._zone is None:
            return
        text = "\n".join(part for part in self._parts if part)
        if text:
            self.sections.append((self._zone, text))
        self._zone = None
        self._parts = []


def load_zone_documents(
    db: CrawlDB | str | os.PathLike[str],
    *,
    urls: Sequence[str] | None = None,
) -> list[ZoneDoc]:
    """Load zone documents from a persisted crawler database.

    IR concept: corpus construction. Each current page becomes one zone
    document per extracted region. The result is ordered by URL, then by
    the order of regions on that page. The same database yields the same
    list.

    ``db`` is a :class:`CrawlDB` or a path to its SQLite file. A missing
    file raises ``FileNotFoundError``. It is not created, and it is not
    reported as an empty corpus. An existing database with no pages is an
    empty corpus and returns an empty list.

    ``urls`` limits the load to those pages. A URL that is not stored
    raises :class:`ZoneCorpusError`. CrawlDB looks up one URL at a time,
    so the full URL list is read from the same connection and each page
    is then loaded with ``get_page`` and ``get_versions``.
    """
    database, owned = _open_database(db)
    try:
        return _load_database(database, urls)
    finally:
        if owned:
            database.close()


def zone_documents_from_page(page: PageRecord, html: str) -> list[ZoneDoc]:
    """Map one crawler page and its stored HTML to zone documents.

    IR concept: field mapping at the corpus boundary. ``page.fetched_at``
    becomes ``crawled_at``. ``page.last_changed_at`` becomes
    ``last_changed_at`` and is never replaced with ``fetched_at``.
    ``content_hash`` and ``g_score`` are copied. ``text_path`` is not a
    zone field; the caller passes the HTML that file contained.
    """
    if not isinstance(page, PageRecord):
        raise TypeError(f"page must be a PageRecord, got {type(page).__name__}")
    if not isinstance(page.url, str) or page.url == "":
        raise ZoneCorpusError("crawler page is missing a url")
    if not isinstance(page.domain, str):
        raise ZoneCorpusError(f"page {page.url} is missing a domain")
    if not isinstance(page.content_hash, str) or page.content_hash == "":
        raise ZoneCorpusError(f"page {page.url} is missing a content_hash")
    if not isinstance(html, str):
        raise TypeError(f"html must be str, got {type(html).__name__}")
    if html.strip() == "":
        raise ZoneCorpusError(f"stored HTML for {page.url} is empty")

    extracted = _extract_page(html)
    if not extracted.sections:
        raise ZoneCorpusError(f"stored HTML for {page.url} has no extractable text")

    crawled_at = _copied_timestamp("fetched_at", page.fetched_at, page.url)
    last_changed_at = _copied_timestamp(
        "last_changed_at", page.last_changed_at, page.url
    )
    documents: list[ZoneDoc] = []
    for ordinal, (zone, text) in enumerate(extracted.sections):
        documents.append(
            ZoneDoc(
                doc_id=_doc_id(page.url, page.content_hash, ordinal, zone),
                url=page.url,
                domain=page.domain,
                title=extracted.title,
                zone=zone,
                text=text,
                state=extracted.state,
                conditions=extracted.conditions,
                g_score=_copied_score(page),
                crawled_at=crawled_at,
                last_changed_at=last_changed_at,
                content_hash=page.content_hash,
            )
        )
    return documents


def _load_database(db: CrawlDB, urls: Sequence[str] | None) -> list[ZoneDoc]:
    """Read each selected page through CrawlDB and zone its current file."""
    documents: list[ZoneDoc] = []
    for url in _select_urls(db, urls):
        page = db.get_page(url)
        if page is None:
            raise ZoneCorpusError(f"no crawler page stored for {url}")
        version = _current_version(page, db.get_versions(url))
        text_path = version.text_path
        if text_path is None or text_path == "":
            raise ZoneCorpusError(
                f"latest version for {page.url} has no text_path; "
                "crawler content is unavailable"
            )
        html = _read_stored_html(text_path, page.content_hash, page.url)
        documents.extend(zone_documents_from_page(page, html))
    return documents


def _open_database(
    db: CrawlDB | str | os.PathLike[str],
) -> tuple[CrawlDB, bool]:
    """Return ``(database, should_close)``.

    CrawlDB creates a missing file. A path is checked first so a missing
    database raises instead of becoming an empty corpus.
    """
    if isinstance(db, CrawlDB):
        return db, False
    if isinstance(db, Path):
        path_str = str(db)
    elif isinstance(db, str):
        path_str = db
    elif isinstance(db, os.PathLike):
        path_str = os.fspath(db)
    else:
        raise TypeError(
            "db must be a CrawlDB or a path to a persisted crawler database, "
            f"got {type(db).__name__}"
        )
    if path_str == ":memory:":
        raise ZoneCorpusError(
            "an in-memory database path is not a persisted crawl; "
            "pass the CrawlDB instance or a database file"
        )
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"crawler database not found: {path_str}. "
            "An empty zone list is not returned when the database is missing."
        )
    return CrawlDB(db_path=path_str), True


def _select_urls(db: CrawlDB, urls: Sequence[str] | None) -> list[str]:
    """URLs to load, in ascending URL order, without duplicates."""
    if urls is None:
        return _stored_urls(db)
    if isinstance(urls, str) or not isinstance(urls, Sequence):
        raise TypeError(
            f"urls must be a sequence of strings, got {type(urls).__name__}"
        )
    chosen: list[str] = []
    for url in urls:
        if not isinstance(url, str) or url == "":
            raise TypeError("urls must contain non-empty strings")
        chosen.append(url)
    return sorted(set(chosen))


def _stored_urls(db: CrawlDB) -> list[str]:
    """Every page URL, ordered by URL.

    IR concept: a corpus scan. CrawlDB's public API fetches one URL at
    a time and has no page list. The list is read from the connection
    CrawlDB already opened, including an in-memory database. Each page
    is still loaded with ``get_page`` and ``get_versions``.
    """
    connection = getattr(db, "_conn", None)
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError(
            "db must be a CrawlDB from bharosa.crawl.db so stored pages can be read"
        )
    try:
        rows = connection.execute(
            "SELECT url FROM pages ORDER BY url ASC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise ZoneCorpusError(f"could not read crawler pages: {exc}") from exc
    urls: list[str] = []
    for row in rows:
        url = row["url"] if isinstance(row, sqlite3.Row) else row[0]
        if not isinstance(url, str) or url == "":
            raise ZoneCorpusError("crawler pages table has a row without a url")
        urls.append(url)
    return urls


def _current_version(
    page: PageRecord, versions: list[VersionRecord]
) -> VersionRecord:
    """Return the latest snapshot when it agrees with the page row.

    An older snapshot is not substituted when the latest row disagrees
    with the page's ``content_hash`` or ``fetched_at``.
    """
    if not versions:
        raise ZoneCorpusError(
            f"no version stored for {page.url}; text_path is unavailable"
        )
    current = versions[-1]
    if current.content_hash != page.content_hash:
        raise ZoneCorpusError(
            f"latest version for {page.url} has content_hash "
            f"{current.content_hash!r}, but the page record has "
            f"{page.content_hash!r}; refusing to load a different snapshot"
        )
    if current.fetched_at != page.fetched_at:
        raise ZoneCorpusError(
            f"latest version for {page.url} has fetched_at "
            f"{current.fetched_at!r}, but the page record has "
            f"{page.fetched_at!r}"
        )
    if current.text_path is None or current.text_path == "":
        raise ZoneCorpusError(
            f"latest version for {page.url} has no text_path; "
            "crawler content is unavailable"
        )
    return current


def _read_stored_html(path_str: str, expected_hash: str, url: str) -> str:
    """Read a version file and require it to match the stored hash.

    IR concept: the zone text has to be the persisted bytes. A missing
    file, a hash mismatch, or bytes that are not UTF-8 is an error.
    """
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"crawler text_path for {url} is not a file: {path_str}"
        )
    data = path.read_bytes()
    actual = compute_content_hash(data)
    if actual != expected_hash:
        raise ZoneCorpusError(
            f"stored file for {url} does not match content_hash "
            f"(file {actual}, record {expected_hash})"
        )
    if data.strip() == b"":
        raise ZoneCorpusError(f"stored file for {url} is empty: {path_str}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ZoneCorpusError(
            f"stored file for {url} is not UTF-8 text: {path_str}"
        ) from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    return text


def _extract_page(html: str) -> _ExtractedPage:
    """Split one HTML document into zones and explicit labels.

    IR concept: zone indexing. A heading that names one allowed zone
    starts that zone. Any other heading is ``other``. ``state`` is set
    only when every explicit state label names the same known state.
    ``conditions`` come only from explicit condition labels. Prose that
    mentions a place or an illness does not fill those fields.
    """
    parser = _HTMLEvents()
    parser.feed(html)
    parser.close()
    builder = _SectionBuilder()
    for event in parser.events:
        kind = event[0]
        if kind == "heading":
            builder.start(_zone_for_heading(str(event[1])))
        elif kind == "text":
            text = str(event[1])
            paragraph_zone = _zone_for_paragraph(text)
            if paragraph_zone is not None:
                builder.start(paragraph_zone)
                continue
            label = _label_line(text)
            if label is not None:
                builder.record_label(label[0], label[1])
            builder.add(text)
        elif kind == "pair":
            name, value = str(event[1]), str(event[2])
            builder.record_label(name, value)
            builder.add(name)
            builder.add(value)
        elif kind == "row":
            cells = tuple(str(cell) for cell in event[1])  # type: ignore[union-attr]
            if len(cells) >= 2:
                builder.record_label(cells[0], cells[1])
            for cell in cells:
                builder.add(cell)
        else:
            raise ZoneCorpusError(f"internal HTML event {kind!r} is not supported")
    builder.finish()
    return _ExtractedPage(
        title=parser.title,
        state=_one_state(builder.state_values),
        conditions=_dedupe(builder.condition_values),
        sections=tuple(builder.sections),
    )


def _zone_for_heading(text: str) -> str:
    """Map a heading to one allowed zone, or ``other`` when it is mixed."""
    matched = _matched_zones(_heading_key(text))
    if len(matched) == 1:
        return matched[0]
    return "other"


def _zone_for_paragraph(text: str) -> str | None:
    """Treat a short block as a heading only when the whole block is one.

    A sentence that merely mentions a zone word stays body text.
    """
    key = _heading_key(text)
    if key == "" or len(key) > 60:
        return None
    matched = _matched_zones(key)
    if len(matched) == 1:
        return matched[0]
    return None


def _matched_zones(key: str) -> list[str]:
    """Zones whose phrase is the heading, or the heading plus a suffix.

    A second zone's phrase anywhere in the heading makes the match
    ambiguous, so both are returned and the caller keeps ``other``.
    """
    if key == "" or len(key) > 80:
        return []
    found: list[str] = []
    contained: list[str] = []
    padded = f" {key} "
    for zone, phrases in _normalised_zone_phrases():
        hit = False
        inside = False
        for phrase in phrases:
            if key == phrase or _suffix_heading(key, phrase):
                hit = True
            if f" {phrase} " in padded:
                inside = True
        if hit:
            found.append(zone)
        elif inside:
            contained.append(zone)
    if contained:
        return found + contained
    return found


def _suffix_heading(key: str, phrase: str) -> bool:
    """True when ``key`` is ``phrase`` followed only by generic suffix words."""
    prefix = phrase + " "
    if not key.startswith(prefix):
        return False
    rest = key[len(prefix) :].split(" ")
    return bool(rest) and all(token in _HEADING_SUFFIXES for token in rest)


def _label_kind(name: str) -> str | None:
    """Return ``state`` or ``conditions`` for an explicit field label."""
    key = _heading_key(name)
    if key in _STATE_LABELS:
        return "state"
    if key in _CONDITION_LABELS:
        return "conditions"
    return None


def _label_line(text: str) -> tuple[str, str] | None:
    """Parse a block that is only ``Label: value``. Anything else is prose."""
    match = _LABEL_LINE.fullmatch(text.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def _one_state(values: list[str]) -> str | None:
    """One canonical state, or none when the labels disagree or are unknown."""
    if not values:
        return None
    found: list[str] = []
    for value in values:
        canonical = _canonical_state(value)
        if canonical is None:
            return None
        if canonical not in found:
            found.append(canonical)
    if len(found) == 1:
        return found[0]
    return None


def _canonical_state(value: str) -> str | None:
    """Map a labeled value onto the closed state list. Unknown stays unset."""
    key = _heading_key(value)
    if key in _STATE_INDEX:
        return _STATE_INDEX[key]
    compact = key.replace(" ", "")
    return _STATE_INDEX.get(compact)


def _condition_items(value: str) -> tuple[str, ...]:
    """Split an explicit condition label. A sentence is not a condition list."""
    cleaned = re.sub(r"\s+", " ", value).strip()
    if cleaned == "" or len(cleaned) > 80:
        return ()
    if any(mark in cleaned for mark in ".?!"):
        return ()
    items: list[str] = []
    for part in re.split(r"\s*[,;]\s*", cleaned):
        item = part.strip()
        if item == "" or len(item) > 40:
            return ()
        items.append(item)
    return tuple(items)


def _dedupe(items: list[str]) -> tuple[str, ...]:
    """Keep the first spelling of each condition, in label order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return tuple(result)


def _copied_score(page: PageRecord) -> float | None:
    """Copy ``g_score``. ``None`` stays ``None`` and is not replaced with 0."""
    score = page.g_score
    if score is None:
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ZoneCorpusError(
            f"page {page.url} has a g_score that is not a real number"
        )
    if not math.isfinite(float(score)):
        raise ZoneCorpusError(f"page {page.url} has a non-finite g_score")
    return float(score)


def _copied_timestamp(field: str, value: object, url: str) -> str | None:
    """Copy a crawler timestamp string. Blank stays missing.

    The string is not rewritten, and this function does not read the
    other timestamp. ``ZoneDoc`` rejects a non-ISO value.
    """
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ZoneCorpusError(
            f"page {url} has {field} {value!r}; expected an ISO datetime string"
        )
    return value


def _doc_id(url: str, content_hash: str, ordinal: int, zone: str) -> str:
    """Stable id for one zone of one stored snapshot.

    IR concept: a document id. The same URL, hash, position, and zone
    always produce the same id. The id does not contain ``text_path``.
    """
    if zone not in _ZONE_SET:
        allowed = ", ".join(ZONES)
        raise ZoneCorpusError(f"zone must be one of {allowed}; got {zone!r}")
    material = f"{url}\n{content_hash}\n{ordinal}\n{zone}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _collapse(parts: list[str]) -> str:
    """Join a tag's text and collapse whitespace. Do not change words."""
    text = " ".join(parts).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _heading_key(text: str) -> str:
    """Case-fold a heading and turn punctuation into spaces.

    Letters, numbers, and combining marks stay, so a Devanagari heading
    is not split into consonants by dropping its vowel signs.
    """
    folded = unicodedata.normalize("NFC", text).casefold().replace("\u00a0", " ")
    kept: list[str] = []
    for char in folded:
        if char.isspace():
            kept.append(" ")
            continue
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            kept.append(char)
        else:
            kept.append(" ")
    return re.sub(r" +", " ", "".join(kept)).strip()


def _normalised_zone_phrases() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Zone phrases after the same heading normalisation as the page."""
    global _NORMALISED_ZONE_PHRASES
    if _NORMALISED_ZONE_PHRASES is None:
        _NORMALISED_ZONE_PHRASES = tuple(
            (zone, tuple(_heading_key(phrase) for phrase in phrases))
            for zone, phrases in _ZONE_PHRASES
        )
    return _NORMALISED_ZONE_PHRASES


_NORMALISED_ZONE_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] | None = None


def _build_state_index() -> dict[str, str]:
    """Closed lookup from a normalised label value to a canonical name."""
    index: dict[str, str] = {}
    for name in _STATE_NAMES:
        key = _heading_key(name)
        index[key] = name
        index[key.replace(" ", "")] = name
    for alias, name in _STATE_ALIASES:
        index[_heading_key(alias)] = name
    return index


_STATE_INDEX = _build_state_index()
