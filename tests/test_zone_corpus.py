"""Corpus adapter tests against a temporary CrawlDB and raw HTML files.

The pages here are fixtures. They are not a live crawl and they are not
government results. This branch does not contain the crawler package, so
the tests load Parth's unmodified ``bharosa.crawl`` from
``origin/parth/crawler`` when it is not already importable. Nothing is
written into ``bharosa/crawl``.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest


def _ensure_crawler_package() -> None:
    """Import CrawlDB from the repo, or from Parth's git ref if needed."""
    try:
        if importlib.util.find_spec("bharosa.crawl.db") is not None:
            return
    except ModuleNotFoundError:
        pass
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "origin/parth/crawler"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        archive = subprocess.check_output(
            ["git", "archive", "origin/parth/crawler", "bharosa/crawl"],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ImportError(
            "bharosa.crawl is not on this branch, and origin/parth/crawler "
            "could not be read. These tests use that unmodified CrawlDB."
        ) from exc
    dest = Path(tempfile.gettempdir()) / "bharosa-parth-crawler" / sha
    if not (dest / "bharosa" / "crawl" / "db.py").is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(dest, filter="data")
    if str(dest) not in sys.path:
        sys.path.append(str(dest))
    importlib.invalidate_caches()


_ensure_crawler_package()

from bharosa.crawl.db import CrawlDB  # noqa: E402
from bharosa.crawl.dedup import compute_content_hash  # noqa: E402
from bharosa.index.params import ZONE_CONTRACT_FIELDS  # noqa: E402
from bharosa.index.zones import search_schemes  # noqa: E402
from bharosa.text.zones import ZoneCorpusError, load_zone_documents  # noqa: E402

_FETCHED = "2026-10-06T10:00:00Z"
_REFETCHED = "2026-10-06T14:00:00Z"
_CONTRACT_KEYS = set(ZONE_CONTRACT_FIELDS)


def _page(body: str, title: str | None = "Scheme fixture page") -> str:
    head = "" if title is None else f"<title>{title}</title>"
    return f"<!DOCTYPE html><html><head>{head}</head><body>{body}</body></html>"


def _store(
    db: CrawlDB,
    raw: Path,
    url: str,
    html: str,
    fetched_at: str,
    *,
    g_score: float = 0.75,
    domain: str = "scheme.example",
    text_path: str | None | object = ...,
) -> None:
    body = html.encode("utf-8")
    digest = compute_content_hash(body)
    path = raw / f"{digest[:16]}.html"
    path.write_bytes(body)
    stored_path: str | None
    if text_path is ...:
        stored_path = str(path)
    else:
        stored_path = text_path  # type: ignore[assignment]
    db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=fetched_at,
        content_hash=digest,
        g_score=g_score,
        text_path=stored_path,
    )


@pytest.fixture
def crawl(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    db = CrawlDB(db_path=str(tmp_path / "bharosa.db"))
    try:
        yield db, raw, tmp_path / "bharosa.db"
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            pass


def test_crawler_page_maps_onto_zone_doc_fields(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/ayushman"
    html = _page(
        "<h2>Eligibility</h2><p>Families listed in the deprivation criteria may be covered.</p>",
        title="Scheme fixture page",
    )
    _store(
        db,
        raw,
        url,
        html,
        _FETCHED,
        g_score=0.75,
        domain="scheme.example",
    )

    docs = load_zone_documents(db)
    page = db.get_page(url)
    assert page is not None
    assert len(docs) == 1
    doc = docs[0]
    assert doc.url == url
    assert doc.domain == "scheme.example"
    assert doc.title == "Scheme fixture page"
    assert doc.zone == "eligibility"
    assert doc.text == "Families listed in the deprivation criteria may be covered."
    assert doc.g_score == 0.75
    assert doc.content_hash == page.content_hash
    assert doc.crawled_at == page.fetched_at
    assert doc.last_changed_at == page.last_changed_at
    assert set(doc.as_contract()) == _CONTRACT_KEYS
    assert "text_path" not in ZoneDoc_fields()
    assert "fetched_at" not in ZoneDoc_fields()
    assert "detected_at" not in ZoneDoc_fields()


def ZoneDoc_fields() -> set[str]:
    from bharosa.index.zones import ZoneDoc

    return set(ZoneDoc.__dataclass_fields__)


def test_fetched_at_becomes_crawled_at_not_an_older_snapshot(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/fetched"
    html = _page("<p>Visible fetched body.</p>")
    _store(db, raw, url, html, _FETCHED)
    _store(db, raw, url, html, _REFETCHED)

    doc = load_zone_documents(db)[0]
    page = db.get_page(url)
    versions = db.get_versions(url)
    assert page is not None
    assert versions[0].fetched_at == _FETCHED
    assert versions[1].fetched_at == _REFETCHED
    assert doc.crawled_at == page.fetched_at == _REFETCHED
    assert doc.crawled_at != versions[0].fetched_at


def test_last_changed_at_stays_distinct_from_crawled_at(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/freshness"
    html = _page("<p>Unchanged visible body.</p>")
    _store(db, raw, url, html, _FETCHED)
    _store(db, raw, url, html, _REFETCHED)

    doc = load_zone_documents(db)[0]
    page = db.get_page(url)
    assert page is not None
    assert page.fetched_at == _REFETCHED
    assert page.last_changed_at == _FETCHED
    assert doc.crawled_at == _REFETCHED
    assert doc.last_changed_at == _FETCHED
    assert doc.crawled_at != doc.last_changed_at


def test_content_hash_is_preserved(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/hash"
    html = _page("<p>Hash visible body.</p>")
    _store(db, raw, url, html, _FETCHED)
    page = db.get_page(url)
    assert page is not None
    version = db.get_versions(url)[-1]
    file_hash = compute_content_hash(Path(version.text_path or "").read_bytes())

    doc = load_zone_documents(db)[0]
    assert doc.content_hash == page.content_hash == version.content_hash == file_hash
    assert len(doc.content_hash) == 64


def test_text_path_content_is_what_the_zone_contains(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/versions"
    _store(
        db,
        raw,
        url,
        _page("<p>Version one alphaunique sentence.</p>"),
        _FETCHED,
    )
    _store(
        db,
        raw,
        url,
        _page("<p>Version two betaunique sentence.</p>"),
        _REFETCHED,
    )

    docs = load_zone_documents(db)
    assert len(docs) == 1
    assert docs[0].text == "Version two betaunique sentence."
    assert "alphaunique" not in docs[0].text
    assert "betaunique" in docs[0].text
    version = db.get_versions(url)[-1]
    stored = Path(version.text_path or "").read_text(encoding="utf-8")
    assert "Version two betaunique sentence." in stored


def test_zone_extraction_for_representative_sections(crawl) -> None:
    db, raw, _db_path = crawl
    html = _page(
        "<h2>Overview</h2><p>Official description kept as other.</p>"
        "<h2>Eligibility</h2><p>Families listed in the deprivation criteria may be covered.</p>"
        "<h2>Benefits</h2><p>Hospitalisation cover is described on this page.</p>"
        "<h2>Documents Required</h2><p>Bring the family card mentioned on this page.</p>"
        "<h2>How to Apply</h2><p>Submit the form at the official desk.</p>"
        "<h2>पात्रता</h2><p>पात्र परिवार।</p>",
        title="Scheme fixture page",
    )
    _store(db, raw, "https://scheme.example/zones", html, _FETCHED)

    docs = load_zone_documents(db)
    assert [(doc.zone, doc.text) for doc in docs] == [
        ("other", "Official description kept as other."),
        ("eligibility", "Families listed in the deprivation criteria may be covered."),
        ("benefits", "Hospitalisation cover is described on this page."),
        ("documents", "Bring the family card mentioned on this page."),
        ("how_to_apply", "Submit the form at the official desk."),
        ("eligibility", "पात्र परिवार।"),
    ]
    assert all(doc.title == "Scheme fixture page" for doc in docs)

    _store(
        db,
        raw,
        "https://scheme.example/paragraph-heading",
        _page("<p>Eligibility</p><p>Paragraph heading body.</p>"),
        _FETCHED,
    )
    paragraph = [
        doc
        for doc in load_zone_documents(db)
        if doc.url.endswith("/paragraph-heading")
    ]
    assert [(doc.zone, doc.text) for doc in paragraph] == [
        ("eligibility", "Paragraph heading body."),
    ]


def test_missing_optional_state_and_conditions_stay_empty(crawl) -> None:
    db, raw, _db_path = crawl
    _store(
        db,
        raw,
        "https://scheme.example/nolabels",
        _page("<h2>Eligibility</h2><p>No labeled place or illness is given.</p>"),
        _FETCHED,
    )

    doc = load_zone_documents(db)[0]
    assert doc.state is None
    assert doc.conditions == ()


def test_explicit_labels_are_kept_and_conflicts_stay_empty(crawl) -> None:
    db, raw, _db_path = crawl
    _store(
        db,
        raw,
        "https://scheme.example/labeled",
        _page(
            "<p>State: Bihar</p>"
            "<p>Condition: Heart, Kidney</p>"
            "<h2>Eligibility</h2>"
            "<p>Resident families.</p>"
            "<p>The note mentions care in Uttar Pradesh.</p>",
            title="Labeled fixture",
        ),
        _FETCHED,
    )
    _store(
        db,
        raw,
        "https://scheme.example/conflict",
        _page(
            "<p>State: Bihar</p><p>State: Assam</p><p>Body text stays visible.</p>",
            title="Conflict fixture",
        ),
        _FETCHED,
    )

    labeled = next(
        doc
        for doc in load_zone_documents(db)
        if doc.url.endswith("/labeled") and doc.zone == "eligibility"
    )
    assert labeled.state == "Bihar"
    assert labeled.conditions == ("Heart", "Kidney")
    assert "Uttar Pradesh" in labeled.text
    assert labeled.state != "Uttar Pradesh"
    other = next(
        doc
        for doc in load_zone_documents(db)
        if doc.url.endswith("/labeled") and doc.zone == "other"
    )
    assert other.state == "Bihar"
    assert other.conditions == ("Heart", "Kidney")

    conflict = next(doc for doc in load_zone_documents(db) if doc.url.endswith("/conflict"))
    assert conflict.state is None
    assert "Bihar" in conflict.text
    assert "Assam" in conflict.text


def test_malformed_or_missing_persisted_content_fails(crawl, tmp_path: Path) -> None:
    db, raw, db_path = crawl
    missing = tmp_path / "absent" / "bharosa.db"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_zone_documents(missing)
    assert not missing.exists()

    empty_db_path = tmp_path / "empty.db"
    empty_db = CrawlDB(db_path=str(empty_db_path))
    empty_db.close()
    assert load_zone_documents(empty_db_path) == []

    url = "https://scheme.example/broken"
    _store(db, raw, url, _page("<p>Present body.</p>"), _FETCHED, text_path=None)
    with pytest.raises(ZoneCorpusError, match="text_path"):
        load_zone_documents(db)

    db._conn.execute("DELETE FROM pages")
    db._conn.execute("DELETE FROM versions")
    db._conn.commit()

    html = _page("<p>Missing file body.</p>")
    digest = compute_content_hash(html.encode("utf-8"))
    missing_file = raw / "missing.html"
    db.upsert_page(
        url=url,
        domain="scheme.example",
        fetched_at=_FETCHED,
        content_hash=digest,
        g_score=0.75,
        text_path=str(missing_file),
    )
    with pytest.raises(FileNotFoundError, match="text_path"):
        load_zone_documents(db)

    db._conn.execute("DELETE FROM pages")
    db._conn.execute("DELETE FROM versions")
    db._conn.commit()
    empty_hash = compute_content_hash(b"")
    empty_file = raw / "empty.html"
    empty_file.write_bytes(b"")
    db.upsert_page(
        url=url,
        domain="scheme.example",
        fetched_at=_FETCHED,
        content_hash=empty_hash,
        g_score=0.75,
        text_path=str(empty_file),
    )
    with pytest.raises(ZoneCorpusError, match="empty"):
        load_zone_documents(db)

    db._conn.execute("DELETE FROM pages")
    db._conn.execute("DELETE FROM versions")
    db._conn.commit()
    bad = raw / "bad.html"
    bad.write_bytes(b"<html><p>other</p></html>")
    db.upsert_page(
        url=url,
        domain="scheme.example",
        fetched_at=_FETCHED,
        content_hash=compute_content_hash(b"<html><p>real</p></html>"),
        g_score=0.75,
        text_path=str(bad),
    )
    with pytest.raises(ZoneCorpusError, match="does not match"):
        load_zone_documents(db)

    db._conn.execute("DELETE FROM pages")
    db._conn.execute("DELETE FROM versions")
    db._conn.commit()
    binary = raw / "binary.html"
    payload = b"\xff\xfe<html>"
    binary.write_bytes(payload)
    db.upsert_page(
        url=url,
        domain="scheme.example",
        fetched_at=_FETCHED,
        content_hash=compute_content_hash(payload),
        g_score=0.75,
        text_path=str(binary),
    )
    with pytest.raises(ZoneCorpusError, match="UTF-8"):
        load_zone_documents(db)

    db._conn.execute("DELETE FROM pages")
    db._conn.execute("DELETE FROM versions")
    db._conn.commit()
    shell = "<html><head><title></title></head><body></body></html>"
    shell_bytes = shell.encode("utf-8")
    shell_path = raw / "shell.html"
    shell_path.write_bytes(shell_bytes)
    db.upsert_page(
        url=url,
        domain="scheme.example",
        fetched_at=_FETCHED,
        content_hash=compute_content_hash(shell_bytes),
        g_score=0.75,
        text_path=str(shell_path),
    )
    with pytest.raises(ZoneCorpusError, match="no extractable text"):
        load_zone_documents(db)

    with pytest.raises(ZoneCorpusError, match="no crawler page"):
        load_zone_documents(db, urls=["https://scheme.example/missing"])

    assert db_path.is_file()


def test_loader_does_not_fabricate_fields(crawl) -> None:
    db, raw, _db_path = crawl
    url = "https://scheme.example/prose"
    html = (
        "<html><body>"
        "<script>secretmarker heart Uttar Pradesh eligibility</script>"
        "<p>This note talks about heart treatment in Uttar Pradesh.</p>"
        "</body></html>"
    )
    _store(db, raw, url, html, _FETCHED, g_score=0.75)
    db._conn.execute(
        "UPDATE pages SET g_score = NULL, last_changed_at = '' WHERE url = ?",
        (url,),
    )
    db._conn.commit()

    docs = load_zone_documents(db)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.zone == "other"
    assert doc.text == "This note talks about heart treatment in Uttar Pradesh."
    assert "secretmarker" not in doc.text
    assert doc.state is None
    assert doc.conditions == ()
    assert doc.title == ""
    assert doc.title != url
    assert doc.g_score is None
    assert doc.crawled_at == _FETCHED
    assert doc.last_changed_at is None
    assert doc.crawled_at != doc.last_changed_at
    assert set(doc.as_contract()) == _CONTRACT_KEYS
    assert "Rs 5 lakh" not in doc.text
    assert doc.url == url
    assert doc.domain == "scheme.example"
    assert doc.content_hash == compute_content_hash(html.encode("utf-8"))


def test_multiple_zone_docs_load_deterministically(crawl) -> None:
    db, raw, db_path = crawl
    url_b = "https://scheme.example/b"
    url_a = "https://scheme.example/a"
    _store(
        db,
        raw,
        url_b,
        _page(
            "<h2>Benefits</h2><p>Cover described here.</p>"
            "<h2>Documents Required</h2><p>Bring the card.</p>",
            title="Page B",
        ),
        "2026-04-01T00:00:00Z",
        g_score=0.25,
    )
    _store(
        db,
        raw,
        url_a,
        _page(
            "<h2>How to Apply</h2><p>Submit the form at the official desk.</p>"
            "<h2>Eligibility</h2><p>Listed families with deprivation criteria may qualify.</p>",
            title="Page A",
        ),
        "2026-03-01T00:00:00Z",
        g_score=0.5,
    )

    first = load_zone_documents(db)
    second = load_zone_documents(db)
    assert first == second
    assert [doc.url for doc in first] == [url_a, url_a, url_b, url_b]
    assert [doc.zone for doc in first] == [
        "how_to_apply",
        "eligibility",
        "benefits",
        "documents",
    ]
    assert first[0].crawled_at == "2026-03-01T00:00:00Z"
    assert first[0].last_changed_at == "2026-03-01T00:00:00Z"
    assert first[2].crawled_at == "2026-04-01T00:00:00Z"
    assert first[0].doc_id != first[1].doc_id
    assert len({doc.doc_id for doc in first}) == 4

    db.close()
    reopened = load_zone_documents(db_path)
    assert reopened == first

    eligibility = next(doc for doc in first if "deprivation" in doc.text)
    hits = search_schemes("deprivation", documents=first, k=5)
    assert len(hits) == 1
    assert hits[0].doc_id == eligibility.doc_id
    assert hits[0].url == eligibility.url
    assert hits[0].zone == "eligibility"
    assert hits[0].text == eligibility.text
    assert hits[0].last_changed_at == eligibility.last_changed_at
