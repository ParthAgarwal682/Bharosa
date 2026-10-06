"""Unit tests for the Bharosa crawler SQLite database module.

Tests page upserts, historical version snapshots, content change logs, and conditional HTTP metadata.
Uses temporary/in-memory SQLite databases for strict isolation.
"""

from __future__ import annotations

import pathlib
import pytest

from bharosa.crawl.db import ChangeRecord, CrawlDB, PageRecord, VersionRecord


@pytest.fixture
def memory_db() -> CrawlDB:
    """Fixture providing an isolated in-memory CrawlDB instance."""
    db = CrawlDB(db_path=":memory:")
    yield db
    db.close()


def test_schema_initialization(memory_db: CrawlDB) -> None:
    """Test database tables are initialized properly."""
    assert memory_db.get_page("http://example.com/nonexistent") is None
    assert memory_db.get_versions("http://example.com/nonexistent") == []
    assert memory_db.get_changes("http://example.com/nonexistent") == []


def test_upsert_initial_page_crawl(memory_db: CrawlDB) -> None:
    """Test initial page insertion creates page, version snapshot, and change event."""
    url = "http://scheme.gov.in/ayushman"
    domain = "scheme.gov.in"
    hash1 = "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890"

    page, changed = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at="2026-10-06T10:00:00Z",
        content_hash=hash1,
        g_score=1.5,
        etag='"v1.0"',
        last_modified="Mon, 05 Oct 2026 10:00:00 GMT",
        status_code=200,
        text_path="data/raw/ayushman_v1.html",
    )

    assert changed is True
    assert page.url == url
    assert page.domain == domain
    assert page.fetched_at == "2026-10-06T10:00:00Z"
    assert page.last_changed_at == "2026-10-06T10:00:00Z"
    assert page.content_hash == hash1
    assert page.etag == '"v1.0"'

    # Verify query
    fetched_page = memory_db.get_page(url)
    assert fetched_page == page

    # Verify version snapshot
    versions = memory_db.get_versions(url)
    assert len(versions) == 1
    assert versions[0].url == url
    assert versions[0].content_hash == hash1
    assert versions[0].text_path == "data/raw/ayushman_v1.html"

    # Verify change log
    changes = memory_db.get_changes(url)
    assert len(changes) == 1
    assert changes[0].url == url
    assert changes[0].old_hash is None
    assert changes[0].new_hash == hash1
    assert "Initial page crawl" in (changes[0].diff_summary or "")


def test_upsert_unchanged_page(memory_db: CrawlDB) -> None:
    """Test re-crawling page with identical content_hash updates fetched_at but preserves last_changed_at."""
    url = "http://scheme.gov.in/rules"
    domain = "scheme.gov.in"
    hash1 = "1111222233334444555566667777888899990000111122223333444455556666"

    # Fetch 1 at 10:00
    memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at="2026-10-06T10:00:00Z",
        content_hash=hash1,
        etag='"v1"',
    )

    # Fetch 2 at 12:00 (content hash UNCHANGED)
    page2, changed2 = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at="2026-10-06T12:00:00Z",
        content_hash=hash1,
        etag='"v1"',
    )

    assert changed2 is False
    assert page2.fetched_at == "2026-10-06T12:00:00Z"
    # last_changed_at remains 10:00:00Z!
    assert page2.last_changed_at == "2026-10-06T10:00:00Z"

    # Should have 2 versions
    versions = memory_db.get_versions(url)
    assert len(versions) == 2
    assert versions[0].fetched_at == "2026-10-06T10:00:00Z"
    assert versions[1].fetched_at == "2026-10-06T12:00:00Z"

    # Should still only have 1 change log entry (initial)
    changes = memory_db.get_changes(url)
    assert len(changes) == 1


def test_upsert_changed_page_detects_hash_difference(memory_db: CrawlDB) -> None:
    """Test re-crawling page with new content_hash logs change and updates last_changed_at."""
    url = "http://scheme.gov.in/benefits"
    domain = "scheme.gov.in"
    hash_old = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    hash_new = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    # Fetch 1
    memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at="2026-10-06T10:00:00Z",
        content_hash=hash_old,
    )

    # Fetch 2 with CHANGED content hash
    page2, changed2 = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at="2026-10-06T15:00:00Z",
        content_hash=hash_new,
        etag='"v2"',
    )

    assert changed2 is True
    assert page2.fetched_at == "2026-10-06T15:00:00Z"
    assert page2.last_changed_at == "2026-10-06T15:00:00Z"
    assert page2.content_hash == hash_new

    # 2 versions
    versions = memory_db.get_versions(url)
    assert len(versions) == 2

    # 2 change entries
    changes = memory_db.get_changes(url)
    assert len(changes) == 2
    assert changes[1].old_hash == hash_old
    assert changes[1].new_hash == hash_new
    assert "Hash updated" in (changes[1].diff_summary or "")


def test_temporary_db_file(tmp_path: pathlib.Path) -> None:
    """Test CrawlDB persistence on a real filesystem temporary DB file."""
    db_file = tmp_path / "test_bharosa.db"
    db = CrawlDB(db_path=str(db_file))

    url = "http://test.org/page"
    db.upsert_page(
        url=url,
        domain="test.org",
        fetched_at="2026-10-06T10:00:00Z",
        content_hash="12345",
    )
    db.close()

    # Re-open DB from file
    db2 = CrawlDB(db_path=str(db_file))
    page = db2.get_page(url)
    assert page is not None
    assert page.domain == "test.org"
    db2.close()
