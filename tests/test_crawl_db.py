"""Unit tests for the Bharosa crawler SQLite database module.

Tests page upserts, historical version snapshots, content change logs, conditional HTTP metadata,
and last_changed_at semantics across first crawl, unchanged recrawl, and changed recrawl.
"""

from __future__ import annotations

import pathlib
import pytest

from bharosa.crawl.db import ChangeRecord, CrawlDB, PageRecord, VersionRecord
from bharosa.crawl.dedup import compute_content_hash


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


def test_sha256_known_input_hash() -> None:
    """Test compute_content_hash produces real SHA-256 hex digest for known input."""
    content = "Hello Bharosa Crawler"
    expected_digest = "c6c245193a4a26f3a0223098d899d135d022169404ba2d52c7865bc5a8ade1bc"

    computed = compute_content_hash(content)
    assert len(computed) == 64
    assert computed == expected_digest

    # Verify deterministic property
    assert compute_content_hash(content) == computed
    assert compute_content_hash("Different Content") != computed


def test_first_crawl_semantics(memory_db: CrawlDB) -> None:
    """Test case A: First crawl sets fetched_at == last_changed_at and no change event row is created."""
    url = "http://scheme.gov.in/ayushman"
    domain = "scheme.gov.in"
    content = "<html>Initial Ayushman Bharat Content</html>"
    hash1 = compute_content_hash(content)

    t1 = "2026-10-06T10:00:00Z"
    page, changed = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=t1,
        content_hash=hash1,
        g_score=1.5,
        etag='"v1.0"',
        last_modified="Mon, 05 Oct 2026 10:00:00 GMT",
        status_code=200,
        text_path="data/raw/hash1.html",
    )

    assert changed is False
    assert page.url == url
    assert page.domain == domain
    assert page.fetched_at == t1
    assert page.last_changed_at == t1
    assert page.content_hash == hash1

    # Verify version snapshot recorded
    versions = memory_db.get_versions(url)
    assert len(versions) == 1
    assert versions[0].fetched_at == t1
    assert versions[0].content_hash == hash1
    assert versions[0].text_path == "data/raw/hash1.html"

    # Verify NO change event row created on first crawl
    changes = memory_db.get_changes(url)
    assert len(changes) == 0


def test_unchanged_recrawl_semantics(memory_db: CrawlDB) -> None:
    """Test case B: Unchanged recrawl updates fetched_at, keeps last_changed_at unchanged, creates no change event."""
    url = "http://scheme.gov.in/rules"
    domain = "scheme.gov.in"
    content = "<html>Static Rules Content</html>"
    hash1 = compute_content_hash(content)

    t1 = "2026-10-06T10:00:00Z"
    t2 = "2026-10-06T14:00:00Z"

    # 1. First crawl at t1
    memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=t1,
        content_hash=hash1,
        etag='"v1"',
    )

    # 2. Unchanged recrawl at t2 (same content_hash)
    page2, changed2 = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=t2,
        content_hash=hash1,
        etag='"v1"',
    )

    assert changed2 is False
    assert page2.fetched_at == t2
    # last_changed_at MUST remain t1!
    assert page2.last_changed_at == t1

    # 2 version snapshots exist
    versions = memory_db.get_versions(url)
    assert len(versions) == 2
    assert versions[0].fetched_at == t1
    assert versions[1].fetched_at == t2

    # NO change event row created
    changes = memory_db.get_changes(url)
    assert len(changes) == 0


def test_changed_recrawl_semantics(memory_db: CrawlDB) -> None:
    """Test case C: Changed recrawl updates fetched_at & last_changed_at, creates a change event row with old/new hash."""
    url = "http://scheme.gov.in/benefits"
    domain = "scheme.gov.in"
    hash_old = compute_content_hash("Original Benefits Content v1")
    hash_new = compute_content_hash("Updated Benefits Content v2")

    t1 = "2026-10-06T10:00:00Z"
    t3 = "2026-10-06T18:00:00Z"

    # 1. First crawl at t1
    memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=t1,
        content_hash=hash_old,
    )

    # 2. Changed recrawl at t3
    page3, changed3 = memory_db.upsert_page(
        url=url,
        domain=domain,
        fetched_at=t3,
        content_hash=hash_new,
        etag='"v2"',
    )

    assert changed3 is True
    assert page3.fetched_at == t3
    assert page3.last_changed_at == t3
    assert page3.content_hash == hash_new

    # 2 version snapshots exist
    versions = memory_db.get_versions(url)
    assert len(versions) == 2

    # Exactly 1 change event row created with detected_at = t3
    changes = memory_db.get_changes(url)
    assert len(changes) == 1
    assert changes[0].url == url
    assert changes[0].detected_at == t3
    assert changes[0].old_hash == hash_old
    assert changes[0].new_hash == hash_new
    assert "Hash updated" in (changes[0].diff_summary or "")


def test_persistent_text_path_verification(tmp_path: pathlib.Path) -> None:
    """Test case E: text_path references an actual valid readable file whose contents match fetched HTML."""
    db_file = tmp_path / "test_persist.db"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    db = CrawlDB(db_path=str(db_file))

    url = "http://scheme.gov.in/page"
    html_content = "<html><body>Empanelled Hospitals List 2026</body></html>"
    content_hash = compute_content_hash(html_content)

    # Persist raw HTML file under data/raw/<content_hash[:16]>.html
    filename = f"{content_hash[:16]}.html"
    text_path = str(raw_dir / filename)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Upsert to CrawlDB
    db.upsert_page(
        url=url,
        domain="scheme.gov.in",
        fetched_at="2026-10-06T10:00:00Z",
        content_hash=content_hash,
        text_path=text_path,
    )

    # Retrieve version from SQLite
    versions = db.get_versions(url)
    assert len(versions) == 1
    stored_text_path = versions[0].text_path
    assert stored_text_path is not None

    # Check referenced file exists
    path_obj = pathlib.Path(stored_text_path)
    assert path_obj.exists()
    assert path_obj.is_file()

    # Read file back and verify contents
    read_back_content = path_obj.read_text(encoding="utf-8")
    assert read_back_content == html_content

    db.close()
