"""Unit tests for the Bharosa live crawl runner module.

Tests missing seed configuration detection, zero-seed safety, and runner integration
with mocked network fetches and persistent raw HTML verification.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest
import yaml

from bharosa.crawl.dedup import compute_content_hash
from bharosa.crawl.fetcher import FetchResult, FetchStatus
from bharosa.crawl.run import CrawlRunner, load_seeds


def test_missing_seed_file_raises_setup_required(tmp_path: pathlib.Path) -> None:
    """Test that load_seeds raises FileNotFoundError with clear setup instructions if file is missing."""
    missing_file = tmp_path / "nonexistent_seeds.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_seeds(str(missing_file))

    assert "SETUP REQUIRED" in str(exc_info.value)
    assert "config/seeds.yaml" in str(exc_info.value)


def test_empty_or_invalid_seed_file_raises_value_error(tmp_path: pathlib.Path) -> None:
    """Test that empty or malformed YAML raises ValueError."""
    bad_file = tmp_path / "bad_seeds.yaml"
    bad_file.write_text("invalid_key: true\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_seeds(str(bad_file))

    assert "Expected top-level 'seeds' key" in str(exc_info.value)

    empty_seeds_file = tmp_path / "empty_seeds.yaml"
    empty_seeds_file.write_text("seeds: []\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info2:
        load_seeds(str(empty_seeds_file))

    assert "No seed entries found" in str(exc_info2.value)


def test_valid_seed_file_loading(tmp_path: pathlib.Path) -> None:
    """Test loading valid seed entries from YAML."""
    seed_file = tmp_path / "seeds.yaml"
    seed_content = {
        "seeds": [
            {"url": "http://scheme.gov.in/p1", "g_score": 2.0},
            {"url": "http://scheme.gov.in/p2", "g_score": 1.0},
        ]
    }
    seed_file.write_text(yaml.dump(seed_content), encoding="utf-8")

    loaded = load_seeds(str(seed_file))
    assert len(loaded) == 2
    assert loaded[0]["url"] == "http://scheme.gov.in/p1"
    assert loaded[0]["g_score"] == 2.0


def test_runner_zero_seed_safety(tmp_path: pathlib.Path) -> None:
    """Test runner completes safely with 0 pages crawled when no seeds are enqueued."""
    db_file = tmp_path / "test_run.db"
    raw_dir = tmp_path / "raw"

    runner = CrawlRunner(
        db_path=str(db_file),
        raw_data_dir=str(raw_dir),
        verbose=False,
    )

    summary = runner.run()
    assert summary["pages_crawled"] == 0
    assert summary["remaining_frontier_size"] == 0
    assert summary["crawl_type"] == "LIVE"
    runner.close()


def test_runner_integration_mocked_and_text_path_persistence(tmp_path: pathlib.Path) -> None:
    """Test runner pipeline integration with mocked HTTP fetcher and persistent text_path verification."""
    db_file = tmp_path / "test_run.db"
    raw_dir = tmp_path / "raw"

    runner = CrawlRunner(
        db_path=str(db_file),
        raw_data_dir=str(raw_dir),
        min_delay_seconds=0.0,
        max_pages=5,
        verbose=False,
    )

    html_bytes = b"<html><body>Mock Scheme Content 2026</body></html>"
    expected_hash = compute_content_hash(html_bytes)

    # Mock fetcher
    mock_fetch_result = FetchResult(
        url="http://test.org/scheme1",
        status=FetchStatus.SUCCESS,
        status_code=200,
        is_success=True,
        is_modified=True,
        content=html_bytes,
        text=html_bytes.decode("utf-8"),
        headers={"ETag": '"v1"'},
        etag='"v1"',
        last_modified="Mon, 05 Oct 2026 10:00:00 GMT",
        fetched_at="2026-10-06T10:00:00Z",
        elapsed_seconds=0.1,
        robots_allowed=True,
    )

    runner.fetcher.is_url_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]
    runner.fetcher.fetch = MagicMock(return_value=mock_fetch_result)  # type: ignore[method-assign]

    seeds = [{"url": "http://test.org/scheme1", "g_score": 1.5}]
    runner.enqueue_seeds(seeds)

    summary = runner.run()

    assert summary["pages_crawled"] == 1
    assert summary["pages_skipped_robots"] == 0
    assert summary["pages_failed"] == 0
    assert summary["crawl_type"] == "LIVE"

    # 1. Verify page persisted in DB pages table
    page = runner.db.get_page("http://test.org/scheme1")
    assert page is not None
    assert page.domain == "test.org"
    assert page.g_score == 1.5
    assert page.etag == '"v1"'
    assert page.content_hash == expected_hash

    # 2. Verify version snapshot and stored text_path
    versions = runner.db.get_versions("http://test.org/scheme1")
    assert len(versions) == 1
    stored_text_path = versions[0].text_path
    assert stored_text_path is not None

    # 3. Verify referenced raw HTML file exists and matches fetched content
    path_obj = pathlib.Path(stored_text_path)
    assert path_obj.exists()
    assert path_obj.is_file()
    assert path_obj.read_bytes() == html_bytes

    runner.close()


def test_runner_304_not_modified_handling(tmp_path: pathlib.Path) -> None:
    """Regression test: 304 Not Modified updates fetched_at while preserving content_hash, last_changed_at, text_path, versions, and change logs."""
    db_file = tmp_path / "test_304.db"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = "http://test.org/scheme304"
    orig_html = "<html><body>Original Content T1</body></html>"
    orig_hash = compute_content_hash(orig_html)
    t1 = "2026-10-06T10:00:00Z"
    t2 = "2026-10-06T14:00:00Z"

    # Save original raw HTML on disk
    html_file = raw_dir / f"{orig_hash[:16]}.html"
    html_file.write_text(orig_html, encoding="utf-8")

    runner = CrawlRunner(
        db_path=str(db_file),
        raw_data_dir=str(raw_dir),
        min_delay_seconds=0.0,
        max_pages=5,
        verbose=False,
    )

    # Establish initial page record in DB at T1
    runner.db.upsert_page(
        url=url,
        domain="test.org",
        fetched_at=t1,
        content_hash=orig_hash,
        g_score=1.0,
        etag='"v1"',
        last_modified="Mon, 05 Oct 2026 10:00:00 GMT",
        status_code=200,
        text_path=str(html_file),
    )

    # Initial state assertions
    p_initial = runner.db.get_page(url)
    assert p_initial is not None
    assert p_initial.fetched_at == t1
    assert p_initial.last_changed_at == t1
    assert len(runner.db.get_versions(url)) == 1
    assert len(runner.db.get_changes(url)) == 0

    # Simulate later fetch returning 304 Not Modified at T2
    mock_304_result = FetchResult(
        url=url,
        status=FetchStatus.NOT_MODIFIED,
        status_code=304,
        is_success=True,
        is_modified=False,
        content=b"",
        text="",
        headers={"ETag": '"v1"'},
        etag='"v1"',
        last_modified="Mon, 05 Oct 2026 10:00:00 GMT",
        fetched_at=t2,
        elapsed_seconds=0.05,
        robots_allowed=True,
    )

    runner.fetcher.is_url_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]
    runner.fetcher.fetch = MagicMock(return_value=mock_304_result)  # type: ignore[method-assign]

    runner.enqueue_seeds([url])
    summary = runner.run()

    assert summary["pages_crawled"] == 1

    # Verify 304 state post-conditions:
    p_after = runner.db.get_page(url)
    assert p_after is not None

    # 1. fetched_at advances to T2
    assert p_after.fetched_at == t2

    # 2. content_hash and last_changed_at remain T1 / original_hash
    assert p_after.last_changed_at == t1
    assert p_after.content_hash == orig_hash

    # 3. No new version row or change row created
    assert len(runner.db.get_versions(url)) == 1
    assert len(runner.db.get_changes(url)) == 0

    # 4. text_path and disk raw HTML remain untouched
    v = runner.db.get_versions(url)[0]
    assert v.text_path == str(html_file)
    assert html_file.exists()
    assert html_file.read_text(encoding="utf-8") == orig_html

    runner.close()

