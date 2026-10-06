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
