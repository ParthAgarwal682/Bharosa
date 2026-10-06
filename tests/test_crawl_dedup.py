"""Unit tests for the Bharosa URL normalization and content deduplication module.

Tests equivalent vs genuinely different URLs, parameter stripping, query sorting,
SHA-256 content hashing, and DedupStore logic.
"""

from __future__ import annotations

import pytest

from bharosa.crawl.dedup import (
    DEFAULT_TRACKING_PARAMS,
    DedupStore,
    compute_content_hash,
    normalize_url,
)


def test_scheme_and_host_lowercasing() -> None:
    """Test scheme and host are lowercased."""
    raw = "HTTP://SCHEME.GOV.IN/Ayushman"
    normalized = normalize_url(raw)
    assert normalized == "http://scheme.gov.in/Ayushman"


def test_default_port_removal() -> None:
    """Test default ports 80 and 443 are stripped, custom ports preserved."""
    assert normalize_url("http://example.com:80/path") == "http://example.com/path"
    assert normalize_url("https://example.com:443/path") == "https://example.com/path"
    assert normalize_url("http://example.com:8080/path") == "http://example.com:8080/path"


def test_fragment_removal() -> None:
    """Test URL fragments (#...) are removed."""
    raw = "http://example.com/scheme#eligibility"
    assert normalize_url(raw) == "http://example.com/scheme"


def test_path_normalization() -> None:
    """Test consecutive slashes are collapsed and trailing slashes stripped for non-root paths."""
    assert normalize_url("http://example.com/a//b///c/") == "http://example.com/a/b/c"
    # Root path remains "/"
    assert normalize_url("http://example.com/") == "http://example.com/"


def test_tracking_parameter_removal() -> None:
    """Test tracking params (utm_*, fbclid, gclid, etc.) are removed while content params are kept."""
    raw = "http://example.com/scheme?utm_source=google&id=101&utm_medium=cpc&state=UP&fbclid=xyz"
    normalized = normalize_url(raw)
    # Content parameters id and state MUST be kept
    assert normalized == "http://example.com/scheme?id=101&state=UP"


def test_query_parameter_sorting() -> None:
    """Test remaining query parameters are sorted alphabetically by key."""
    raw = "http://example.com/search?z=3&a=1&m=2"
    normalized = normalize_url(raw)
    assert normalized == "http://example.com/search?a=1&m=2&z=3"


def test_equivalent_urls_produce_identical_normalized_string() -> None:
    """Test that variant representations of the same resource normalize to identical string."""
    url1 = "HTTP://EXAMPLE.COM:80/scheme/?utm_source=newsletter&state=UP&id=42#documents"
    url2 = "http://example.com/scheme?id=42&state=UP"

    assert normalize_url(url1) == normalize_url(url2) == "http://example.com/scheme?id=42&state=UP"


def test_genuinely_different_urls_remain_distinct() -> None:
    """Test conservative filtering keeps distinct pages separate."""
    url1 = "http://example.com/scheme?state=UP"
    url2 = "http://example.com/scheme?state=MP"
    url3 = "http://example.com/scheme?id=1"
    url4 = "http://example.com/scheme?id=2"

    assert normalize_url(url1) != normalize_url(url2)
    assert normalize_url(url3) != normalize_url(url4)


def test_stable_content_hashing() -> None:
    """Test SHA-256 content hashing for str and bytes input."""
    text1 = "Official Scheme Guidelines"
    text2 = "Official Scheme Guidelines"
    text3 = "Modified Scheme Guidelines"

    h1 = compute_content_hash(text1)
    h2 = compute_content_hash(text2)
    h3 = compute_content_hash(text3)

    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h3

    # Bytes input
    assert compute_content_hash(text1.encode("utf-8")) == h1


def test_content_hash_rejects_invalid_types() -> None:
    """Test content hash raises TypeError for non str/bytes."""
    with pytest.raises(TypeError):
        compute_content_hash(12345)  # type: ignore[arg-type]


def test_dedup_store_urls_and_content() -> None:
    """Test DedupStore tracks seen normalized URLs and content hashes."""
    store = DedupStore()

    raw_url = "HTTP://EXAMPLE.COM/page?utm_source=test&id=5#top"
    norm_url = "http://example.com/page?id=5"

    assert not store.is_url_seen(raw_url)
    assert store.mark_url_seen(raw_url) == norm_url
    assert store.is_url_seen(norm_url)
    assert store.is_url_seen(raw_url)
    assert len(store) == 1

    content = "<html>Scheme details...</html>"
    chash = compute_content_hash(content)

    assert not store.is_content_seen(chash)
    assert store.mark_content_seen(chash, url=raw_url) is True
    assert store.is_content_seen(chash)
    assert store.mark_content_seen(chash, url=raw_url) is False
    assert store.get_url_for_content(chash) == norm_url


def test_dedup_store_clear() -> None:
    """Test clearing DedupStore resets state."""
    store = DedupStore()
    store.mark_url_seen("http://example.com/a")
    store.mark_content_seen("hash123", url="http://example.com/a")

    store.clear()
    assert len(store) == 0
    assert not store.is_url_seen("http://example.com/a")
    assert not store.is_content_seen("hash123")
