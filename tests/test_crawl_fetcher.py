"""Unit tests for the Bharosa crawler fetcher module.

All tests use mocked HTTP responses and require no internet connection.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from bharosa.crawl.fetcher import (
    DEFAULT_USER_AGENT,
    Fetcher,
    FetchResult,
    FetchStatus,
    RobotsCache,
    RobotsDisallowedError,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """Fixture providing a mocked requests.Session."""
    session = MagicMock(spec=requests.Session)
    return session


def create_mock_response(
    status_code: int = 200,
    text: str = "<html><body>Test Page</body></html>",
    content: bytes = b"<html><body>Test Page</body></html>",
    headers: dict[str, str] | None = None,
    reason: str = "OK",
) -> MagicMock:
    """Helper to construct a mock requests.Response object."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.headers = headers or {}
    resp.reason = reason
    return resp


def test_user_agent_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test User-Agent selection order: explicit param > env var > default."""
    # Default
    monkeypatch.delenv("CRAWL_USER_AGENT", raising=False)
    fetcher_default = Fetcher(min_delay_seconds=0.0)
    assert fetcher_default.get_user_agent() == DEFAULT_USER_AGENT

    # Env var override
    monkeypatch.setenv("CRAWL_USER_AGENT", "EnvBot/1.0")
    fetcher_env = Fetcher(min_delay_seconds=0.0)
    assert fetcher_env.get_user_agent() == "EnvBot/1.0"

    # Parameter override
    fetcher_param = Fetcher(user_agent="CustomBot/2.0", min_delay_seconds=0.0)
    assert fetcher_param.get_user_agent() == "CustomBot/2.0"


def test_robots_txt_disallows_path(mock_session: MagicMock) -> None:
    """Test that URLs disallowed by robots.txt are blocked without fetching target URL."""
    robots_txt_body = "User-agent: *\nDisallow: /private/\n"
    robots_resp = create_mock_response(200, text=robots_txt_body)

    # Return robots.txt when requested
    mock_session.get.return_value = robots_resp

    fetcher = Fetcher(
        session=mock_session,
        min_delay_seconds=0.0,
        user_agent="BharosaBot/1.0",
    )

    url_disallowed = "http://example.com/private/secret.html"

    # Runtime check
    assert not fetcher.is_url_allowed(url_disallowed)

    # Fetch attempt
    res = fetcher.fetch(url_disallowed, enforce_delay=False)

    assert res.status == FetchStatus.DISALLOWED_BY_ROBOTS
    assert res.status_code == 403
    assert not res.is_success
    assert not res.robots_allowed
    assert "Disallowed by robots.txt" in (res.error_message or "")

    # Ensure get was only called once for robots.txt, not for target URL
    assert mock_session.get.call_count == 1
    call_url = mock_session.get.call_args[0][0]
    assert call_url == "http://example.com/robots.txt"


def test_robots_raise_on_disallowed(mock_session: MagicMock) -> None:
    """Test that raise_on_disallowed=True raises RobotsDisallowedError."""
    robots_txt_body = "User-agent: *\nDisallow: /admin/\n"
    mock_session.get.return_value = create_mock_response(200, text=robots_txt_body)

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    url = "http://example.com/admin/dashboard"

    with pytest.raises(RobotsDisallowedError) as exc_info:
        fetcher.fetch(url, raise_on_disallowed=True, enforce_delay=False)

    assert exc_info.value.url == url
    assert "Crawling disallowed by robots.txt" in str(exc_info.value)


def test_robots_allow_and_successful_fetch(mock_session: MagicMock) -> None:
    """Test fetching a permitted URL returns a successful FetchResult."""
    robots_txt = "User-agent: *\nAllow: /\n"
    robots_resp = create_mock_response(200, text=robots_txt)

    page_headers = {
        "ETag": '"v123"',
        "Last-Modified": "Wed, 21 Oct 2026 07:28:00 GMT",
        "Content-Type": "text/html",
    }
    page_resp = create_mock_response(
        200,
        text="<h1>PM-JAY Scheme</h1>",
        content=b"<h1>PM-JAY Scheme</h1>",
        headers=page_headers,
    )

    # First call for robots.txt, second for page
    mock_session.get.side_effect = [robots_resp, page_resp]

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    url = "http://scheme.gov.in/pmjay"

    assert fetcher.is_url_allowed(url)

    res = fetcher.fetch(url, enforce_delay=False)

    assert res.status == FetchStatus.SUCCESS
    assert res.status_code == 200
    assert res.is_success
    assert res.is_modified
    assert res.text == "<h1>PM-JAY Scheme</h1>"
    assert res.etag == '"v123"'
    assert res.last_modified == "Wed, 21 Oct 2026 07:28:00 GMT"
    assert res.robots_allowed
    assert res.retries_used == 0


def test_conditional_request_304_not_modified(mock_session: MagicMock) -> None:
    """Test conditional GET returns 304 Not Modified when content is unchanged."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    not_modified_resp = create_mock_response(
        304,
        text="",
        content=b"",
        headers={"ETag": '"v123"'},
        reason="Not Modified",
    )

    mock_session.get.side_effect = [robots_resp, not_modified_resp]

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    url = "http://scheme.gov.in/rules"

    res = fetcher.fetch(
        url,
        etag='"v123"',
        last_modified="Wed, 21 Oct 2026 07:28:00 GMT",
        enforce_delay=False,
    )

    assert res.status == FetchStatus.NOT_MODIFIED
    assert res.status_code == 304
    assert res.is_success
    assert not res.is_modified
    assert res.content == b""
    assert res.etag == '"v123"'

    # Verify conditional headers were sent
    page_call_args = mock_session.get.call_args_list[1]
    sent_headers = page_call_args[1].get("headers", {})
    assert sent_headers.get("If-None-Match") == '"v123"'
    assert sent_headers.get("If-Modified-Since") == "Wed, 21 Oct 2026 07:28:00 GMT"


def test_retry_on_500_server_error(mock_session: MagicMock) -> None:
    """Test bounded retries when encountering transient 500 server errors."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    err_resp = create_mock_response(500, text="Internal Error", reason="Server Error")
    ok_resp = create_mock_response(200, text="Recovered Page", reason="OK")

    # 1. robots.txt -> 2. 500 Error -> 3. Retry 200 OK
    mock_session.get.side_effect = [robots_resp, err_resp, ok_resp]

    fetcher = Fetcher(
        session=mock_session,
        min_delay_seconds=0.0,
        max_retries=2,
        backoff_factor=0.001,
    )
    url = "http://example.com/unstable"

    res = fetcher.fetch(url, enforce_delay=False)

    assert res.status == FetchStatus.SUCCESS
    assert res.status_code == 200
    assert res.retries_used == 1
    assert res.text == "Recovered Page"


def test_retry_exhaustion_on_persistent_500(mock_session: MagicMock) -> None:
    """Test that persistent 500 errors exhaust retries and return HTTP_ERROR."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    err_resp = create_mock_response(503, text="Service Unavailable", reason="Unavailable")

    mock_session.get.side_effect = [robots_resp, err_resp, err_resp, err_resp]

    fetcher = Fetcher(
        session=mock_session,
        min_delay_seconds=0.0,
        max_retries=2,
        backoff_factor=0.001,
    )

    res = fetcher.fetch("http://example.com/down", enforce_delay=False)

    assert res.status == FetchStatus.HTTP_ERROR
    assert res.status_code == 503
    assert not res.is_success
    assert res.retries_used == 2


def test_timeout_handling(mock_session: MagicMock) -> None:
    """Test timeout exception returns TIMEOUT status with status_code=0."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    timeout_exc = requests.Timeout("Connection timed out")

    mock_session.get.side_effect = [robots_resp, timeout_exc, timeout_exc]

    fetcher = Fetcher(
        session=mock_session,
        min_delay_seconds=0.0,
        max_retries=1,
        backoff_factor=0.001,
    )

    res = fetcher.fetch("http://example.com/slow", enforce_delay=False)

    assert res.status == FetchStatus.TIMEOUT
    assert res.status_code == 0
    assert not res.is_success
    assert res.retries_used == 1
    assert "Timeout error" in (res.error_message or "")


def test_network_error_handling(mock_session: MagicMock) -> None:
    """Test network connection failure returns NETWORK_ERROR status."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    conn_exc = requests.ConnectionError("Failed to establish a new connection")

    mock_session.get.side_effect = [robots_resp, conn_exc]

    fetcher = Fetcher(
        session=mock_session,
        min_delay_seconds=0.0,
        max_retries=0,
    )

    res = fetcher.fetch("http://example.com/unreachable", enforce_delay=False)

    assert res.status == FetchStatus.NETWORK_ERROR
    assert res.status_code == 0
    assert not res.is_success
    assert "Network error" in (res.error_message or "")


def test_robots_404_allows_all(mock_session: MagicMock) -> None:
    """Test that a 404 response for robots.txt permits crawling all URLs."""
    robots_404 = create_mock_response(404, text="Not Found")
    page_200 = create_mock_response(200, text="Page content")

    mock_session.get.side_effect = [robots_404, page_200]

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    url = "http://example.com/any_page"

    assert fetcher.is_url_allowed(url)

    res = fetcher.fetch(url, enforce_delay=False)
    assert res.is_success
    assert res.robots_allowed


def test_robots_403_disallows_all(mock_session: MagicMock) -> None:
    """Test that a 403 Forbidden response for robots.txt blocks crawling."""
    robots_403 = create_mock_response(403, text="Forbidden")

    mock_session.get.return_value = robots_403

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    url = "http://example.com/blocked_domain_page"

    assert not fetcher.is_url_allowed(url)

    res = fetcher.fetch(url, enforce_delay=False)
    assert res.status == FetchStatus.DISALLOWED_BY_ROBOTS
    assert not res.robots_allowed


def test_invalid_url_structure() -> None:
    """Test fetching a malformed URL returns INVALID_URL status."""
    fetcher = Fetcher(min_delay_seconds=0.0)
    res = fetcher.fetch("not_a_valid_url", enforce_delay=False)

    assert res.status == FetchStatus.INVALID_URL
    assert res.status_code == 0
    assert not res.is_success
    assert not res.robots_allowed


def test_robots_cache_clearing(mock_session: MagicMock) -> None:
    """Test resetting fetcher state clears cached robots parsers."""
    robots_resp = create_mock_response(200, text="User-agent: *\nAllow: /\n")
    mock_session.get.return_value = robots_resp

    fetcher = Fetcher(session=mock_session, min_delay_seconds=0.0)
    fetcher.is_url_allowed("http://example.com/page1")
    assert mock_session.get.call_count == 1

    # Second call to same origin uses cache
    fetcher.is_url_allowed("http://example.com/page2")
    assert mock_session.get.call_count == 1

    # Clear cache and check again -> triggers fresh fetch
    fetcher.reset_state()
    fetcher.is_url_allowed("http://example.com/page3")
    assert mock_session.get.call_count == 2
