"""Fetcher module for Bharosa crawler with politeness and robots.txt compliance.

Provides HTTP fetching with per-host robots.txt caching, rate limiting delays,
custom User-Agent configuration, timeouts, bounded retries, conditional HTTP
headers (ETag / If-Modified-Since), and structured result/status objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
import time
from typing import Any
import urllib.parse
import urllib.robotparser

import requests

DEFAULT_USER_AGENT = "BharosaStudentCrawler/0.1 (contact: student@college.edu)"
DEFAULT_MIN_DELAY_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


class FetchStatus(str, Enum):
    """Explicit status categorisation for a fetch operation."""

    SUCCESS = "success"
    NOT_MODIFIED = "not_modified"
    DISALLOWED_BY_ROBOTS = "disallowed_by_robots"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INVALID_URL = "invalid_url"


class RobotsDisallowedError(Exception):
    """Raised when crawling a URL is disallowed by robots.txt."""

    def __init__(self, url: str, user_agent: str) -> None:
        self.url = url
        self.user_agent = user_agent
        super().__init__(
            f"Crawling disallowed by robots.txt for URL: {url} (User-Agent: {user_agent})"
        )


class FetchError(Exception):
    """Base exception for HTTP fetch failures."""

    pass


@dataclass(frozen=True)
class FetchResult:
    """Structured result of an HTTP fetch attempt.

    Attributes:
        url: Target URL that was requested.
        status: Categorised status of the fetch operation.
        status_code: HTTP status code (0 if fetch did not complete).
        is_success: True if HTTP 2xx or 304.
        is_modified: False if HTTP 304 Not Modified, True otherwise.
        content: Raw response bytes.
        text: Response decoded as string.
        headers: Case-insensitive dictionary of response headers.
        etag: ETag response header if present.
        last_modified: Last-Modified response header if present.
        fetched_at: ISO 8601 UTC timestamp when fetch was completed/attempted.
        elapsed_seconds: Elapsed time for the HTTP request(s).
        error_message: Detailed error string if fetch failed or was disallowed.
        robots_allowed: False if disallowed by robots.txt, True otherwise.
        retries_used: Number of retry attempts executed.
    """

    url: str
    status: FetchStatus
    status_code: int
    is_success: bool
    is_modified: bool
    content: bytes
    text: str
    headers: dict[str, str]
    etag: str | None
    last_modified: str | None
    fetched_at: str
    elapsed_seconds: float
    error_message: str | None = None
    robots_allowed: bool = True
    retries_used: int = 0


class RobotsCache:
    """Caches parsed robots.txt rules per scheme+host origin."""

    def __init__(
        self,
        session: requests.Session,
        user_agent: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._user_agent = user_agent
        self._timeout = timeout
        # Mapping: origin ("https://domain.com") -> (RobotFileParser, crawl_delay)
        self._cache: dict[
            str, tuple[urllib.robotparser.RobotFileParser, float | None]
        ] = {}

    def get_parser(
        self, url: str
    ) -> tuple[urllib.robotparser.RobotFileParser, float | None]:
        """Fetch (if uncached) and return the RobotFileParser and crawl_delay for a URL's host."""
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme or "http"
        host = parsed.netloc

        if not host:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse([])
            return parser, None

        origin = f"{scheme}://{host}"
        if origin in self._cache:
            return self._cache[origin]

        robots_url = f"{origin}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)

        crawl_delay_val: float | None = None
        try:
            resp = self._session.get(
                robots_url,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
            if resp.status_code in (401, 403):
                # Forbidden robots.txt -> disallow all
                parser.parse(["User-agent: *", "Disallow: /"])
            elif resp.status_code in (404, 410):
                # Missing robots.txt -> allow all
                parser.parse([])
            elif resp.status_code == 200 and resp.text is not None:
                parser.parse(resp.text.splitlines())
            else:
                # Other status codes (5xx, etc.) -> conservative disallow
                parser.parse(["User-agent: *", "Disallow: /"])
        except Exception:
            # Network or parsing failure -> conservative disallow
            parser.parse(["User-agent: *", "Disallow: /"])

        try:
            delay = parser.crawl_delay(self._user_agent)
            if delay is not None:
                crawl_delay_val = float(delay)
        except Exception:
            crawl_delay_val = None

        self._cache[origin] = (parser, crawl_delay_val)
        return parser, crawl_delay_val

    def is_allowed(self, url: str) -> bool:
        """Check if robots.txt allows user_agent to crawl url."""
        parser, _ = self.get_parser(url)
        return bool(parser.can_fetch(self._user_agent, url))

    def clear(self) -> None:
        """Clear cached robots.txt rules."""
        self._cache.clear()


class Fetcher:
    """Polite HTTP Fetcher with robots.txt enforcement, rate limiting, retries, and conditional headers."""

    def __init__(
        self,
        user_agent: str | None = None,
        min_delay_seconds: float = DEFAULT_MIN_DELAY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        session: requests.Session | None = None,
    ) -> None:
        env_ua = os.getenv("CRAWL_USER_AGENT")
        self.user_agent = user_agent or env_ua or DEFAULT_USER_AGENT
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self.backoff_factor = max(0.0, float(backoff_factor))

        self.session = session or requests.Session()
        self.robots_cache = RobotsCache(
            session=self.session,
            user_agent=self.user_agent,
            timeout=self.timeout_seconds,
        )
        self._last_fetch_times: dict[str, float] = {}

    def get_user_agent(self) -> str:
        """Return active User-Agent string."""
        return self.user_agent

    def is_url_allowed(self, url: str) -> bool:
        """Runtime check if URL is permitted by target host's robots.txt."""
        return self.robots_cache.is_allowed(url)

    def reset_state(self) -> None:
        """Clear cached robots.txt parsers and per-host rate limit timestamps."""
        self.robots_cache.clear()
        self._last_fetch_times.clear()

    def _enforce_politeness_delay(
        self, origin: str, extra_delay: float | None
    ) -> float:
        """Sleep if necessary to respect min_delay_seconds and robots.txt Crawl-delay."""
        required_delay = self.min_delay_seconds
        if extra_delay is not None:
            required_delay = max(required_delay, extra_delay)

        now = time.time()
        last_time = self._last_fetch_times.get(origin, 0.0)
        elapsed = now - last_time
        slept = 0.0

        if elapsed < required_delay and required_delay > 0.0:
            slept = required_delay - elapsed
            time.sleep(slept)

        self._last_fetch_times[origin] = time.time()
        return slept

    def fetch(
        self,
        url: str,
        etag: str | None = None,
        last_modified: str | None = None,
        raise_on_disallowed: bool = False,
        enforce_delay: bool = True,
    ) -> FetchResult:
        """Fetch URL with robots.txt check, politeness delay, conditional headers, and retries."""
        now_iso = datetime.now(timezone.utc).isoformat()
        parsed = urllib.parse.urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            return FetchResult(
                url=url,
                status=FetchStatus.INVALID_URL,
                status_code=0,
                is_success=False,
                is_modified=True,
                content=b"",
                text="",
                headers={},
                etag=None,
                last_modified=None,
                fetched_at=now_iso,
                elapsed_seconds=0.0,
                error_message=f"Invalid URL structure: '{url}'",
                robots_allowed=False,
                retries_used=0,
            )

        origin = f"{parsed.scheme}://{parsed.netloc}"

        # 1. Check robots.txt
        parser, crawl_delay = self.robots_cache.get_parser(url)
        allowed = bool(parser.can_fetch(self.user_agent, url))

        if not allowed:
            if raise_on_disallowed:
                raise RobotsDisallowedError(url, self.user_agent)
            return FetchResult(
                url=url,
                status=FetchStatus.DISALLOWED_BY_ROBOTS,
                status_code=403,
                is_success=False,
                is_modified=True,
                content=b"",
                text="",
                headers={},
                etag=None,
                last_modified=None,
                fetched_at=now_iso,
                elapsed_seconds=0.0,
                error_message=f"Disallowed by robots.txt for User-Agent '{self.user_agent}'",
                robots_allowed=False,
                retries_used=0,
            )

        # 2. Enforce politeness rate limiting
        if enforce_delay:
            self._enforce_politeness_delay(origin, crawl_delay)

        # 3. Build HTTP request headers
        req_headers = {"User-Agent": self.user_agent}
        if etag:
            req_headers["If-None-Match"] = etag
        if last_modified:
            req_headers["If-Modified-Since"] = last_modified

        # 4. Perform HTTP GET with bounded retry policy
        start_time = time.time()
        retries_used = 0
        last_error_msg: str | None = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                retries_used += 1
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

            try:
                response = self.session.get(
                    url,
                    headers=req_headers,
                    timeout=self.timeout_seconds,
                    allow_redirects=True,
                )
                elapsed = time.time() - start_time
                resp_headers = dict(response.headers)
                resp_etag = resp_headers.get("ETag") or resp_headers.get("etag")
                resp_last_mod = resp_headers.get("Last-Modified") or resp_headers.get(
                    "last-modified"
                )

                if response.status_code == 304:
                    return FetchResult(
                        url=url,
                        status=FetchStatus.NOT_MODIFIED,
                        status_code=304,
                        is_success=True,
                        is_modified=False,
                        content=b"",
                        text="",
                        headers=resp_headers,
                        etag=resp_etag or etag,
                        last_modified=resp_last_mod or last_modified,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=elapsed,
                        error_message=None,
                        robots_allowed=True,
                        retries_used=retries_used,
                    )

                if 200 <= response.status_code < 300:
                    return FetchResult(
                        url=url,
                        status=FetchStatus.SUCCESS,
                        status_code=response.status_code,
                        is_success=True,
                        is_modified=True,
                        content=response.content,
                        text=response.text,
                        headers=resp_headers,
                        etag=resp_etag,
                        last_modified=resp_last_mod,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=elapsed,
                        error_message=None,
                        robots_allowed=True,
                        retries_used=retries_used,
                    )

                # Server error 5xx: retry if attempts remaining
                if response.status_code >= 500 and attempt < self.max_retries:
                    last_error_msg = (
                        f"HTTP {response.status_code}: {response.reason}"
                    )
                    continue

                # 4xx or final failed 5xx attempt: return non-success result
                return FetchResult(
                    url=url,
                    status=FetchStatus.HTTP_ERROR,
                    status_code=response.status_code,
                    is_success=False,
                    is_modified=True,
                    content=response.content,
                    text=response.text,
                    headers=resp_headers,
                    etag=resp_etag,
                    last_modified=resp_last_mod,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=elapsed,
                    error_message=f"HTTP {response.status_code}: {response.reason}",
                    robots_allowed=True,
                    retries_used=retries_used,
                )

            except requests.Timeout as err:
                last_error_msg = f"Timeout error: {err}"
                if attempt < self.max_retries:
                    continue
                elapsed = time.time() - start_time
                return FetchResult(
                    url=url,
                    status=FetchStatus.TIMEOUT,
                    status_code=0,
                    is_success=False,
                    is_modified=True,
                    content=b"",
                    text="",
                    headers={},
                    etag=None,
                    last_modified=None,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=elapsed,
                    error_message=last_error_msg,
                    robots_allowed=True,
                    retries_used=retries_used,
                )
            except requests.RequestException as err:
                last_error_msg = f"Network error: {err}"
                if attempt < self.max_retries:
                    continue
                elapsed = time.time() - start_time
                return FetchResult(
                    url=url,
                    status=FetchStatus.NETWORK_ERROR,
                    status_code=0,
                    is_success=False,
                    is_modified=True,
                    content=b"",
                    text="",
                    headers={},
                    etag=None,
                    last_modified=None,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=elapsed,
                    error_message=last_error_msg,
                    robots_allowed=True,
                    retries_used=retries_used,
                )

        # Fallback if loop finishes without returning
        elapsed = time.time() - start_time
        return FetchResult(
            url=url,
            status=FetchStatus.HTTP_ERROR,
            status_code=0,
            is_success=False,
            is_modified=True,
            content=b"",
            text="",
            headers={},
            etag=None,
            last_modified=None,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=elapsed,
            error_message=last_error_msg or "Unknown error after retries",
            robots_allowed=True,
            retries_used=retries_used,
        )
