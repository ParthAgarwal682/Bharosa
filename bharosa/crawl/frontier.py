"""Frontier module for Bharosa crawler with priority queue and politeness rate limiting.

Provides deterministic task ordering, per-host politeness delay tracking,
and priority computation based on configured importance and staleness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import time
import urllib.parse

DEFAULT_MIN_HOST_DELAY = 3.0


@dataclass
class CrawlTask:
    """Represents a URL task in the crawl frontier.

    Attributes:
        url: Normalized target URL.
        priority: Priority score (higher value = fetched sooner).
        base_importance: Configured importance weight (default 1.0).
        staleness: Time elapsed since last crawl (default 0.0).
        added_at: Timestamp when task was added to frontier.
        sequence_id: Sequence ID for deterministic tie-breaking.
    """

    url: str
    priority: float
    base_importance: float = 1.0
    staleness: float = 0.0
    added_at: float = field(default_factory=time.time)
    sequence_id: int = 0

    @property
    def host(self) -> str:
        """Extract scheme://netloc origin host key."""
        parsed = urllib.parse.urlparse(self.url)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or "unknown"
        return f"{scheme}://{netloc}"


class CrawlFrontier:
    """Priority queue frontier with deterministic tie-breaking and per-host politeness."""

    def __init__(self, min_host_delay: float = DEFAULT_MIN_HOST_DELAY) -> None:
        self.min_host_delay = max(0.0, float(min_host_delay))
        # Heap entries: (-priority, sequence_id, url, CrawlTask)
        self._heap: list[tuple[float, int, str, CrawlTask]] = []
        self._seq_counter: int = 0
        self._seen_urls: set[str] = set()
        self._last_fetched_times: dict[str, float] = {}

    def add_url(
        self,
        url: str,
        base_importance: float | None = None,
        staleness: float | None = None,
        priority: float | None = None,
        added_at: float | None = None,
    ) -> bool:
        """Add a URL to the frontier if not already present.

        Args:
            url: Target URL to crawl.
            base_importance: Configured importance weight (if provided).
            staleness: Elapsed staleness value (if provided).
            priority: Explicit priority score override. If None, combines base_importance + staleness.
            added_at: Fixed creation timestamp for testing/reproducibility.

        Returns:
            True if URL was newly added, False if duplicate.
        """
        if url in self._seen_urls:
            return False

        imp = base_importance if base_importance is not None else 1.0
        stale = staleness if staleness is not None else 0.0

        if priority is None:
            computed_priority = imp + stale
        else:
            computed_priority = float(priority)

        self._seq_counter += 1
        t_added = added_at if added_at is not None else time.time()
        task = CrawlTask(
            url=url,
            priority=computed_priority,
            base_importance=imp,
            staleness=stale,
            added_at=t_added,
            sequence_id=self._seq_counter,
        )

        self._seen_urls.add(url)
        # Store in heap with -priority for max-priority behavior
        heapq.heappush(self._heap, (-computed_priority, self._seq_counter, url, task))
        return True

    def get_next_task(self, now: float | None = None) -> CrawlTask | None:
        """Retrieve the highest priority task whose host is not in politeness cooldown.

        Args:
            now: Current timestamp (defaults to time.time()).

        Returns:
            The next ready CrawlTask, or None if frontier is empty or all tasks are in host cooldown.
        """
        if not self._heap:
            return None

        current_time = now if now is not None else time.time()

        skipped: list[tuple[float, int, str, CrawlTask]] = []
        selected_task: CrawlTask | None = None

        while self._heap:
            neg_pri, seq, url, task = heapq.heappop(self._heap)
            host = task.host
            last_fetch = self._last_fetched_times.get(host, 0.0)
            elapsed = current_time - last_fetch

            if elapsed >= self.min_host_delay or last_fetch == 0.0:
                selected_task = task
                self._last_fetched_times[host] = current_time
                break
            else:
                skipped.append((neg_pri, seq, url, task))

        # Re-insert any tasks that were skipped due to host cooldown
        for item in skipped:
            heapq.heappush(self._heap, item)

        return selected_task

    def mark_fetched(self, url: str, fetched_at: float | None = None) -> None:
        """Record an explicit fetch timestamp for host politeness tracking."""
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or "unknown"
        origin = f"{scheme}://{netloc}"
        self._last_fetched_times[origin] = (
            fetched_at if fetched_at is not None else time.time()
        )

    def is_host_ready(self, url_or_host: str, now: float | None = None) -> bool:
        """Check if a host is eligible for fetching based on min_host_delay."""
        current_time = now if now is not None else time.time()
        if "://" in url_or_host:
            parsed = urllib.parse.urlparse(url_or_host)
            scheme = parsed.scheme or "http"
            netloc = parsed.netloc or "unknown"
            origin = f"{scheme}://{netloc}"
        else:
            origin = url_or_host

        last_fetch = self._last_fetched_times.get(origin, 0.0)
        return (current_time - last_fetch) >= self.min_host_delay or last_fetch == 0.0

    def time_until_any_ready(self, now: float | None = None) -> float | None:
        """Return seconds until the next host in frontier becomes ready, or 0.0 if any is ready."""
        if not self._heap:
            return None

        current_time = now if now is not None else time.time()
        min_wait: float | None = None

        for _, _, _, task in self._heap:
            host = task.host
            last_fetch = self._last_fetched_times.get(host, 0.0)
            if last_fetch == 0.0:
                return 0.0
            wait_time = self.min_host_delay - (current_time - last_fetch)
            if wait_time <= 0.0:
                return 0.0
            if min_wait is None or wait_time < min_wait:
                min_wait = wait_time

        return min_wait

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        """Clear frontier queue and tracked state."""
        self._heap.clear()
        self._seen_urls.clear()
        self._last_fetched_times.clear()
        self._seq_counter = 0
