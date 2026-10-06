"""Scheduler module for Bharosa crawler managing recrawl intervals.

Provides a unified interface for fixed-interval and adaptive-interval crawling
schedulers.

Behavior Note:
The AdaptiveScheduler adjusts recrawl intervals dynamically (halving interval
upon detected content change, increasing by factor 1.5 when unchanged, bounded by
min and max bounds). Freshness improvements are subject to empirical evaluation in
eval/freshness_eval.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

DEFAULT_INTERVAL_SECONDS = 3600.0  # 1 hour
DEFAULT_MIN_INTERVAL_SECONDS = 300.0  # 5 minutes
DEFAULT_MAX_INTERVAL_SECONDS = 86400.0  # 24 hours
DEFAULT_SHRINK_FACTOR = 0.5  # Halve interval on change
DEFAULT_GROWTH_FACTOR = 1.5  # Expand interval by 1.5x on no change


class BaseScheduler(ABC):
    """Abstract base class establishing common scheduler interface."""

    @abstractmethod
    def get_interval(self, url: str) -> float:
        """Return the current recrawl interval (in seconds) for a URL."""
        pass

    @abstractmethod
    def update_interval(self, url: str, changed: bool) -> float:
        """Update recrawl interval based on whether page content changed.

        Args:
            url: Target URL.
            changed: True if page content changed since last fetch, False if unchanged.

        Returns:
            The updated recrawl interval (in seconds).
        """
        pass

    def get_next_due_time(self, url: str, last_crawled_at: float) -> float:
        """Calculate the next timestamp when recrawl is due."""
        return last_crawled_at + self.get_interval(url)

    def is_due(self, url: str, last_crawled_at: float, now: float) -> bool:
        """Check if URL is due for recrawling at timestamp `now`."""
        return now >= self.get_next_due_time(url, last_crawled_at)


class FixedIntervalScheduler(BaseScheduler):
    """Static recrawl scheduler returning a constant interval for all URLs."""

    def __init__(
        self,
        default_interval: float = DEFAULT_INTERVAL_SECONDS,
        custom_intervals: Mapping[str, float] | None = None,
    ) -> None:
        self.default_interval = max(1.0, float(default_interval))
        self._custom_intervals = (
            dict(custom_intervals) if custom_intervals is not None else {}
        )

    def get_interval(self, url: str) -> float:
        """Return constant configured interval for URL."""
        return self._custom_intervals.get(url, self.default_interval)

    def update_interval(self, url: str, changed: bool) -> float:
        """Fixed interval scheduler does not alter interval on change/no-change."""
        return self.get_interval(url)


class AdaptiveScheduler(BaseScheduler):
    """Adaptive recrawl scheduler updating intervals based on observed page changes.

    Adjusts recrawl interval after every fetch attempt:
    - If changed is True: interval = max(min_interval, current_interval * shrink_factor)
    - If changed is False: interval = min(max_interval, current_interval * growth_factor)
    """

    def __init__(
        self,
        default_interval: float = DEFAULT_INTERVAL_SECONDS,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_interval: float = DEFAULT_MAX_INTERVAL_SECONDS,
        shrink_factor: float = DEFAULT_SHRINK_FACTOR,
        growth_factor: float = DEFAULT_GROWTH_FACTOR,
    ) -> None:
        self.default_interval = float(default_interval)
        self.min_interval = max(0.1, float(min_interval))
        self.max_interval = max(self.min_interval, float(max_interval))
        self.shrink_factor = max(0.01, float(shrink_factor))
        self.growth_factor = max(1.0, float(growth_factor))

        self._intervals: dict[str, float] = {}

    def get_interval(self, url: str) -> float:
        """Return current interval for URL, defaulting to initial default_interval."""
        return self._intervals.get(url, self.default_interval)

    def update_interval(self, url: str, changed: bool) -> float:
        """Adjust interval based on change observation and enforce min/max bounds."""
        current = self.get_interval(url)

        if changed:
            # Page changed -> crawl sooner
            new_val = current * self.shrink_factor
        else:
            # Page unchanged -> crawl less frequently
            new_val = current * self.growth_factor

        # Enforce bounds
        bounded_val = max(self.min_interval, min(self.max_interval, new_val))
        self._intervals[url] = bounded_val
        return bounded_val

    def reset_url(self, url: str) -> None:
        """Reset interval for a URL back to initial default."""
        self._intervals.pop(url, None)

    def clear(self) -> None:
        """Reset all tracked URL intervals."""
        self._intervals.clear()
