"""Unit tests for the Bharosa crawler scheduler module.

Tests FixedIntervalScheduler and AdaptiveScheduler behavior, interval adjustments,
min/max bounds, and common interface compliance.
"""

from __future__ import annotations

import pytest

from bharosa.crawl.scheduler import (
    AdaptiveScheduler,
    BaseScheduler,
    FixedIntervalScheduler,
)


def test_fixed_interval_scheduler() -> None:
    """Test FixedIntervalScheduler returns static intervals regardless of changes."""
    sched = FixedIntervalScheduler(default_interval=3600.0)
    url = "http://example.com/page"

    assert sched.get_interval(url) == 3600.0
    assert sched.update_interval(url, changed=True) == 3600.0
    assert sched.update_interval(url, changed=False) == 3600.0

    last_crawled = 1000.0
    assert sched.get_next_due_time(url, last_crawled) == 4600.0
    assert not sched.is_due(url, last_crawled, now=4599.0)
    assert sched.is_due(url, last_crawled, now=4600.0)


def test_fixed_interval_custom_mapping() -> None:
    """Test FixedIntervalScheduler respects per-URL custom initial mapping."""
    sched = FixedIntervalScheduler(
        default_interval=3600.0,
        custom_intervals={"http://custom.com": 1800.0},
    )

    assert sched.get_interval("http://custom.com") == 1800.0
    assert sched.get_interval("http://other.com") == 3600.0


def test_adaptive_scheduler_shrink_on_change() -> None:
    """Test AdaptiveScheduler reduces interval when content changes until min_interval."""
    sched = AdaptiveScheduler(
        default_interval=3600.0,
        min_interval=300.0,
        max_interval=86400.0,
        shrink_factor=0.5,
    )
    url = "http://example.com/dynamic"

    assert sched.get_interval(url) == 3600.0

    # 1st change: 3600 * 0.5 = 1800
    inv1 = sched.update_interval(url, changed=True)
    assert inv1 == 1800.0

    # 2nd change: 1800 * 0.5 = 900
    inv2 = sched.update_interval(url, changed=True)
    assert inv2 == 900.0

    # 3rd change: 900 * 0.5 = 450
    inv3 = sched.update_interval(url, changed=True)
    assert inv3 == 450.0

    # 4th change: 450 * 0.5 = 225 -> bounded to min_interval 300.0
    inv4 = sched.update_interval(url, changed=True)
    assert inv4 == 300.0


def test_adaptive_scheduler_growth_on_no_change() -> None:
    """Test AdaptiveScheduler expands interval when content is unchanged until max_interval."""
    sched = AdaptiveScheduler(
        default_interval=1000.0,
        min_interval=100.0,
        max_interval=2000.0,
        growth_factor=1.5,
    )
    url = "http://example.com/static"

    # 1st no-change: 1000 * 1.5 = 1500
    inv1 = sched.update_interval(url, changed=False)
    assert inv1 == 1500.0

    # 2nd no-change: 1500 * 1.5 = 2250 -> bounded to max_interval 2000.0
    inv2 = sched.update_interval(url, changed=False)
    assert inv2 == 2000.0


def test_adaptive_scheduler_per_url_isolation() -> None:
    """Test AdaptiveScheduler tracks intervals for different URLs independently."""
    sched = AdaptiveScheduler(default_interval=1000.0, shrink_factor=0.5)

    url_a = "http://example.com/a"
    url_b = "http://example.com/b"

    sched.update_interval(url_a, changed=True)  # becomes 500
    assert sched.get_interval(url_a) == 500.0
    assert sched.get_interval(url_b) == 1000.0


def test_common_interface_type_check() -> None:
    """Test both schedulers inherit from BaseScheduler and share signature."""
    s1: BaseScheduler = FixedIntervalScheduler()
    s2: BaseScheduler = AdaptiveScheduler()

    url = "http://example.com"

    assert isinstance(s1.get_interval(url), float)
    assert isinstance(s2.get_interval(url), float)
    assert isinstance(s1.update_interval(url, True), float)
    assert isinstance(s2.update_interval(url, True), float)
