"""Unit tests for the Bharosa crawl frontier module.

Tests priority queue mechanics, deterministic tie-breaking, duplicate suppression,
and per-host politeness rate limiting.
"""

from __future__ import annotations

import pytest

from bharosa.crawl.frontier import CrawlFrontier, CrawlTask


def test_priority_queue_ordering() -> None:
    """Test higher priority tasks are popped before lower priority tasks."""
    frontier = CrawlFrontier(min_host_delay=0.0)

    frontier.add_url("http://domain1.com/low", priority=1.0)
    frontier.add_url("http://domain2.com/high", priority=10.0)
    frontier.add_url("http://domain3.com/medium", priority=5.0)

    assert len(frontier) == 3

    t1 = frontier.get_next_task(now=0.0)
    assert t1 is not None
    assert t1.url == "http://domain2.com/high"
    assert t1.priority == 10.0

    t2 = frontier.get_next_task(now=0.0)
    assert t2 is not None
    assert t2.url == "http://domain3.com/medium"
    assert t2.priority == 5.0

    t3 = frontier.get_next_task(now=0.0)
    assert t3 is not None
    assert t3.url == "http://domain1.com/low"
    assert t3.priority == 1.0

    assert len(frontier) == 0
    assert frontier.get_next_task(now=0.0) is None


def test_deterministic_tie_breaking() -> None:
    """Test tied priorities break ties deterministically in FIFO sequence order."""
    frontier = CrawlFrontier(min_host_delay=0.0)

    frontier.add_url("http://h1.com/p1", priority=5.0)
    frontier.add_url("http://h2.com/p2", priority=5.0)
    frontier.add_url("http://h3.com/p3", priority=5.0)

    order = []
    while len(frontier) > 0:
        task = frontier.get_next_task(now=0.0)
        assert task is not None
        order.append(task.url)

    assert order == ["http://h1.com/p1", "http://h2.com/p2", "http://h3.com/p3"]


def test_duplicate_url_rejection() -> None:
    """Test duplicate URLs are rejected and not re-inserted."""
    frontier = CrawlFrontier(min_host_delay=0.0)

    assert frontier.add_url("http://example.com/doc", priority=1.0) is True
    assert frontier.add_url("http://example.com/doc", priority=10.0) is False
    assert len(frontier) == 1

    task = frontier.get_next_task(now=0.0)
    assert task is not None
    assert task.url == "http://example.com/doc"
    assert task.priority == 1.0


def test_priority_computation_from_importance_and_staleness() -> None:
    """Test priority computes base_importance + staleness when priority is not explicitly given."""
    frontier = CrawlFrontier(min_host_delay=0.0)

    frontier.add_url(
        "http://example.com/page",
        base_importance=2.5,
        staleness=1.5,
    )

    task = frontier.get_next_task(now=0.0)
    assert task is not None
    assert task.base_importance == 2.5
    assert task.staleness == 1.5
    assert task.priority == 4.0


def test_per_host_politeness_cooldown_skipping() -> None:
    """Test frontier skips hosts currently in politeness cooldown to select ready hosts."""
    # Min delay 3.0 seconds per host
    frontier = CrawlFrontier(min_host_delay=3.0)

    # Add 2 tasks for hostA, 1 task for hostB
    frontier.add_url("http://hostA.com/page1", priority=10.0)
    frontier.add_url("http://hostA.com/page2", priority=9.0)
    frontier.add_url("http://hostB.com/page1", priority=8.0)

    t_base = 100.0

    # Pop 1: hostA page1 (pri 10.0) -> hostA fetched at t=100.0
    t1 = frontier.get_next_task(now=t_base)
    assert t1 is not None
    assert t1.url == "http://hostA.com/page1"

    # Pop 2 at t=101.0: hostA page2 (pri 9.0) is in cooldown (elapsed 1.0 < 3.0)
    # Frontier should skip hostA page2 and pop hostB page1 (pri 8.0) instead!
    t2 = frontier.get_next_task(now=t_base + 1.0)
    assert t2 is not None
    assert t2.url == "http://hostB.com/page1"

    # Pop 3 at t=101.5: Both hostA and hostB are now in cooldown
    # get_next_task returns None because no host is ready
    t3_none = frontier.get_next_task(now=t_base + 1.5)
    assert t3_none is None

    # Time until any ready should be (100.0 + 3.0) - 101.5 = 1.5s
    assert pytest.approx(frontier.time_until_any_ready(now=t_base + 1.5), 0.001) == 1.5

    # Pop 4 at t=103.5: hostA cooldown has expired (103.5 - 100.0 = 3.5 >= 3.0)
    # hostA page2 (pri 9.0) is now eligible
    t4 = frontier.get_next_task(now=t_base + 3.5)
    assert t4 is not None
    assert t4.url == "http://hostA.com/page2"

    assert len(frontier) == 0


def test_mark_fetched_external_tracking() -> None:
    """Test mark_fetched updates host timestamp externally."""
    frontier = CrawlFrontier(min_host_delay=5.0)

    frontier.add_url("http://gov.in/scheme", priority=5.0)

    frontier.mark_fetched("http://gov.in/index", fetched_at=1000.0)

    assert not frontier.is_host_ready("http://gov.in/scheme", now=1002.0)
    assert frontier.is_host_ready("http://gov.in/scheme", now=1006.0)


def test_clear_resets_frontier() -> None:
    """Test clear resets state completely."""
    frontier = CrawlFrontier(min_host_delay=3.0)
    frontier.add_url("http://example.com/p1", priority=1.0)
    frontier.mark_fetched("http://example.com/p1", fetched_at=100.0)

    frontier.clear()
    assert len(frontier) == 0
    assert frontier.get_next_task(now=101.0) is None
    # Seen URLs set should be cleared
    assert frontier.add_url("http://example.com/p1", priority=1.0) is True
