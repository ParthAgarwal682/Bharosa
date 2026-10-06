"""Minimal Live Crawl Runner for Bharosa.

Consumes team-provided seed configurations (e.g. config/seeds.yaml), enforces
robots.txt checking, respects host politeness delays, normalizes and deduplicates URLs,
and persists metadata & content references into the SQLite CrawlDB.

LIVE CRAWL DISCLAIMER:
This runner produces LIVE crawl evidence only. It never generates or incorporates
synthetic/simulated freshness evaluation results.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Sequence
import urllib.parse

import yaml

# Ensure parent path in sys.path if run as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bharosa.crawl.db import CrawlDB, PageRecord
from bharosa.crawl.dedup import DedupStore, compute_content_hash, normalize_url
from bharosa.crawl.fetcher import Fetcher, FetchStatus
from bharosa.crawl.frontier import CrawlFrontier
from bharosa.crawl.scheduler import AdaptiveScheduler


def load_seeds(seed_file_path: str) -> list[dict[str, Any]]:
    """Load seed configuration from YAML file.

    Returns:
        List of seed dicts: [{'url': '...', 'g_score': 1.0, ...}, ...]

    Raises:
        FileNotFoundError: If seed_file_path does not exist.
        ValueError: If seed_file_path is invalid or contains no seed URLs.
    """
    if not os.path.exists(seed_file_path):
        raise FileNotFoundError(
            f"Seed configuration file not found at '{seed_file_path}'.\n"
            "SETUP REQUIRED: Please create 'config/seeds.yaml' containing team-approved seed URLs.\n"
            "Example format:\n"
            "seeds:\n"
            "  - url: '<TEAM_APPROVED_URL>'\n"
            "    g_score: 1.0\n"
        )

    with open(seed_file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict) or "seeds" not in data:
        raise ValueError(
            f"Invalid seed configuration in '{seed_file_path}'.\n"
            "Expected top-level 'seeds' key containing a list of seed entries."
        )

    seeds = data["seeds"]
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(
            f"No seed entries found in '{seed_file_path}'.\n"
            "Please populate the 'seeds' list with team-approved URLs before running the crawler."
        )

    return seeds


class CrawlRunner:
    """Orchestrates polite live crawling over team-provided seeds."""

    def __init__(
        self,
        db_path: str = "data/bharosa.db",
        raw_data_dir: str = "data/raw",
        min_delay_seconds: float = 3.0,
        user_agent: str | None = None,
        max_pages: int = 100,
        verbose: bool = True,
    ) -> None:
        self.db_path = db_path
        self.raw_data_dir = raw_data_dir
        self.min_delay_seconds = min_delay_seconds
        self.max_pages = max_pages
        self.verbose = verbose

        self.fetcher = Fetcher(
            user_agent=user_agent,
            min_delay_seconds=min_delay_seconds,
        )
        self.frontier = CrawlFrontier(min_host_delay=min_delay_seconds)
        self.dedup_store = DedupStore()
        self.db = CrawlDB(db_path=self.db_path)
        self.scheduler = AdaptiveScheduler()

    def log(self, message: str) -> None:
        """Print verbose log message if verbose logging is enabled."""
        if self.verbose:
            print(f"[LIVE CRAWLER] {message}")

    def enqueue_seeds(self, seeds: list[dict[str, Any] | str]) -> int:
        """Enqueue loaded seeds into frontier."""
        added_count = 0
        for seed_entry in seeds:
            raw_url = (
                seed_entry.get("url")
                if isinstance(seed_entry, dict)
                else str(seed_entry)
            )
            g_score = (
                seed_entry.get("g_score", 1.0)
                if isinstance(seed_entry, dict)
                else 1.0
            )

            if not raw_url:
                continue

            norm_url = normalize_url(raw_url)
            added = self.frontier.add_url(norm_url, base_importance=g_score)
            if added:
                added_count += 1
                self.log(f"Enqueued seed: {norm_url} (g_score: {g_score})")
            else:
                self.log(f"Skipped duplicate seed: {norm_url}")

        return added_count

    def run(self) -> dict[str, Any]:
        """Execute live crawl loop over enqueued tasks until max_pages or frontier empty.

        Returns:
            Summary dictionary of live crawl results.
        """
        os.makedirs(self.raw_data_dir, exist_ok=True)

        pages_crawled = 0
        pages_skipped_robots = 0
        pages_failed = 0

        self.log(f"Starting live crawl loop (max_pages={self.max_pages})...")

        while pages_crawled < self.max_pages and len(self.frontier) > 0:
            now = time.time()
            task = self.frontier.get_next_task(now=now)

            if task is None:
                wait_seconds = self.frontier.time_until_any_ready(now=now)
                if wait_seconds is not None and wait_seconds > 0.0:
                    self.log(
                        f"All host queues in politeness cooldown. Sleeping {wait_seconds:.2f}s..."
                    )
                    time.sleep(wait_seconds)
                continue

            url = task.url
            self.log("--------------------------------------------------")
            self.log(f"Dequeued task: {url} (priority: {task.priority:.2f})")

            # 1. Check robots.txt decision at runtime
            allowed = self.fetcher.is_url_allowed(url)
            self.log(f"Robots.txt decision for {url}: ALLOWED={allowed}")

            if not allowed:
                pages_skipped_robots += 1
                self.log(f"Skipping disallowed URL: {url}")
                continue

            # 2. Perform Fetch with politeness rate limiting and conditional headers
            existing_page = self.db.get_page(url)
            etag = existing_page.etag if existing_page else None
            last_modified = existing_page.last_modified if existing_page else None

            self.log(f"Fetching URL: {url} (conditional etag={etag})...")
            result = self.fetcher.fetch(
                url=url,
                etag=etag,
                last_modified=last_modified,
                raise_on_disallowed=False,
                enforce_delay=True,
            )

            self.log(
                f"Fetch result: status={result.status.value}, http_code={result.status_code}, "
                f"elapsed={result.elapsed_seconds:.2f}s, retries={result.retries_used}"
            )

            if not result.is_success:
                pages_failed += 1
                self.log(f"Fetch failed: {result.error_message}")
                continue

            # 3. Handle response
            if result.status == FetchStatus.NOT_MODIFIED:
                self.log(f"Page content not modified (HTTP 304): {url}")
                self.scheduler.update_interval(url, changed=False)
                pages_crawled += 1
                continue

            content_hash = compute_content_hash(result.content)
            self.dedup_store.mark_url_seen(url)
            self.dedup_store.mark_content_seen(content_hash, url=url)

            filename = f"{content_hash[:16]}.html"
            text_path = os.path.join(self.raw_data_dir, filename)
            with open(text_path, "wb") as f:
                f.write(result.content)

            page_rec, changed = self.db.upsert_page(
                url=url,
                domain=urllib.parse.urlparse(url).netloc,
                fetched_at=result.fetched_at,
                content_hash=content_hash,
                g_score=task.base_importance,
                etag=result.etag,
                last_modified=result.last_modified,
                status_code=result.status_code,
                text_path=text_path,
            )

            self.scheduler.update_interval(url, changed=changed)
            pages_crawled += 1
            self.log(
                f"Persisted page: {url} (content_hash={content_hash[:8]}, "
                f"changed={changed}, text_path={text_path})"
            )

        summary = {
            "crawl_type": "LIVE",
            "pages_crawled": pages_crawled,
            "pages_skipped_robots": pages_skipped_robots,
            "pages_failed": pages_failed,
            "remaining_frontier_size": len(self.frontier),
            "db_path": self.db_path,
        }
        self.log(f"Live crawl complete: {summary}")
        return summary

    def close(self) -> None:
        """Close DB connection."""
        self.db.close()


def main() -> None:
    """CLI entrypoint for minimal live crawl runner."""
    parser = argparse.ArgumentParser(description="Bharosa Minimal Live Crawl Runner")
    parser.add_argument(
        "--seeds",
        default="config/seeds.yaml",
        help="Path to YAML seed configuration file (default: config/seeds.yaml)",
    )
    parser.add_argument(
        "--db",
        default="data/bharosa.db",
        help="Path to SQLite database (default: data/bharosa.db)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages to fetch in this run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose crawling logs",
    )
    args = parser.parse_args()

    try:
        seeds = load_seeds(args.seeds)
    except (FileNotFoundError, ValueError) as err:
        print("=" * 70)
        print("LIVE CRAWL RUNNER - SETUP REQUIRED")
        print("=" * 70)
        print(str(err))
        print("=" * 70)
        sys.exit(1)

    runner = CrawlRunner(
        db_path=args.db,
        max_pages=args.max_pages,
        verbose=args.verbose,
    )

    try:
        runner.enqueue_seeds(seeds)
        summary = runner.run()
        print("\nLive Crawl Summary:", summary)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
