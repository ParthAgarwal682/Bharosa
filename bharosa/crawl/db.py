"""SQLite Database persistence for Bharosa crawler.

Manages persistent storage for crawled pages, historical version snapshots,
and detected content change logs using SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import sqlite3


@dataclass(frozen=True)
class PageRecord:
    """Database record for a crawled page identity and state."""

    url: str
    domain: str
    g_score: float = 0.0
    fetched_at: str = ""
    last_changed_at: str = ""
    content_hash: str = ""
    etag: str | None = None
    last_modified: str | None = None
    status_code: int = 0


@dataclass(frozen=True)
class VersionRecord:
    """Database record for a historical page version snapshot."""

    id: int | None
    url: str
    fetched_at: str
    content_hash: str
    etag: str | None = None
    last_modified: str | None = None
    status_code: int = 200
    text_path: str | None = None


@dataclass(frozen=True)
class ChangeRecord:
    """Database record for a detected content change event."""

    id: int | None
    url: str
    detected_at: str
    old_hash: str | None
    new_hash: str
    diff_summary: str | None = None


class CrawlDB:
    """SQLite storage interface for crawler pages, versions, and change logs."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize database connection and create tables if missing.

        Args:
            db_path: Path to SQLite database file or ':memory:' for transient DB.
        """
        self.db_path = db_path or "data/bharosa.db"
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create pages, versions, and changes tables if they do not exist."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pages (
                    url TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    g_score REAL DEFAULT 0.0,
                    fetched_at TEXT NOT NULL,
                    last_changed_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    etag TEXT,
                    last_modified TEXT,
                    status_code INTEGER DEFAULT 0
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    etag TEXT,
                    last_modified TEXT,
                    status_code INTEGER DEFAULT 200,
                    text_path TEXT,
                    FOREIGN KEY (url) REFERENCES pages(url) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    old_hash TEXT,
                    new_hash TEXT NOT NULL,
                    diff_summary TEXT,
                    FOREIGN KEY (url) REFERENCES pages(url) ON DELETE CASCADE
                );
                """
            )

    def upsert_page(
        self,
        url: str,
        domain: str,
        fetched_at: str,
        content_hash: str,
        g_score: float = 0.0,
        etag: str | None = None,
        last_modified: str | None = None,
        status_code: int = 200,
        text_path: str | None = None,
    ) -> tuple[PageRecord, bool]:
        """Insert or update a page record, record version snapshot, and log change if hash changed.

        Returns:
            Tuple of (PageRecord, changed: bool).
        """
        existing = self.get_page(url)
        now_iso = fetched_at or datetime.now(timezone.utc).isoformat()

        changed = False
        old_hash: str | None = None

        if existing is None:
            changed = True
            last_changed_at = now_iso
        else:
            old_hash = existing.content_hash
            if old_hash != content_hash:
                changed = True
                last_changed_at = now_iso
            else:
                last_changed_at = existing.last_changed_at

        page = PageRecord(
            url=url,
            domain=domain,
            g_score=g_score,
            fetched_at=now_iso,
            last_changed_at=last_changed_at,
            content_hash=content_hash,
            etag=etag,
            last_modified=last_modified,
            status_code=status_code,
        )

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO pages (url, domain, g_score, fetched_at, last_changed_at, content_hash, etag, last_modified, status_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    domain = excluded.domain,
                    g_score = excluded.g_score,
                    fetched_at = excluded.fetched_at,
                    last_changed_at = excluded.last_changed_at,
                    content_hash = excluded.content_hash,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    status_code = excluded.status_code;
                """,
                (
                    page.url,
                    page.domain,
                    page.g_score,
                    page.fetched_at,
                    page.last_changed_at,
                    page.content_hash,
                    page.etag,
                    page.last_modified,
                    page.status_code,
                ),
            )

            # Insert version snapshot
            self._conn.execute(
                """
                INSERT INTO versions (url, fetched_at, content_hash, etag, last_modified, status_code, text_path)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    page.url,
                    page.fetched_at,
                    page.content_hash,
                    page.etag,
                    page.last_modified,
                    page.status_code,
                    text_path,
                ),
            )

            # Log change event if hash changed or initial fetch
            if changed:
                diff = (
                    "Initial page crawl"
                    if old_hash is None
                    else f"Hash updated: {old_hash[:8]} -> {content_hash[:8]}"
                )
                self._conn.execute(
                    """
                    INSERT INTO changes (url, detected_at, old_hash, new_hash, diff_summary)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (page.url, now_iso, old_hash, content_hash, diff),
                )

        return page, changed

    def get_page(self, url: str) -> PageRecord | None:
        """Retrieve a page record by URL."""
        cursor = self._conn.execute("SELECT * FROM pages WHERE url = ?", (url,))
        row = cursor.fetchone()
        if not row:
            return None
        return PageRecord(
            url=row["url"],
            domain=row["domain"],
            g_score=row["g_score"],
            fetched_at=row["fetched_at"],
            last_changed_at=row["last_changed_at"],
            content_hash=row["content_hash"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            status_code=row["status_code"],
        )

    def get_versions(self, url: str) -> list[VersionRecord]:
        """Retrieve all version snapshots for a URL ordered by id ascending."""
        cursor = self._conn.execute(
            "SELECT * FROM versions WHERE url = ? ORDER BY id ASC", (url,)
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                VersionRecord(
                    id=row["id"],
                    url=row["url"],
                    fetched_at=row["fetched_at"],
                    content_hash=row["content_hash"],
                    etag=row["etag"],
                    last_modified=row["last_modified"],
                    status_code=row["status_code"],
                    text_path=row["text_path"],
                )
            )
        return results

    def get_changes(self, url: str | None = None) -> list[ChangeRecord]:
        """Retrieve recorded change events, optionally filtered by URL."""
        if url:
            cursor = self._conn.execute(
                "SELECT * FROM changes WHERE url = ? ORDER BY id ASC", (url,)
            )
        else:
            cursor = self._conn.execute("SELECT * FROM changes ORDER BY id ASC")

        results = []
        for row in cursor.fetchall():
            results.append(
                ChangeRecord(
                    id=row["id"],
                    url=row["url"],
                    detected_at=row["detected_at"],
                    old_hash=row["old_hash"],
                    new_hash=row["new_hash"],
                    diff_summary=row["diff_summary"],
                )
            )
        return results

    def close(self) -> None:
        """Close SQLite database connection."""
        self._conn.close()
