"""URL normalization and content deduplication for Bharosa crawler.

Provides URL canonicalisation (scheme/host lowercasing, fragment stripping,
tracking query parameter filtering, and parameter sorting), stable SHA-256
content hashing, and duplicate detection tracking.
"""

from __future__ import annotations

import hashlib
import urllib.parse

# Documented block list of analytics / tracking parameters to strip during URL normalization.
# Conservative: Only includes known tracking params. Content parameters (page, id, state, lang) are preserved.
DEFAULT_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "msclkid",
        "ref",
        "ref_src",
        "ref_url",
        "fb_action_ids",
        "fb_action_types",
        "mc_eid",
        "_hsenc",
        "_hsmi",
    }
)


def normalize_url(
    url: str,
    tracking_params: frozenset[str] | set[str] | None = None,
    strip_trailing_slash: bool = True,
) -> str:
    """Normalize a URL to its canonical form.

    Steps:
    1. Lowercase scheme and netloc (host).
    2. Remove default port numbers (80 for http, 443 for https).
    3. Strip fragment (#...).
    4. Collapse multiple consecutive slashes and handle trailing slash.
    5. Filter out tracking query parameters from specified block list.
    6. Sort remaining query parameters deterministically.

    Args:
        url: Raw URL string.
        tracking_params: Block list of query parameter names to remove. Defaults to DEFAULT_TRACKING_PARAMS.
        strip_trailing_slash: If True, removes trailing slash for non-root paths (e.g. /path/ -> /path).

    Returns:
        Canonical normalized URL string.
    """
    if not url or not isinstance(url, str):
        return ""

    cleaned_url = url.strip()
    parsed = urllib.parse.urlparse(cleaned_url)

    scheme = (parsed.scheme or "http").lower()
    netloc = (parsed.netloc or "").lower()

    if not netloc:
        return cleaned_url

    # Remove default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize path
    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")

    if strip_trailing_slash and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Normalize query parameters
    block_set = (
        tracking_params if tracking_params is not None else DEFAULT_TRACKING_PARAMS
    )
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)

    filtered_params = [
        (k, v) for k, v in query_params if k.lower() not in block_set
    ]
    # Sort query parameters by key, then value for deterministic URL string
    filtered_params.sort(key=lambda item: (item[0], item[1]))

    new_query = urllib.parse.urlencode(filtered_params, doseq=True)

    # Reconstruct normalized URL without fragment
    normalized = urllib.parse.urlunparse(
        (scheme, netloc, path, parsed.params, new_query, "")
    )
    return normalized


def compute_content_hash(content: str | bytes) -> str:
    """Compute a stable 64-character SHA-256 hex hash of page content."""
    if isinstance(content, str):
        raw_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw_bytes = content
    else:
        raise TypeError(f"Content must be str or bytes, got {type(content).__name__}")

    return hashlib.sha256(raw_bytes).hexdigest()


class DedupStore:
    """In-memory store for tracking seen URLs and content hashes."""

    def __init__(self) -> None:
        self._seen_urls: set[str] = set()
        self._seen_content_hashes: dict[str, str] = {}  # hash -> first seen normalized url

    def is_url_seen(self, url: str) -> bool:
        """Check if normalized URL has been seen."""
        norm = normalize_url(url)
        return norm in self._seen_urls

    def mark_url_seen(self, url: str) -> str:
        """Mark URL as seen. Returns the normalized URL."""
        norm = normalize_url(url)
        self._seen_urls.add(norm)
        return norm

    def is_content_seen(self, content_hash: str) -> bool:
        """Check if exact content hash has been seen."""
        return content_hash in self._seen_content_hashes

    def mark_content_seen(self, content_hash: str, url: str = "") -> bool:
        """Record a content hash. Returns True if newly seen, False if already existed."""
        if content_hash in self._seen_content_hashes:
            return False
        norm_url = normalize_url(url) if url else ""
        self._seen_content_hashes[content_hash] = norm_url
        return True

    def get_url_for_content(self, content_hash: str) -> str | None:
        """Return the first seen URL associated with a content hash, or None."""
        return self._seen_content_hashes.get(content_hash)

    def __len__(self) -> int:
        return len(self._seen_urls)

    def clear(self) -> None:
        """Clear seen URLs and content hashes."""
        self._seen_urls.clear()
        self._seen_content_hashes.clear()
