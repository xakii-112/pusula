"""SEC EDGAR client: XBRL company facts, filing index and entity search. No key, no fee.

``data.sec.gov`` (company facts, submissions) and ``efts.sec.gov`` (entity search) accept a
descriptive User-Agent without personal contact details; ``www.sec.gov`` does not (HTTP 403
"undeclared automated tool", measured 2026-10-24), so nothing here uses it. Requests are
throttled below SEC's fair-access limit of 10 per second and retried on 429/5xx, honouring
``Retry-After`` in both of its forms (seconds or an HTTP date). Every response is returned
with the SHA-256 of its raw bytes, which the fundamentals store records as provenance.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import Any

from pusula.core.errors import DataError

USER_AGENT = "pusula-research/0.1 (point-in-time fundamentals research)"
DATA_HOST = "https://data.sec.gov"
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_RETRY = (429, 500, 502, 503, 504)


def retry_after_seconds(value: str | None, now: dt.datetime | None = None) -> float | None:
    """``Retry-After`` as seconds to wait; ``None`` if absent or unreadable."""
    if not value:
        return None
    v = value.strip()
    if v.isdigit():
        return float(v)
    try:
        when = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    ref = now or dt.datetime.now(dt.UTC)
    return max(0.0, (when - ref).total_seconds())


class SecClient:
    def __init__(
        self,
        transport: object | None = None,
        *,
        rate: float = 8.0,
        retries: int = 4,
        user_agent: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        import httpx

        ua = user_agent or os.environ.get("PUSULA_SEC_USER_AGENT") or USER_AGENT
        self._client = httpx.Client(
            timeout=60.0,
            transport=transport,  # type: ignore[arg-type]
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
        )
        self._interval = 1.0 / rate
        self._retries = retries
        self._sleep = sleep
        self._clock = clock
        self._last = -1e9
        self.requests = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SecClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        wait = self._last + self._interval - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last = self._clock()

    def get_bytes(self, url: str, params: dict[str, str] | None = None) -> bytes | None:
        """Body of a 200 response; ``None`` for 404 (no such entity); raises otherwise."""
        last = "?"
        for attempt in range(self._retries + 1):
            self._throttle()
            self.requests += 1
            r = self._client.get(url, params=params)
            if r.status_code == 200:
                return bytes(r.content)
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}"
            if r.status_code not in _RETRY:
                break
            hint = retry_after_seconds(r.headers.get("Retry-After"))
            self._sleep(hint if hint is not None else 2.0 * (2**attempt))
        raise DataError(f"SEC GET {url}: {last}")

    def get_json(self, url: str, params: dict[str, str] | None = None) -> tuple[Any, str] | None:
        body = self.get_bytes(url, params)
        if body is None:
            return None
        return json.loads(body), hashlib.sha256(body).hexdigest()

    def companyfacts(self, cik: int) -> tuple[dict[str, Any], str] | None:
        return self.get_json(f"{DATA_HOST}/api/xbrl/companyfacts/CIK{cik:010d}.json")

    def submissions(self, cik: int, *, archive: bool = True) -> tuple[dict[str, Any], str] | None:
        """Filing index with the archive pages merged into ``filings.recent`` (the recent block
        holds only the last ~1000 filings, a year or two for a company with many Form 4s).
        ``archive=False`` reads the recent block only (enough to detect new filings)."""
        got = self.get_json(f"{DATA_HOST}/submissions/CIK{cik:010d}.json")
        if got is None:
            return None
        sub, digest = got
        h = hashlib.sha256(digest.encode())
        recent = sub.get("filings", {}).get("recent", {})
        for page in sub.get("filings", {}).get("files", []) if archive else []:
            extra = self.get_json(f"{DATA_HOST}/submissions/{page['name']}")
            if extra is None:
                raise DataError(f"submissions archive page missing: {page['name']}")
            block, d = extra
            h.update(d.encode())
            for k, v in block.items():
                if isinstance(v, list) and isinstance(recent.get(k), list):
                    recent[k].extend(v)
        return sub, h.hexdigest()

    def search_entities(self, name: str) -> list[tuple[int, str]]:
        """EDGAR company-name search: ``[(cik, entity name), ...]`` best match first."""
        got = self.get_json(SEARCH_URL, {"keysTyped": name})
        if got is None:
            return []
        hits = got[0].get("hits", {}).get("hits", [])
        out = []
        for h in hits:
            try:
                out.append((int(h["_id"]), str(h["_source"]["entity"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out
