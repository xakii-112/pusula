"""Alpaca market-data REST client (historical daily bars, SIP feed).

* Pagination follows ``next_page_token`` until exhausted; a page is never assumed complete.
* A token-bucket rate limiter keeps below the plan limit; HTTP 429/5xx retry with backoff.
* ``asof`` pins the *entity* a ticker referred to on that date. Alpaca maps renames through
  time (FB -> META) and keeps delisted securities, which is what makes a survivorship-free
  universe possible: a removed index member is requested as of its last day in the index.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import httpx
import pandas as pd

from pusula.core.errors import DataError
from pusula.data.settings import AlpacaCredentials

DATA_URL = "https://data.alpaca.markets"
BAR_COLUMNS = ("open", "high", "low", "close", "volume", "vwap", "trades")
# The free plan refuses SIP data from the last 15 minutes; a one-minute margin for clock skew.
SIP_EMBARGO = dt.timedelta(minutes=16)


def end_param(end: dt.date, now: dt.datetime | None = None) -> str:
    """The ``end`` of a bar request. A bare date means the end of that day, so on the evening of
    a session (when the nightly run asks for that session) it reaches into the embargo and the
    API answers 403 "subscription does not permit querying recent SIP data"; every scheduled
    run failed that way until 2026-10-30. A timestamp before the embargo returns the same bar."""
    cutoff = (now or dt.datetime.now(dt.UTC)) - SIP_EMBARGO
    # 05:00 UTC the next day is at or after midnight in New York, summer and winter
    day_end = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(5), tzinfo=dt.UTC)
    return end.isoformat() if day_end <= cutoff else cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


class RateLimiter:
    """Token bucket: at most ``per_minute`` acquisitions in any rolling minute (on average)."""

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.last = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            time.sleep((1.0 - self.tokens) / self.rate)


@dataclass(frozen=True, slots=True)
class BarRequest:
    symbols: tuple[str, ...]
    start: dt.date
    end: dt.date
    asof: dt.date | None = None
    adjustment: str = "all"
    feed: str = "sip"


class AlpacaClient:
    def __init__(
        self,
        creds: AlpacaCredentials,
        *,
        per_minute: int = 180,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 6,
    ) -> None:
        self._client = httpx.Client(
            base_url=DATA_URL,
            headers={"APCA-API-KEY-ID": creds.key_id, "APCA-API-SECRET-KEY": creds.secret},
            timeout=60.0,
            transport=transport,
        )
        self._limiter = RateLimiter(per_minute)
        self._max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AlpacaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str]) -> dict[str, object]:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                r = self._client.get(path, params=params)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if r.status_code == 200:
                payload = r.json()
                if not isinstance(payload, dict):
                    raise DataError(f"unexpected payload type from {path}")
                return payload
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise DataError(f"alpaca {path} -> HTTP {r.status_code}: {r.text[:200]}")
        raise DataError("unreachable retry loop exit")  # pragma: no cover

    def iter_bar_pages(self, req: BarRequest) -> Iterator[dict[str, list[dict[str, object]]]]:
        params = {
            "symbols": ",".join(req.symbols),
            "timeframe": "1Day",
            "start": req.start.isoformat(),
            "end": end_param(req.end),
            "adjustment": req.adjustment,
            "feed": req.feed,
            "limit": "10000",
            "sort": "asc",
        }
        if req.asof is not None:
            params["asof"] = req.asof.isoformat()
        token: str | None = None
        while True:
            p = dict(params)
            if token:
                p["page_token"] = token
            payload = self._get("/v2/stocks/bars", p)
            bars = payload.get("bars") or {}
            if not isinstance(bars, dict):
                raise DataError("bars payload is not a mapping")
            yield bars
            nxt = payload.get("next_page_token")
            if not nxt:
                return
            token = str(nxt)

    def daily_bars(self, req: BarRequest) -> dict[str, pd.DataFrame]:
        """Daily bars per requested symbol (absent symbols map to an empty frame)."""
        rows: dict[str, list[dict[str, object]]] = {s: [] for s in req.symbols}
        for page in self.iter_bar_pages(req):
            for sym, bars in page.items():
                rows.setdefault(sym, []).extend(bars)
        return {s: bars_to_frame(b) for s, b in rows.items()}

    def trades(
        self,
        symbol: str,
        start: dt.datetime,
        end: dt.datetime,
        feed: str = "sip",
        max_pages: int = 5,
    ) -> pd.DataFrame:
        """Historical consolidated trades in [start, end): price and size, by time."""
        rows: list[dict[str, object]] = []
        token: str | None = None
        for _ in range(max_pages):
            p = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "feed": feed,
                "limit": "10000",
            }
            if token:
                p["page_token"] = token
            payload = self._get(f"/v2/stocks/{symbol}/trades", p)
            got = payload.get("trades") or []
            if not isinstance(got, list):
                raise DataError("trades payload is not a list")
            rows += got
            token = payload.get("next_page_token")  # type: ignore[assignment]
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=["price", "size"], index=pd.DatetimeIndex([], tz="UTC"))
        index = pd.to_datetime([str(r["t"]) for r in rows], utc=True, format="ISO8601")
        frame = pd.DataFrame(
            {"price": [float(r["p"]) for r in rows], "size": [float(r["s"]) for r in rows]},  # type: ignore[arg-type]
            index=index.as_unit("ns"),
        )
        return frame.sort_index()

    def quotes(
        self,
        symbol: str,
        start: dt.datetime,
        end: dt.datetime,
        feed: str = "sip",
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """Historical consolidated quotes (NBBO updates) for one symbol in [start, end)."""
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": feed,
            "limit": "10000",
        }
        rows: list[dict[str, object]] = []
        token: str | None = None
        for _ in range(max_pages):
            p = dict(params)
            if token:
                p["page_token"] = token
            payload = self._get(f"/v2/stocks/{symbol}/quotes", p)
            q = payload.get("quotes") or []
            if not isinstance(q, list):
                raise DataError("quotes payload is not a list")
            rows.extend(q)
            nxt = payload.get("next_page_token")
            if not nxt:
                break
            token = str(nxt)
        if not rows:
            return pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"])
        df = pd.DataFrame(rows)
        idx = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True, format="ISO8601"), name="t")
        # .to_numpy(): building a frame from Series with a *new* index would align on the old
        # RangeIndex and silently produce all-NaN columns.
        return pd.DataFrame(
            {
                "bid": df["bp"].astype(float).to_numpy(),
                "ask": df["ap"].astype(float).to_numpy(),
                "bid_size": df["bs"].astype(float).to_numpy(),
                "ask_size": df["as"].astype(float).to_numpy(),
            },
            index=idx,
        )


def bars_to_frame(bars: Sequence[dict[str, object]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=list(BAR_COLUMNS), index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(bars)
    out = pd.DataFrame(
        {
            "open": df["o"].astype(float),
            "high": df["h"].astype(float),
            "low": df["l"].astype(float),
            "close": df["c"].astype(float),
            "volume": df["v"].astype(float),
            "vwap": df["vw"].astype(float) if "vw" in df else float("nan"),
            "trades": df["n"].astype(float) if "n" in df else float("nan"),
        }
    )
    # Daily bars are stamped at midnight New York time (04:00/05:00Z); the session is the NY date.
    ts = pd.to_datetime(df["t"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    out.index = pd.DatetimeIndex(ts.dt.tz_localize(None).dt.normalize(), name="date")
    if out.index.has_duplicates:
        raise DataError("duplicate session stamps in bar payload")
    return out.sort_index()


def chunked(items: Iterable[str], size: int) -> Iterator[tuple[str, ...]]:
    buf: list[str] = []
    for it in items:
        buf.append(it)
        if len(buf) == size:
            yield tuple(buf)
            buf = []
    if buf:
        yield tuple(buf)
