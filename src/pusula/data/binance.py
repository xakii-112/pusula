"""Binance public market data: no key, no account, nothing here can trade.

* :class:`BinanceClient`: exchange information (delisted pairs included, which is what makes a
  survivorship-free crypto universe possible), daily klines, order-book snapshots, aggregate trades.
* :func:`record_book`: the order book as it happened. Binance publishes no history of its book, so
  a queue-position study needs its own recording: the 100 ms diff stream, every aggregate trade and
  a REST snapshot to anchor the diffs (the documented procedure for a local book), written as gzip
  JSON lines with the local receive time. A crash loses at most the last unflushed seconds.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import httpx
import pandas as pd

from pusula.core.errors import DataError

REST = "https://api.binance.com"
WS = "wss://stream.binance.com:9443/stream?streams="
KLINE_LIMIT = 1000
AGG_LIMIT = 1000
MIN_GAP = 0.06  # seconds between requests: well inside the 6,000 weight a minute
RETRY_STATUS = frozenset({418, 429, 500, 502, 503, 504})
RETRIES = 6
FLUSH_SECONDS = 30.0
SNAPSHOT_SECONDS = 1800.0
SNAPSHOT_DEPTH = 1000


class BinanceClient:
    def __init__(
        self, transport: httpx.BaseTransport | None = None, min_gap: float = MIN_GAP
    ) -> None:
        self._http = httpx.Client(base_url=REST, timeout=30.0, transport=transport)
        self._gap = min_gap
        self._last = 0.0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(RETRIES):
            wait = self._last + self._gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == RETRIES - 1:
                    raise DataError(f"binance {path}: {type(exc).__name__}") from exc
                time.sleep(2.0**attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                raise DataError(f"binance {path}: HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(float(r.headers.get("Retry-After", 2.0**attempt)))
        raise DataError(f"binance {path}: gave up after {RETRIES} attempts")

    def symbols(self) -> list[dict[str, Any]]:
        """Every spot pair the exchange knows, trading or not."""
        info = self.get("/api/v3/exchangeInfo")
        return list(info["symbols"])

    def klines(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily bars in [start, end] (UTC days), the last unfinished day excluded by the caller."""
        rows: list[list[Any]] = []
        cursor = int(start.timestamp() * 1000)
        stop = int(end.timestamp() * 1000)
        while cursor <= stop:
            page = self.get(
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": cursor,
                    "endTime": stop,
                    "limit": KLINE_LIMIT,
                },
            )
            if not page:
                break
            rows += page
            cursor = int(page[-1][0]) + 86_400_000
            if len(page) < KLINE_LIMIT:
                break
        return klines_frame(rows)

    def depth(self, symbol: str, limit: int = SNAPSHOT_DEPTH) -> dict[str, Any]:
        out: dict[str, Any] = self.get("/api/v3/depth", {"symbol": symbol, "limit": limit})
        return out


def fetch_instruments(client: BinanceClient, symbols: Sequence[str], path: Path) -> None:
    """Tick and lot of each symbol, from the exchange's filters, for reading a recording."""
    wanted = set(symbols)
    out = {}
    for s in client.symbols():
        if s["symbol"] in wanted:
            filters = {f["filterType"]: f for f in s["filters"]}
            out[s["symbol"]] = {
                "tick": filters["PRICE_FILTER"]["tickSize"],
                "lot": filters["LOT_SIZE"]["stepSize"],
            }
    missing = wanted - set(out)
    if missing:
        raise DataError(f"binance does not list {sorted(missing)}")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


def klines_frame(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Binance kline rows as a frame indexed by the UTC day: quote volume is dollars traded."""
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "quote_volume", "trades"],
            index=pd.DatetimeIndex([], name="date"),
            dtype=float,
        )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows], unit="ms"),
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "quote_volume": [float(r[7]) for r in rows],
            "trades": [float(r[8]) for r in rows],
        }
    )
    return df.set_index("date").sort_index()


# ---------------------------------------------------------------------------- book recording
class _Sink:
    """One gzip JSON-lines file per symbol, sync-flushed so that a crash leaves a readable file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f: IO[bytes] = open(path, "ab")  # noqa: SIM115  (closed by .close())
        self._z = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)

    def write(self, kind: str, data: Any) -> None:
        line = json.dumps({"t": time.time_ns(), "k": kind, "d": data}, separators=(",", ":"))
        self._f.write(self._z.compress(line.encode() + b"\n"))

    def flush(self) -> None:
        self._f.write(self._z.flush(zlib.Z_SYNC_FLUSH))
        self._f.flush()

    def close(self) -> None:
        self._f.write(self._z.flush(zlib.Z_FINISH))
        self._f.close()


def read_recording(path: Path) -> Any:
    """Yield (receive_ns, kind, data) of a recording; a file cut off by a crash ends where it was
    cut, and that is not an error: everything before it was written whole."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                if not line.endswith("\n"):
                    return
                rec = json.loads(line)
                yield rec["t"], rec["k"], rec["d"]
        except EOFError:
            return


async def _snapshot(client: BinanceClient, sink: _Sink, symbol: str) -> None:
    snap = await asyncio.to_thread(client.depth, symbol)
    sink.write("snapshot", snap)


async def _record_once(
    symbols: Sequence[str], sinks: dict[str, _Sink], client: BinanceClient, deadline: float
) -> None:
    import websockets

    streams = "/".join(f"{s.lower()}@depth@100ms/{s.lower()}@aggTrade" for s in symbols)
    async with websockets.connect(WS + streams, max_size=2**24, ping_interval=None) as ws:
        for s in symbols:
            sinks[s].write("segment", {"started": True})
        # the stream is open before the snapshots: the diffs that anchor them are all on file
        for s in symbols:
            await _snapshot(client, sinks[s], s)
        last_flush = last_snap = time.monotonic()
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
            msg = json.loads(raw)
            sym, kind = msg["stream"].split("@")[0].upper(), msg["data"]["e"]
            sinks[sym].write("depth" if kind == "depthUpdate" else "trade", msg["data"])
            now = time.monotonic()
            if now - last_flush >= FLUSH_SECONDS:
                for sink in sinks.values():
                    sink.flush()
                last_flush = now
            if now - last_snap >= SNAPSHOT_SECONDS:
                for s in symbols:
                    await _snapshot(client, sinks[s], s)
                last_snap = now


async def _record(symbols: Sequence[str], out_dir: Path, seconds: float) -> dict[str, Path]:
    import websockets

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    sinks = {s: _Sink(out_dir / f"{s}-{stamp}.jsonl.gz") for s in symbols}
    deadline = time.monotonic() + seconds
    try:
        with BinanceClient() as client:
            while time.monotonic() < deadline:
                try:
                    await _record_once(symbols, sinks, client, deadline)
                except (TimeoutError, OSError, websockets.WebSocketException, DataError) as exc:
                    # Binance closes every connection within 24 hours; a new segment re-anchors
                    print(f"[record] reconnecting after {type(exc).__name__}: {exc}", flush=True)
                    await asyncio.sleep(2.0)
    finally:
        for sink in sinks.values():
            sink.close()
    return {s: sink.path for s, sink in sinks.items()}


def record_book(symbols: Sequence[str], out_dir: Path, seconds: float) -> dict[str, Path]:
    """Record for ``seconds``; returns the file of each symbol."""
    return asyncio.run(_record(symbols, out_dir, seconds))
