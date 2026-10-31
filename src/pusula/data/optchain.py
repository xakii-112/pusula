"""Strike-level option chains with their quotes: the data a full option engine needs.

``data.optskew`` reduced DoltHub's chain table to three numbers per stock and day, so the only way
to price an option afterwards was one ATM volatility. This keeps every quote, with the chain's own
greeks, for expiries within ``MAX_DTE`` days: read once as the same compressed export, one Parquet
part per stretch of the table, so a backtest reads only the dates it asks for.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pusula.core.errors import DataError
from pusula.data.optskew import DATABASE_ZIP, _StreamReader, zip_member

COLUMNS = [
    "date",
    "act_symbol",
    "expiration",
    "strike",
    "call_put",
    "bid",
    "ask",
    "vol",
    "delta",
    "gamma",
    "theta",
    "vega",
]
MAX_DTE = 200
CHUNK_ROWS = 2_000_000


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rows with a two-sided quote within the expiry range, in compact types."""
    ok = (chunk["ask"] >= chunk["bid"]) & (chunk["bid"] >= 0) & (chunk["ask"] > 0)
    c = chunk[ok].copy()
    c["date"] = pd.to_datetime(c["date"])
    c["expiration"] = pd.to_datetime(c["expiration"])
    c = c[(c["expiration"] - c["date"]).dt.days.between(0, MAX_DTE)]
    c = c.rename(columns={"act_symbol": "symbol"})
    for name in ("symbol", "call_put"):
        c[name] = c[name].astype("category")
    for name in ("strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega"):
        c[name] = c[name].astype("float32" if name != "strike" else "float64")
    return c


def _write_parts(raw: io.RawIOBase, store: Path, log: bool) -> tuple[int, str, str]:
    reader = pd.read_csv(
        io.BufferedReader(raw, 1 << 20),
        usecols=COLUMNS,
        chunksize=CHUNK_ROWS,
        dtype={"act_symbol": str},
    )
    rows, first, last = 0, "", ""
    for k, chunk in enumerate(reader):
        part = clean_chunk(chunk)
        if part.empty:
            continue
        part.to_parquet(store / f"part-{k:05d}.parquet", index=False, compression="zstd")
        rows += len(part)
        first = first or str(part["date"].min().date())
        last = str(part["date"].max().date())
        if log and k % 10 == 0:
            print(f"[optchain] chunk {k}: {rows:,} rows, through {last}", flush=True)
    return rows, first, last


def bootstrap_chain(store: Path, transport: object | None = None, url: str = DATABASE_ZIP) -> Path:
    """Stream the option-chain table once (about an hour, a few GB on disk)."""
    import httpx

    store.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with (
        httpx.Client(timeout=600.0, transport=transport) as client,  # type: ignore[arg-type]
        client.stream("GET", url, follow_redirects=True) as r,
    ):
        if r.status_code != 200:
            raise DataError(f"DoltHub export {url}: HTTP {r.status_code}")
        raw = _StreamReader(zip_member(r.iter_bytes(1 << 20), "option_chain.csv", sha))
        rows, first, last = _write_parts(raw, store, log=True)
    meta = {
        "source": url + " (option_chain.csv)",
        "source_sha256": sha.hexdigest(),
        "rows": rows,
        "start": first,
        "end": last,
        "max_dte": MAX_DTE,
    }
    (store / "chain.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return store


def read_member_chain(
    store: Path, symbols: Sequence[str], start: str, end: str, dte: tuple[int, int]
) -> pd.DataFrame:
    """The columns a quoted backtest reads, for some symbols, part by part: the table has 120M
    rows, and what a run needs of them fits in memory only if the rest never gets there."""
    wanted = ["date", "symbol", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta"]
    filters = [
        ("date", ">=", pd.Timestamp(start)),
        ("date", "<=", pd.Timestamp(end)),
        ("symbol", "in", list(symbols)),
    ]
    frames = []
    for part in sorted(store.glob("part-*.parquet")):
        f = pd.read_parquet(part, columns=wanted, filters=filters)
        if f.empty:
            continue
        days = (f["expiration"] - f["date"]).dt.days
        f = f[days.between(*dte)].assign(dte=days)
        f["symbol"] = f["symbol"].astype(str)
        f["call_put"] = f["call_put"].astype(str)
        frames.append(f)
    if not frames:
        raise DataError(f"no quotes in {store} for {start} to {end}")
    return pd.concat(frames, ignore_index=True)


def read_chain(
    store: Path,
    start: str | None = None,
    end: str | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Quotes dated in [start, end], optionally for some symbols only."""
    parts = sorted(store.glob("part-*.parquet"))
    if not parts:
        raise DataError(f"no option chain store in {store}: run `pusula data option-chain`")
    filters: list[tuple[str, str, object]] = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    if symbols:
        filters.append(("symbol", "in", list(symbols)))
    frames = [pd.read_parquet(p, filters=filters or None) for p in parts]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "expiration", "strike", "call_put"])
    out = pd.concat(frames, ignore_index=True)
    for name in ("symbol", "call_put"):
        out[name] = out[name].astype(str)
    return out
