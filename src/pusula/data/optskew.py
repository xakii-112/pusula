"""Option-surface features per stock and day, from DoltHub ``post-no-preference/options``
``option_chain`` (bid, ask, implied vol and greeks per contract, 2019-02 onward; weekly
Saturday snapshots of Friday's close at first, daily later).

The table is ~10 GB as CSV. It is streamed once and reduced on the fly to one row per
(date, stock); the chain itself is never stored. The CSV is in primary-key order
(date, symbol, expiration, strike, side), so a (date, symbol) group that straddles a chunk
boundary is carried into the next chunk. The SHA-256 of the streamed bytes is recorded.

Features, defined by delta, not moneyness: the chain has no underlying price, and mixing raw
strikes with split-adjusted prices would be wrong. Contracts with a non-positive bid, an ask
below the bid or an implied vol outside (1%, 500%) are dropped.

* ``smirk``: IV of the put with delta closest to -0.20 (within -0.35..-0.10) minus IV of the
  call with delta closest to 0.50 (0.35..0.65), at the expiration closest to 30 days (10..60).
  A steep smirk predicts low returns (Xing, Zhang & Zhao 2010).
* ``vol_spread``: mean IV(call) - IV(put) over strikes quoted on both sides with call delta in
  0.25..0.75, same expiration. Equal-weighted: the table has no open interest (Cremers &
  Weinbaum 2010 weight by it). A high spread predicts high returns.
* ``term_slope``: ATM call IV at the expiration closest to 90 days (60..180) minus at the
  30-day one. Upward slope predicts high returns (Vasquez 2017).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.core.errors import DataError
from pusula.data.panel import FloatArray, Panel

CHAIN_CSV = "https://www.dolthub.com/csv/post-no-preference/options/master/option_chain"
DATABASE_ZIP = "https://www.dolthub.com/csv/post-no-preference/options/master"  # all tables
FEATURES = ("smirk", "vol_spread", "term_slope")
COLUMNS = ["date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta"]


KEY = ["date", "act_symbol"]


def _empty() -> pd.DataFrame:
    """Typed empty result (an untyped one turns concatenated feature columns into objects)."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype=str),
            "symbol": pd.Series(dtype=str),
            **{f: pd.Series(dtype=np.float64) for f in FEATURES},
        }
    )


def _pick(
    df: pd.DataFrame, col: str, target: float, lo: float, hi: float, keep: list[str]
) -> pd.DataFrame:
    """Per (date, symbol), the row whose ``col`` is closest to ``target`` within [lo, hi]; ties
    go to the first row in file order (stable sort)."""
    x = df[(df[col] >= lo) & (df[col] <= hi)]
    x = x.assign(_d=(x[col] - target).abs()).sort_values([*KEY, "_d"], kind="mergesort")
    return x.drop_duplicates(KEY)[[*KEY, *keep]]


def reduce_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, symbol) with ``FEATURES`` from a chunk of chain rows (vectorised;
    ``tests/test_optskew.py`` holds it to a per-group reference implementation)."""
    ok = (
        (chain["bid"] > 0)
        & (chain["ask"] >= chain["bid"])
        & (chain["vol"] > 0.01)
        & (chain["vol"] < 5.0)
    )
    c = chain[ok].reset_index(drop=True)
    if c.empty:
        return _empty()
    c["dte"] = (pd.to_datetime(c["expiration"]) - pd.to_datetime(c["date"])).dt.days
    base = c[KEY].drop_duplicates()
    exps = c[[*KEY, "expiration", "dte"]].drop_duplicates([*KEY, "expiration"])
    e30 = _pick(exps, "dte", 30, 10, 60, ["expiration"])
    g30 = c.merge(e30, on=[*KEY, "expiration"])
    calls, puts = g30[g30["call_put"] == "Call"], g30[g30["call_put"] == "Put"]
    atm = _pick(calls, "delta", 0.50, 0.35, 0.65, ["vol"]).rename(columns={"vol": "atm"})
    otm = _pick(puts, "delta", -0.20, -0.35, -0.10, ["vol"]).rename(columns={"vol": "otm"})
    pair = calls[(calls["delta"] >= 0.25) & (calls["delta"] <= 0.75)].merge(
        puts, on=[*KEY, "expiration", "strike"], suffixes=("_c", "_p")
    )
    spread = (
        pair.assign(vol_spread=pair["vol_c"] - pair["vol_p"])
        .groupby(KEY, sort=False)["vol_spread"]
        .mean()
        .reset_index()
    )
    e90 = _pick(exps, "dte", 90, 60, 180, ["expiration"])
    g90 = c.merge(e90, on=[*KEY, "expiration"])
    atm90 = _pick(g90[g90["call_put"] == "Call"], "delta", 0.50, 0.35, 0.65, ["vol"])
    out = (
        base.merge(atm, on=KEY, how="left")
        .merge(otm, on=KEY, how="left")
        .merge(spread, on=KEY, how="left")
        .merge(atm90.rename(columns={"vol": "atm90"}), on=KEY, how="left")
    )
    out["smirk"] = out["otm"] - out["atm"]
    out["term_slope"] = out["atm90"] - out["atm"]
    return out.rename(columns={"act_symbol": "symbol"})[["date", "symbol", *FEATURES]]


class _StreamReader(io.RawIOBase):
    """A readable file over an iterator of byte chunks (an HTTP stream), hashing as it reads."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._it = chunks
        self._buf = b""
        self.sha = hashlib.sha256()
        self.bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
            self.sha.update(self._buf)
            self.bytes += len(self._buf)
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def reduce_stream(raw: io.RawIOBase, chunk_rows: int = 2_000_000, log: bool = True) -> pd.DataFrame:
    """Reduce a chain CSV stream chunk by chunk, carrying the last (date, symbol) group."""
    parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    reader = pd.read_csv(
        io.BufferedReader(raw, 1 << 20),
        usecols=COLUMNS,
        chunksize=chunk_rows,
        dtype={"act_symbol": str},
    )
    for k, read in enumerate(reader):
        rows = read if carry.empty else pd.concat([carry, read], ignore_index=True)
        last = (rows["date"].iloc[-1], rows["act_symbol"].iloc[-1])
        tail = (rows["date"] == last[0]) & (rows["act_symbol"] == last[1])
        carry = rows[tail]
        parts.append(reduce_chain(rows[~tail]))
        if log and k % 10 == 0:
            print(f"[optskew] chunk {k}: through {last[0]}", flush=True)
    if not carry.empty:
        parts.append(reduce_chain(carry))
    parts = [p for p in parts if not p.empty]
    out = pd.concat(parts, ignore_index=True) if parts else _empty()
    if out.duplicated(["date", "symbol"]).any():
        raise DataError("option chain was not in primary-key order: a (date, symbol) repeats")
    return out


def zip_member(chunks: Iterator[bytes], member: str, sha: Any) -> Iterator[bytes]:
    """Decompressed bytes of one member of a *streaming* ZIP archive (local headers in order,
    sizes in data descriptors after the data), read without the central directory: members
    before it are inflated to find their end and discarded; reading stops at its end. The
    compressed bytes consumed are hashed into ``sha``."""
    import struct
    import zlib

    it = iter(chunks)
    buf = b""

    def more() -> bool:
        nonlocal buf
        try:
            b = next(it)
        except StopIteration:
            return False
        sha.update(b)
        buf += b
        return True

    while True:
        while len(buf) < 30:
            if not more():
                raise DataError(f"ZIP stream ended before member {member!r}")
        if buf[:4] != b"PK\x03\x04":
            raise DataError(f"not a ZIP local header (member {member!r} not found)")
        method = struct.unpack("<H", buf[8:10])[0]
        nlen, xlen = struct.unpack("<HH", buf[26:30])
        while len(buf) < 30 + nlen + xlen:
            if not more():
                raise DataError("ZIP stream ended inside a local header")
        name = buf[30 : 30 + nlen].decode("utf-8", errors="replace")
        if method != 8:
            raise DataError(f"ZIP member {name!r}: compression method {method} not supported")
        buf = buf[30 + nlen + xlen :]
        d = zlib.decompressobj(-15)
        while True:
            data = d.decompress(buf)
            buf = b""
            if name == member and data:
                yield data
            if d.eof:
                break
            if not more():
                raise DataError(f"ZIP stream ended inside member {name!r}")
        if name == member:
            return
        buf = d.unused_data
        # skip the data descriptor (12-24 bytes, optional signature) up to the next header
        while (k := buf.find(b"PK\x03\x04")) < 0:
            if not more():
                raise DataError(f"ZIP stream ended before member {member!r}")
        buf = buf[k:]


def bootstrap_skew(
    store_dir: Path, transport: object | None = None, url: str = DATABASE_ZIP
) -> Path:
    """Stream the option-chain table once and store the daily features (a few MB). Reads the
    database as DoltHub's compressed ZIP export (4x fewer bytes than the table's CSV endpoint,
    which DoltHub cuts after about an hour, before a 10 GB table is through) and stops at the
    end of ``option_chain.csv``."""
    import httpx

    store_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with (
        httpx.Client(timeout=600.0, transport=transport) as client,  # type: ignore[arg-type]
        client.stream("GET", url, follow_redirects=True) as r,
    ):
        if r.status_code != 200:
            raise DataError(f"DoltHub export {url}: HTTP {r.status_code}")
        raw = _StreamReader(zip_member(r.iter_bytes(1 << 20), "option_chain.csv", sha))
        df = reduce_stream(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = store_dir / "option_skew.parquet"
    df.to_parquet(out, index=False)
    meta = {
        "source": url + " (option_chain.csv)",
        "source_sha256": sha.hexdigest(),  # compressed bytes read, through the member's end
        "csv_sha256": raw.sha.hexdigest(),
        "csv_bytes": raw.bytes,
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "coverage": {f: float(df[f].notna().mean()) for f in FEATURES},
    }
    (store_dir / "option_skew.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def skew_fields(panel: Panel, store_dir: Path, max_stale_days: int = 7) -> dict[str, FloatArray]:
    """T x N features aligned to the panel: the latest observation dated <= the session and at
    most ``max_stale_days`` old (a Saturday snapshot serves the following week's sessions)."""
    df = pd.read_parquet(store_dir / "option_skew.parquet")
    target = panel.dates.astype("datetime64[D]")
    out = {f: np.full((panel.n_dates, panel.n_symbols), np.nan) for f in FEATURES}
    by_sym = dict(list(df.groupby("symbol")))
    for j, sid in enumerate(panel.symbols):
        g = by_sym.get(sid.split("@")[0])
        if g is None:
            continue
        obs = g["date"].to_numpy().astype("datetime64[D]")
        pos = np.searchsorted(obs, target, side="right") - 1
        ok = pos >= 0
        age = np.where(ok, (target - obs[np.maximum(pos, 0)]).astype(int), 10**9)
        live = ok & (age <= max_stale_days)
        for f in FEATURES:
            vals = g[f].to_numpy(dtype=np.float64)
            out[f][:, j] = np.where(live, vals[np.maximum(pos, 0)], np.nan)
    return out
