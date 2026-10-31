"""Daily short-sale volume per stock from FINRA's Reg SHO files (consolidated NMS, since
2018-08-01): ``cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt``. Free, no key.

What the numbers are: shares sold short and total shares in trades reported to FINRA's
facilities (off-exchange: wholesalers, dark pools, ATSs), not the consolidated tape. A large
part is market makers selling short to fill buy orders, so the *level* differs structurally
across stocks; the informative part is a stock's short share relative to its own recent
history (Diether, Lee & Werner 2009: short sellers increase their activity after gains and
predict next-week returns).

Point in time: the file for session D is published after the close (observed 17:16 ET), so its
numbers are first usable at the close of the next session; the panel fields are shifted by one
session. FINRA names a stock by the ticker it traded under *that day*, while a security id
carries today's ticker: a member that was renamed is read under its earlier tickers up to the
change (``data.ticker_history``), and a row counts only on a session on which the security has
a price, so a ticker's earlier owner cannot fill a column before its company listed. FINRA
writes share classes with a slash (BF/B), the universe with a dot (BF.B). Every symbol is
stored; the SHA-256 of every file's bytes is chained.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pusula.core.calendar import sessions
from pusula.core.errors import DataError
from pusula.data.panel import FloatArray, Panel

URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
FIRST = dt.date(2018, 8, 1)
USER_AGENT = "pusula-research/0.1 (short-sale volume research)"
FIELDS = ("short_volume", "finra_volume")


def parse_day(body: bytes) -> pd.DataFrame:
    """One FINRA file -> rows (date, symbol, short, total) for every symbol in it."""
    df = pd.read_csv(io.BytesIO(body), sep="|", dtype={"Symbol": str, "Date": str})
    df = df[df["Date"].str.fullmatch(r"\d{8}", na=False)]  # drops a trailing count line
    df["symbol"] = df["Symbol"].str.replace("/", ".", regex=False)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], format="%Y%m%d"),
            "symbol": df["symbol"].to_numpy(),
            "short": pd.to_numeric(df["ShortVolume"], errors="coerce").to_numpy(),
            "total": pd.to_numeric(df["TotalVolume"], errors="coerce").to_numpy(),
        }
    )


def _fetcher(
    transport: object | None = None, rate: float = 5.0
) -> Callable[[dt.date], bytes | None]:
    import httpx

    client = httpx.Client(timeout=60.0, transport=transport, headers={"User-Agent": USER_AGENT})  # type: ignore[arg-type]
    last = [0.0]

    def get(day: dt.date) -> bytes | None:
        wait = last[0] + 1.0 / rate - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        last[0] = time.monotonic()
        for attempt in range(4):
            r = client.get(URL.format(date=day.strftime("%Y%m%d")))
            if r.status_code == 200:
                return bytes(r.content)
            if r.status_code in (403, 404):  # no file for that session (FINRA answers 403)
                return None
            time.sleep(2.0 * (2**attempt))
        raise DataError(f"FINRA short volume {day}: HTTP {r.status_code}")

    return get


def update_shortvol(
    store_dir: Path,
    end: dt.date,
    fetch: Callable[[dt.date], bytes | None] | None = None,
    log: bool = True,
) -> int:
    """Append the sessions after the last stored one up to ``end``; returns sessions added.
    Builds the store from ``FIRST`` when it does not exist yet. Every symbol is kept, not only
    today's tickers: a member that traded under another ticker earlier (FB before META) is
    found there through ``data.ticker_history``.

    The chain is per file (c_k = SHA-256(c_{k-1} || SHA-256(file_k))), so it does not depend
    on how many runs built the store. A session without a file counts as missing only once a
    later session has one: the newest file may simply not be published yet, and is asked for
    again on the next run."""
    store_dir.mkdir(parents=True, exist_ok=True)
    path, meta_p = store_dir / "short_volume.parquet", store_dir / "short_volume.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    old = pd.read_parquet(path) if path.exists() else None
    start = (
        FIRST if old is None else (pd.Timestamp(old["date"].max()).date() + dt.timedelta(days=1))
    )
    days = [d.item() for d in sessions(start, end)]
    get = fetch or _fetcher()
    chain = bytes.fromhex(meta.get("chain_sha256", ""))
    new, missing, pending = [], list(meta.get("missing", [])), []
    for k, day in enumerate(days):
        body = get(day)
        if body is None:
            pending.append(day.isoformat())
            continue
        missing += pending
        pending = []
        chain = hashlib.sha256(chain + hashlib.sha256(body).digest()).digest()
        new.append(parse_day(body))
        if log and k % 250 == 0:
            print(f"[shortvol] {day} ({k + 1}/{len(days)})", flush=True)
    if not new:
        return 0
    add = pd.concat(new, ignore_index=True)
    out = add if old is None else pd.concat([old, add], ignore_index=True)
    if out.duplicated(["date", "symbol"]).any():
        raise DataError("duplicate (date, symbol) rows in the short-volume store")
    out = out.astype({"symbol": "category"}).sort_values(["symbol", "date"], ignore_index=True)
    out.to_parquet(path, index=False, row_group_size=1_000_000)
    meta.update(
        {
            "source": URL.format(date="YYYYMMDD"),
            "first": str(out["date"].min().date()),
            "last": str(out["date"].max().date()),
            "rows": len(out),
            "symbols": int(out["symbol"].nunique()),
            "missing": missing,  # sessions without a file (holiday mismatches, outages)
            "chain_sha256": chain.hex(),
        }
    )
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return len(new)


def ticker_segments(
    symbols: Sequence[str], earlier: Mapping[str, Sequence[tuple[str, str]]]
) -> pd.DataFrame:
    """(symbol, lo, hi, j): FINRA ticker ``symbol`` carries security ``j`` on sessions in
    [lo, hi). ``earlier[sid]`` lists (ticker, first session of the next ticker) in date order;
    the id's own ticker applies after the last of them, or always."""
    rows: list[tuple[str, np.datetime64, np.datetime64, int]] = []
    for j, sid in enumerate(symbols):
        lo = np.datetime64("1900-01-01")
        for tick, until in earlier.get(sid, ()):
            rows.append((tick, lo, np.datetime64(until), j))
            lo = np.datetime64(until)
        rows.append((sid.split("@")[0], lo, np.datetime64("2262-01-01"), j))
    return pd.DataFrame(rows, columns=["symbol", "lo", "hi", "j"])


def shortvol_fields(
    panel: Panel,
    store_dir: Path,
    earlier: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> dict[str, FloatArray]:
    """T x N short and total FINRA volume, shifted one session: the value at row t is the file
    of session t-1 (published after that close), read under the ticker the security had on
    session t-1 (``earlier`` defaults to ``data.ticker_history.EARLIER_TICKERS``)."""
    if earlier is None:
        from pusula.data.ticker_history import EARLIER_TICKERS

        earlier = EARLIER_TICKERS
    seg = ticker_segments(panel.symbols, earlier)
    df = pd.read_parquet(
        store_dir / "short_volume.parquet", filters=[("symbol", "in", sorted(set(seg["symbol"])))]
    )
    days = df["date"].to_numpy().astype("datetime64[D]")
    row = np.searchsorted(panel.dates, days)
    on_session = (row < panel.n_dates) & (panel.dates[np.minimum(row, panel.n_dates - 1)] == days)
    usable = on_session & (row + 1 < panel.n_dates)  # usable at the next session's close
    d = df.loc[usable, ["symbol", "date", "short", "total"]].assign(r=row[usable])
    m = d.astype({"symbol": str}).merge(seg, on="symbol")
    day = m["date"].to_numpy().astype("datetime64[D]")
    r, j = m["r"].to_numpy(), m["j"].to_numpy()
    keep = (day >= m["lo"].to_numpy()) & (day < m["hi"].to_numpy())
    keep &= np.isfinite(panel["close"][r, j])  # the security itself traded that session
    r, j, m = r[keep] + 1, j[keep], m[keep]
    out = {f: np.full((panel.n_dates, panel.n_symbols), np.nan) for f in FIELDS}
    out["short_volume"][r, j] = m["short"].to_numpy(np.float64)
    out["finra_volume"][r, j] = m["total"].to_numpy(np.float64)
    return out
