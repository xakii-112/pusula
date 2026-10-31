"""Implied-volatility history (30-day ATM IV per underlying), point in time.

Source: DoltHub ``post-no-preference/options`` ``volatility_history`` (2019-02 onward), cached
by NihaiAlgoritma. It is imported **read-only** into pusula's own store with the source path
and SHA-256 recorded, then aligned to the research panel: the IV of session t is the latest
observation dated <= t and at most ``max_stale_days`` old (no back-filling, no interpolation
across the future).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pusula.core.errors import DataError
from pusula.data.panel import FloatArray, Panel

LEGACY_DOLT = Path(
    os.environ.get(
        "PUSULA_LEGACY_IV",
        str(Path.home() / ".pusula" / "legacy" / "volatility_history__master.parquet"),
    )
)


def import_iv(dest_dir: Path, source: Path = LEGACY_DOLT) -> Path:
    if not source.exists():
        raise DataError(f"IV source not found: {source}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "iv_history.parquet"
    df = pd.read_parquet(source, columns=["date", "act_symbol", "iv_current"])
    df = df.rename(columns={"act_symbol": "symbol", "iv_current": "iv"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["iv"] > 0.01) & (df["iv"] < 5.0)].sort_values(["symbol", "date"])
    df.to_parquet(out, index=False)
    meta = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
    }
    (dest_dir / "iv_history.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def iv_panel(panel: Panel, store_dir: Path, max_stale_days: int = 7) -> FloatArray:
    """T x N IV aligned to the panel (security ids map to their ticker)."""
    df = pd.read_parquet(store_dir / "iv_history.parquet")
    wide = df.pivot_table(index="date", columns="symbol", values="iv", aggfunc="last").sort_index()
    obs_days = wide.index.values.astype("datetime64[D]")
    target_days = panel.dates.astype("datetime64[D]")
    pos = np.searchsorted(obs_days, target_days, side="right") - 1  # last obs <= session
    values = wide.to_numpy(dtype=np.float64)
    columns = {str(c): k for k, c in enumerate(wide.columns)}
    out = np.full((panel.n_dates, panel.n_symbols), np.nan)
    for j, sid in enumerate(panel.symbols):
        k = columns.get(sid.split("@")[0])
        if k is None:
            continue
        col = values[:, k]
        # per-column forward fill of observation index (an IV print may be missing on a date)
        has = np.isfinite(col)
        last_obs = np.where(has, np.arange(col.size), -1)
        last_obs = np.maximum.accumulate(last_obs)
        src = np.where(pos >= 0, last_obs[np.maximum(pos, 0)], -1)
        ok = src >= 0
        age = np.where(ok, (target_days - obs_days[np.maximum(src, 0)]).astype(int), 10**9)
        out[:, j] = np.where(ok & (age <= max_stale_days), col[np.maximum(src, 0)], np.nan)
    return out


DOLT_BRANCH = "master"
DOLT_API = f"https://www.dolthub.com/api/v1alpha1/post-no-preference/options/{DOLT_BRANCH}"
_RETRYABLE = (
    "context deadline exceeded",
    "connection reset",
    "too many connections",
    "timeout",
    "temporarily unavailable",
)


class DoltClient:
    """DoltHub SQL API (1000-row pages). A response is either complete or an error: partial
    rows returned with a server error are never used (NihaiAlgoritma learnt this the hard way)."""

    def __init__(
        self, transport: object | None = None, polite_delay: float = 0.5, retries: int = 4
    ) -> None:
        import httpx

        self._client = httpx.Client(timeout=90.0, transport=transport)  # type: ignore[arg-type]
        self.delay = polite_delay
        self.retries = retries

    def close(self) -> None:
        self._client.close()

    def query(self, sql: str) -> list[dict[str, object]]:
        import time
        from urllib.parse import quote

        last = "?"
        for attempt in range(self.retries + 1):
            r = self._client.get(f"{DOLT_API}?q={quote(sql, safe='')}")
            if r.status_code in (403, 429):
                last = f"HTTP {r.status_code} (rate limit)"
                time.sleep(30.0)
                continue
            if r.status_code != 200:
                last = f"HTTP {r.status_code}"
            else:
                payload = r.json()
                status = payload.get("query_execution_status")
                if status == "Success":
                    if self.delay:
                        time.sleep(self.delay)
                    return list(payload.get("rows") or [])
                last = str(payload.get("query_execution_message", status))
                if status == "RowLimit":
                    raise DataError(f"query hit the row limit; paginate it: {sql[:100]}")
                if not any(m in last.lower() for m in _RETRYABLE):
                    raise DataError(f"DoltHub query failed: {last}")
            time.sleep(2.0 * (2**attempt))
        raise DataError(f"DoltHub query failed after {self.retries + 1} attempts: {last}")

    def paged(self, sql: str, page: int = 1000) -> list[dict[str, object]]:
        """``sql`` must end with a deterministic ORDER BY (primary-key order)."""
        rows: list[dict[str, object]] = []
        for k in range(10_000):
            chunk = self.query(f"{sql} LIMIT {page} OFFSET {k * page}")
            rows.extend(chunk)
            if len(chunk) < page:
                return rows
        raise DataError("pagination did not terminate")


def update_iv(store_dir: Path, client: DoltClient | None = None, log: bool = True) -> int:
    """Append the sessions published since the last stored date; returns rows added.

    Only new dates are fetched (rows of a published date are not revised here: the store is an
    append-only point-in-time record). The Dolt commit the rows came from is recorded.
    """
    path, meta_p = store_dir / "iv_history.parquet", store_dir / "iv_history.json"
    df = pd.read_parquet(path)
    last = pd.Timestamp(df["date"].max()).date().isoformat()
    own = client is None
    cli = client or DoltClient()
    try:
        dates = [
            str(r["date"])[:10]
            for r in cli.query(
                f"SELECT DISTINCT date FROM volatility_history WHERE date > '{last}' ORDER BY date"
            )
        ]
        new = []
        for d in dates:
            rows = cli.paged(
                f"SELECT date, act_symbol, iv_current FROM volatility_history "
                f"WHERE date = '{d}' ORDER BY act_symbol"
            )
            new.extend(rows)
            if log:
                print(f"[iv] {d}: {len(rows)} rows", flush=True)
        # the branch head, only when there is something to record it for. Not from dolt_log:
        # sorting the whole commit log by date hit the server's deadline (~56 s, 2026-10-27,
        # which stopped that night's run); the branch table answers in under a second
        head = (
            cli.query(f"SELECT hash FROM dolt_branches WHERE name = '{DOLT_BRANCH}'") if new else []
        )
    finally:
        if own:
            cli.close()
    if not new:
        return 0
    add = pd.DataFrame(new).rename(columns={"act_symbol": "symbol", "iv_current": "iv"})
    add["date"] = pd.to_datetime(add["date"])
    add["iv"] = pd.to_numeric(add["iv"], errors="coerce")
    add = add[(add["iv"] > 0.01) & (add["iv"] < 5.0)]
    if add.duplicated(["date", "symbol"]).any():
        raise DataError("duplicate (date, symbol) rows in the DoltHub update")
    out = pd.concat([df, add[["date", "symbol", "iv"]]], ignore_index=True).sort_values(
        ["symbol", "date"]
    )
    out.to_parquet(path, index=False)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    meta.setdefault("updates", []).append(
        {
            "source": DOLT_API,
            "dolt_commit": str(head[0]["hash"]) if head else None,
            "from": dates[0],
            "to": dates[-1],
            "rows": len(add),
        }
    )
    meta.update({"rows": len(out), "end": str(out["date"].max().date())})
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return len(add)


DOLT_CSV = "https://www.dolthub.com/csv/post-no-preference/options/master/volatility_history"


def bootstrap_iv(store_dir: Path, transport: object | None = None, url: str = DOLT_CSV) -> Path:
    """Build the implied-vol store from scratch from DoltHub's whole-table CSV (one streamed
    request, ~200 MB; the SQL API is the wrong tool for bulk reads). The download goes to a
    temporary file and is only used once complete; its SHA-256 is recorded."""
    import httpx

    store_dir.mkdir(parents=True, exist_ok=True)
    tmp = store_dir / "volatility_history.csv.partial"
    h = hashlib.sha256()
    with (
        httpx.Client(timeout=300.0, transport=transport) as client,  # type: ignore[arg-type]
        client.stream("GET", url, follow_redirects=True) as r,
    ):
        if r.status_code != 200:
            raise DataError(f"DoltHub CSV {url}: HTTP {r.status_code}")
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                h.update(chunk)
    df = pd.read_csv(tmp, usecols=["date", "act_symbol", "iv_current"])
    tmp.unlink()
    df = df.rename(columns={"act_symbol": "symbol", "iv_current": "iv"})
    df["date"] = pd.to_datetime(df["date"])
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df = df[(df["iv"] > 0.01) & (df["iv"] < 5.0)].sort_values(["symbol", "date"])
    if df.empty:
        raise DataError("DoltHub CSV contained no usable implied vols")
    out = store_dir / "iv_history.parquet"
    df.to_parquet(out, index=False)
    meta = {
        "source": url,
        "source_sha256": h.hexdigest(),
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
    }
    (store_dir / "iv_history.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out
