"""End-to-end data build: point-in-time universe -> bars -> research panel.

``build_store`` downloads (or refreshes) daily bars for every security that was an S&P 500
member at any time since ``since``, each requested as of the entity it was at the time, plus
benchmark ETFs. ``research_panel`` loads them into one :class:`Panel` with the membership
mask ``in_universe``, applies the causal quality repairs and returns the quality report.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from pusula.core.errors import DataError
from pusula.data import universe as U
from pusula.data.alpaca import AlpacaClient, BarRequest, chunked
from pusula.data.panel import Panel
from pusula.data.quality import QualityReport, check_and_repair
from pusula.data.settings import alpaca_credentials, atomic_write_text, data_dir
from pusula.data.store import BarStore

BENCHMARKS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "RSP",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)
# QuantBot's 21 assets (Desktop/QuantBot/quantbot/config.py), picked there by a return-blind rule:
# candidates in a fixed asset-class rotation, skipping any too correlated with one already chosen.
# Without its crypto pairs and Borsa Istanbul's index, which this data feed does not carry. Every
# one of them trades over the whole of the store's history.
CROSS_ASSET: tuple[str, ...] = (
    "DBA",
    "DBB",
    "DBC",
    "EEM",
    "EMB",
    "EWJ",
    "FXY",
    "GLD",
    "HYG",
    "LQD",
    "QQQ",
    "SPY",
    "TIP",
    "TLT",
    "UUP",
    "VNQ",
    "XLE",
    "XLU",
)
DATA_START = dt.date(2016, 1, 1)


@dataclass(frozen=True, slots=True)
class Security:
    security_id: str
    ticker: str
    asof: dt.date | None
    sector: str


def securities(hist: U.IndexHistory, since: dt.date) -> tuple[list[Security], list[U.Spell]]:
    spells = hist.spells(since)
    secs: dict[str, Security] = {}
    for sp in spells:
        sid = sp.security_id
        if sid not in secs:
            secs[sid] = Security(
                sid,
                sp.ticker,
                sp.asof,
                hist.sectors.get(sp.ticker, "Unknown") if sp.end is None else "Unknown",
            )
    return sorted(secs.values(), key=lambda s: s.security_id), spells


def _default_client() -> AlpacaClient:
    creds = alpaca_credentials()
    if creds is None:
        raise DataError(
            "no Alpaca credentials (set ALPACA_API_KEY/ALPACA_SECRET_KEY or PUSULA_ENV_FILE)"
        )
    return AlpacaClient(creds)


def build_store(
    root: Path | None = None,
    *,
    since: dt.date = DATA_START,
    end: dt.date | None = None,
    refresh_universe: bool = False,
    refetch: bool = False,
    log: bool = True,
    client: AlpacaClient | None = None,
    history: U.IndexHistory | None = None,
) -> BarStore:
    """``client`` and ``history`` are injectable (tests); by default they come from the
    environment's credentials and the cached Wikipedia snapshot."""
    root = root or data_dir()
    client = client or _default_client()
    hist = history or U.load_or_fetch(root / "universe", refresh=refresh_universe)
    secs, spells = securities(hist, since)
    secs += [Security(b, b, None, "ETF") for b in BENCHMARKS]
    store = BarStore(root)
    # The store only ever holds completed sessions (the free SIP plan also refuses the last 15 min).
    end = end or last_completed_session()
    todo = [
        s
        for s in secs
        if refetch or not store.has(s.security_id) or store.record(s.security_id).rows == 0
    ]
    groups: dict[dt.date | None, list[Security]] = defaultdict(list)
    for s in todo:
        groups[s.asof].append(s)
    fetched_at = dt.datetime.now(dt.UTC)
    n_done = 0
    with client:
        for asof, group in sorted(
            groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or dt.date.min)
        ):
            by_ticker = {s.ticker: s for s in group}
            for chunk in chunked(sorted(by_ticker), 40):
                req = BarRequest(chunk, DATA_START, end, asof=asof)
                frames = client.daily_bars(req)
                # Multi-symbol requests can silently drop a symbol whose entity mapping collides
                # with another in the same request (seen: DOW next to DD, 2026-10-23). Anything
                # that came back empty is asked for again on its own.
                for tk in chunk:
                    if frames[tk].empty and len(chunk) > 1:
                        frames[tk] = client.daily_bars(
                            BarRequest((tk,), DATA_START, end, asof=asof)
                        )[tk]
                for tk in chunk:
                    s = by_ticker[tk]
                    store.put(
                        s.security_id,
                        s.ticker,
                        s.asof,
                        frames[tk],
                        adjustment=req.adjustment,
                        feed=req.feed,
                        fetched_at=fetched_at,
                    )
                n_done += len(chunk)
                store.flush()
                if log:
                    print(f"[data] {n_done}/{len(todo)} securities (asof={asof})", flush=True)
    write_meta(root, hist, secs, spells)
    return store


def write_meta(
    root: Path, hist: U.IndexHistory, secs: list[Security], spells: list[U.Spell]
) -> None:
    meta = {
        "universe_snapshot": hist.snapshot_date.isoformat(),
        "universe_sha256": hist.source_sha256,
        "n_securities": len(secs),
        "n_spells": len(spells),
        "sectors": {s.security_id: s.sector for s in secs},
        "spells": [
            [sp.security_id, sp.start.isoformat(), sp.end.isoformat() if sp.end else None]
            for sp in spells
        ],
    }
    atomic_write_text(root / "universe_meta.json", json.dumps(meta, indent=1))


def update_store(
    root: Path | None = None,
    *,
    end: dt.date | None = None,
    log: bool = True,
    client: AlpacaClient | None = None,
) -> list[str]:
    """Append newly completed sessions for securities whose spell is open (and the ETFs).

    The last stored week is re-requested and compared: with split/dividend-adjusted prices a
    new corporate action rescales the whole history, so any mismatch on the overlap triggers a
    full re-download of that security instead of a silent splice. Returns the ids re-downloaded.
    """
    root = root or data_dir()
    client = client or _default_client()
    store = BarStore(root)
    meta = json.loads((root / "universe_meta.json").read_text(encoding="utf-8"))
    open_ids = [sid for sid, _, b in meta["spells"] if b is None] + list(BENCHMARKS)
    open_ids += [t for t in CROSS_ASSET if t not in BENCHMARKS and store.has(t)]
    end = end or last_completed_session()
    rebuilt: list[str] = []
    now = dt.datetime.now(dt.UTC)
    with client:
        for chunk in chunked(open_ids, 40):
            have = {sid: store.get(sid) for sid in chunk if store.has(sid)}
            tickers = {sid.split("@")[0]: sid for sid in chunk}
            since = min(
                (df.index.max() for df in have.values() if len(df)),
                default=pd.Timestamp(DATA_START),
            )
            start = (since - pd.Timedelta(days=10)).date()
            frames = client.daily_bars(BarRequest(tuple(tickers), start, end))
            for tk, sid in tickers.items():
                new = frames.get(tk)
                if new is None or new.empty:
                    new = client.daily_bars(BarRequest((tk,), start, end))[tk]
                old = have.get(sid)
                if old is None or old.empty:
                    continue
                overlap = old.index.intersection(new.index)
                o, n = old.loc[overlap, "close"], new.loc[overlap, "close"]
                if len(overlap) and not np.allclose(o.to_numpy(), n.to_numpy(), rtol=1e-6, atol=0):
                    full = client.daily_bars(BarRequest((tk,), DATA_START, end))[tk]
                    store.put(sid, tk, None, full, adjustment="all", feed="sip", fetched_at=now)
                    rebuilt.append(sid)
                    continue
                merged = pd.concat([old, new.loc[new.index.difference(old.index)]]).sort_index()
                store.put(sid, tk, None, merged, adjustment="all", feed="sip", fetched_at=now)
            store.flush()
    if log:
        print(
            f"[data] updated {len(open_ids)} open securities to {end}; re-downloaded {rebuilt}",
            flush=True,
        )
    return rebuilt


def fetch_etfs(
    tickers: tuple[str, ...],
    root: Path | None = None,
    *,
    end: dt.date | None = None,
    client: AlpacaClient | None = None,
) -> list[str]:
    """Download the listed ETFs the store does not have yet; ``update_store`` keeps them current."""
    root = root or data_dir()
    store = BarStore(root)
    todo = [t for t in tickers if not store.has(t) or store.record(t).rows == 0]
    end = end or last_completed_session()
    now = dt.datetime.now(dt.UTC)
    with client or _default_client() as c:
        for chunk in chunked(todo, 40):
            req = BarRequest(chunk, DATA_START, end)
            frames = c.daily_bars(req)
            for tk in chunk:
                store.put(
                    tk,
                    tk,
                    None,
                    frames[tk],
                    adjustment=req.adjustment,
                    feed=req.feed,
                    fetched_at=now,
                )
            store.flush()
    return todo


def last_completed_session(now: dt.datetime | None = None) -> dt.date:
    """Today if the US session has closed (+30 min for the consolidated tape), else yesterday."""
    from zoneinfo import ZoneInfo

    from pusula.core import calendar as cal

    ny = (now or dt.datetime.now(dt.UTC)).astimezone(ZoneInfo("America/New_York"))
    d = ny.date()
    if cal.is_session(d) and ny.time() >= dt.time(16, 30):
        return d
    return cal.previous_session(d)


@dataclass(frozen=True, slots=True)
class ResearchData:
    panel: Panel  # index members only (columns = security ids), with ``in_universe``
    benchmarks: Panel  # ETFs
    quality: QualityReport
    sectors: dict[str, str]
    data_digest: str
    empty_securities: tuple[str, ...]


def research_panel(root: Path | None = None, end: str | None = None) -> ResearchData:
    root = root or data_dir()
    store = BarStore(root)
    meta = json.loads((root / "universe_meta.json").read_text(encoding="utf-8"))
    spells = [
        U.Spell(
            sid.split("@")[0], dt.date.fromisoformat(a), dt.date.fromisoformat(b) if b else None
        )
        for sid, a, b in meta["spells"]
    ]
    if any(sp.security_id != sid for sp, (sid, _, _) in zip(spells, meta["spells"], strict=True)):
        raise DataError("universe_meta security ids do not match the id rule; rebuild the store")
    ids = sorted({sp.security_id for sp in spells})
    nonempty = [i for i in ids if store.has(i) and store.record(i).rows > 0]
    empty = tuple(sorted(set(ids) - set(nonempty)))
    raw = store.panel(nonempty, start=DATA_START.isoformat(), end=end)
    # keep sessions on which most securities traded (drops stray off-calendar prints)
    live_frac = np.isfinite(raw["close"]).mean(axis=1)
    keep = live_frac >= 0.5 * float(np.median(live_frac))
    raw = Panel(raw.dates[keep], raw.symbols, {k: v[keep] for k, v in raw.fields.items()})
    mask = U.membership_mask(spells, raw.dates, raw.symbols)
    mask = np.where(np.isfinite(raw["close"]), mask, 0.0)
    cleaned, report = check_and_repair(raw.with_fields(in_universe=mask))
    bench_ids = [b for b in BENCHMARKS if store.has(b) and store.record(b).rows > 0]
    bench = store.panel(bench_ids, start=DATA_START.isoformat(), end=end)
    bidx = np.isin(bench.dates, cleaned.dates)
    bench = Panel(bench.dates[bidx], bench.symbols, {k: v[bidx] for k, v in bench.fields.items()})
    if bench.n_dates != cleaned.n_dates:
        full = {}
        for k in bench.fields:
            arr = np.full((cleaned.n_dates, bench.n_symbols), np.nan)
            pos = np.searchsorted(cleaned.dates, bench.dates)
            arr[pos] = bench[k]
            full[k] = arr
        bench = Panel(cleaned.dates, bench.symbols, full)
    digest = store.manifest_digest([*nonempty, *bench_ids])
    return ResearchData(cleaned, bench, report, dict(meta["sectors"]), digest, empty)


def etf_data(
    tickers: tuple[str, ...], root: Path | None = None, end: str | None = None
) -> ResearchData:
    """A fixed list of ETFs as a research panel: each is a member from its first bar. A fund has no
    index to leave, so there is no survivorship to rebuild, as long as the list was fixed in advance
    (it is part of what a family registers)."""
    root = root or data_dir()
    store = BarStore(root)
    missing = [t for t in tickers if not store.has(t) or store.record(t).rows == 0]
    if missing:
        raise DataError(f"no bars for {missing}: run `pusula data cross-asset`")
    raw = store.panel(list(tickers), start=DATA_START.isoformat(), end=end)
    live_frac = np.isfinite(raw["close"]).mean(axis=1)
    keep = live_frac >= 0.5 * float(np.median(live_frac))
    raw = Panel(raw.dates[keep], raw.symbols, {k: v[keep] for k, v in raw.fields.items()})
    mask = np.isfinite(raw["close"]).astype(np.float64)
    cleaned, report = check_and_repair(raw.with_fields(in_universe=mask))
    sectors = dict.fromkeys(tickers, "ETF")
    return ResearchData(cleaned, cleaned, report, sectors, store.manifest_digest(list(tickers)), ())


def fetch_raw_bars(
    root: Path | None = None,
    *,
    end: dt.date | None = None,
    client: AlpacaClient | None = None,
) -> int:
    """Unadjusted daily bars of every security, in a store of their own: an option's strike is in
    the prices of its day, which the split- and dividend-adjusted store does not hold."""
    root = root or data_dir()
    hist = U.load_or_fetch(root / "universe")
    secs, _ = securities(hist, DATA_START)
    store = BarStore(root / "raw")
    end = end or last_completed_session()
    todo = [s for s in secs if not store.has(s.security_id)]
    groups: dict[dt.date | None, list[Security]] = defaultdict(list)
    for s in todo:
        groups[s.asof].append(s)
    fetched_at = dt.datetime.now(dt.UTC)
    with client or _default_client() as c:
        for asof, group in sorted(
            groups.items(), key=lambda kv: (kv[0] is not None, kv[0] or dt.date.min)
        ):
            by_ticker = {s.ticker: s for s in group}
            for chunk in chunked(sorted(by_ticker), 40):
                req = BarRequest(chunk, DATA_START, end, asof=asof, adjustment="raw")
                frames = c.daily_bars(req)
                for tk in chunk:
                    s = by_ticker[tk]
                    store.put(
                        s.security_id,
                        s.ticker,
                        s.asof,
                        frames[tk],
                        adjustment="raw",
                        feed=req.feed,
                        fetched_at=fetched_at,
                    )
                store.flush()
    return len(todo)


def raw_closes(panel: Panel, root: Path | None = None) -> npt.NDArray[np.float64]:
    """T x N unadjusted closes aligned to ``panel``; NaN where the store has none."""
    store = BarStore((root or data_dir()) / "raw")
    out = np.full((panel.n_dates, panel.n_symbols), np.nan)
    index = pd.DatetimeIndex(panel.dates.astype("datetime64[ns]"))
    for j, sid in enumerate(panel.symbols):
        if store.has(sid) and store.record(sid).rows > 0:
            out[:, j] = store.get(sid)["close"].reindex(index).to_numpy(dtype=np.float64)
    return out


def is_crypto(universe: tuple[str, ...]) -> bool:
    return bool(universe) and universe[0].startswith("crypto:")


def universe_data(
    universe: tuple[str, ...], root: Path | None = None, end: str | None = None
) -> ResearchData:
    """A family's own universe: a list of ETFs, or the crypto universe the marker names."""
    if is_crypto(universe):
        from pusula.data.crypto import crypto_data

        return crypto_data(root, end)
    return etf_data(universe, root, end)


def with_implied_vol(rd: ResearchData, root: Path | None = None) -> ResearchData:
    """``rd`` with the panel field ``iv`` (30-day ATM implied vol, point in time) attached when
    the implied-vol store exists (imported from NihaiAlgoritma's cache on first use; elsewhere,
    e.g. in a container, run ``pusula data iv-update`` once to bootstrap it from DoltHub)."""
    from dataclasses import replace

    from pusula.data.iv import LEGACY_DOLT, import_iv, iv_panel

    store = (root or data_dir()) / "iv"
    if not (store / "iv_history.parquet").exists():
        if not LEGACY_DOLT.exists():
            return rd
        import_iv(store)
    return replace(rd, panel=rd.panel.with_fields(iv=iv_panel(rd.panel, store)))


def with_short_volume(rd: ResearchData, root: Path | None = None) -> ResearchData:
    """``rd`` with FINRA short-sale volume (``data.shortvol.FIELDS``, lagged one session)
    attached when the store exists (`pusula data short-volume`); unchanged otherwise."""
    from dataclasses import replace

    from pusula.data.shortvol import shortvol_fields

    store = (root or data_dir()) / "shortvol"
    if not (store / "short_volume.parquet").exists():
        return rd
    return replace(rd, panel=rd.panel.with_fields(**shortvol_fields(rd.panel, store)))


def with_option_surface(rd: ResearchData, root: Path | None = None) -> ResearchData:
    """``rd`` with the option-surface fields (``data.optskew.FEATURES``) attached when the store
    exists (`pusula data option-surface`, a one-off ~10 GB stream); unchanged otherwise."""
    from dataclasses import replace

    from pusula.data.optskew import skew_fields

    store = (root or data_dir()) / "optskew"
    if not (store / "option_skew.parquet").exists():
        return rd
    return replace(rd, panel=rd.panel.with_fields(**skew_fields(rd.panel, store)))


def with_fundamentals(rd: ResearchData, root: Path | None = None) -> ResearchData:
    """``rd`` with the company-filing event fields (``data.fundamentals.FIELDS``) attached when
    the fundamentals store exists (``pusula data fundamentals``); unchanged otherwise."""
    from dataclasses import replace

    from pusula.data.fundamentals import fundamental_fields, store_dir

    root = root or data_dir()
    if not (store_dir(root) / "manifest.json").exists():
        return rd
    return replace(rd, panel=rd.panel.with_fields(**fundamental_fields(rd.panel, root)))
