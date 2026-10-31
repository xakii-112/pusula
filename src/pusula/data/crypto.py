"""A survivorship-free crypto universe from Binance's public data.

**Universe.** Every USDT spot pair the exchange has ever listed, delisted ones included, minus
stablecoins, fiat pairs and leveraged tokens. On the first session of each month the members are
the ``TOP_N`` pairs by mean dollar volume over the previous ``ADV_WINDOW`` sessions, known at the
time; a coin that dies mid-month stays until its prices end, and the backtester closes it there.

**Relistings.** A pair whose bars stop for ``RELIST_GAP`` days or more ends there, and what follows
is a new security, ``SYMBOL#2``. Binance reuses names: LUNAUSDT came back on 2022-05-31 as another
coin at 177,400 times the last price, and token swaps (COCOS, DREP, QUICK, SUN, BNX) do the same at
ratios of up to 1,300. The first results of ``us-crypto-v1`` held such a jump and showed it as a
return. The rule looks only at the gap, so it also ends a few plain halts (CVC, KEY, FTT).

**Sessions.** Bars are folded onto the NYSE calendar: Saturday, Sunday and Monday make Monday's bar.
Nothing is lost (returns compound across the fold), and everything built on sessions keeps working
as it is: annualisation, the risk-free rate, factor regressions against US equities. A book trades
on session days only, which a real desk need not do. A session is complete at 00:00 UTC of the next
day, so the nightly run sees it a day after the US market closes.

**Units.** Each coin's prices are rescaled by a constant so that its first bar is 100, and its
volumes by the inverse. Returns and dollar volume do not change. The paper ledger holds whole units
at micro-dollar prices; neither a one-unit Bitcoin nor a coin at $0.00001 survives that otherwise.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.core import calendar as cal
from pusula.core.errors import DataError
from pusula.data.binance import BinanceClient
from pusula.data.panel import FloatArray, Panel
from pusula.data.pipeline import ResearchData
from pusula.data.quality import check_and_repair
from pusula.data.settings import data_dir
from pusula.data.store import BarStore

UNIVERSE = "crypto:binance-usdt-top100"
TOP_N = 100
ADV_WINDOW = 30
MIN_HISTORY = 60
MIN_ADV = 1_000_000.0  # dollars a day: below this a position cannot be built at any size
DATA_START = pd.Timestamp("2018-01-01")
FIRST_PRICE = 100.0
RELIST_GAP = 3  # days without a bar
BTC = "BTCUSDT"

STABLE = frozenset(
    {
        "USDC",
        "BUSD",
        "TUSD",
        "DAI",
        "FDUSD",
        "USDP",
        "PAX",
        "USDS",
        "USDSB",
        "GUSD",
        "SUSD",
        "USTC",
        "UST",
        "USDD",
        "AEUR",
        "EUR",
        "GBP",
        "AUD",
        "TRY",
        "BRL",
        "RUB",
        "NGN",
        "ZAR",
        "UAH",
        "BIDR",
        "IDRT",
        "BVND",
        "EURI",
        "XUSD",
        "USD1",
        "BFUSD",
        "RLUSD",
        "USDE",
        "PYUSD",
    }
)
_LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")


def is_member_pair(info: dict[str, Any]) -> bool:
    """A USDT pair of a coin whose price is not meant to stand still, nor to be levered."""
    base = str(info["baseAsset"])
    if info.get("quoteAsset") != "USDT" or base in STABLE:
        return False
    return not (len(base) > 3 and base.endswith(_LEVERAGED))


def store_of(root: Path | None = None) -> BarStore:
    return BarStore((root or data_dir()) / "crypto")


def _to_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Binance's day into the bar store's columns; VWAP from the day's own two volumes."""
    out = frame[["open", "high", "low", "close", "volume", "trades"]].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vwap"] = (frame["quote_volume"] / frame["volume"]).where(frame["volume"] > 0)
    return out


def update_crypto(
    root: Path | None = None,
    end: dt.date | None = None,
    client: BinanceClient | None = None,
    log: bool = True,
) -> int:
    """Fetch what each pair lacks up to ``end`` (default: yesterday, the last finished UTC day).
    Spot klines are final and not adjusted, so a pair only ever grows."""
    store = store_of(root)
    end = end or (dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1))
    stop = pd.Timestamp(end)
    fetched_at = dt.datetime.now(dt.UTC)
    added = 0
    own = client or BinanceClient()
    try:
        pairs = sorted(str(s["symbol"]) for s in own.symbols() if is_member_pair(s))
        for k, symbol in enumerate(pairs):
            old = store.get(symbol) if store.has(symbol) else None
            start = (
                DATA_START if old is None or old.empty else old.index.max() + pd.Timedelta(days=1)
            )
            if start > stop:
                continue
            new = _to_bars(own.klines(symbol, start, stop))
            if new.empty and old is not None:
                continue
            merged = new if old is None or old.empty else pd.concat([old, new]).sort_index()
            store.put(
                symbol,
                symbol,
                None,
                merged,
                adjustment="raw",
                feed="binance",
                fetched_at=fetched_at,
            )
            added += len(new)
            if k % 50 == 0:
                store.flush()
                if log:
                    print(f"[crypto] {k}/{len(pairs)} pairs", flush=True)
    finally:
        if client is None:
            own.close()
    store.flush()
    return added


def session_of(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first NYSE session on or after each UTC day."""
    lo, hi = days.min().date(), days.max().date() + dt.timedelta(days=10)
    sessions = pd.DatetimeIndex(cal.sessions(lo, hi))
    return sessions[sessions.searchsorted(days)]


def fold_to_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """One UTC-daily coin onto sessions. A folded bar opens with the first day, closes with the
    last, and its VWAP is dollars over units. Only sessions whose last day is on file count."""
    if frame.empty:
        return frame
    dollars = (frame["vwap"] * frame["volume"]).fillna(0.0)
    keys = session_of(pd.DatetimeIndex(frame.index))
    g = frame.groupby(keys)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "trades": g["trades"].sum(),
        }
    )
    out["dollars"] = dollars.groupby(keys).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vwap"] = (out["dollars"] / out["volume"]).where(out["volume"] > 0)
    return out.drop(columns="dollars")[out.columns.drop("dollars")]


def split_relistings(symbol: str, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A pair's bars as one security per stretch without a ``RELIST_GAP``."""
    gaps = pd.Series(frame.index).diff().dt.days.to_numpy()
    cuts = [0, *np.nonzero(gaps >= RELIST_GAP)[0].tolist(), len(frame)]
    out: dict[str, pd.DataFrame] = {}
    for k, (a, b) in enumerate(pairwise(cuts)):
        out[symbol if k == 0 else f"{symbol}#{k + 1}"] = frame.iloc[a:b]
    return out


def _rescaled(frame: pd.DataFrame) -> pd.DataFrame:
    first = float(frame["close"].dropna().iloc[0])
    if first <= 0:
        raise DataError("a first price of zero")
    k = FIRST_PRICE / first
    out = frame.copy()
    for name in ("open", "high", "low", "close", "vwap"):
        out[name] = out[name] * k
    out["volume"] = out["volume"] / k
    return out


def monthly_members(dollar_volume: FloatArray, dates: np.ndarray, closes: FloatArray) -> FloatArray:
    """T x N membership: reconstituted on each month's first session from data up to the session
    before it. A member stays until its price ends."""
    T, N = dollar_volume.shape
    mask = np.zeros((T, N))
    months = dates.astype("datetime64[M]")
    finite = np.isfinite(closes)
    current = np.zeros(N, dtype=bool)
    for t in range(T):
        if t >= 1 and months[t] != months[t - 1]:
            lo = max(0, t - ADV_WINDOW)
            window = dollar_volume[lo:t]
            seen = np.isfinite(window).sum(axis=0)
            with np.errstate(invalid="ignore"):
                adv = np.nansum(window, axis=0) / np.maximum(seen, 1)
            history = finite[:t].sum(axis=0)
            ok = (seen > 0) & (adv >= MIN_ADV) & (history >= MIN_HISTORY) & finite[t - 1]
            ranked = np.argsort(-np.where(ok, adv, -1.0))[:TOP_N]
            current = np.zeros(N, dtype=bool)
            current[ranked[ok[ranked]]] = True
        mask[t] = current & finite[t]
    return mask


def crypto_data(root: Path | None = None, end: str | None = None) -> ResearchData:
    store = store_of(root)
    ids = sorted(store.ids())
    if not ids:
        raise DataError("no crypto store: run `pusula data crypto`")
    bars: dict[str, pd.DataFrame] = {}
    latest_raw = pd.Timestamp.min
    for sid in ids:
        raw = store.get(sid)
        raw = raw.dropna(subset=["close"]) if len(raw) else raw
        if end is not None and len(raw):
            raw = raw.loc[: pd.Timestamp(end)]
        if len(raw):
            latest_raw = max(latest_raw, raw.index.max())
            for name, stretch in split_relistings(sid, raw).items():
                bars[name] = _rescaled(fold_to_sessions(stretch))
    symbols = sorted(bars)
    wide = {
        f: pd.DataFrame({s: bars[s][f] for s in symbols}).sort_index()
        for f in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    index = wide["close"].index
    # the last session counts only if its own UTC day is on file (Monday needs Monday)
    last_day = max(bars[s].index.max() for s in symbols)
    if last_day > latest_raw:
        index = index[index < last_day]
    wide = {f: v.reindex(index) for f, v in wide.items()}
    fields = {f: v.to_numpy(dtype=np.float64) for f, v in wide.items()}
    dates = index.to_numpy().astype("datetime64[D]")
    dollars = fields["vwap"] * fields["volume"]
    mask = monthly_members(dollars, dates, fields["close"])
    panel = Panel(dates, tuple(symbols), {**fields, "in_universe": mask})
    cleaned, report = check_and_repair(panel)
    return ResearchData(
        cleaned,
        cleaned,
        report,
        dict.fromkeys(symbols, "Crypto"),
        store.manifest_digest(ids),
        (),
    )


def crypto_factors(panel: Panel, french: pd.DataFrame) -> pd.DataFrame:
    """The factors a crypto book must beat: the US market, Bitcoin, and the equal-weight crypto
    market. On FF5+MOM alone any crypto holding would be "alpha": it barely loads on them."""
    close = panel["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = close[1:] / close[:-1] - 1.0
    idx = pd.DatetimeIndex(panel.dates[1:].astype("datetime64[ns]"))
    member = panel["in_universe"][:-1] > 0  # the members known before the day
    held = np.where(member, r, np.nan)
    count = np.isfinite(held).sum(axis=1)  # no members yet in the first weeks: no market to measure
    with np.errstate(invalid="ignore", divide="ignore"):
        ew = np.where(count > 0, np.nansum(held, axis=1) / count, np.nan)
    out = french.reindex(idx)[["mkt_rf", "rf"]].copy()
    rf = out["rf"].ffill().fillna(0.0).to_numpy()
    out["rf"] = rf
    out["btc"] = r[:, panel.symbol_index(BTC)] - rf
    out["cmkt"] = ew - rf
    return out


CRYPTO_FACTORS: tuple[str, ...] = ("mkt_rf", "btc", "cmkt")


def listed_pairs(client: BinanceClient) -> Sequence[str]:
    return sorted(str(s["symbol"]) for s in client.symbols() if is_member_pair(s))
