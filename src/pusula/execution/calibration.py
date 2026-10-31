"""Calibrate the spread model against real consolidated quotes.

For a sample of (security, session) pairs, the quoted half-spread is measured from SIP NBBO
updates in short intraday windows (time-weighted by update count is avoided: the median over
updates in each window, then the median over windows). The model's prediction for the same pair
uses only daily-bar information known at the previous close: ``sigma / sqrt(trades)``.
``spread_k`` is the median ratio (robust to the heavy-tailed quote data); the log-log slope
checks the functional form.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pusula.alpha import ops
from pusula.data.alpaca import AlpacaClient
from pusula.data.panel import Panel

NY = ZoneInfo("America/New_York")
WINDOWS = ((10, 30), (12, 30), (14, 30))  # ET; avoid the open/close auctions


@dataclass(frozen=True, slots=True)
class SpreadSample:
    security: str
    date: str
    quoted_half_spread: float
    model_base: float  # sigma / sqrt(trades), as of the previous close


@dataclass(frozen=True, slots=True)
class SpreadCalibration:
    k_median: float
    loglog_slope: float
    loglog_intercept: float
    n: int
    median_quoted_bps: float
    p90_quoted_bps: float
    by_year: dict[int, float]


def measured_half_spread(
    client: AlpacaClient, symbol: str, day: dt.date, minutes: int = 2
) -> float:
    vals = []
    for hh, mm in WINDOWS:
        start = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY)
        q = client.quotes(symbol, start, start + dt.timedelta(minutes=minutes), max_pages=1)
        q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
        if len(q) < 5:
            continue
        mid = (q["bid"] + q["ask"]) / 2.0
        vals.append(float(np.median((q["ask"] - q["bid"]) / mid / 2.0)))
    return float(np.median(vals)) if vals else math.nan


def model_base(panel: Panel, halflife: float = 21.0, window: int = 21) -> np.ndarray:
    r = ops.log_return(panel["close"])
    sigma = np.sqrt(ops.ewm_var(r, halflife, min_periods=10))
    n_tr = ops.rolling_mean(panel["trades"], window, window // 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ops.lag(sigma / np.sqrt(n_tr), 1)  # known at the previous close


def calibrate(samples: list[SpreadSample]) -> SpreadCalibration:
    df = pd.DataFrame([asdict(s) for s in samples])
    df = df[
        np.isfinite(df.quoted_half_spread)
        & np.isfinite(df.model_base)
        & (df.quoted_half_spread > 0)
        & (df.model_base > 0)
    ]
    ratio = df.quoted_half_spread / df.model_base
    x, y = np.log(df.model_base.to_numpy()), np.log(df.quoted_half_spread.to_numpy())
    slope, intercept = np.polyfit(x, y, 1) if len(df) > 2 else (math.nan, math.nan)
    years = pd.to_datetime(df.date).dt.year
    by_year = {int(yr): float(np.median(ratio[years == yr])) for yr in sorted(years.unique())}
    return SpreadCalibration(
        k_median=float(np.median(ratio)),
        loglog_slope=float(slope),
        loglog_intercept=float(intercept),
        n=len(df),
        median_quoted_bps=float(np.median(df.quoted_half_spread) * 1e4),
        p90_quoted_bps=float(np.quantile(df.quoted_half_spread, 0.9) * 1e4),
        by_year=by_year,
    )


def sample_quotes(
    panel: Panel,
    client: AlpacaClient,
    days: list[str],
    per_day: int = 40,
    seed: int = 0,
    log: bool = True,
) -> list[SpreadSample]:
    """Measure quoted half-spreads for random index members on the given sessions."""
    rng = np.random.default_rng(seed)
    base = model_base(panel)
    out: list[SpreadSample] = []
    for d in days:
        t = panel.date_index(d)
        members = np.nonzero((panel["in_universe"][t] > 0) & np.isfinite(base[t]))[0]
        pick = rng.choice(members, size=min(per_day, members.size), replace=False)
        for j in pick:
            sid = panel.symbols[j]
            ticker = sid.split("@")[0]
            hs = measured_half_spread(client, ticker, dt.date.fromisoformat(d))
            out.append(SpreadSample(sid, d, hs, float(base[t, j])))
        if log:
            done = [s for s in out if s.date == d and math.isfinite(s.quoted_half_spread)]
            print(f"[spread] {d}: {len(done)}/{len(pick)} measured", flush=True)
    return out
