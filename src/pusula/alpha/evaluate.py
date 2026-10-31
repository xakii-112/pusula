"""Signal evaluation: information coefficients, decay, turnover, quantile spreads.

Timing convention (the same one the backtester uses): a score at row t is known at the close
of session t; the earliest trade is at the open of session ``t + lag``. The forward return of
row t over horizon h is therefore ``px[t + lag + h] / px[t + lag] - 1`` at the execution price
(open by default). Measuring close-to-close from t would credit the signal with the overnight
move it could never have traded: the classic one-bar look-ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pusula.alpha import ops
from pusula.data.panel import FloatArray, Panel
from pusula.stats.inference import nw_lags, nw_mean_test


def forward_returns(
    panel: Panel, horizon: int = 1, lag: int = 1, price: str = "open"
) -> FloatArray:
    px = panel[price]
    T = px.shape[0]
    out = np.full_like(px, np.nan)
    a, b = lag, lag + horizon
    if b < T:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[: T - b] = px[b:] / px[a : T - b + a] - 1.0
    return out


def daily_ic(
    score: FloatArray,
    fwd: FloatArray,
    mask: FloatArray | None = None,
    method: str = "spearman",
    min_names: int = 20,
) -> FloatArray:
    """Per-date cross-sectional correlation between score and forward return."""
    s = np.where(mask > 0, score, np.nan) if mask is not None else score
    ok = np.isfinite(s) & np.isfinite(fwd)
    if method == "spearman":
        s = ops.cs_rank(np.where(ok, s, np.nan))
        f = ops.cs_rank(np.where(ok, fwd, np.nan))
    else:
        s = np.where(ok, s, np.nan)
        f = np.where(ok, fwd, np.nan)
    n = ok.sum(axis=1)
    sd = np.where(ok, s - ops.cs_mean(s), 0.0)
    fd = np.where(ok, f - ops.cs_mean(f), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = (sd * fd).sum(axis=1)
        den = np.sqrt((sd * sd).sum(axis=1) * (fd * fd).sum(axis=1))
        ic = num / den
    return np.where(n >= min_names, ic, np.nan)


@dataclass(frozen=True, slots=True)
class ICSummary:
    mean: float
    std: float
    ir_annual: float
    t_nw: float
    hit_rate: float
    n_days: int
    avg_names: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__slots__}


def summarize_ic(ic: FloatArray, n_names: FloatArray | None = None, horizon: int = 1) -> ICSummary:
    x = ic[np.isfinite(ic)]
    if x.size < 3:
        return ICSummary(math.nan, math.nan, math.nan, math.nan, math.nan, int(x.size), math.nan)
    # overlapping h-day labels: NW with at least h lags
    test = nw_mean_test(x, lags=None if horizon <= 1 else max(horizon, nw_lags(int(x.size))))
    sd = float(x.std(ddof=1))
    return ICSummary(
        mean=float(x.mean()),
        std=sd,
        ir_annual=float(x.mean() / sd * math.sqrt(252 / horizon)) if sd > 0 else math.nan,
        t_nw=test.t,
        hit_rate=float((x > 0).mean()),
        n_days=int(x.size),
        avg_names=float(np.mean(n_names[np.isfinite(n_names)]))
        if n_names is not None
        else math.nan,
    )


def ic_decay(
    score: FloatArray,
    panel: Panel,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 21),
    lag: int = 1,
    mask: FloatArray | None = None,
) -> dict[int, float]:
    """Mean IC of the score against the h-day forward return for each horizon."""
    out = {}
    for h in horizons:
        ic = daily_ic(score, forward_returns(panel, h, lag), mask)
        out[h] = float(np.mean(ic[np.isfinite(ic)])) if np.isfinite(ic).any() else math.nan
    return out


def rank_autocorrelation(score: FloatArray, mask: FloatArray | None = None, k: int = 1) -> float:
    """Mean cross-sectional rank correlation between the score at t and t-k (turnover proxy)."""
    r = ops.cs_rank(score, mask)
    ic = daily_ic(r, ops.lag(r, k), None, method="pearson", min_names=10)
    return float(np.mean(ic[np.isfinite(ic)])) if np.isfinite(ic).any() else math.nan


def quantile_spread(
    score: FloatArray, fwd1: FloatArray, mask: FloatArray | None = None, n_q: int = 5
) -> FloatArray:
    """Daily mean forward return per score quantile (T x n_q), equal-weighted."""
    r = ops.cs_rank(score, mask)
    out = np.full((score.shape[0], n_q), np.nan)
    bucket = np.floor(r * n_q)
    bucket = np.where(bucket >= n_q, n_q - 1, bucket)
    for q in range(n_q):
        sel = (bucket == q) & np.isfinite(fwd1)
        cnt = sel.sum(axis=1)
        tot = np.where(sel, fwd1, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, q] = np.where(cnt > 0, tot / cnt, np.nan)
    return out


def trailing_ic(
    score: FloatArray,
    panel: Panel,
    horizon: int,
    lag: int = 1,
    window: int = 252,
    prior_days: int = 252,
    min_obs: int = 63,
    cap: float = 0.10,
    mask: FloatArray | None = None,
) -> FloatArray:
    """Causal estimate of a score's information coefficient, per decision row.

    The rank IC of row ``s`` is known only once its forward return has been realised, at row
    ``s + lag + horizon``; the estimate at ``t`` therefore averages the ICs of rows
    ``<= t - lag - horizon`` over a trailing window, shrinks the average towards zero by
    ``n / (n + prior_days)`` (a zero-IC prior worth ``prior_days`` observations) and floors it at
    zero: a signal that has not been working is not traded, rather than traded backwards.
    """
    ic = daily_ic(score, forward_returns(panel, horizon, lag), mask)[:, None]
    known = ops.lag(ic, lag + horizon)
    mean = ops.rolling_mean(known, window, min_obs)[:, 0]
    n = ops.rolling_count(known, window)[:, 0]
    with np.errstate(invalid="ignore"):
        shrunk = mean * n / (n + prior_days)
    return np.clip(np.where(np.isfinite(shrunk), shrunk, 0.0), 0.0, cap)
