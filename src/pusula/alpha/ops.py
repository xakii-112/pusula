"""Causal, NaN-aware time-series and cross-sectional operators on T x N arrays.

Every time-series operator's output at row t depends on input rows <= t only. Windows
require ``min_periods`` finite observations; otherwise the output is NaN (no partial
windows passed off as full ones).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

FloatArray = npt.NDArray[np.float64]


def lag(x: FloatArray, k: int = 1) -> FloatArray:
    """out[t] = x[t - k] (k >= 0)."""
    if k < 0:
        raise ValueError("negative lag would read the future")
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k == 0:
        out[:] = x
    elif k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def ffill(x: FloatArray) -> FloatArray:
    """Carry the last finite value forward in time (causal)."""
    out = x.astype(np.float64, copy=True)
    for i in range(1, out.shape[0]):
        m = ~np.isfinite(out[i])
        out[i, m] = out[i - 1, m]
    return out


def hold_last_event(value: FloatArray, event: npt.NDArray[np.bool_], hold: int) -> FloatArray:
    """At row t, ``value`` at the most recent event row s <= t if t - s <= ``hold`` (NaN if that
    value is NaN: a newer event without a value is not covered by an older one)."""
    T = value.shape[0]
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(event, rows, -1), axis=0)
    picked = np.take_along_axis(value, np.maximum(last, 0), axis=0)
    return np.where((last >= 0) & (rows - last <= hold), picked, np.nan)


def log_return(close: FloatArray, k: int = 1) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.where(close > 0, np.log(close), np.nan)
    return lc - lag(lc, k)


def _cum(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    finite = np.isfinite(x)
    v = np.where(finite, x, 0.0)
    zeros = np.zeros((1, x.shape[1]))
    return (
        np.concatenate([zeros, np.cumsum(v, axis=0)]),
        np.concatenate([zeros, np.cumsum(finite, axis=0, dtype=np.float64)]),
    )


def rolling_sum(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    mp = window if min_periods is None else min_periods
    cs, cn = _cum(x)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(0, hi - window)
    s = cs[hi] - cs[lo]
    n = cn[hi] - cn[lo]
    return np.where(n >= mp, s, np.nan)


def rolling_count(x: FloatArray, window: int) -> FloatArray:
    _, cn = _cum(x)
    T = x.shape[0]
    hi = np.arange(1, T + 1)
    lo = np.maximum(0, hi - window)
    return cn[hi] - cn[lo]


def rolling_mean(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    mp = window if min_periods is None else min_periods
    s = rolling_sum(x, window, mp)
    n = rolling_count(x, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n


def rolling_std(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    """Sample standard deviation (ddof=1), computed from demeaned sums for stability."""
    mp = window if min_periods is None else min_periods
    n = rolling_count(x, window)
    m = rolling_mean(x, window, mp)
    s2 = rolling_sum(x * x, window, mp)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - n * m * m) / (n - 1)
    return np.sqrt(np.maximum(var, 0.0))


def rolling_cov(
    x: FloatArray, y: FloatArray, window: int, min_periods: int | None = None
) -> FloatArray:
    """Rolling sample covariance over rows where both are finite. ``y`` may be T x 1."""
    yb = np.broadcast_to(y, x.shape)
    both = np.isfinite(x) & np.isfinite(yb)
    xv = np.where(both, x, np.nan)
    yv = np.where(both, yb, np.nan)
    mp = window if min_periods is None else min_periods
    n = rolling_count(xv, window)
    sx = rolling_sum(xv, window, mp)
    sy = rolling_sum(yv, window, mp)
    sxy = rolling_sum(xv * yv, window, mp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sxy - sx * sy / n) / (n - 1)


def rolling_max(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    mp = window if min_periods is None else min_periods
    return pd.DataFrame(x).rolling(window, min_periods=mp).max().to_numpy()


def rolling_min(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    mp = window if min_periods is None else min_periods
    return pd.DataFrame(x).rolling(window, min_periods=mp).min().to_numpy()


def ewm_mean(x: FloatArray, halflife: float, min_periods: int = 1) -> FloatArray:
    df = pd.DataFrame(x)
    return df.ewm(halflife=halflife, min_periods=min_periods, ignore_na=True).mean().to_numpy()


def ewm_var(x: FloatArray, halflife: float, min_periods: int = 20) -> FloatArray:
    """Exponentially weighted second moment around zero (RiskMetrics style)."""
    df = pd.DataFrame(x * x)
    return df.ewm(halflife=halflife, min_periods=min_periods, ignore_na=True).mean().to_numpy()


# ---------------------------------------------------------------------- cross-sectional
def cs_count(x: FloatArray) -> FloatArray:
    return np.isfinite(x).sum(axis=1, keepdims=True).astype(np.float64)


def cs_mean(x: FloatArray) -> FloatArray:
    """Row mean over finite entries (T x 1); NaN for rows without any (no warning)."""
    n = cs_count(x)
    tot = np.where(np.isfinite(x), x, 0.0).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, tot / n, np.nan)


def cs_std(x: FloatArray, ddof: int = 1) -> FloatArray:
    n = cs_count(x)
    d = np.where(np.isfinite(x), x - cs_mean(x), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > ddof, np.sqrt((d * d).sum(axis=1, keepdims=True) / (n - ddof)), np.nan)


def _masked(x: FloatArray, mask: FloatArray | None) -> FloatArray:
    if mask is None:
        return x.astype(np.float64, copy=True)
    return np.where(mask > 0, x, np.nan)


def cs_rank(x: FloatArray, mask: FloatArray | None = None) -> FloatArray:
    """Cross-sectional rank scaled to (0, 1): (rank - 0.5) / n, ties averaged."""
    v = _masked(x, mask)
    ranks = pd.DataFrame(v).rank(axis=1, method="average", na_option="keep").to_numpy()
    n = np.isfinite(v).sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (ranks - 0.5) / n, np.nan)


def cs_gauss(x: FloatArray, mask: FloatArray | None = None) -> FloatArray:
    """Normal scores: inverse-normal of cross-sectional ranks (robust to outliers)."""
    return np.asarray(special.ndtri(cs_rank(x, mask)), dtype=np.float64)


def cs_zscore(x: FloatArray, mask: FloatArray | None = None, clip: float = 3.0) -> FloatArray:
    v = _masked(x, mask)
    sd = cs_std(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (v - cs_mean(v)) / np.where(sd > 0, sd, np.nan)
    return np.clip(z, -clip, clip)


def cs_demean(x: FloatArray, mask: FloatArray | None = None) -> FloatArray:
    v = _masked(x, mask)
    return v - cs_mean(v)


def cs_neutralize(
    x: FloatArray, exposures: list[FloatArray], mask: FloatArray | None = None, ridge: float = 1e-8
) -> FloatArray:
    """Residual of a per-date cross-sectional regression of ``x`` on an intercept + exposures."""
    v = _masked(x, mask)
    T, _ = v.shape
    out = np.full_like(v, np.nan)
    E = np.stack(exposures, axis=2) if exposures else np.zeros((*v.shape, 0))
    for t in range(T):
        ok = np.isfinite(v[t]) & np.all(np.isfinite(E[t]), axis=1)
        if ok.sum() <= E.shape[2] + 2:
            continue
        X = np.column_stack([np.ones(ok.sum()), E[t][ok]])
        y = v[t][ok]
        coef = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
        out[t, ok] = y - X @ coef
    return out
