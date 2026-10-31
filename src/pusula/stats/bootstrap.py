"""Stationary bootstrap (Politis & Romano 1994) for dependent series."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> npt.NDArray[np.intp]:
    """One resample of indices 0..n-1 with geometric block lengths of mean ``mean_block``."""
    if n <= 0:
        return np.empty(0, dtype=np.intp)
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.intp)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if restart[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_ci(
    x: npt.ArrayLike,
    stat: Callable[[npt.NDArray[np.float64]], float],
    *,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile interval of ``stat``."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    point = stat(a)
    draws = np.array([stat(a[stationary_indices(a.size, mean_block, rng)]) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class SharpeDifference:
    diff: float  # annualised Sharpe of a minus Sharpe of b
    lo: float
    hi: float
    p_better: float  # share of resamples in which a beats b
    n: int


def sharpe_difference(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    alpha: float = 0.05,
    seed: int = 0,
    periods: int = 252,
) -> SharpeDifference:
    """Paired stationary bootstrap of a Sharpe-ratio difference: both series are resampled on the
    same days, so their correlation and autocorrelation stay in the interval (a normal-theory
    test such as Memmel's assumes neither fat tails nor serial dependence away)."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 3:
        return SharpeDifference(math.nan, math.nan, math.nan, math.nan, n)

    def stat(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float:
        su, sv = u.std(ddof=1), v.std(ddof=1)
        if su <= 0 or sv <= 0:
            return math.nan
        return float((u.mean() / su - v.mean() / sv) * math.sqrt(periods))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = stationary_indices(n, mean_block, rng)
        draws[i] = stat(x[idx], y[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return SharpeDifference(stat(x, y), math.nan, math.nan, math.nan, n)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return SharpeDifference(stat(x, y), float(lo), float(hi), float((draws > 0).mean()), n)
