"""Sharpe ratio inference that accounts for non-normality and selection.

References
----------
* Lo (2002), "The Statistics of Sharpe Ratios".
* Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR, MinTRL).
* Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" (DSR, expected max SR).

All Sharpe ratios here are **per period** (not annualised) unless the name says otherwise:
the inference formulas are stated per observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


def _clean(r: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(r, dtype=np.float64)
    return x[np.isfinite(x)]


def sharpe(r: npt.ArrayLike) -> float:
    """Per-period Sharpe ratio (mean / sample std, ddof=1). NaN if undefined."""
    x = _clean(r)
    if x.size < 2:
        return math.nan
    sd = float(x.std(ddof=1))
    # A (numerically) constant series took no risk: its Sharpe ratio is undefined, not the ratio
    # of two round-off errors.
    if not sd > 1e-12:
        return math.nan
    return float(x.mean()) / sd


def annualised_sharpe(r: npt.ArrayLike, periods: int = TRADING_DAYS) -> float:
    return sharpe(r) * math.sqrt(periods)


@dataclass(frozen=True, slots=True)
class Moments:
    n: int
    sr: float
    skew: float
    kurt: float  # Pearson (non-excess) kurtosis; 3 for a normal


def moments(r: npt.ArrayLike) -> Moments:
    x = _clean(r)
    if x.size < 3:
        return Moments(int(x.size), math.nan, math.nan, math.nan)
    return Moments(
        n=int(x.size),
        sr=sharpe(x),
        skew=float(sps.skew(x, bias=False)),
        kurt=float(sps.kurtosis(x, fisher=False, bias=False)),
    )


def sr_std(sr: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of the per-period SR estimator under non-normal iid returns (Mertens)."""
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)
    return math.sqrt(max(var, 0.0))


def psr(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark) given the observed SR."""
    if n < 3 or not math.isfinite(sr):
        return math.nan
    se = sr_std(sr, n, skew, kurt)
    if se == 0:
        return 1.0 if sr > sr_benchmark else 0.0
    return float(sps.norm.cdf((sr - sr_benchmark) / se))


def psr_of(r: npt.ArrayLike, sr_benchmark: float = 0.0) -> float:
    m = moments(r)
    return psr(m.sr, m.n, m.skew, m.kurt, sr_benchmark)


def expected_max_sr(n_trials: int, var_trials: float) -> float:
    """E[max of n_trials SR estimates] under the null that every true SR is zero.

    ``var_trials`` is the cross-sectional variance of the trials' per-period SRs.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    sd = math.sqrt(max(var_trials, 0.0))
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(r: npt.ArrayLike, n_trials: int, var_trials: float) -> float:
    """Deflated Sharpe Ratio: PSR against the expected maximum SR of ``n_trials`` null trials."""
    m = moments(r)
    return psr(m.sr, m.n, m.skew, m.kurt, expected_max_sr(n_trials, var_trials))


def min_track_record(
    sr: float, skew: float, kurt: float, sr_benchmark: float = 0.0, prob: float = 0.95
) -> float:
    """Minimum number of observations for PSR(sr_benchmark) >= prob (MinTRL)."""
    if not sr > sr_benchmark:
        return math.inf
    z = sps.norm.ppf(prob)
    return float(
        1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) * (z / (sr - sr_benchmark)) ** 2
    )


def max_drawdown(r: npt.ArrayLike) -> float:
    """Maximum peak-to-trough decline of the compounded equity curve (a positive fraction)."""
    x = _clean(r)
    if x.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.max(1.0 - eq / peak))


def annualised_return(r: npt.ArrayLike, periods: int = TRADING_DAYS) -> float:
    x = _clean(r)
    if x.size == 0:
        return math.nan
    growth = float(np.prod(1.0 + x))
    if growth <= 0:
        return -1.0
    return growth ** (periods / x.size) - 1.0


def annualised_vol(r: npt.ArrayLike, periods: int = TRADING_DAYS) -> float:
    x = _clean(r)
    return float(x.std(ddof=1) * math.sqrt(periods)) if x.size > 1 else math.nan
