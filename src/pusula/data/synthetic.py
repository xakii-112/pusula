"""Synthetic equity market with a known answer.

Returns follow a factor model with fat-tailed specific shocks and a *planted* predictive
signal of chosen strength::

    r[t+1, i] = beta_i * m[t+1] + sum_k B[i, k] g[t+1, k] + ic * sigma_i * z[t, i] + eps[t+1, i]

``z`` is a persistent AR(1) cross-sectional score published as the field ``planted`` at the
close of ``t``. With ``ic = 0`` the market is pure noise. The research stack must recover the
signal when it exists and must refuse to promote anything when it does not: that is how the
gates themselves are tested.

Securities list and delist at random, so survivorship handling is exercised too.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from pusula.core import calendar as cal
from pusula.data.panel import Panel


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    n_symbols: int = 200
    start: str = "2015-01-02"
    end: str = "2020-12-31"
    n_factors: int = 4
    ic: float = 0.0  # signal strength: predicted share of a name's daily vol
    signal_persistence: float = 0.97  # AR(1) coefficient of the planted score
    market_vol: float = 0.011
    market_drift: float = 0.0003  # ~7.5% a year equity premium (plus the vol drag correction)
    factor_vol: float = 0.005
    specific_vol_range: tuple[float, float] = (0.010, 0.030)
    tail_df: float = 5.0
    listing_churn: float = 0.15  # fraction of names with a late listing or early delisting
    seed: int = 0


def generate(spec: SyntheticSpec) -> Panel:
    rng = np.random.default_rng(spec.seed)
    dates = cal.sessions(dt.date.fromisoformat(spec.start), dt.date.fromisoformat(spec.end))
    T, N, K = dates.size, spec.n_symbols, spec.n_factors
    beta = rng.normal(1.0, 0.3, N)
    B = rng.normal(0.0, 1.0, (N, K))
    spec_vol = rng.uniform(*spec.specific_vol_range, N)
    total_vol = np.sqrt(
        (beta * spec.market_vol) ** 2 + (B**2).sum(1) * spec.factor_vol**2 + spec_vol**2
    )

    m = spec.market_drift + rng.standard_t(spec.tail_df, T) * spec.market_vol / np.sqrt(
        spec.tail_df / (spec.tail_df - 2)
    )
    g = rng.normal(0.0, spec.factor_vol, (T, K))
    t_scale = np.sqrt(spec.tail_df / (spec.tail_df - 2))
    eps = rng.standard_t(spec.tail_df, (T, N)) / t_scale * spec_vol

    phi = spec.signal_persistence
    z = np.empty((T, N))
    z[0] = rng.normal(size=N)
    shocks = rng.normal(size=(T, N)) * np.sqrt(1 - phi**2)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + shocks[t]

    r = beta[None, :] * m[:, None] + g @ B.T + eps
    r[1:] += spec.ic * total_vol[None, :] * z[:-1]
    r = np.clip(r, -0.5, 0.5)

    close = 50.0 * np.exp(np.cumsum(np.log1p(r), axis=0))
    gap = rng.normal(0.0, 0.3, (T, N)) * np.abs(r)
    open_ = close / (1 + r) * (1 + gap)
    open_[0] = close[0] / (1 + r[0])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vwap = (open_ + close + hi + lo) / 4.0
    adv_shares = rng.lognormal(np.log(2e6), 0.6, N)
    volume = adv_shares[None, :] * rng.lognormal(0.0, 0.35, (T, N))

    alive = np.ones((T, N), dtype=bool)
    churn = rng.random(N) < spec.listing_churn
    for j in np.nonzero(churn)[0]:
        if rng.random() < 0.5:
            alive[: rng.integers(T // 10, T // 2), j] = False
        else:
            alive[rng.integers(T // 2, T - T // 10) :, j] = False

    def dead(a: np.ndarray) -> np.ndarray:
        out = a.astype(np.float64).copy()
        out[~alive] = np.nan
        return out

    symbols = tuple(f"S{j:04d}" for j in range(N))
    zpub = z.copy()
    zpub[~alive] = np.nan
    return Panel(
        dates,
        symbols,
        {
            "open": dead(open_),
            "high": dead(hi),
            "low": dead(lo),
            "close": dead(close),
            "vwap": dead(vwap),
            "volume": dead(volume),
            "trades": dead(volume / 100.0),
            "planted": zpub,
            "in_universe": alive.astype(np.float64),
        },
    )
