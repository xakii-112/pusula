"""A stock's volatility surface from one day's quotes: smile per expiry, interpolation in between.

Each slice is an SVI fit to out-of-the-money quotes, against the forward that put-call parity
implies (the chain has no spot, and a parity forward carries the dividends the market expects).
Weights favour tight quotes. Between expiries, total variance is linear in time at fixed
log-moneyness, which keeps a surface without calendar arbitrage free of it. A slice with too few
usable quotes is left out, and asking for a maturity beyond the slices raises, not extrapolates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from pusula.options import black, svi

DAYS = 365.0
MIN_ATM_PAIRS = 1
MAX_QUOTE_SPREAD = 1.0  # of mid: wider than this says nothing about value
VOL_RANGE = (0.01, 5.0)


class SurfaceError(ValueError):
    """Not enough quotes for what was asked."""


def parity_forward(chain: pd.DataFrame, T: float, r: float) -> float:
    """``F = K + e^{rT} (C - P)`` at the strike where calls and puts are closest in price."""
    calls = chain[chain["call_put"] == "Call"].set_index("strike")
    puts = chain[chain["call_put"] == "Put"].set_index("strike")
    both = calls.index.intersection(puts.index)
    if len(both) < MIN_ATM_PAIRS:
        raise SurfaceError("no strike quoted on both sides")
    c = (calls.loc[both, "bid"] + calls.loc[both, "ask"]) / 2.0
    p = (puts.loc[both, "bid"] + puts.loc[both, "ask"]) / 2.0
    gaps = (c - p).astype(np.float64)
    K = float(gaps.abs().idxmin())
    return K + math.exp(r * T) * float(gaps.loc[K])


@dataclass(frozen=True, slots=True)
class SliceFit:
    expiry: pd.Timestamp
    T: float
    forward: float
    params: svi.SVIParams
    rmse_vol: float  # at the quotes, in points of volatility
    n_quotes: int
    butterfly_free: bool  # over the quoted range
    k_lo: float = -0.5
    k_hi: float = 0.5
    lifted: float = 0.0  # total variance added to stay above the previous slice
    butterfly_free_wings: bool = True  # also far beyond the quotes, where nothing is quoted

    def vol(self, K: float | np.ndarray) -> np.ndarray:
        k = np.log(np.asarray(K, dtype=np.float64) / self.forward)
        return self.params.implied_vol(k, self.T)


def fit_slice(
    chain: pd.DataFrame, asof: pd.Timestamp, r: float, floor: svi.SVIParams | None = None
) -> SliceFit:
    """``chain``: one symbol, one date, one expiry, with bid, ask, strike, call_put, vol. ``floor``
    is the previous expiry's smile, which this one's total variance must not fall below."""
    expiry = pd.Timestamp(chain["expiration"].iloc[0])
    T = (expiry - asof).days / DAYS
    if T <= 0:
        raise SurfaceError("expired")
    F = parity_forward(chain, T, r)
    otm = chain[
        ((chain["call_put"] == "Call") & (chain["strike"] >= F))
        | ((chain["call_put"] == "Put") & (chain["strike"] < F))
    ]
    mid = (otm["bid"] + otm["ask"]) / 2.0
    spread = (otm["ask"] - otm["bid"]) / mid.where(mid > 0)
    keep = (
        (otm["bid"] > 0)
        & (spread <= MAX_QUOTE_SPREAD)
        & otm["vol"].between(VOL_RANGE[0], VOL_RANGE[1])
    )
    q = otm[keep]
    if len(q) < svi.MIN_POINTS:
        raise SurfaceError(f"{len(q)} usable quotes")
    k = np.log(q["strike"].to_numpy(dtype=np.float64) / F)
    vols = q["vol"].to_numpy(dtype=np.float64)
    weights = 1.0 / np.maximum(spread[keep].to_numpy(dtype=np.float64), 0.02)
    fit = svi.fit_svi(k, vols * vols * T, weights, floor)
    fitted = fit.params.implied_vol(k, T)
    return SliceFit(
        expiry,
        T,
        F,
        fit.params,
        float(np.sqrt(np.mean((fitted - vols) ** 2))),
        len(q),
        svi.butterfly_arbitrage_free(fit.params, np.linspace(fit.k_lo, fit.k_hi, 201)),
        fit.k_lo,
        fit.k_hi,
        fit.lifted,
        svi.butterfly_arbitrage_free(fit.params),
    )


@dataclass(frozen=True, slots=True)
class VolSurface:
    symbol: str
    asof: pd.Timestamp
    slices: tuple[SliceFit, ...]  # ascending in T

    @property
    def calendar_free(self) -> bool:
        """No fall in total variance over the range both slices of a pair were quoted on."""
        for a, b in pairwise(self.slices):
            lo, hi = max(a.k_lo, b.k_lo), min(a.k_hi, b.k_hi)
            if hi > lo and svi.calendar_violations(
                [(a.T, a.params), (b.T, b.params)], np.linspace(lo, hi, 101)
            ):
                return False
        return True

    def vol(self, K: float, T: float) -> float:
        """Implied volatility at strike ``K`` and maturity ``T`` (years), within the slices."""
        first, last = self.slices[0], self.slices[-1]
        if T <= first.T + 1e-9:
            if T < first.T * 0.5:
                raise SurfaceError("maturity well before the first slice")
            return float(first.vol(K))
        if T >= last.T - 1e-9:
            if T > last.T + 1e-9:
                raise SurfaceError("maturity beyond the last slice")
            return float(last.vol(K))
        i = int(np.searchsorted([s.T for s in self.slices], T))
        a, b = self.slices[i - 1], self.slices[i]
        weight = (T - a.T) / (b.T - a.T)
        # the same log-moneyness on each slice, each against its own forward
        forward = a.forward + weight * (b.forward - a.forward)
        k = math.log(K / forward)
        w = (1.0 - weight) * float(a.params.total_variance(k)) + weight * float(
            b.params.total_variance(k)
        )
        return math.sqrt(max(w, 0.0) / T)

    def price(self, K: float, T: float, r: float, cp: int = 1) -> float:
        """European value from the surface (Black-76 on the interpolated forward)."""
        fs = [s.forward for s in self.slices]
        ts = [s.T for s in self.slices]
        F = float(np.interp(T, ts, fs))
        vol = self.vol(K, T)
        return float(black.price(F * math.exp(-r * T), K, T, r, 0.0, vol, cp))


def build_surface(chain: pd.DataFrame, symbol: str, asof: pd.Timestamp, r: float) -> VolSurface:
    """Every expiry of ``chain`` that supports a fit; raises when none does."""
    fits: list[SliceFit] = []
    for _, group in chain.groupby("expiration", sort=True, observed=True):
        try:
            fits.append(fit_slice(group, asof, r, fits[-1].params if fits else None))
        except SurfaceError:
            continue
    if not fits:
        raise SurfaceError(f"{symbol} {asof.date()}: no expiry with enough quotes")
    return VolSurface(symbol, asof, tuple(sorted(fits, key=lambda s: s.T)))
