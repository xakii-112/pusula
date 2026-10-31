"""Black-Scholes-Merton pricing, Greeks and implied volatility (European, continuous yield).

Vectorised over numpy arrays. ``cp`` is +1 for calls and -1 for puts. Time in years.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import special

FloatArray = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike


def _ncdf(x: FloatArray) -> FloatArray:
    return np.asarray(special.ndtr(x), dtype=np.float64)


def _npdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _d1d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, vol: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    S_, K_, T_, r_, q_, v_ = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q, vol))
    sq = v_ * np.sqrt(T_)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * v_ * v_) * T_) / sq
    return d1, d1 - sq


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    vol: ArrayLike,
    cp: ArrayLike = 1,
) -> FloatArray:
    S_, K_, T_, r_, q_, v_, c = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q, vol, cp))
    expiry = np.maximum(c * (S_ - K_), 0.0)
    fwd_intrinsic = np.maximum(c * (S_ * np.exp(-q_ * T_) - K_ * np.exp(-r_ * T_)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1d2(S_, K_, np.maximum(T_, 1e-12), r_, q_, np.maximum(v_, 1e-12))
        val = c * (S_ * np.exp(-q_ * T_) * _ncdf(c * d1) - K_ * np.exp(-r_ * T_) * _ncdf(c * d2))
    return np.where(T_ <= 0, expiry, np.where(v_ <= 0, fwd_intrinsic, val))


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    vol: ArrayLike,
    cp: ArrayLike = 1,
) -> FloatArray:
    c = np.asarray(cp, dtype=np.float64)
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return (
        np.exp(-np.asarray(q, dtype=np.float64) * np.asarray(T, dtype=np.float64))
        * c
        * _ncdf(c * d1)
    )


def gamma(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, vol: ArrayLike
) -> FloatArray:
    S_, T_, q_, v_ = (np.asarray(a, dtype=np.float64) for a in (S, T, q, vol))
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return np.exp(-q_ * T_) * _npdf(d1) / (S_ * v_ * np.sqrt(T_))


def vega(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, vol: ArrayLike
) -> FloatArray:
    """dPrice / dVol (per 1.00 of vol)."""
    S_, T_, q_ = (np.asarray(a, dtype=np.float64) for a in (S, T, q))
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return S_ * np.exp(-q_ * T_) * _npdf(d1) * np.sqrt(T_)


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    vol: ArrayLike,
    cp: ArrayLike = 1,
) -> FloatArray:
    """dPrice / dt (per year, calendar time passing)."""
    S_, K_, T_, r_, q_, v_, c = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q, vol, cp))
    d1, d2 = _d1d2(S_, K_, T_, r_, q_, v_)
    term1 = -S_ * np.exp(-q_ * T_) * _npdf(d1) * v_ / (2.0 * np.sqrt(T_))
    return (
        term1
        - c * r_ * K_ * np.exp(-r_ * T_) * _ncdf(c * d2)
        + c * q_ * S_ * np.exp(-q_ * T_) * _ncdf(c * d1)
    )


def implied_vol(
    target: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike = 1,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> FloatArray:
    """Safeguarded Newton (falls back to bisection) on a bracket; NaN outside no-arbitrage bounds.

    In-the-money options are inverted through their out-of-the-money counterpart (put-call
    parity): an ITM option's price is mostly intrinsic value, its vega is tiny, and solving it
    directly is ill-conditioned.
    """
    P, S_, K_, T_, r_, q_, c = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (target, S, K, T, r, q, cp))
    )
    fwd_diff = S_ * np.exp(-q_ * T_) - K_ * np.exp(-r_ * T_)  # C - P
    itm = c * fwd_diff > 0
    P = np.where(itm, P - c * fwd_diff, P)
    c = np.where(itm, -c, c)
    a = np.full(P.shape, lo)
    b = np.full(P.shape, hi)
    pa = price(S_, K_, T_, r_, q_, a, c) - P
    pb = price(S_, K_, T_, r_, q_, b, c) - P
    ok = (pa <= 0) & (pb >= 0) & np.isfinite(P)
    x = np.where(ok, 0.3, np.nan)
    for _ in range(max_iter):
        fx = price(S_, K_, T_, r_, q_, x, c) - P
        a = np.where(fx < 0, x, a)
        b = np.where(fx >= 0, x, b)
        vg = vega(S_, K_, T_, r_, q_, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - fx / vg
        inside = (newton > a) & (newton < b) & np.isfinite(newton)
        nxt = np.where(inside, newton, 0.5 * (a + b))
        done = np.abs(nxt - x) < tol
        x = np.where(ok, nxt, np.nan)
        if np.all(done | ~ok):
            break
    return x


def listed_strikes(spot: float, n_each_side: int = 20) -> FloatArray:
    """Approximate listed US equity strike grid around ``spot``."""
    if spot < 25:
        step = 0.5
    elif spot < 200:
        step = 1.0 if spot < 100 else 2.5
    elif spot < 500:
        step = 5.0
    else:
        step = 10.0
    centre = round(spot / step) * step
    return centre + step * np.arange(-n_each_side, n_each_side + 1, dtype=np.float64)


def strike_for_delta(
    spot: float, T: float, r: float, q: float, vol: float, target_delta: float, cp: int = 1
) -> float:
    """Listed strike whose BSM delta is closest to ``target_delta`` (calls: positive)."""
    ks = listed_strikes(spot)
    ks = ks[ks > 0]
    d = delta(spot, ks, T, r, q, vol, cp)
    return float(ks[int(np.argmin(np.abs(d - target_delta)))])
