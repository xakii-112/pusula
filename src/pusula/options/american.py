"""American options on a binomial tree, with a continuous yield and discrete cash dividends.

US single-name options are American; ``black`` prices them as European, which is fine for a call on
a stock paying nothing and wrong for a deep put or a call before an ex-date. Cox-Ross-Rubinstein
with Richardson-style averaging of two adjacent step counts, which removes most of the tree's
odd-even oscillation. Discrete dividends use the escrowed model: the tree runs on the stock less the
present value of the dividends still to come, and the exercise value adds that back, so it stays
recombining.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

from pusula.options import black

STEPS = 200
GREEK_STEPS = 800
VOL_LO, VOL_HI = 1e-4, 5.0
TINY = 1e-12

Dividend = tuple[float, float]  # (time in years, cash amount)


def _pv_dividends(dividends: Sequence[Dividend], r: float, t: float, T: float) -> float:
    """Present value at ``t`` of the dividends paid in ``(t, T]``."""
    return sum(d * math.exp(-r * (s - t)) for s, d in dividends if t < s <= T)


def _tree(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    cp: int,
    american: bool,
    dividends: Sequence[Dividend],
    steps: int,
) -> float:
    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"tree probability {p:.4f} outside (0, 1): too few steps for this vol")
    base = S - _pv_dividends(dividends, r, 0.0, T)
    if base <= 0:
        raise ValueError("dividends worth more than the stock")
    powers = u ** (2.0 * np.arange(steps + 1) - steps)
    value = np.maximum(cp * (base * powers - K), 0.0)
    for n in range(steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        if american:
            t = n * dt
            spot = base * u ** (2.0 * np.arange(n + 1) - n) + _pv_dividends(dividends, r, t, T)
            value = np.maximum(value, cp * (spot - K))
    return float(value[0])


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    cp: int = 1,
    american: bool = True,
    dividends: Sequence[Dividend] = (),
    steps: int = STEPS,
) -> float:
    """``cp`` is +1 for a call, -1 for a put."""
    if T <= 0:
        return max(cp * (S - K), 0.0)
    if vol <= 0:
        raise ValueError("volatility must be positive")
    a = _tree(S, K, T, r, q, vol, cp, american, dividends, steps)
    b = _tree(S, K, T, r, q, vol, cp, american, dividends, steps + 1)
    return 0.5 * (a + b)


def early_exercise_premium(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    cp: int = 1,
    dividends: Sequence[Dividend] = (),
) -> float:
    """What the right to exercise early is worth, measured on the same tree."""
    return price(S, K, T, r, q, vol, cp, True, dividends) - price(
        S, K, T, r, q, vol, cp, False, dividends
    )


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    cp: int = 1,
    dividends: Sequence[Dividend] = (),
) -> dict[str, float]:
    """Delta, gamma, vega (per 1.00 of vol) and theta (per year) by central differences."""
    bump = max(S * 0.03, TINY)  # wide enough to see past the tree's own grid
    up, mid, dn = (
        price(x, K, T, r, q, vol, cp, True, dividends, GREEK_STEPS) for x in (S + bump, S, S - bump)
    )
    v = max(vol * 1e-3, 1e-5)
    vu = price(S, K, T, r, q, vol + v, cp, True, dividends)
    vd = price(S, K, T, r, q, vol - v, cp, True, dividends)
    dt = min(T * 0.5, 1.0 / 365.0)
    later = [(s - dt, d) for s, d in dividends if s > dt]
    next_day = price(S, K, T - dt, r, q, vol, cp, True, later)
    return {
        "delta": (up - dn) / (2.0 * bump),
        "gamma": (up - 2.0 * mid + dn) / (bump * bump),
        "vega": (vu - vd) / (2.0 * v),
        "theta": (next_day - mid) / dt,
    }


def implied_vol(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    cp: int = 1,
    dividends: Sequence[Dividend] = (),
) -> float:
    """The volatility at which the American price equals ``target``; NaN if none exists (below the
    exercise value, or a price no volatility in range reaches)."""
    if T <= 0 or target <= 0 or target < max(cp * (S - K), 0.0) - TINY:
        return math.nan

    def gap(v: float) -> float:
        return price(S, K, T, r, q, v, cp, True, dividends, steps=100) - target

    lo = max(VOL_LO, 2.0 * abs(r - q) * math.sqrt(T / 100.0) + 1e-3)
    if gap(lo) > 0 or gap(VOL_HI) < 0:
        return math.nan
    return float(brentq(gap, lo, VOL_HI, xtol=1e-6, maxiter=100))


def european_check(S: float, K: float, T: float, r: float, q: float, vol: float, cp: int) -> float:
    """Tree minus closed form, for tests and for a cheap sanity check at run time."""
    closed = float(black.price(S, K, T, r, q, vol, cp))
    return price(S, K, T, r, q, vol, cp, american=False) - closed
