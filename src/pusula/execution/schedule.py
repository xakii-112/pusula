"""Parent-order scheduling.

* :func:`almgren_chriss` - the optimal liquidation trajectory of Almgren & Chriss (2000) for
  a risk-averse trader with linear temporary impact:
  ``x_k = X sinh(kappa (T - t_k)) / sinh(kappa T)``.
  ``kappa -> 0`` (risk neutral) gives a straight line (TWAP); large ``kappa`` front-loads.
* :func:`volume_profile_slices` - split a parent order across intraday bins in proportion to an
  expected volume curve (VWAP-style), capped at a participation rate per bin; what cannot be
  done today is reported, never silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# A typical U-shaped US equity intraday volume curve, 13 half-hour bins (09:30-16:00).
US_VOLUME_PROFILE = np.array(
    [0.125, 0.085, 0.07, 0.062, 0.058, 0.055, 0.054, 0.055, 0.058, 0.064, 0.072, 0.085, 0.157]
)
US_VOLUME_PROFILE = US_VOLUME_PROFILE / US_VOLUME_PROFILE.sum()


def almgren_chriss(
    shares: float, n_steps: int, sigma: float, eta: float, risk_aversion: float
) -> FloatArray:
    """Holdings after each of ``n_steps`` equal intervals (length ``n_steps + 1``, from ``shares``
    to 0). ``sigma`` is the per-interval price volatility (currency per share), ``eta`` the
    temporary impact (currency per share per share/interval)."""
    if n_steps < 1:
        raise ValueError("n_steps >= 1")
    if risk_aversion <= 0 or sigma <= 0:
        return shares * (1.0 - np.arange(n_steps + 1) / n_steps)
    kappa_sq = risk_aversion * sigma * sigma / eta
    # discrete-time kappa: 2 (cosh(kappa) - 1) = kappa_tilde^2
    kappa = math.acosh(1.0 + 0.5 * kappa_sq)
    t = np.arange(n_steps + 1)
    return shares * np.sinh(kappa * (n_steps - t)) / math.sinh(kappa * n_steps)


@dataclass(frozen=True, slots=True)
class SliceResult:
    slices: FloatArray  # signed shares per bin
    unfilled: float  # signed shares that exceed today's participation capacity


def volume_profile_slices(
    shares: float,
    adv_shares: float,
    max_participation: float = 0.1,
    profile: FloatArray = US_VOLUME_PROFILE,
) -> SliceResult:
    expected = adv_shares * profile
    cap = max_participation * expected
    want = abs(shares) * profile
    got = np.minimum(want, cap)
    # re-distribute what one bin could not take to later bins with spare capacity
    spill = float((want - got).sum())
    for i in range(profile.size):
        room = cap[i] - got[i]
        if spill <= 0:
            break
        take = min(room, spill)
        got[i] += take
        spill -= take
    sign = 1.0 if shares >= 0 else -1.0
    return SliceResult(sign * got, sign * max(spill, 0.0))
