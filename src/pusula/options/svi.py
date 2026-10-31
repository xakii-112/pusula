"""Raw SVI smiles (Gatheral 2004), fitted quasi-explicitly, with static-arbitrage checks.

Total implied variance at log-moneyness ``k = ln(K / F)``::

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))

For fixed ``(m, sigma)`` the model is linear in ``(a, d, c)`` with ``y = (k - m) / sigma``:
``w = a + d y + c sqrt(y^2 + 1)``, ``c = b sigma``, ``d = rho c`` (Zeliade Systems 2009). The inner
problem is a small bounded least squares and the outer one searches ``(m, sigma)``, which is far
more stable than five free parameters. From meridian (Desktop/jane), which had no real chain to fit;
here it runs on quotes, so the fit is weighted and a thin slice is refused, not extrapolated.

Quotes alone do not make a surface free of arbitrage, so a fit is pushed towards it: the search
over ``(m, sigma)`` pays for negative ``g(k)`` within the quoted range, and a slice fitted after an
earlier one is lifted, if it must be, to the earlier one's total variance over that range.

A slice is free of butterfly arbitrage iff Durrleman's ``g(k)`` is non-negative (Gatheral & Jacquier
2014); calendar arbitrage is absent iff total variance does not fall with maturity at any ``k``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy.optimize import lsq_linear, minimize

FloatArray = npt.NDArray[np.float64]
MIN_POINTS = 5
BUTTERFLY_PENALTY = 1e3  # per unit of squared negative g, in units of the mean total variance
RANGE_MARGIN = 0.05  # log-moneyness beyond the quotes that a fit also answers for


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0 or not -1 < self.rho < 1 or self.sigma <= 0:
            raise ValueError("SVI needs b >= 0, |rho| < 1, sigma > 0")

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho**2)

    def total_variance(self, k: npt.ArrayLike) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))

    def implied_vol(self, k: npt.ArrayLike, expiry: float) -> FloatArray:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / expiry)

    def durrleman_g(self, k: npt.ArrayLike) -> FloatArray:
        kk = np.asarray(k, dtype=np.float64)
        x = kk - self.m
        root = np.sqrt(x * x + self.sigma**2)
        w = self.total_variance(kk)
        w1 = self.b * (self.rho + x / root)
        w2 = self.b * self.sigma**2 / root**3
        return (1.0 - kk * w1 / (2.0 * w)) ** 2 - (w1**2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


@dataclass(frozen=True, slots=True)
class SVIFit:
    params: SVIParams
    rmse: float  # in total variance, unweighted
    n_points: int
    k_lo: float = -0.5
    k_hi: float = 0.5
    lifted: float = 0.0  # total variance added to stay above the previous slice


def _inner(
    k: FloatArray, w: FloatArray, sw: FloatArray, m: float, s: float
) -> tuple[float, FloatArray]:
    y = (k - m) / s
    design = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    res = lsq_linear(
        design * sw[:, None],
        w * sw,
        bounds=([0.0, -4.0 * s, 0.0], [float(w.max()), 4.0 * s, 4.0 * s]),
    )
    theta = np.asarray(res.x, dtype=np.float64)
    # |d| <= c is the no-arbitrage wing condition; projecting onto it keeps the fit feasible
    theta[1] = float(np.clip(theta[1], -theta[2], theta[2]))
    r = (design @ theta - w) * sw
    return float(r @ r), theta


def _from_theta(theta: FloatArray, m: float, s: float) -> SVIParams:
    a, d, c = (float(x) for x in theta)
    rho = float(np.clip(d / c, -0.999999, 0.999999)) if c > 0 else 0.0
    return SVIParams(a=a, b=c / s, rho=rho, m=m, sigma=s)


def fit_svi(
    log_moneyness: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    floor: SVIParams | None = None,
) -> SVIFit:
    k = np.asarray(log_moneyness, dtype=np.float64)
    w = np.asarray(total_variance, dtype=np.float64)
    if k.ndim != 1 or k.shape != w.shape or k.size < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} (k, w) points")
    if np.any(w <= 0) or not (np.isfinite(k).all() and np.isfinite(w).all()):
        raise ValueError("total variances must be positive and finite")
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(wt / wt.mean())
    span = max(float(k.max() - k.min()), 1e-3)
    k_lo, k_hi = float(k.min()) - RANGE_MARGIN, float(k.max()) + RANGE_MARGIN
    grid = np.linspace(k_lo, k_hi, 41)
    scale = float(np.mean(w)) ** 2 * k.size

    def outer(x: FloatArray) -> float:
        m, s = float(x[0]), math.exp(float(x[1]))
        loss, theta = _inner(k, w, sw, m, s)
        if (
            theta[0] + theta[2] * math.sqrt(1.0 - min((theta[1] / max(theta[2], 1e-12)) ** 2, 1.0))
            <= 0
        ):
            return loss + scale
        g = _from_theta(theta, m, s).durrleman_g(grid)
        short = np.minimum(np.nan_to_num(g, nan=-1.0), 0.0)
        return loss + BUTTERFLY_PENALTY * scale * float(np.mean(short * short))

    best: tuple[float, FloatArray] | None = None
    for m0 in (float(k.min()), float(np.median(k)), float(k.max())):
        for s0 in (0.1 * span, 0.5 * span):
            res = minimize(
                outer,
                np.array([m0, math.log(s0)]),
                method="Nelder-Mead",
                options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 200},
            )
            if best is None or res.fun < best[0]:
                best = (float(res.fun), np.asarray(res.x, dtype=np.float64))
    assert best is not None
    m, s = float(best[1][0]), math.exp(float(best[1][1]))
    _, theta = _inner(k, w, sw, m, s)
    params = _from_theta(theta, m, s)
    lifted = 0.0
    if floor is not None:
        shortfall = float(np.max(floor.total_variance(grid) - params.total_variance(grid)))
        if shortfall > 0:
            lifted = shortfall
            params = SVIParams(params.a + shortfall, params.b, params.rho, params.m, params.sigma)
    resid = params.total_variance(k) - w
    return SVIFit(params, float(np.sqrt(np.mean(resid**2))), int(k.size), k_lo, k_hi, lifted)


def butterfly_arbitrage_free(
    params: SVIParams, k_grid: npt.ArrayLike | None = None, tol: float = 1e-9
) -> bool:
    grid = np.linspace(-1.5, 1.5, 601) if k_grid is None else np.asarray(k_grid, dtype=np.float64)
    return bool(params.min_variance > 0 and np.all(params.durrleman_g(grid) >= -tol))


def calendar_violations(
    slices: Sequence[tuple[float, SVIParams]], k_grid: npt.ArrayLike | None = None
) -> list[tuple[float, float, float]]:
    """(T1, T2, worst shortfall) for each adjacent pair whose total variance falls somewhere."""
    grid = np.linspace(-0.5, 0.5, 201) if k_grid is None else np.asarray(k_grid, dtype=np.float64)
    out: list[tuple[float, float, float]] = []
    for (t1, p1), (t2, p2) in pairwise(sorted(slices, key=lambda s: s[0])):
        shortfall = float(np.max(p1.total_variance(grid) - p2.total_variance(grid)))
        if shortfall > 0:
            out.append((t1, t2, shortfall))
    return out
