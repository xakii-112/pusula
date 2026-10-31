"""Hierarchical risk parity and the instrument diversification multiplier.

Estimated on volatility-normalised weekly returns, as in Nihai3: weekly sums are less noisy than
daily correlations, and normalising by each security's lagged volatility makes the covariance a
correlation, so the weights spread risk rather than dollars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from pusula.data.panel import FloatArray

BLOCK = 5
MIN_BLOCKS = 26


def vol_normalised_returns(close: FloatArray, halflife: float = 11.2) -> FloatArray:
    """Daily simple return over the EWMA volatility known the day before (causal). The half-life
    is the RiskMetrics lambda 0.94."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = close[1:] / close[:-1] - 1.0
    r = np.vstack([np.full((1, close.shape[1]), np.nan), r])
    var = pd.DataFrame(r * r).ewm(halflife=halflife, min_periods=20, ignore_na=True).mean()
    sig = np.sqrt(var.to_numpy())
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), sig[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(prev > 0, r / prev, np.nan)


def weekly_blocks(z: FloatArray, end: int, length: int) -> FloatArray:
    """Rows ``end - length + 1 .. end`` summed into blocks of five, counted back from ``end`` (a
    block with no observation of a column is NaN for it)."""
    seg = z[max(0, end - length + 1) : end + 1]
    k = seg.shape[0] // BLOCK
    if k == 0:
        return np.empty((0, z.shape[1]))
    blk = seg[seg.shape[0] - k * BLOCK :].reshape(k, BLOCK, -1)
    valid = np.isfinite(blk).any(axis=1)
    return np.where(valid, np.nansum(blk, axis=1), np.nan)


def pairwise_corr(x: FloatArray, min_obs: int = MIN_BLOCKS) -> FloatArray:
    """Pairwise-complete Pearson correlation; NaN for pairs with fewer than ``min_obs`` blocks."""
    m = np.isfinite(x).astype(np.float64)
    x0 = np.where(m > 0, x, 0.0)
    n = m.T @ m
    sa = x0.T @ m
    saa = (x0 * x0).T @ m
    sab = x0.T @ x0
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = sa / n, sa.T / n
        rho = (sab / n - ma * mb) / np.sqrt((saa / n - ma * ma) * (saa.T / n - mb * mb))
    rho[n < min_obs] = np.nan
    np.fill_diagonal(rho, 1.0)
    return np.clip(rho, -1.0, 1.0)


def weekly_correlation(z: FloatArray, end: int, length: int) -> FloatArray:
    """Correlation of the columns of ``z`` from rows <= ``end``, with an unknown pair set to the
    mean of the known ones. Nihai3 dropped to equal weights as soon as one pair was unknown; in
    a universe of 500 stocks a single recent listing would switch the method off."""
    rho = pairwise_corr(weekly_blocks(z, end, length))
    off = rho[~np.eye(rho.shape[0], dtype=bool)]
    known = off[np.isfinite(off)]
    fill = float(known.mean()) if known.size else 1.0
    return np.where(np.isfinite(rho), rho, fill)


def nearest_psd_corr(rho: FloatArray) -> FloatArray:
    """Clip eigenvalues at (almost) zero and rescale the diagonal back to one."""
    r = (rho + rho.T) / 2.0
    vals, vecs = np.linalg.eigh(r)
    r = (vecs * np.maximum(vals, 1e-8)) @ vecs.T
    d = np.sqrt(np.diag(r))
    r = r / np.outer(d, d)
    np.fill_diagonal(r, 1.0)
    return r


def hrp(rho: FloatArray) -> FloatArray:
    """Hierarchical Risk Parity (Lopez de Prado 2016) for unit-variance assets: single-linkage
    clusters, quasi-diagonal order, recursive bisection on the variance of each equal-weighted
    cluster. Weights are positive and sum to one."""
    n = rho.shape[0]
    if n == 1:
        return np.ones(1)
    r = nearest_psd_corr(rho)
    dist = np.sqrt(np.clip(0.5 * (1.0 - r), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            c0, c1 = c[:h], c[h:]
            v0 = float(r[np.ix_(c0, c0)].mean())
            v1 = float(r[np.ix_(c1, c1)].mean())
            a = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w[c0] *= a
            w[c1] *= 1.0 - a
            nxt += [c0, c1]
        clusters = nxt
    return w / w.sum()


def idm(w: FloatArray, rho: FloatArray, cap: float) -> float:
    """Instrument diversification multiplier 1/sqrt(w' rho+ w), negative correlations floored at
    zero (Carver 2015): how much lower risk the mix has than its average member."""
    r = np.where(np.isfinite(rho), np.maximum(rho, 0.0), 1.0)
    q = float(w @ r @ w)
    return float(min(1.0 / np.sqrt(max(q, 1e-12)), cap))
