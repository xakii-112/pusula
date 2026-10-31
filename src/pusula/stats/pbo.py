"""Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

Bailey, Borwein, López de Prado & Zhu (2014), "The Probability of Backtest Overfitting".
Given a T x N matrix of per-period returns of N configurations, split time into S blocks; for
every half/half split, pick the in-sample best configuration and record the relative rank of
the same configuration out of sample. PBO is the fraction of splits where the IS winner ranks
below the OOS median. For pure noise PBO is about 0.5; a genuine, stable edge pushes it to 0.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    logits: npt.NDArray[np.float64]
    n_splits: int
    perf_degradation_slope: float  # OOS ~ IS regression slope of the selected config


def _sr(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd, 0.0)


def pbo_cscv(returns: npt.ArrayLike, n_blocks: int = 16) -> PBOResult:
    M = np.asarray(returns, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("need a T x N matrix with N >= 2 configurations")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    M = M[np.all(np.isfinite(M), axis=1)]
    T, N = M.shape
    if 2 * n_blocks > T:
        raise ValueError("not enough observations for the number of blocks")
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    logits = []
    is_best, oos_of_best = [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        sr_is, sr_oos = _sr(M[is_idx]), _sr(M[oos_idx])
        best = int(np.argmax(sr_is))
        # relative rank in (0, 1): fraction of configs the IS winner beats OOS
        rank = (np.sum(sr_oos < sr_oos[best]) + 0.5 * (np.sum(sr_oos == sr_oos[best]) - 1) + 1) / (
            N + 1
        )
        logits.append(math.log(rank / (1.0 - rank)))
        is_best.append(sr_is[best])
        oos_of_best.append(sr_oos[best])
    lg = np.asarray(logits)
    slope = float(np.polyfit(is_best, oos_of_best, 1)[0]) if len(set(is_best)) > 1 else math.nan
    return PBOResult(float(np.mean(lg <= 0.0)), lg, int(lg.size), slope)
