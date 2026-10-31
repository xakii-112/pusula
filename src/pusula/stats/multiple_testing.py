"""Family-wise and false-discovery-rate corrections."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def holm(pvalues: npt.ArrayLike, alpha: float = 0.05) -> npt.NDArray[np.bool_]:
    """Holm-Bonferroni step-down: which hypotheses are rejected at FWER ``alpha``."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def benjamini_hochberg(pvalues: npt.ArrayLike, q: float = 0.05) -> npt.NDArray[np.bool_]:
    """Benjamini-Hochberg step-up: rejections controlling FDR at ``q`` (independent tests)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        reject[order[: k + 1]] = True
    return reject
