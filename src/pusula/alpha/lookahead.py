"""Mechanical look-ahead detection.

A computation ``f`` is causal iff, for every cutoff c, ``f(panel)[: c + 1]`` equals
``f(panel.upto(c))``: appending the future must not change the past. This test is run for every
registered signal and for every stage of the pipeline that produces per-date outputs (risk
forecasts, ML predictions, target portfolios). It catches centred windows, full-sample
normalisation, ``shift(-1)``, global ranks and "fit once on everything" models alike.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pusula.core.errors import LookaheadError
from pusula.data.panel import FloatArray, Panel, nan_equal


def check_causal(
    fn: Callable[[Panel], FloatArray],
    panel: Panel,
    cutoffs: Sequence[int],
    atol: float = 1e-10,
    name: str = "computation",
) -> None:
    full = fn(panel)
    if full.shape[0] != panel.n_dates:
        raise LookaheadError(f"{name}: output has {full.shape[0]} rows for {panel.n_dates} dates")
    for c in cutoffs:
        trunc = fn(panel.upto(c))
        head = full[: c + 1]
        if not nan_equal(head, trunc, atol=atol):
            diff = ~((np.isnan(head) & np.isnan(trunc)) | (np.abs(head - trunc) <= atol))
            rows = np.nonzero(diff.any(axis=tuple(range(1, diff.ndim))))[0]
            raise LookaheadError(
                f"{name}: rows {rows[:5].tolist()} change when data after row {c} is appended"
            )


def default_cutoffs(n: int, k: int = 4, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    lo = max(1, n // 5)
    return sorted({int(x) for x in rng.integers(lo, n - 1, size=k)} | {n // 2})
