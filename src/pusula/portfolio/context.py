"""The decision interface shared by research backtests, the paper simulator and live runs.

A strategy sees exactly one :class:`DecisionContext` per rebalance and returns target weights.
It never sees prices after the decision close: the engine builds the context from data up to
row ``t`` only. This is the single code path that NihaiAlgoritma lacked (its live loop and its
backtest were separate programs and diverged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pusula.data.panel import Panel
from pusula.risk.model import FactorRisk

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DecisionContext:
    t: int
    date: np.datetime64
    equity: float
    w_current: FloatArray  # N, drifted weights at the decision close
    eligible: BoolArray  # N, point-in-time universe & tradable
    close: FloatArray  # N
    half_spread: FloatArray  # N, fraction
    adv_dollars: FloatArray  # N
    sigma: FloatArray  # N, daily vol forecast
    risk: FactorRisk | None
    beta: FloatArray  # N, beta to the equal-weight universe (NaN outside the risk model)

    def dense_loadings(self, k: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(B, F, d) expanded to all N columns (zero loadings / NaN variance outside the model)."""
        n = self.w_current.size
        B = np.zeros((n, k))
        d = np.full(n, np.nan)
        if self.risk is None:
            return B, np.eye(k) * 1e-8, d
        kk = min(k, self.risk.B.shape[1])
        B[self.risk.cols, :kk] = self.risk.B[:, :kk]
        F = np.eye(k) * 1e-8
        F[:kk, :kk] = self.risk.F[:kk, :kk]
        d[self.risk.cols] = self.risk.d
        return B, F, d


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def rebalance_every(self) -> int: ...

    def prepare(self, panel: Panel) -> None:
        """Precompute causal inputs (signals, forecasts) for the whole panel."""
        ...

    def decide(self, ctx: DecisionContext) -> FloatArray:
        """Target weights (N) for the context's decision close."""
        ...
