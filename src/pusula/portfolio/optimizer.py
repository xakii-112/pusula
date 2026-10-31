"""Cost-aware mean-variance portfolio optimiser.

    maximise   alpha'w - lambda * risk(w) - sum c_i |w_i - w0_i| - sum k_i |w_i - w0_i|^1.5
    subject to sum |w| <= G, net bounds, |w_i| <= ub_i, |w_i - w0_i| <= tcap_i,
               |beta'w| <= b_max, sqrt(risk(w)) <= vol_target (optional)

with ``risk(w) = ||F^{1/2} B'w||^2 + ||D^{1/2} w||^2`` in factor form (O(n k), never the dense
n x n covariance). The problem is built once over all panel columns with parameters (cvxpy
DPP), so each rebalance only re-binds numbers; ineligible names get ``ub = 0``. The
``|dw|^1.5`` term is the square-root impact law integrated over the trade; it makes the
optimiser trade large, illiquid positions slowly instead of paying for immediacy.

Feasibility is guaranteed by construction: gross and net bounds are widened to what today's
trade caps can reach from the current portfolio (a portfolio that is too large is reduced as
fast as liquidity allows, not declared infeasible). If the beta or volatility constraint still
cannot be met, a relaxed problem without them is solved and the result is flagged
``relaxed``; the risk gate reports the breach.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from pusula.core.errors import SolverError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    risk_aversion: float = 5.0  # lambda on daily variance x horizon
    max_gross: float = 2.0
    net_lo: float = 0.0
    net_hi: float = 0.0
    long_only: bool = False
    max_weight: float = 0.02
    beta_max: float | None = 0.05
    vol_target_daily: float | None = None  # hard ex-ante risk cap
    cost_aversion: float = 1.0
    n_factors: int = 12


@dataclass(frozen=True, slots=True)
class OptimizerInput:
    alpha: FloatArray  # N expected return over the holding horizon (fraction)
    w0: FloatArray  # N current weights
    B: FloatArray  # N x k (zero rows for names outside the risk model)
    F: FloatArray  # k x k
    d: FloatArray  # N specific variance (daily)
    eligible: npt.NDArray[np.bool_]
    lin_cost: FloatArray  # N cost per unit |dw|
    impact: FloatArray  # N k_i for |dw|^1.5
    trade_cap: FloatArray  # N max |dw|
    beta: FloatArray  # N
    horizon_days: float = 5.0
    vol_target_daily: float | None = None  # overrides the spec's target for this solve


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    w: FloatArray
    status: str
    objective: float
    expected_alpha: float
    expected_cost: float
    ex_ante_vol_daily: float
    relaxed: bool


@dataclass(slots=True)
class _Built:
    problem: cp.Problem
    w: cp.Variable
    risk: cp.Expression
    cost: cp.Expression


class PortfolioOptimizer:
    def __init__(self, n: int, spec: OptimizerSpec) -> None:
        k = spec.n_factors
        self.n, self.k, self.spec = n, k, spec
        self.p_alpha = cp.Parameter(n)
        self.p_w0 = cp.Parameter(n)
        self.p_Bt = cp.Parameter((k, n))  # F^{1/2} B'  (pre-multiplied)
        self.p_sd = cp.Parameter(n, nonneg=True)  # sqrt(d)
        self.p_lin = cp.Parameter(n, nonneg=True)
        self.p_imp = cp.Parameter(n, nonneg=True)
        self.p_ub = cp.Parameter(n, nonneg=True)
        self.p_tcap = cp.Parameter(n, nonneg=True)
        self.p_beta = cp.Parameter(n)
        self.p_BtL = cp.Parameter((k, n))  # sqrt(lambda * horizon) * F^{1/2} B'
        self.p_sdL = cp.Parameter(n, nonneg=True)
        self.p_gross = cp.Parameter(nonneg=True)
        self.p_net_lo = cp.Parameter()
        self.p_net_hi = cp.Parameter()
        self.p_vol = cp.Parameter(nonneg=True)  # ex-ante volatility cap (daily)
        self._strict = self._build(relaxed=False)
        self._relaxed: _Built | None = None

    def _build(self, relaxed: bool) -> _Built:
        s = self.spec
        w = cp.Variable(self.n)
        dw = cp.Variable(self.n)  # trade; a variable so that cost terms stay DPP
        risk_pen = cp.sum_squares(self.p_BtL @ w) + cp.sum_squares(cp.multiply(self.p_sdL, w))
        cost = self.p_lin @ cp.abs(dw) + self.p_imp @ cp.power(cp.abs(dw), 1.5)
        obj = self.p_alpha @ w - risk_pen - s.cost_aversion * cost
        cons = [
            dw == w - self.p_w0,
            cp.abs(w) <= self.p_ub,
            cp.abs(dw) <= self.p_tcap,
            cp.norm1(w) <= self.p_gross,
            cp.sum(w) >= self.p_net_lo,
            cp.sum(w) <= self.p_net_hi,
        ]
        if s.long_only:
            cons.append(w >= 0)
        if not relaxed:
            if s.beta_max is not None:
                cons.append(cp.abs(self.p_beta @ w) <= s.beta_max)
            if s.vol_target_daily is not None:
                cons.append(
                    cp.norm(cp.hstack([self.p_Bt @ w, cp.multiply(self.p_sd, w)]), 2) <= self.p_vol
                )
        prob = cp.Problem(cp.Maximize(obj), cons)
        if not prob.is_dpp():
            raise SolverError("optimiser problem is not DPP")
        return _Built(prob, w, risk_pen, cost)

    def _bind(self, inp: OptimizerInput) -> None:
        s = self.spec
        el = inp.eligible
        w0 = inp.w0
        ub = np.where(el, s.max_weight, 0.0)
        tcap = np.where(np.isfinite(inp.trade_cap), inp.trade_cap, 0.0)
        # A name may never be stuck above its bound: allow reducing it to the bound today.
        tcap = np.maximum(tcap, np.maximum(np.abs(w0) - ub, 0.0))
        # Per-name feasible interval today (non-empty because tcap >= |w0| - ub).
        min_w = np.maximum(w0 - tcap, -ub)
        max_w = np.minimum(w0 + tcap, ub)
        if s.long_only:
            min_w = np.maximum(min_w, 0.0)
            max_w = np.maximum(max_w, 0.0)
        min_abs = np.where(
            (min_w <= 0) & (max_w >= 0), 0.0, np.minimum(np.abs(min_w), np.abs(max_w))
        )
        self.p_gross.value = max(s.max_gross, float(min_abs.sum()) + 1e-9)
        net_min, net_max = float(min_w.sum()), float(max_w.sum())
        self.p_net_lo.value = min(max(s.net_lo, net_min), net_max)
        self.p_net_hi.value = max(min(s.net_hi, net_max), net_min)
        L = np.linalg.cholesky(inp.F + 1e-12 * np.eye(inp.F.shape[0]))
        Bt = L.T @ np.where(np.isfinite(inp.B), inp.B, 0.0).T
        self.p_alpha.value = np.where(el & np.isfinite(inp.alpha), inp.alpha, 0.0)
        self.p_w0.value = w0
        self.p_Bt.value = Bt
        self.p_sd.value = np.sqrt(np.where(np.isfinite(inp.d), np.maximum(inp.d, 0.0), 0.0))
        self.p_lin.value = np.where(np.isfinite(inp.lin_cost), inp.lin_cost, 0.0)
        self.p_imp.value = np.where(np.isfinite(inp.impact), inp.impact, 0.0)
        self.p_ub.value = ub
        self.p_tcap.value = tcap
        self.p_beta.value = np.where(np.isfinite(inp.beta), inp.beta, 0.0)
        target = inp.vol_target_daily if inp.vol_target_daily is not None else s.vol_target_daily
        self.p_vol.value = float(target) if target is not None else 1.0
        lam = np.sqrt(s.risk_aversion * inp.horizon_days)
        self.p_BtL.value = lam * Bt
        self.p_sdL.value = lam * np.asarray(self.p_sd.value)

    def _run(self, built: _Built) -> bool:
        try:
            # warm_start=False: results must not depend on the history of earlier solves
            # (a resumed paper book must reproduce an uninterrupted one bit for bit)
            built.problem.solve(solver=cp.CLARABEL, warm_start=False)
        except cp.SolverError:
            return False
        return (
            built.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and built.w.value is not None
        )

    def solve(self, inp: OptimizerInput) -> OptimizerResult:
        self._bind(inp)
        built, relaxed = self._strict, False
        if not self._run(built):
            if self._relaxed is None:
                self._relaxed = self._build(relaxed=True)
            built, relaxed = self._relaxed, True
            if not self._run(built):
                raise SolverError(f"optimiser failed (status {built.problem.status})")
        w = np.asarray(built.w.value, dtype=np.float64)
        w[np.abs(w) < 1e-7] = 0.0
        cost_v = built.cost.value
        fx = np.asarray(self.p_Bt.value) @ w
        risk_v = float(fx @ fx + ((np.asarray(self.p_sd.value) * w) ** 2).sum())
        return OptimizerResult(
            w=w,
            status=str(built.problem.status),
            objective=float(built.problem.value),
            expected_alpha=float(np.asarray(self.p_alpha.value) @ w),
            expected_cost=float(cost_v) if cost_v is not None else float("nan"),
            ex_ante_vol_daily=float(np.sqrt(max(risk_v, 0.0))),
            relaxed=relaxed,
        )
