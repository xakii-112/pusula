"""Portfolio strategies.

* :class:`OptimizedAlpha`: composite (or ML) alpha -> expected returns via Grinold's
  "alpha = IC x volatility x score" -> cost-aware optimiser. The institutional core.
* :class:`TrendFollowing`: NihaiAlgoritma's live ``quant_trend`` lifted from single-name signals
  to a portfolio: multi-horizon trend sign, inverse-volatility sizing, a portfolio volatility
  target from the factor model. Long-only by default, as in the original.
* :class:`ForecastRiskParity`: Nihai3's portfolio engine: position proportional to a continuous
  forecast, hierarchical risk parity between securities, a volatility target, and a no-trade
  buffer instead of stops.
* :class:`EqualWeight`: the universe benchmark every strategy must beat after costs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pusula.alpha.evaluate import trailing_ic
from pusula.core.errors import SolverError
from pusula.data.panel import FloatArray, Panel
from pusula.execution.costs import CostModel, impact_coef, linear_cost_rate, participation_cap
from pusula.portfolio import hrp as H
from pusula.portfolio.context import DecisionContext
from pusula.portfolio.optimizer import OptimizerInput, OptimizerSpec, PortfolioOptimizer
from pusula.risk.measures import realised_to_model_vol
from pusula.risk.model import simple_returns

AlphaFn = Callable[[Panel], FloatArray]


@dataclass
class OptimizedAlpha:
    alpha_fn: AlphaFn
    spec: OptimizerSpec
    label: str = "optimized_alpha"
    ic: float = 0.02  # assumed per-period information coefficient (pre-registered)
    horizon_days: float = 5.0
    every: int = 5
    cost_model: CostModel = field(default_factory=CostModel)
    ic_mode: str = "fixed"  # fixed | adaptive
    ic_window: int = 252
    ic_prior_days: int = 252
    vol_feedback: bool = False
    vol_window: int = 63
    scores: FloatArray | None = None
    ic_series: FloatArray | None = None
    solver_failures: int = 0
    relaxed_solves: int = 0
    _opt: PortfolioOptimizer | None = None
    _returns: FloatArray | None = None

    @property
    def name(self) -> str:
        return self.label

    @property
    def rebalance_every(self) -> int:
        return self.every

    def prepare(self, panel: Panel) -> None:
        self.scores = self.alpha_fn(panel)
        self._opt = PortfolioOptimizer(panel.n_symbols, self.spec)
        if self.ic_mode == "adaptive":
            self.ic_series = trailing_ic(
                self.scores,
                panel,
                max(1, round(self.horizon_days)),
                window=self.ic_window,
                prior_days=self.ic_prior_days,
                mask=panel.get("in_universe"),
            )
        elif self.ic_mode != "fixed":
            raise ValueError(f"unknown ic_mode {self.ic_mode!r}")
        if self.vol_feedback:
            self._returns = simple_returns(panel["close"])

    def ic_at(self, t: int) -> float:
        if self.ic_series is not None:
            return float(self.ic_series[t])
        return self.ic

    def vol_target_at(self, ctx: DecisionContext) -> float | None:
        """The spec's ex-ante target, divided by the realised/model volatility ratio of the book
        currently held (clipped to [0.67, 1.5]); unchanged without feedback or a book."""
        target = self.spec.vol_target_daily
        if not self.vol_feedback or target is None or ctx.risk is None or self._returns is None:
            return target
        w = ctx.w_current
        if np.abs(w).sum() < 0.1 or ctx.t < self.vol_window:
            return target
        window = self._returns[ctx.t - self.vol_window + 1 : ctx.t + 1]
        ratio = realised_to_model_vol(window, w, ctx.risk, w[ctx.risk.cols])
        if not np.isfinite(ratio):
            return target
        return target / float(np.clip(ratio, 0.67, 1.5))

    def expected_returns(self, ctx: DecisionContext) -> FloatArray:
        assert self.scores is not None
        z = self.scores[ctx.t]
        sig = np.where(np.isfinite(ctx.sigma), ctx.sigma, np.nan)
        alpha = self.ic_at(ctx.t) * sig * np.sqrt(self.horizon_days) * z
        return np.where(ctx.eligible & np.isfinite(alpha), alpha, 0.0)

    def decide(self, ctx: DecisionContext) -> FloatArray:
        assert self._opt is not None, "prepare() first"
        if ctx.risk is None:
            return ctx.w_current.copy()
        B, F, d = ctx.dense_loadings(self.spec.n_factors)
        in_model = np.zeros(ctx.w_current.size, dtype=bool)
        in_model[ctx.risk.cols] = True
        el = (
            ctx.eligible & in_model & np.isfinite(self.scores[ctx.t])
            if self.scores is not None
            else ctx.eligible
        )
        inp = OptimizerInput(
            alpha=self.expected_returns(ctx),
            w0=ctx.w_current,
            B=B,
            F=F,
            d=np.where(np.isfinite(d), d, np.nanmedian(d) if np.isfinite(d).any() else 1e-4),
            eligible=el,
            lin_cost=linear_cost_rate(ctx.half_spread, self.cost_model),
            impact=impact_coef(ctx.sigma, ctx.adv_dollars, ctx.equity, self.cost_model),
            trade_cap=participation_cap(ctx.adv_dollars, ctx.equity, self.cost_model),
            beta=ctx.beta,
            horizon_days=self.horizon_days,
            vol_target_daily=self.vol_target_at(ctx),
        )
        try:
            res = self._opt.solve(inp)
        except SolverError:
            self.solver_failures += 1
            return ctx.w_current.copy()
        self.relaxed_solves += int(res.relaxed)
        return res.w


@dataclass
class TrendFollowing:
    score_fn: AlphaFn
    label: str = "trend"
    every: int = 5
    long_only: bool = True
    vol_target_annual: float = 0.15
    max_weight: float = 0.05
    max_gross: float = 1.0
    scores: FloatArray | None = None

    @property
    def name(self) -> str:
        return self.label

    @property
    def rebalance_every(self) -> int:
        return self.every

    def prepare(self, panel: Panel) -> None:
        self.scores = self.score_fn(panel)

    def decide(self, ctx: DecisionContext) -> FloatArray:
        assert self.scores is not None
        s = np.where(ctx.eligible & np.isfinite(self.scores[ctx.t]), self.scores[ctx.t], 0.0)
        if self.long_only:
            s = np.maximum(s, 0.0)
        sig = np.where(np.isfinite(ctx.sigma) & (ctx.sigma > 0), ctx.sigma, np.inf)
        raw = s / sig  # inverse-vol: equal risk per unit of trend conviction
        if not np.any(raw != 0):
            return np.zeros_like(ctx.w_current)
        w = raw / np.abs(raw).sum() * self.max_gross
        w = np.clip(w, -self.max_weight, self.max_weight)
        if ctx.risk is not None:
            v = ctx.risk.vol(w[ctx.risk.cols])
            if v > 0:
                w *= min(1.0, self.vol_target_annual / v)
        return w


@dataclass(frozen=True, slots=True)
class ForecastSpec:
    """Nihai3's registered values (Carver 2015; Lopez de Prado 2016), except where the universe
    differs: long-only and unlevered, as ``trend_heritage``, and the same 15% volatility target,
    so that the comparison with it isolates the portfolio construction."""

    target_vol: float = 0.15
    weighting: str = "hrp"  # hrp | equal
    corr_window: int = 756
    corr_every: int = 21
    min_history: int = 252
    idm_cap: float = 2.5
    vol_cap_mult: float = 1.5  # scale the book down when the risk model forecasts more than this
    max_weight: float = 0.05
    max_gross: float = 1.0
    buffer: float = 0.10  # share of a security's full-forecast position traded around
    long_only: bool = True


@dataclass
class ForecastRiskParity:
    """w = F/10 * W * IDM * target_vol / sigma, traded to the edge of a no-trade band.

    Nihai3 found that its gain over NihaiAlgoritma came from here and not from any signal (the
    old sign score through this engine took a Sharpe of 0.52 to 1.06), but on today's survivors.
    ``us-trend-v1`` asks the question again on the point-in-time universe with calibrated costs.

    The between-security weights are a function of the panel at a fixed schedule of rows, not of
    the run's history, so a resumed book equals an uninterrupted one.
    """

    forecast_fn: AlphaFn
    spec: ForecastSpec = field(default_factory=ForecastSpec)
    label: str = "forecast_risk_parity"
    every: int = 1
    forecast: FloatArray | None = None
    _z: FloatArray | None = None
    _members: FloatArray | None = None
    _history: FloatArray | None = None
    _weights: dict[int, tuple[FloatArray, float]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.label

    @property
    def rebalance_every(self) -> int:
        return self.every

    def prepare(self, panel: Panel) -> None:
        self.forecast = self.forecast_fn(panel)
        self._z = H.vol_normalised_returns(panel["close"])
        self._members = panel["in_universe"]
        self._history = np.cumsum(np.isfinite(panel["close"]), axis=0)
        self._weights = {}

    def weights_at(self, t: int) -> tuple[FloatArray, float]:
        """(W, IDM) of the schedule's row at or before ``t``, from rows <= that row only."""
        assert self.forecast is not None
        assert self._z is not None
        assert self._members is not None
        assert self._history is not None
        s = self.spec
        row = t - t % s.corr_every
        if row in self._weights:
            return self._weights[row]
        N = self.forecast.shape[1]
        cols = np.nonzero(
            (self._members[row] > 0)
            & np.isfinite(self.forecast[row])
            & (self._history[row] >= s.min_history)
        )[0]
        W = np.zeros(N)
        idm = 1.0
        if cols.size == 1:
            W[cols] = 1.0
        elif cols.size > 1:
            rho = H.weekly_correlation(self._z[:, cols], row, s.corr_window)
            W[cols] = H.hrp(rho) if s.weighting == "hrp" else 1.0 / cols.size
            idm = H.idm(W[cols], rho, s.idm_cap)
        self._weights[row] = (W, idm)
        return W, idm

    def decide(self, ctx: DecisionContext) -> FloatArray:
        assert self.forecast is not None
        s = self.spec
        W, idm = self.weights_at(ctx.t)
        F = self.forecast[ctx.t]
        sig_ann = ctx.sigma * math.sqrt(252.0)
        live = ctx.eligible & (W > 0) & np.isfinite(F) & np.isfinite(sig_ann) & (sig_ann > 0)
        unit = np.zeros_like(W)
        unit[live] = W[live] * idm * s.target_vol / sig_ann[live]
        f = np.where(live, F, 0.0)
        if s.long_only:
            f = np.maximum(f, 0.0)
        target = np.clip(f / 10.0 * unit, -s.max_weight, s.max_weight)
        scale = 1.0
        if ctx.risk is not None and np.any(target != 0):
            v = ctx.risk.vol(target[ctx.risk.cols])
            if v > s.vol_cap_mult * s.target_vol:
                scale = s.vol_cap_mult * s.target_vol / v
        gross = float(np.abs(target).sum()) * scale
        if gross > s.max_gross:
            scale *= s.max_gross / gross
        target *= scale
        # no trade inside the band; outside it, trade only to its edge
        band = s.buffer * unit * scale
        out = np.where(live, np.clip(ctx.w_current, target - band, target + band), 0.0)
        gross = float(np.abs(out).sum())
        return out * (s.max_gross / gross) if gross > s.max_gross else out


@dataclass
class HoldOne:
    """All of the book in one security: Bitcoin is to crypto what the market is to stocks."""

    security: str
    label: str = "hold"
    every: int = 21

    @property
    def name(self) -> str:
        return self.label

    @property
    def rebalance_every(self) -> int:
        return self.every

    def prepare(self, panel: Panel) -> None:
        self._column = panel.symbol_index(self.security)

    def decide(self, ctx: DecisionContext) -> FloatArray:
        w = np.zeros_like(ctx.w_current)
        if ctx.eligible[self._column]:
            w[self._column] = 1.0
        return w


@dataclass
class EqualWeight:
    label: str = "equal_weight"
    every: int = 21

    @property
    def name(self) -> str:
        return self.label

    @property
    def rebalance_every(self) -> int:
        return self.every

    def prepare(self, panel: Panel) -> None:
        return None

    def decide(self, ctx: DecisionContext) -> FloatArray:
        n = int(ctx.eligible.sum())
        return np.where(ctx.eligible, 1.0 / max(n, 1), 0.0)
