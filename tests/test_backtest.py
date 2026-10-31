"""Backtester: accounting identities, timing, costs, capacity, delisting, end-to-end causality."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pusula.alpha import ops
from pusula.alpha.signals import Planted
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.execution.costs import CostModel
from pusula.portfolio.context import DecisionContext
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import EqualWeight, OptimizedAlpha
from pusula.stats import sharpe as S


def _planted_fn(p: Panel) -> FloatArray:
    m = p["in_universe"]
    return ops.cs_zscore(ops.cs_gauss(Planted().compute(p), m), m)


def _opt() -> OptimizedAlpha:
    return OptimizedAlpha(
        _planted_fn, OptimizerSpec(max_weight=0.03, beta_max=0.05), ic=0.03, every=5
    )


@pytest.fixture(scope="module")
def small_alpha() -> Panel:
    return generate(
        SyntheticSpec(n_symbols=100, start="2016-01-04", end="2018-12-31", ic=0.03, seed=5)
    )


def test_equity_identity(small_alpha: Panel) -> None:
    res = Backtester(small_alpha, BacktestConfig(start="2017-01-03", kill_switch=False)).run(
        EqualWeight()
    )
    eq = res.equity
    np.testing.assert_allclose(eq[1:] / eq[:-1] - 1.0, res.returns[1:], rtol=1e-12)
    # gross - net = (costs + borrow) / previous equity
    np.testing.assert_allclose(
        res.gross_returns[1:] - res.returns[1:],
        (res.costs[1:] + res.borrow[1:]) / eq[:-1],
        atol=1e-12,
    )


def test_zero_cost_equal_weight_tracks_index_return(small_alpha: Panel) -> None:
    free = CostModel(
        spread_floor_bps=0.0,
        spread_cap_bps=0.0,
        fallback_half_spread_bps=0.0,
        impact_y=0.0,
        commission_bps=0.0,
        sec_fee_rate=0.0,
        borrow_annual=0.0,
        max_participation=1.0,
    )
    cfg = BacktestConfig(
        start="2017-01-03", kill_switch=False, cost_model=free, exec_price="close", lag=0
    )
    res = Backtester(small_alpha, cfg).run(EqualWeight(every=1))
    # daily-rebalanced equal weight == mean simple return of the names held
    r = small_alpha["close"][1:] / small_alpha["close"][:-1] - 1.0
    t0 = int(np.searchsorted(small_alpha.dates, np.datetime64("2017-01-03")))
    W = res.weights.astype(np.float64)
    expected = np.nansum(W[:-1] * r[t0 : t0 + W.shape[0] - 1], axis=1)
    np.testing.assert_allclose(res.returns[2:], expected[1:], atol=2e-6)


def test_planted_alpha_profits_and_costs_bite(small_alpha: Panel) -> None:
    cfg = BacktestConfig(start="2017-01-03")
    base = Backtester(small_alpha, cfg).run(_opt())
    s = base.summary()
    assert s["sharpe"] > 2.0
    assert abs(s["avg_beta"]) < 0.1
    assert s["sharpe_gross"] > s["sharpe"]
    dear = Backtester(small_alpha, replace(cfg, cost_model=CostModel().stressed(5.0))).run(_opt())
    assert dear.summary()["sharpe"] < s["sharpe"]


def test_noise_does_not_profit() -> None:
    noise = generate(
        SyntheticSpec(n_symbols=100, start="2016-01-04", end="2018-12-31", ic=0.0, seed=6)
    )
    res = Backtester(noise, BacktestConfig(start="2017-01-03")).run(_opt())
    assert S.psr_of(res.returns[1:]) < 0.95


def test_execution_lag_reduces_planted_edge(small_alpha: Panel) -> None:
    a = (
        Backtester(small_alpha, BacktestConfig(start="2017-01-03", lag=1))
        .run(_opt())
        .summary()["sharpe"]
    )
    b = (
        Backtester(small_alpha, BacktestConfig(start="2017-01-03", lag=3))
        .run(_opt())
        .summary()["sharpe"]
    )
    assert b < a


def test_participation_cap_limits_fills(small_alpha: Panel) -> None:
    tiny = CostModel(max_participation=1e-5)
    res = Backtester(
        small_alpha, BacktestConfig(start="2017-01-03", cost_model=tiny, kill_switch=False)
    ).run(EqualWeight())
    assert res.summary()["unfilled"] > 0.5
    assert res.gross_exposure[-1] < 0.5


def test_delisted_positions_are_closed() -> None:
    p = generate(
        SyntheticSpec(n_symbols=60, start="2016-01-04", end="2018-12-31", seed=7, listing_churn=0.5)
    )
    res = Backtester(p, BacktestConfig(start="2016-03-01", kill_switch=False)).run(
        EqualWeight(every=5)
    )
    assert any("delisted" in e for e in res.events)
    last = np.isfinite(p["close"])[::-1].argmax(axis=0)
    gone = last > 0  # symbols whose data ends before the panel end
    final_w = res.weights[-1]
    assert np.all(np.abs(final_w[gone]) < 1e-9)


def test_end_to_end_decisions_are_causal(small_alpha: Panel) -> None:
    """Decisions up to row c must not change when data after c is appended."""
    seen: dict[int, np.ndarray] = {}

    class Spy(OptimizedAlpha):
        def decide(self, ctx: DecisionContext) -> np.ndarray:
            w = super().decide(ctx)
            seen[ctx.t] = w.copy()
            return w

    cfg = BacktestConfig(start="2017-01-03", end="2018-12-31")
    Backtester(small_alpha, cfg).run(
        Spy(_planted_fn, OptimizerSpec(max_weight=0.03), ic=0.03, every=5)
    )
    full = dict(seen)
    seen.clear()
    cut = small_alpha.date_index("2018-03-29")
    Backtester(small_alpha.upto(cut), replace(cfg, end=None)).run(
        Spy(_planted_fn, OptimizerSpec(max_weight=0.03), ic=0.03, every=5)
    )
    common = [t for t in seen if t < cut]
    assert len(common) > 20
    for t in common:
        np.testing.assert_allclose(seen[t], full[t], atol=1e-6)


def test_book_halted_in_cash_has_no_sharpe() -> None:
    """Regression (us-equity-v1, 2026-10-24): a kill-switched book earning T-bills showed a raw
    Sharpe of 80. Statistics must be computed on excess returns."""
    p = generate(SyntheticSpec(n_symbols=40, start="2016-01-04", end="2018-12-31", seed=12))
    rf = 0.0002 + np.random.default_rng(0).normal(0, 2e-6, p.n_dates)  # ~5% a year, daily prints
    cfg = BacktestConfig(
        start="2016-03-01", kill_switch=True, max_drawdown=0.02, max_daily_loss=0.5
    )
    res = Backtester(p, cfg, rf).run(EqualWeight(every=5))
    assert res.halted_on is not None
    after = res.dates > np.datetime64(res.halted_on) + np.timedelta64(10, "D")
    raw = res.returns[after]
    assert S.annualised_sharpe(raw) > 20  # the artefact the old metric reported
    assert np.isnan(S.annualised_sharpe(res.excess_returns[after]))  # no risk, no Sharpe
    assert abs(res.summary()["sharpe"]) < 3.0


def test_cost_stress_charges_more_without_changing_decisions(small_alpha: Panel) -> None:
    """The 2x-cost variant must charge double while the strategy keeps believing the model
    costs; otherwise a cost-aware optimiser simply trades less and the stress tests nothing."""
    base_cfg = BacktestConfig(start="2017-01-03", kill_switch=False)
    stressed = replace(base_cfg, cost_model=base_cfg.cost_model.stressed(2.0))

    def strat() -> OptimizedAlpha:
        return OptimizedAlpha(
            _planted_fn,
            OptimizerSpec(max_weight=0.03),
            ic=0.03,
            every=5,
            cost_model=base_cfg.cost_model,
        )

    a = Backtester(small_alpha, base_cfg).run(strat())
    b = Backtester(small_alpha, stressed).run(strat())
    assert np.sum(b.costs) == pytest.approx(2.0 * np.sum(a.costs), rel=0.05)
    assert np.mean(b.turnover) == pytest.approx(np.mean(a.turnover), rel=0.02)
    assert b.summary()["sharpe"] < a.summary()["sharpe"]
