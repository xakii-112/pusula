from __future__ import annotations

import numpy as np
import pytest

from pusula.portfolio.optimizer import OptimizerInput, OptimizerSpec, PortfolioOptimizer


def _inp(n: int = 60, k: int = 3, seed: int = 0, **kw: object) -> OptimizerInput:
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 0.01, (n, k))
    B[:, 0] = rng.normal(1, 0.2, n) * 0.01
    base = dict(
        alpha=rng.normal(0, 0.002, n),
        w0=np.zeros(n),
        B=B,
        F=np.eye(k),
        d=np.full(n, 0.015**2),
        eligible=np.ones(n, dtype=bool),
        lin_cost=np.full(n, 3e-4),
        impact=np.full(n, 1e-3),
        trade_cap=np.full(n, 0.05),
        beta=B[:, 0] / 0.01,
        horizon_days=5.0,
    )
    base.update(kw)
    return OptimizerInput(**base)  # type: ignore[arg-type]


def test_constraints_hold() -> None:
    spec = OptimizerSpec(
        max_gross=1.5, max_weight=0.04, beta_max=0.02, n_factors=3, risk_aversion=2.0
    )
    inp = _inp()
    res = PortfolioOptimizer(60, spec).solve(inp)
    w = res.w
    assert np.abs(w).sum() <= 1.5 + 1e-6
    assert np.abs(w).max() <= 0.04 + 1e-6
    assert abs(w.sum()) <= 1e-6
    assert abs(inp.beta @ w) <= 0.02 + 1e-6
    assert res.expected_alpha > 0
    assert not res.relaxed


def test_ineligible_names_are_sold_and_not_bought() -> None:
    el = np.ones(60, dtype=bool)
    el[:5] = False
    w0 = np.zeros(60)
    w0[0] = 0.03  # held, but no longer eligible
    inp = _inp(eligible=el, w0=w0, trade_cap=np.full(60, 0.01))
    res = PortfolioOptimizer(60, OptimizerSpec(n_factors=3, beta_max=None)).solve(inp)
    assert np.all(np.abs(res.w[1:5]) < 1e-6)
    assert res.w[0] == pytest.approx(0.0, abs=1e-6)  # forced to zero despite the 1% trade cap


def test_higher_costs_mean_less_turnover() -> None:
    spec = OptimizerSpec(n_factors=3, beta_max=None)
    lo = PortfolioOptimizer(60, spec).solve(_inp(lin_cost=np.full(60, 1e-5))).w
    hi = PortfolioOptimizer(60, spec).solve(_inp(lin_cost=np.full(60, 2e-3))).w
    assert np.abs(hi).sum() < np.abs(lo).sum()


def test_vol_target_is_respected() -> None:
    spec = OptimizerSpec(n_factors=3, beta_max=None, vol_target_daily=0.002, risk_aversion=0.01)
    res = PortfolioOptimizer(60, spec).solve(_inp())
    assert res.ex_ante_vol_daily <= 0.002 * 1.001


def test_long_only_fully_invested_reachable_over_days() -> None:
    spec = OptimizerSpec(
        n_factors=3,
        long_only=True,
        net_lo=1.0,
        net_hi=1.0,
        max_gross=1.0,
        max_weight=0.05,
        beta_max=None,
    )
    opt = PortfolioOptimizer(60, spec)
    w = np.zeros(60)
    for _ in range(4):  # 10% per name per day cap: cannot be fully invested on day one
        res = opt.solve(_inp(w0=w, trade_cap=np.full(60, 0.01)))
        assert np.all(res.w >= -1e-8)
        w = res.w
    assert w.sum() == pytest.approx(1.0, abs=1e-4)


def test_dpp_reuse_is_fast_enough() -> None:
    import time

    opt = PortfolioOptimizer(300, OptimizerSpec(n_factors=3))
    opt.solve(_inp(n=300))
    t = time.perf_counter()
    for s in range(3):
        opt.solve(_inp(n=300, seed=s))
    assert (time.perf_counter() - t) / 3 < 2.0
