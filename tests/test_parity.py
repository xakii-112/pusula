"""Research backtester vs. production paper engine on the same market and strategy."""

from __future__ import annotations

import numpy as np
import pytest

from pusula.alpha import ops
from pusula.alpha.signals import Planted
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import EqualWeight, OptimizedAlpha


def _fn(p: Panel) -> FloatArray:
    m = p["in_universe"]
    return ops.cs_zscore(ops.cs_gauss(Planted().compute(p), m), m)


@pytest.fixture(scope="module")
def market() -> Panel:
    return generate(
        SyntheticSpec(n_symbols=60, start="2016-01-04", end="2017-12-29", ic=0.03, seed=11)
    )


@pytest.mark.parametrize("which", ["equal_weight", "optimized"])
def test_paper_engine_matches_backtester(market: Panel, which: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = BacktestConfig(start="2017-01-03", kill_switch=False)

    def strat():  # type: ignore[no-untyped-def]
        if which == "equal_weight":
            return EqualWeight(every=5)
        return OptimizedAlpha(_fn, OptimizerSpec(max_weight=0.05, beta_max=0.05), ic=0.03, every=5)

    bt = Backtester(market, cfg)
    ref = bt.run(strat())
    j = Journal(tmp_path / "paper.jsonl")
    eng = PaperEngine(Backtester(market, cfg), strat(), "parity", journal=j)
    t0 = market.date_index("2017-01-03")
    eng.run(t0, market.n_dates - 1)
    eq = np.array([eng.equity_curve[e] for e in range(t0, market.n_dates)], dtype=float) / 1e6
    r_paper = eq[1:] / eq[:-1] - 1.0
    r_ref = ref.returns[1:]
    assert np.max(np.abs(r_paper - r_ref)) < 5e-4  # share rounding only
    assert np.corrcoef(r_paper, r_ref)[0, 1] > 0.999
    assert abs(eq[-1] / eq[0] - ref.equity[-1] / ref.equity[0]) < 0.01
    # the journal alone reconstructs the book exactly
    led, marks = Journal(tmp_path / "paper.jsonl").replay()
    assert led.equity(marks) == eng.equity_curve[market.n_dates - 1]
    led.check_identity(marks)
