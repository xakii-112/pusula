"""us-equity-v3 machinery: adaptive IC, realised-volatility feedback, registration compatibility."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pusula.alpha import ops
from pusula.alpha.evaluate import trailing_ic
from pusula.alpha.lookahead import check_causal
from pusula.alpha.signals import Planted
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.portfolio.context import DecisionContext
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import OptimizedAlpha
from pusula.research.families import FAMILIES
from pusula.research.study import Trial, trial_config
from pusula.risk.measures import realised_to_model_vol
from pusula.risk.model import RiskModelSpec, estimate


def _fn(p: Panel) -> FloatArray:
    m = p["in_universe"]
    return ops.cs_zscore(ops.cs_gauss(Planted().compute(p), m), m)


def _adaptive() -> OptimizedAlpha:
    spec = OptimizerSpec(max_weight=0.03, vol_target_daily=0.10 / np.sqrt(252), risk_aversion=1.0)
    return OptimizedAlpha(
        _fn,
        spec,
        ic_mode="adaptive",
        ic_window=126,
        ic_prior_days=126,
        vol_feedback=True,
        every=5,
        horizon_days=5.0,
    )


def test_trailing_ic_is_causal_and_finds_the_signal(alpha_panel: Panel, noise_panel: Panel) -> None:
    def ic_of(p: Panel) -> FloatArray:
        return trailing_ic(_fn(p), p, 5, window=126, prior_days=126, mask=p["in_universe"])[:, None]

    check_causal(ic_of, alpha_panel, [400, 700], name="trailing_ic")
    good = ic_of(alpha_panel)[-200:, 0]
    bad = ic_of(noise_panel)[-200:, 0]
    assert good.mean() > 0.02
    # flooring at zero biases a noise estimate upwards (~0.01 here); the signal must stand out
    assert good.mean() > 2.0 * bad.mean()
    assert np.all(good >= 0) and np.all(bad >= 0)


def test_realised_to_model_vol_ratio() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (600, 60))
    fr = estimate(
        R, 599, np.ones(60, dtype=bool), RiskModelSpec(n_factors=3, window=500, vra_halflife=None)
    )
    assert fr is not None
    w = np.full(60, 1.0 / 60)
    assert realised_to_model_vol(R[-63:], w, fr, w) == pytest.approx(1.0, abs=0.35)
    assert realised_to_model_vol(2.0 * R[-63:], w, fr, w) == pytest.approx(
        2.0 * realised_to_model_vol(R[-63:], w, fr, w), rel=1e-9
    )


def test_adaptive_decisions_are_causal(alpha_panel: Panel) -> None:
    seen: dict[int, np.ndarray] = {}

    class Spy(OptimizedAlpha):
        def decide(self, ctx: DecisionContext) -> np.ndarray:
            w = super().decide(ctx)
            seen[ctx.t] = w.copy()
            return w

    def spy() -> Spy:
        a = _adaptive()
        return Spy(
            a.alpha_fn,
            a.spec,
            ic_mode="adaptive",
            ic_window=126,
            ic_prior_days=126,
            vol_feedback=True,
            every=5,
            horizon_days=5.0,
        )

    cfg = BacktestConfig(start="2017-01-03", kill_switch=False)
    Backtester(alpha_panel, cfg).run(spy())
    full = dict(seen)
    seen.clear()
    cut = alpha_panel.date_index("2018-06-29")
    Backtester(alpha_panel.upto(cut), cfg).run(spy())
    common = [t for t in seen if t < cut]
    assert len(common) > 50
    for t in common:
        np.testing.assert_allclose(seen[t], full[t], atol=1e-6)


def test_vol_feedback_moves_realised_vol_towards_target(alpha_panel: Panel) -> None:
    cfg = BacktestConfig(start="2017-01-03", kill_switch=False)
    plain = replace(_adaptive(), vol_feedback=False)
    a = Backtester(alpha_panel, cfg).run(plain).summary()["ann_vol"]
    b = Backtester(alpha_panel, cfg).run(_adaptive()).summary()["ann_vol"]
    assert abs(b - 0.10) <= abs(a - 0.10) + 0.01


def test_adaptive_book_resumes_bit_for_bit(tmp_path: Path) -> None:
    p = generate(
        SyntheticSpec(n_symbols=40, start="2016-01-04", end="2017-06-30", ic=0.03, seed=21)
    )
    cfg = BacktestConfig(kill_switch=False)
    t0, mid, t1 = p.date_index("2017-01-03"), p.date_index("2017-03-31"), p.n_dates - 1
    cont = PaperEngine(Backtester(p, cfg), _adaptive(), "b", journal=Journal(tmp_path / "a.jsonl"))
    for e in range(t0, t1 + 1):
        cont.step(e)
    first = PaperEngine(Backtester(p, cfg), _adaptive(), "b", journal=Journal(tmp_path / "b.jsonl"))
    for e in range(t0, mid + 1):
        first.step(e)
    state = json.loads(json.dumps(first.control_state()))
    resumed = PaperEngine(
        Backtester(p, cfg), _adaptive(), "b", journal=Journal(tmp_path / "b.jsonl")
    )
    resumed.restore(state)
    for e in range(mid + 1, t1 + 1):
        resumed.step(e)
    assert (
        Journal(tmp_path / "a.jsonl").events[-1].hash
        == Journal(tmp_path / "b.jsonl").events[-1].hash
    )


def test_new_trial_fields_do_not_change_old_registrations() -> None:
    t = Trial("x", "equal_weight", "h")
    assert set(trial_config(t)) == {
        "name",
        "kind",
        "hypothesis",
        "signals",
        "ml",
        "optimizer",
        "ic",
        "horizon_days",
        "rebalance_every",
        "long_only",
        "vol_target_annual",
        "benchmark",
    }
    assert trial_config(replace(t, ic_mode="adaptive"))["ic_mode"] == "adaptive"
    assert "us-equity-v1" in FAMILIES
