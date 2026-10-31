"""Nihai3's portfolio engine in pusula: the forecast signals, HRP, the no-trade buffer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pusula.alpha.lookahead import check_causal, default_cutoffs
from pusula.alpha.signals import TrendForecast, TrendSignForecast
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.execution.costs import CostModel
from pusula.portfolio import hrp as H
from pusula.portfolio.context import DecisionContext
from pusula.portfolio.strategies import ForecastRiskParity, ForecastSpec
from pusula.research.families import us_trend_v1
from pusula.research.study import make_strategy, registered_config


@pytest.fixture(scope="module")
def panel() -> Panel:
    return generate(SyntheticSpec(n_symbols=40, start="2016-01-04", end="2018-12-31", seed=4))


@pytest.mark.parametrize("sig", [TrendForecast(), TrendSignForecast()])
def test_forecasts_are_causal_scaled_and_capped(
    sig: TrendForecast | TrendSignForecast, panel: Panel
) -> None:
    check_causal(sig.compute, panel, default_cutoffs(panel.n_dates), name=sig.name)
    f = sig.compute(panel)
    tail = f[-250:]
    assert np.nanmax(np.abs(tail)) <= 20.0
    assert 5.0 < float(np.nanmean(np.abs(tail))) < 20.0


def test_forecast_scale_ignores_non_members(panel: Panel) -> None:
    """A security's bars from before it joined the index set no scale: that it would join later
    was not known then."""
    member = panel["in_universe"].copy()
    member[:, 0] = 0.0
    close = panel["close"].copy()
    close[300:, 0] *= np.exp(np.cumsum(np.full(close.shape[0] - 300, 0.05)))  # a wild non-member
    base = TrendForecast().compute(panel.with_fields(in_universe=member))
    wild = TrendForecast().compute(panel.with_fields(in_universe=member, close=close))
    np.testing.assert_allclose(base[-1, 1:], wild[-1, 1:])


def test_hrp_weights_are_positive_and_split_risk_between_clusters() -> None:
    rho = np.full((5, 5), 0.05)
    rho[:4, :4] = 0.9  # four near-copies and one independent asset
    np.fill_diagonal(rho, 1.0)
    w = H.hrp(rho)
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1.0)
    assert w[4] > 0.3  # equal weight would give it 0.2
    assert w[:4].max() <= w[4] + 1e-12


def test_hrp_is_deterministic_and_handles_one_asset() -> None:
    rng = np.random.default_rng(1)
    rho = np.corrcoef(rng.normal(size=(200, 30)), rowvar=False)
    np.testing.assert_array_equal(H.hrp(rho), H.hrp(rho))
    assert H.hrp(np.ones((1, 1))).tolist() == [1.0]


def test_idm_is_one_for_clones_sqrt_n_for_independent_and_capped() -> None:
    w = np.full(4, 0.25)
    assert H.idm(w, np.ones((4, 4)), 2.5) == pytest.approx(1.0)
    assert H.idm(w, np.eye(4), 2.5) == pytest.approx(2.0)
    assert H.idm(np.full(25, 0.04), np.eye(25), 2.5) == 2.5


def test_an_unknown_correlation_does_not_switch_hrp_off() -> None:
    z = np.random.default_rng(2).normal(size=(800, 6))
    z[:790, 5] = np.nan  # a recent listing: too few blocks for any pair
    rho = H.weekly_correlation(z, 799, 756)
    assert np.isfinite(rho).all()
    assert np.all(H.hrp(rho) > 0)


def _ctx(strategy: ForecastRiskParity, panel: Panel, t: int, w: FloatArray) -> DecisionContext:
    bt = Backtester(panel, BacktestConfig())
    strategy.prepare(panel)
    return bt.context(t, 1e7, w)


def _flat_forecast(p: Panel) -> FloatArray:
    return np.full((p.n_dates, p.n_symbols), 10.0)


def test_buffer_holds_inside_the_band_and_trades_to_its_edge(panel: Panel) -> None:
    s = ForecastRiskParity(
        _flat_forecast, ForecastSpec(weighting="equal", max_gross=10.0, vol_cap_mult=100.0)
    )
    t = panel.n_dates - 1
    ctx = _ctx(s, panel, t, np.zeros(panel.n_symbols))
    first = s.decide(ctx)
    live = first > 0
    assert live.any()
    target = first / 0.9  # from flat, the book stops at the band's lower edge: 90% of the target
    # inside the band: no trade
    held = target * 1.05
    again = s.decide(replace(ctx, w_current=held))
    np.testing.assert_allclose(again[live], held[live])
    # far above it: down to the band's upper edge, not to the target
    down = s.decide(replace(ctx, w_current=target * 3.0))
    np.testing.assert_allclose(down[live], target[live] * 1.1)


def test_book_is_long_only_within_gross_and_flat_in_ineligible_names(panel: Panel) -> None:
    s = ForecastRiskParity(TrendForecast().compute, ForecastSpec())
    ctx = _ctx(s, panel, panel.n_dates - 1, np.zeros(panel.n_symbols))
    ctx = replace(
        ctx,
        eligible=ctx.eligible & (np.arange(panel.n_symbols) != 3),
        w_current=np.full(panel.n_symbols, 0.01),
    )
    w = s.decide(ctx)
    assert np.all(w >= 0)
    assert np.abs(w).sum() <= 1.0 + 1e-9
    assert w[3] == 0.0


def test_weights_do_not_depend_on_what_was_computed_before(panel: Panel) -> None:
    """The schedule's weights are a function of the panel: a resumed book asks for the same row
    without having seen the earlier ones."""
    a = ForecastRiskParity(TrendForecast().compute, ForecastSpec())
    b = ForecastRiskParity(TrendForecast().compute, ForecastSpec())
    a.prepare(panel)
    b.prepare(panel)
    for t in range(panel.n_dates - 60, panel.n_dates):
        a.weights_at(t)
    wa, ia = a.weights_at(panel.n_dates - 1)
    wb, ib = b.weights_at(panel.n_dates - 1)
    np.testing.assert_array_equal(wa, wb)
    assert ia == ib


def test_family_builds_strategies_and_hashes_its_forecast_config() -> None:
    spec = us_trend_v1()
    assert {t.name: t.kind for t in spec.trials}["forecast_hrp"] == "forecast"
    a, b = (registered_config(spec, t) for t in spec.trials[:2])
    assert a["trial"]["forecast"]["weighting"] == "hrp"
    assert a["trial"]["signals"] != b["trial"]["signals"]
    s = make_strategy(spec.trials[0], CostModel())
    assert isinstance(s, ForecastRiskParity)
    assert s.rebalance_every == 1


def test_a_backtest_of_the_engine_stays_invested_and_unlevered(panel: Panel) -> None:
    cfg = BacktestConfig(start="2018-01-02", kill_switch=False)
    s = ForecastRiskParity(TrendForecast().compute, ForecastSpec())
    res = Backtester(panel, cfg).run(s)
    assert res.gross_exposure[-100:].mean() > 0.05
    assert res.gross_exposure[-100:].max() <= 1.02  # drift between rebalances only
    assert np.isfinite(res.returns).all()
