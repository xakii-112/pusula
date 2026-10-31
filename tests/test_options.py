from __future__ import annotations

import numpy as np
import pytest

from pusula.alpha.signals import TimeSeriesMomentum
from pusula.data.panel import Panel
from pusula.options import black
from pusula.options.overlay import CallOverlaySpec, run_call_overlay


def test_hull_reference_values() -> None:
    # Hull, Options Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, T=0.5, vol=20%
    assert float(black.price(42, 40, 0.5, 0.1, 0.0, 0.2, 1)) == pytest.approx(4.76, abs=0.005)
    assert float(black.price(42, 40, 0.5, 0.1, 0.0, 0.2, -1)) == pytest.approx(0.81, abs=0.005)


def test_put_call_parity_with_yield() -> None:
    rng = np.random.default_rng(0)
    S, K = rng.uniform(50, 150, 50), rng.uniform(50, 150, 50)
    T, r, q, v = (
        rng.uniform(0.05, 2, 50),
        rng.uniform(0, 0.06, 50),
        rng.uniform(0, 0.04, 50),
        rng.uniform(0.1, 0.8, 50),
    )
    c = black.price(S, K, T, r, q, v, 1)
    p = black.price(S, K, T, r, q, v, -1)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-9)


@pytest.mark.parametrize("cp", [1, -1])
def test_greeks_match_finite_differences(cp: int) -> None:
    S, K, T, r, q, v = 100.0, 95.0, 0.4, 0.03, 0.01, 0.35
    h = 1e-4
    f = lambda **kw: float(black.price(**{**dict(S=S, K=K, T=T, r=r, q=q, vol=v, cp=cp), **kw}))  # noqa: E731
    assert float(black.delta(S, K, T, r, q, v, cp)) == pytest.approx(
        (f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6
    )
    assert float(black.gamma(S, K, T, r, q, v)) == pytest.approx(
        (f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4
    )
    assert float(black.vega(S, K, T, r, q, v)) == pytest.approx(
        (f(vol=v + h) - f(vol=v - h)) / (2 * h), rel=1e-6
    )
    assert float(black.theta(S, K, T, r, q, v, cp)) == pytest.approx(
        -(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5
    )


def test_implied_vol_round_trip_and_bounds() -> None:
    S = np.full(6, 100.0)
    # (K=60, T=0.1, vol=0.15 is unidentifiable: its time value is 5e-28, below double precision)
    K = np.array([85.0, 80, 100, 120, 150, 100])
    T = np.array([0.25, 0.5, 1.0, 0.25, 2.0, 0.02])
    vol = np.array([0.15, 0.3, 0.5, 0.9, 0.25, 1.5])
    px = black.price(S, K, T, 0.02, 0.0, vol, 1)
    np.testing.assert_allclose(black.implied_vol(px, S, K, T, 0.02, 0.0, 1), vol, rtol=1e-6)
    pp = black.price(S, K, T, 0.02, 0.0, vol, -1)
    np.testing.assert_allclose(black.implied_vol(pp, S, K, T, 0.02, 0.0, -1), vol, rtol=1e-6)
    # below intrinsic -> no solution
    assert np.isnan(black.implied_vol(np.array([0.01]), 100.0, 50.0, 1.0, 0.0, 0.0, 1))[0]


def test_strike_for_delta_and_grid() -> None:
    k = black.strike_for_delta(187.3, 30 / 365, 0.04, 0.0, 0.3, 0.5)
    assert k in black.listed_strikes(187.3)
    assert abs(float(black.delta(187.3, k, 30 / 365, 0.04, 0.0, 0.3)) - 0.5) < 0.05


def test_call_overlay_payoffs_are_exact(alpha_panel: Panel) -> None:
    score = TimeSeriesMomentum().compute(alpha_panel)
    iv = np.full((alpha_panel.n_dates, alpha_panel.n_symbols), 0.30)
    rf = np.zeros(alpha_panel.n_dates)
    t0 = 300
    res = run_call_overlay(
        alpha_panel,
        score,
        iv,
        rf,
        t0,
        alpha_panel.n_dates - 1,
        CallOverlaySpec(max_names=20, dividend_yield=0.0, option_half_spread=0.0),
    )
    assert np.isfinite(res.equity).all()
    assert res.n_positions.max() > 0
    assert len(res.payoff_ratio) > 50
    # a call's payoff / premium is >= 0 and the average is plausible for ATM monthly calls
    pr = np.asarray(res.payoff_ratio)
    assert np.all(pr >= 0)
    assert 0.3 < float(np.mean(pr)) < 3.0
