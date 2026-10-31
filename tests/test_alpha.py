from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pusula.alpha import ops
from pusula.alpha.combine import CompositeSpec, SignalSpec, composite
from pusula.alpha.evaluate import daily_ic, forward_returns, quantile_spread, summarize_ic
from pusula.alpha.lookahead import check_causal, default_cutoffs
from pusula.alpha.signals import REGISTRY, Planted, build
from pusula.core.errors import LookaheadError
from pusula.data.panel import Panel


def _ref(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x)


def test_rolling_ops_match_pandas() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 7))
    x[rng.random(x.shape) < 0.1] = np.nan
    w, mp = 20, 15
    np.testing.assert_allclose(
        ops.rolling_mean(x, w, mp),
        _ref(x).rolling(w, min_periods=mp).mean(),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ops.rolling_std(x, w, mp), _ref(x).rolling(w, min_periods=mp).std(), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        ops.rolling_sum(x, w, mp), _ref(x).rolling(w, min_periods=mp).sum(), rtol=1e-10, atol=1e-12
    )
    y = rng.normal(size=(300, 1))
    ref = _ref(x).rolling(w, min_periods=mp).cov(pd.Series(y[:, 0]))
    np.testing.assert_allclose(ops.rolling_cov(x, y, w, mp), ref, rtol=1e-8, atol=1e-10)


def test_lag_rejects_negative() -> None:
    with pytest.raises(ValueError, match="future"):
        ops.lag(np.zeros((3, 2)), -1)


def test_cs_rank_and_gauss() -> None:
    x = np.array([[1.0, 2.0, 3.0, np.nan], [4.0, 4.0, 1.0, 2.0]])
    r = ops.cs_rank(x)
    np.testing.assert_allclose(r[0, :3], [1 / 6, 3 / 6, 5 / 6])
    assert np.isnan(r[0, 3])
    g = ops.cs_gauss(x)
    assert abs(np.nansum(g[0])) < 1e-12


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_signal_is_causal(name: str, noise_panel: Panel) -> None:
    sig = build(name)
    check_causal(sig.compute, noise_panel, default_cutoffs(noise_panel.n_dates), name=name)


def test_composite_is_causal(noise_panel: Panel) -> None:
    spec = CompositeSpec((SignalSpec("momentum"), SignalSpec("reversal", smooth_halflife=3.0)))
    check_causal(lambda p: composite(spec, p), noise_panel, [400, 700], name="composite")


def test_lookahead_detector_catches_leaks(noise_panel: Panel) -> None:
    def leaky_shift(p: Panel) -> np.ndarray:
        c = p["close"]
        out = np.full_like(c, np.nan)
        out[:-1] = c[1:] / c[:-1]
        return out

    def leaky_global_norm(p: Panel) -> np.ndarray:
        c = p["close"]
        return (c - np.nanmean(c, axis=0)) / np.nanstd(c, axis=0)

    for fn in (leaky_shift, leaky_global_norm):
        with pytest.raises(LookaheadError):
            check_causal(fn, noise_panel, [300, 600])


def test_forward_return_timing() -> None:
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-06"))
    px = np.array([[10.0], [11.0], [12.0], [13.0], [14.0]])
    p = Panel(dates, ("A",), {"open": px})
    f = forward_returns(p, horizon=1, lag=1)
    # decision at row 0 trades at open of row 1 (11) and exits at open of row 2 (12)
    assert f[0, 0] == pytest.approx(12 / 11 - 1)
    assert np.isnan(f[3, 0]) and np.isnan(f[4, 0])


def test_planted_signal_is_recovered_and_noise_is_not(
    alpha_panel: Panel, noise_panel: Panel
) -> None:
    for panel, expect in ((alpha_panel, True), (noise_panel, False)):
        mask = panel["in_universe"]
        fwd = forward_returns(panel, 1, 1)
        ic = daily_ic(Planted().compute(panel), fwd, mask)
        s = summarize_ic(ic)
        if expect:
            assert s.mean > 0.02 and s.t_nw > 4
        else:
            assert abs(s.t_nw) < 3


def test_quantile_spread_monotone_on_planted(alpha_panel: Panel) -> None:
    q = quantile_spread(
        Planted().compute(alpha_panel),
        forward_returns(alpha_panel, 1, 1),
        alpha_panel["in_universe"],
    )
    means = np.nanmean(q, axis=0)
    assert means[-1] > means[0]
    assert np.all(np.diff(means) > -2e-4)


def test_option_signal_is_causal(noise_panel: Panel) -> None:
    from pusula.alpha.signals import OPTION_SIGNALS

    rng = np.random.default_rng(0)
    iv = 0.2 + 0.05 * rng.random(noise_panel["close"].shape)
    panel = noise_panel.with_fields(iv=iv)
    for name in OPTION_SIGNALS:
        check_causal(build(name).compute, panel, default_cutoffs(panel.n_dates), name=name)
