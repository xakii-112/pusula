from __future__ import annotations

import numpy as np
import pytest

from pusula.data.panel import Panel
from pusula.risk.limits import KillSwitch, RiskLimits, gate_targets
from pusula.risk.measures import historical_var, scenario_pnl
from pusula.risk.model import RiskModelSpec, estimate, simple_returns
from pusula.risk.validate import bias_statistics


def _factor_returns(
    T: int = 600, N: int = 150, K: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (N, K)) * 0.008
    B[:, 0] = rng.normal(1.0, 0.25, N) * 0.01  # market
    f = rng.normal(0, 1, (T, K))
    spec_vol = rng.uniform(0.008, 0.02, N)
    R = f @ B.T + rng.normal(0, 1, (T, N)) * spec_vol
    true_cov = B @ B.T + np.diag(spec_vol**2)
    return R, true_cov


def test_risk_model_recovers_portfolio_vols() -> None:
    R, cov = _factor_returns()
    fr = estimate(
        R,
        R.shape[0] - 1,
        np.ones(R.shape[1], dtype=bool),
        RiskModelSpec(
            n_factors=3,
            vol_halflife=1e9,
            factor_halflife=1e9,
            specific_halflife=1e9,
            window=600,
            specific_shrinkage=0.0,
            vra_halflife=None,
        ),
    )
    assert fr is not None
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = rng.normal(size=R.shape[1])
        w /= np.abs(w).sum()
        true = float(np.sqrt(w @ cov @ w))
        assert fr.vol(w, annualise=False) == pytest.approx(true, rel=0.15)


def test_vra_reacts_to_a_volatility_shock() -> None:
    R, _ = _factor_returns(T=600, seed=4)
    R2 = R.copy()
    R2[-15:] *= 3.0  # the last three weeks are three times as volatile
    m = np.ones(150, dtype=bool)
    w = np.full(150, 1.0 / 150)
    base = estimate(R2, 599, m, RiskModelSpec(n_factors=3, window=500, vra_halflife=None))
    adj = estimate(R2, 599, m, RiskModelSpec(n_factors=3, window=500, vra_halflife=10.0))
    calm = estimate(R, 599, m, RiskModelSpec(n_factors=3, window=500, vra_halflife=10.0))
    assert base is not None and adj is not None and calm is not None
    assert adj.vol(w) > 1.5 * base.vol(w)
    assert calm.vol(w) == pytest.approx(
        estimate(R, 599, m, RiskModelSpec(n_factors=3, window=500, vra_halflife=None)).vol(w),  # type: ignore[union-attr]
        rel=0.35,
    )


def test_risk_model_identities() -> None:
    R, _ = _factor_returns(seed=2)
    fr = estimate(R, 599, np.ones(150, dtype=bool), RiskModelSpec(n_factors=5, window=400))
    assert fr is not None
    w = np.full(fr.n, 1.0 / fr.n)
    np.testing.assert_allclose(fr.variance(w), w @ fr.covariance() @ w, rtol=1e-10)
    assert fr.marginal_contribution(w).sum() == pytest.approx(fr.vol(w, annualise=False), rel=1e-10)
    assert float(w @ fr.beta(w)) == pytest.approx(1.0, rel=1e-10)


def test_risk_model_is_causal() -> None:
    R, _ = _factor_returns(seed=3)
    m = np.ones(150, dtype=bool)
    spec = RiskModelSpec(n_factors=4, window=300)
    a = estimate(R, 400, m, spec)
    R2 = R.copy()
    R2[401:] = 99.0  # garbage after the decision row
    b = estimate(R2, 400, m, spec)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a.B, b.B)
    np.testing.assert_array_equal(a.d, b.d)


def test_bias_statistic_near_one_out_of_sample(noise_panel: Panel) -> None:
    r = simple_returns(noise_panel["close"])
    members = noise_panel["in_universe"] > 0
    reps = bias_statistics(
        r, members, RiskModelSpec(n_factors=4, window=252), start=300, step=10, n_random=3
    )
    by = {b.portfolio: b for b in reps}
    for name in ("equal_weight", "random_long_short"):
        assert 0.75 < by[name].bias < 1.3, (name, by[name])


def test_mean_variance_weights_match_a_dense_solve() -> None:
    from pusula.risk.validate import mean_variance_weights, solve_sigma

    R, _ = _factor_returns(seed=5)
    fr = estimate(R, 599, np.ones(150, dtype=bool), RiskModelSpec(n_factors=4, window=400))
    assert fr is not None
    x = np.random.default_rng(0).normal(size=fr.n)
    np.testing.assert_allclose(solve_sigma(fr, x), np.linalg.solve(fr.covariance(), x), rtol=1e-8)
    wn = mean_variance_weights(fr, x, neutral=True)
    assert abs(wn.sum()) < 1e-10 and np.abs(wn).sum() == pytest.approx(1.0)
    assert mean_variance_weights(fr, np.ones(fr.n), neutral=False).sum() == pytest.approx(1.0)


def test_eigen_adjustment_corrects_the_smallest_eigenportfolio() -> None:
    """Menchero-Wang-Orr: in small samples the estimated covariance under-states the variance of
    its own smallest eigenportfolio; the adjustment must move that forecast towards the truth."""
    from pusula.risk.model import eigen_adjusted

    rng = np.random.default_rng(7)
    k, T = 10, 60
    true_var = np.linspace(1.0, 10.0, k)
    w = np.full(T, 1.0 / T)
    raw_bias, adj_bias = [], []
    for _ in range(200):
        f = rng.standard_normal((T, k)) * np.sqrt(true_var)
        fc = f - f.mean(0)
        S = fc.T @ fc / T
        lam, U = np.linalg.eigh(S)
        u = U[:, 0]  # smallest estimated eigenportfolio
        truth = float(u @ (true_var * u))
        raw_bias.append(truth / lam[0])
        adj = eigen_adjusted(S, w, a=1.0, sims=50)
        adj_bias.append(truth / float(u @ adj @ u))
    assert np.mean(raw_bias) > 1.3  # the raw sample under-forecasts it badly
    assert abs(np.mean(adj_bias) - 1.0) < abs(np.mean(raw_bias) - 1.0) / 2
    adj = eigen_adjusted(S, w, a=1.2, sims=50)
    np.testing.assert_allclose(adj, adj.T)
    np.testing.assert_array_equal(adj, eigen_adjusted(S, w, a=1.2, sims=50))  # seeded


def test_industry_factors_capture_a_cluster_the_statistical_factors_miss() -> None:
    """Ten names share a weak common shock: invisible to 3 PCs, it inflates the true risk of a
    portfolio concentrated in them; industry factors put it back into the forecast."""
    R, cov = _factor_returns(T=600, N=150, K=3, seed=8)
    rng = np.random.default_rng(9)
    shock = rng.normal(0, 0.006, R.shape[0])
    R[:, :10] += shock[:, None]
    cov = cov.copy()
    cov[:10, :10] += 0.006**2
    groups = np.full(150, -1, dtype=np.int64)
    groups[:10] = 7
    m = np.ones(150, dtype=bool)
    base_spec = RiskModelSpec(n_factors=3, window=600, vra_halflife=None, specific_shrinkage=0.0)
    ind_spec = RiskModelSpec(
        n_factors=3, window=600, vra_halflife=None, specific_shrinkage=0.0, industry_min_size=5
    )
    base = estimate(R, 599, m, base_spec)
    ind = estimate(R, 599, m, ind_spec, groups)
    assert base is not None and ind is not None and ind.B.shape[1] == 4
    w = np.zeros(150)
    w[:10] = 0.1  # concentrated in the cluster
    true = float(np.sqrt(w @ cov @ w))
    assert abs(ind.vol(w, annualise=False) / true - 1) < abs(
        base.vol(w, annualise=False) / true - 1
    )
    # off by default: identical to the model without groups
    same = estimate(R, 599, m, base_spec, groups)
    assert same is not None
    np.testing.assert_array_equal(same.B, base.B)
    np.testing.assert_array_equal(same.d, base.d)


def test_gate_repairs_and_reports() -> None:
    syms = ("A", "B", "C", "D")
    cur = np.array([0.0, 0.02, -0.01, 0.0])
    tgt = np.array([0.30, 0.02, -0.05, 0.10])
    price = np.array([50.0, 50.0, 50.0, 2.0])  # D below the $5 floor
    adv = np.full(4, 1e9)
    res = gate_targets(
        tgt,
        cur,
        syms,
        RiskLimits(max_weight=0.05, hard_to_borrow=frozenset({"C"})),
        price=price,
        adv_dollars=adv,
        equity=1e6,
    )
    assert res.accepted
    assert res.weights[0] == pytest.approx(0.05)
    assert res.weights[2] == pytest.approx(-0.01)  # HTB short not increased
    assert res.weights[3] == 0.0  # sub-$5 name cannot be bought
    assert len(res.breaches) >= 3


def test_gate_rejects_nonfinite() -> None:
    cur = np.zeros(3)
    res = gate_targets(
        np.array([0.1, np.nan, 0.0]),
        cur,
        ("A", "B", "C"),
        RiskLimits(),
        price=np.full(3, 10.0),
        adv_dollars=np.full(3, 1e9),
        equity=1e6,
    )
    assert not res.accepted
    np.testing.assert_array_equal(res.weights, cur)


def test_gate_participation_cap() -> None:
    res = gate_targets(
        np.array([0.05]),
        np.zeros(1),
        ("A",),
        RiskLimits(max_participation=0.05),
        price=np.array([10.0]),
        adv_dollars=np.array([1e6]),
        equity=1e7,
    )
    # 5% of $1M ADV = $50k = 0.5% of $10M equity
    assert res.weights[0] == pytest.approx(0.005)


def test_kill_switch_latches() -> None:
    ks = KillSwitch(max_drawdown=0.1, max_daily_loss=0.08, derisk_drawdown=0.05, derisk_scale=0.5)
    assert ks.update(100.0, 100.0) == 1.0
    assert ks.update(94.0, 100.0) == 0.5  # 6% drawdown: de-risk
    assert ks.update(89.0, 94.0) == 0.0  # 11% drawdown: trip
    assert ks.update(120.0, 89.0) == 0.0  # stays tripped
    ks.reset(120.0)
    assert ks.update(120.0, 120.0) == 1.0


def test_var_and_scenarios() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (1000, 5))
    w = np.full(5, 0.2)
    rep = historical_var(w, R)
    assert 0 < rep.var95 < rep.var99 <= rep.es975 * 1.5
    assert scenario_pnl(
        w, np.array([-0.1, np.nan, 0.0, 0.0, 0.0]), np.ones(5), -0.2
    ) == pytest.approx(0.2 * (-0.1 - 0.2))
