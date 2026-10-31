"""Statistical machinery, verified against Monte Carlo ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pusula.stats import sharpe as S
from pusula.stats.bootstrap import bootstrap_ci
from pusula.stats.cv import cpcv_splits, purged_kfold, walk_forward
from pusula.stats.inference import nw_mean_test, ols_hac
from pusula.stats.multiple_testing import benjamini_hochberg, holm
from pusula.stats.pbo import pbo_cscv


def test_psr_is_calibrated_under_the_null() -> None:
    rng = np.random.default_rng(0)
    n, reps = 500, 4000
    x = rng.standard_t(5, size=(reps, n)) * 0.01
    hits = 0
    for row in x:
        if S.psr_of(row) > 0.95:
            hits += 1
    rate = hits / reps
    assert 0.035 < rate < 0.065, rate


def test_expected_max_sr_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    n_obs, n_trials, reps = 250, 50, 400
    maxima = []
    srs_all = []
    for _ in range(reps):
        r = rng.normal(0, 0.01, size=(n_trials, n_obs))
        srs = r.mean(1) / r.std(1, ddof=1)
        maxima.append(srs.max())
        srs_all.append(srs)
    var = float(np.var(np.concatenate(srs_all), ddof=1))
    approx = S.expected_max_sr(n_trials, var)
    assert approx == pytest.approx(np.mean(maxima), rel=0.05)


def test_dsr_deflates_the_best_of_many() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, size=(100, 500))
    srs = r.mean(1) / r.std(1, ddof=1)
    best = r[int(np.argmax(srs))]
    assert S.psr_of(best) > 0.95  # naive: "significant"
    assert S.dsr(best, 100, float(np.var(srs, ddof=1))) < 0.9  # deflated: not


def test_min_track_record_consistent_with_psr() -> None:
    sr, sk, ku = 0.1, -0.5, 6.0
    n = S.min_track_record(sr, sk, ku, prob=0.95)
    assert S.psr(sr, math.ceil(n), sk, ku) >= 0.95
    assert S.psr(sr, math.floor(n) - 1, sk, ku) < 0.95


def test_max_drawdown() -> None:
    assert S.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(0.5)
    assert S.max_drawdown([0.01, 0.01]) == 0.0


def test_newey_west_controls_size_under_autocorrelation() -> None:
    rng = np.random.default_rng(3)
    rej_nw = rej_iid = 0
    reps = 600
    for _ in range(reps):
        e = rng.normal(size=600)
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, e.size):
            x[i] = 0.6 * x[i - 1] + e[i]
        rej_nw += abs(nw_mean_test(x, lags=12).t) > 1.96
        rej_iid += abs(x.mean() / (x.std(ddof=1) / math.sqrt(x.size))) > 1.96
    assert rej_iid / reps > 0.25  # naive t-test is badly oversized
    assert rej_nw / reps < 0.12


def test_ols_hac_recovers_coefficients() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 2))
    y = 0.001 + X @ np.array([0.5, -0.2]) + rng.normal(0, 0.1, 2000)
    res = ols_hac(y, X, names=("a", "b"))
    assert res.get("a")[0] == pytest.approx(0.5, abs=0.01)
    assert res.get("b")[0] == pytest.approx(-0.2, abs=0.01)
    assert res.r2 > 0.9


def test_multiple_testing() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert holm(p, 0.05).sum() == 1
    assert benjamini_hochberg(p, 0.05).sum() == 2
    rng = np.random.default_rng(5)
    fdp = []
    for _ in range(300):
        null = rng.uniform(size=90)
        alt = rng.beta(0.1, 5, size=10)
        rej = benjamini_hochberg(np.concatenate([null, alt]), 0.1)
        fdp.append(rej[:90].sum() / max(1, rej.sum()))
    assert np.mean(fdp) <= 0.12


def test_purged_kfold_has_no_label_overlap() -> None:
    h = 5
    for train, test in purged_kfold(300, 5, horizon=h, embargo=3):
        lo, hi = test.min(), test.max()
        assert not np.any((train + h >= lo) & (train <= hi + 3))


def test_walk_forward_trains_strictly_before_test() -> None:
    for train, test in walk_forward(1000, first_train=300, step=100, horizon=10, embargo=5):
        assert train.max() + 10 + 5 < test.min()


def test_cpcv_split_count() -> None:
    splits = list(cpcv_splits(600, 6, 2, horizon=5))
    assert len(splits) == 15


def test_pbo_noise_vs_signal() -> None:
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 0.01, size=(1600, 20))
    assert pbo_cscv(noise, 12).pbo > 0.3  # IS winner is not reliably good OOS
    good = noise.copy()
    good[:, 0] += 0.004  # one configuration with a real, stable edge
    assert pbo_cscv(good, 12).pbo < 0.05


def test_bootstrap_ci_covers_mean() -> None:
    rng = np.random.default_rng(7)
    cover = 0
    for i in range(60):
        x = rng.normal(0.5, 1.0, 300)
        _, lo, hi = bootstrap_ci(x, lambda a: float(a.mean()), n_boot=300, mean_block=5, seed=i)
        cover += lo <= 0.5 <= hi
    assert cover >= 51
