"""Research and production orchestration end to end on synthetic markets."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula.alpha.combine import SignalSpec
from pusula.data.panel import Panel
from pusula.data.pipeline import ResearchData
from pusula.data.quality import check_and_repair
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.research import study as S
from pusula.research.gates import GateSpec
from pusula.research.registry import Registry
from pusula.research.report import render

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def market() -> Panel:
    return generate(
        SyntheticSpec(n_symbols=60, start="2016-01-04", end="2018-12-31", ic=0.03, seed=31)
    )


def _factors(dates: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    df = pd.DataFrame(rng.normal(0, 0.008, (idx.size, 6)), index=idx, columns=cols)
    df["rf"] = 0.0001
    return df


def _spec() -> S.StudySpec:
    opt = OptimizerSpec(
        max_weight=0.05, beta_max=0.05, vol_target_daily=0.10 / np.sqrt(252), n_factors=4
    )
    trials = (
        S.Trial(
            "planted_ls",
            "optimized",
            "the planted signal pays",
            signals=(SignalSpec("planted"),),
            optimizer=opt,
            ic=0.03,
            horizon_days=5.0,
            rebalance_every=5,
        ),
        S.Trial(
            "momentum_ls",
            "optimized",
            "momentum pays (it should not here)",
            signals=(SignalSpec("momentum"),),
            optimizer=opt,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        S.Trial("equal_weight", "equal_weight", "benchmark", rebalance_every=21, benchmark=True),
    )
    from pusula.risk.model import RiskModelSpec

    bt = BacktestConfig(
        start="2017-01-03",
        end="2018-12-31",
        kill_switch=False,
        risk_spec=RiskModelSpec(n_factors=4, window=200),
    )
    return S.StudySpec(
        "synth-v1",
        trials,
        ("2017-01-03", "2017-12-29"),
        ("2018-01-02", "2018-12-31"),
        GateSpec(),
        bt,
    )


def test_study_end_to_end(tmp_path: Path, market: Panel) -> None:
    spec = _spec()
    reg = Registry(tmp_path / "ledger.jsonl")
    hashes = S.register(spec, reg, T0)
    assert S.verify_registration(spec, reg) == hashes
    rf = np.full(market.n_dates, 0.0001)
    runs = S.run_backtests(spec, lambda: (market, rf), workers=1)
    assert set(runs["planted_ls"]) == {"base", "lag2", "cost2"} and set(runs["equal_weight"]) == {
        "base"
    }
    outs = S.evaluate_study(spec, runs, _factors(market.dates), reg.program_trials("synth-"))
    by = {o.name: o for o in outs}
    assert by["planted_ls"].holdout["sharpe"] > 1.5
    assert by["planted_ls"].holdout["sharpe"] > by["momentum_ls"].holdout["sharpe"]
    assert by["planted_ls"].cost2_holdout["t_nw"] < by["planted_ls"].holdout["t_nw"]
    assert np.isfinite(by["planted_ls"].dsr) and by["equal_weight"].verdict == {}
    S.record(spec, reg, outs, hashes, "digest", T0 + dt.timedelta(hours=1), code_files={"x": "y"})
    md = render(reg, "synth-v1")
    assert "planted_ls" in md and "benchmark" in md and "Gate detail" in md
    # an edited trial must not run under the old registration
    edited = S.StudySpec(
        spec.family,
        (
            S.Trial(
                "planted_ls",
                "optimized",
                "changed",
                signals=(SignalSpec("planted"),),
                optimizer=spec.trials[0].optimizer,
                ic=0.05,
            ),
            *spec.trials[1:],
        ),
        spec.dev,
        spec.holdout,
        spec.gates,
        spec.backtest,
    )
    from pusula.core.errors import RegistryError

    with pytest.raises(RegistryError, match="differs from the registered"):
        S.verify_registration(edited, reg)
    # re-evaluation from the ledger reproduces the stored DSR when nothing else changed
    re = S.reevaluate(reg, "synth-v1", "synth-", T0 + dt.timedelta(hours=2), "check", code_files={})
    assert re["planted_ls"]["dsr"] == pytest.approx(by["planted_ls"].dsr, rel=1e-9)


def test_forward_books_two_nights(
    tmp_path: Path, market: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pusula.live.forward as F

    monkeypatch.setattr(F, "rf_series", lambda p: np.zeros(p.n_dates))
    spec = _spec()
    when = {t.name: dt.datetime(2018, 6, 1, 22, tzinfo=dt.UTC) for t in spec.trials}

    def rd(p: Panel) -> ResearchData:
        return ResearchData(p, p, check_and_repair(p)[1], {}, "d", ())

    night1 = F.step_books(
        spec, when, rd=rd(market.upto(market.date_index("2018-09-28"))), root=tmp_path, log=False
    )
    night2 = F.step_books(spec, when, rd=rd(market), root=tmp_path, log=False)
    assert night1["as_of"] == "2018-09-28" and night2["as_of"] == "2018-12-31"
    for name in ("planted_ls", "equal_weight"):
        c1, c2 = night1["books"][name]["equity_curve"], night2["books"][name]["equity_curve"]
        assert c1[0][0] == "2018-06-04" and c2[: len(c1)] == c1  # the past is never rewritten
        assert c2[-1][0] == "2018-12-31"
    state = json.loads((tmp_path / "planted_ls" / "state.json").read_text())
    assert state["last_step"] == "2018-12-31"
    assert (tmp_path / "daily_report.md").exists()
    # a third run on the same data processes nothing
    night3 = F.step_books(spec, when, rd=rd(market), root=tmp_path, log=False)
    assert (
        night3["books"]["planted_ls"]["equity_curve"]
        == night2["books"]["planted_ls"]["equity_curve"]
    )


def test_options_study_runs_and_records(
    tmp_path: Path, market: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pusula.research import options_study as O

    monkeypatch.setattr(O, "START", "2017-01-03")
    monkeypatch.setattr(O, "END", "2018-12-31")
    monkeypatch.setattr(O, "DEV", ("2017-01-03", "2017-12-29"))
    monkeypatch.setattr(O, "HOLDOUT", ("2018-01-02", "2018-12-31"))
    reg = Registry(tmp_path / "ledger.jsonl")
    O.register(reg, T0)
    iv = np.full((market.n_dates, market.n_symbols), 0.3)
    out = O.run(
        reg,
        market,
        iv,
        np.zeros(market.n_dates),
        _factors(market.dates),
        "d",
        T0 + dt.timedelta(hours=1),
        code_files={},
    )
    assert set(out) == {t.name for t in O.TRIALS}
    assert out["stock_mirror_t21"]["verdict"]["benchmark"] is True
    assert all("checks" in out[t.name]["verdict"] for t in O.TRIALS if not t.benchmark)
    assert reg.latest_results(O.FAMILY)


def test_spread_calibration_recovers_known_coefficient() -> None:
    from pusula.execution.calibration import SpreadSample, calibrate

    rng = np.random.default_rng(0)
    base = rng.uniform(1e-5, 2e-4, 400)
    samples = [
        SpreadSample(
            f"S{i}", f"20{16 + i % 10}-06-15", float(2.3 * b * np.exp(rng.normal(0, 0.1))), float(b)
        )
        for i, b in enumerate(base)
    ]
    cal = calibrate(samples)
    assert cal.k_median == pytest.approx(2.3, rel=0.05)
    assert cal.loglog_slope == pytest.approx(1.0, abs=0.05)
    assert cal.n == 400 and len(cal.by_year) == 10
