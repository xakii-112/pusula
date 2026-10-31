from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pusula.alpha import ops
from pusula.alpha.signals import Planted
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.live.forward import book_snapshot, first_session_after
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import OptimizedAlpha
from pusula.research.registry import Registry
from pusula.web.app import create_app


def _fn(p: Panel) -> FloatArray:
    m = p["in_universe"]
    return ops.cs_zscore(ops.cs_gauss(Planted().compute(p), m), m)


def test_book_snapshot_reports_risk_and_stress(tmp_path: Path) -> None:
    p = generate(SyntheticSpec(n_symbols=60, start="2016-01-04", end="2017-06-30", ic=0.03, seed=8))
    eng = PaperEngine(
        Backtester(p, BacktestConfig(kill_switch=False)),
        OptimizedAlpha(_fn, OptimizerSpec(max_weight=0.05), ic=0.03, every=5),
        "b",
        journal=Journal(tmp_path / "j.jsonl"),
    )
    for e in range(p.date_index("2017-01-03"), p.n_dates):
        eng.step(e)
    snap = book_snapshot(eng, p, SimpleNamespace(sectors={}))  # type: ignore[arg-type]
    assert snap["n_positions"] > 10
    assert 0.0 < snap["risk"]["ex_ante_vol"] < 1.0
    assert snap["risk"]["var99"] >= snap["risk"]["var95"]
    assert len(snap["equity_curve"]) == p.n_dates - p.date_index("2017-01-03")
    json.dumps(snap)  # serialisable for the dashboard


def test_forward_books_open_after_registration() -> None:
    p = generate(SyntheticSpec(n_symbols=5, start="2016-01-04", end="2016-02-29", seed=1))
    i = first_session_after(p, dt.datetime(2016, 1, 15, 22, 0, tzinfo=dt.UTC))
    assert str(p.dates[i]) == "2016-01-19"  # Fri registration -> MLK Monday closed -> Tuesday


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    reg = Registry(tmp_path / "ledger.jsonl")
    h = reg.preregister(
        family="us-equity-v1",
        name="t",
        hypothesis="h",
        config={"p": 1},
        gates={},
        windows={"dev": ["a", "b"], "holdout": ["c", "d"]},
        now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        code="c",
    )
    metrics = {
        "holdout": {"sharpe": 1.0},
        "dev": {},
        "factor": {"alpha_annual": 0.01},
        "summary": {},
    }
    reg.record_result(
        h,
        metrics=metrics,
        verdict={"promoted": False, "failed": ["dsr"]},
        data_digest="d",
        now=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
    )
    return TestClient(create_app(tmp_path / "ledger.jsonl"))


def test_web_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json()["ledger_ok"] is True
    r = client.get("/api/results/us-equity-v1").json()
    assert r["trials"] == 1 and r["results"][0]["verdict"]["failed"] == ["dsr"]
    assert "pusula" in client.get("/").text
    assert client.get("/api/forward/nope").status_code == 404
    # read-only: no mutating verbs
    assert client.post("/api/results/us-equity-v1").status_code == 405


def test_dashboard_has_no_state_changing_routes() -> None:
    app = create_app(Path("ledger.jsonl"))
    methods = {m for r in app.routes for m in getattr(r, "methods", set())}
    assert methods <= {"GET", "HEAD"}
    assert np.isfinite(1.0)


def test_daily_report_rendering_and_stale_iv_warning() -> None:
    from pusula.live.report import render_daily

    book = {
        "equity": 10_100_000.0,
        "equity_curve": [["2026-10-21", 1e7], ["2026-10-22", 1.01e7]],
        "gross": 2.0,
        "net": 0.01,
        "halted": False,
        "kill_reason": "",
        "risk": {"ex_ante_vol": 0.1, "var99": 0.012},
        "stress": [{"scenario": "2020 COVID crash", "pnl": -0.03}, {"scenario": "x", "pnl": 0.01}],
    }
    snap = {
        "family": "f",
        "as_of": "2026-10-22",
        "books": {"b": book},
        "data": {"quality": "ok", "digest": "abcdef0123456", "iv_last": "2026-10-21"},
    }
    md = render_daily(snap)
    assert (
        "| b | 10,100,000 | 1.00% | 1.00% | 2.00 | +0.01 | 10.0% | 1.20% | "
        "2020 COVID crash -3.0% | running |" in md
    )
    assert "WARNING" not in md and "REHEARSAL" not in md
    stale = render_daily(
        {**snap, "rehearsal": True, "data": {**snap["data"], "iv_last": "2026-08-26"}}
    )
    assert "implied vols last observed 2026-08-26 (57 days old)" in stale
    assert "REHEARSAL" in stale
    # a stopped FINRA feed, and a book trading without one of its signals
    missing = {**book, "signal_coverage": {"momentum": 0.998, "iv_rv_spread": 0.12}}
    md = render_daily(
        {
            **snap,
            "books": {"b": missing},
            "data": {**snap["data"], "short_last": "2026-10-16"},
        }
    )
    assert "short volume last observed 2026-10-16 (6 days old)" in md
    assert "`b`: signal `iv_rv_spread` scores 12% of index members" in md
    assert "| b | momentum 99.8%, iv_rv_spread 12.0% |" in md
    assert "signal `momentum`" not in md


def test_signal_coverage_counts_index_members_on_the_last_session() -> None:
    from pusula.alpha.combine import SignalSpec
    from pusula.core.calendar import sessions
    from pusula.data.panel import Panel
    from pusula.live.forward import signal_coverage

    dates = sessions(dt.date(2026, 10, 1), dt.date(2026, 10, 9))
    planted = np.ones((dates.size, 4))
    planted[-1, :2] = np.nan  # two of the members have no score on the last session
    member = np.ones((dates.size, 4))
    member[-1, 3] = 0.0  # a non-member does not count either way
    p = Panel(dates, ("A@x", "B@x", "C@x", "D@x"), {"planted": planted, "in_universe": member})
    trial = SimpleNamespace(name="t", signals=(SignalSpec("planted"),), ml=None)
    bench = SimpleNamespace(name="ew", signals=(), ml=None)
    cov = signal_coverage(SimpleNamespace(trials=(trial, bench)), p)  # type: ignore[arg-type]
    assert cov == {"t": {"planted": pytest.approx(1 / 3)}}  # C of the members A, B, C
