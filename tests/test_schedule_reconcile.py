from __future__ import annotations

import numpy as np
import pytest

from pusula.engine.ledger import Ledger
from pusula.engine.reconcile import reconcile
from pusula.execution.schedule import US_VOLUME_PROFILE, almgren_chriss, volume_profile_slices


def test_almgren_chriss_limits() -> None:
    x = almgren_chriss(1000.0, 10, sigma=0.5, eta=0.01, risk_aversion=0.0)
    np.testing.assert_allclose(x, np.linspace(1000, 0, 11))  # risk neutral: TWAP
    y = almgren_chriss(1000.0, 10, sigma=0.5, eta=0.01, risk_aversion=1e-3)
    assert y[0] == pytest.approx(1000.0) and y[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(y) < 0)
    assert y[5] < x[5]  # risk aversion front-loads the selling
    z = almgren_chriss(1000.0, 10, sigma=0.5, eta=0.01, risk_aversion=1e-1)
    assert z[1] < y[1]


def test_volume_profile_slices_respect_participation() -> None:
    small = volume_profile_slices(10_000, adv_shares=10_000_000, max_participation=0.1)
    np.testing.assert_allclose(small.slices, 10_000 * US_VOLUME_PROFILE)
    assert small.unfilled == 0
    big = volume_profile_slices(-2_000_000, adv_shares=10_000_000, max_participation=0.1)
    assert np.all(big.slices <= 0)
    assert np.all(np.abs(big.slices) <= 0.1 * 10_000_000 * US_VOLUME_PROFILE + 1e-6)
    assert abs(big.slices.sum()) + abs(big.unfilled) == pytest.approx(2_000_000)
    assert big.unfilled == pytest.approx(-1_000_000)


def test_reconcile_breaks() -> None:
    led = Ledger()
    led.contribute(10**12)
    led.fill("AAPL@2016-01-01", 100, 150_000_000, 0)
    led.fill("MSFT@2016-01-01", -20, 300_000_000, 0)
    ok = reconcile(led, {"AAPL": 100, "MSFT": -20})
    assert ok.clean
    bad = reconcile(led, {"AAPL": 90, "TSLA": 5})
    assert not bad.clean
    assert {(b.security, b.diff) for b in bad.breaks} == {("AAPL", -10), ("MSFT", 20), ("TSLA", 5)}
    assert bad.as_event()["clean"] is False


def test_paper_engine_halts_on_reconciliation_break(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pusula.data.synthetic import SyntheticSpec, generate
    from pusula.engine.backtest import BacktestConfig, Backtester
    from pusula.engine.journal import Journal
    from pusula.engine.paper import PaperEngine
    from pusula.portfolio.strategies import EqualWeight

    p = generate(SyntheticSpec(n_symbols=10, start="2016-01-04", end="2016-03-31", seed=2))
    eng = PaperEngine(
        Backtester(p, BacktestConfig(kill_switch=False)),
        EqualWeight(every=5),
        "b",
        journal=Journal(tmp_path / "j.jsonl"),
    )
    for e in range(0, 15):
        eng.step(e)
    held = {s.split("@")[0]: pos.qty for s, pos in eng.ledger.positions.items()}
    assert eng.reconcile_with(held, "2016-01-26")
    held[next(iter(held))] += 1
    assert not eng.reconcile_with(held, "2016-01-26")
    assert eng.halted
    kinds = [e.kind for e in Journal(tmp_path / "j.jsonl")]
    assert kinds.count("reconcile") == 2 and kinds[-1] == "risk"
