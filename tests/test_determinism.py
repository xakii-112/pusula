"""Same inputs, same bits; and the forward engine resumes exactly where it stopped."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pusula.alpha import ops
from pusula.alpha.signals import Planted
from pusula.core.hashing import array_digest
from pusula.data.panel import FloatArray, Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import OptimizedAlpha


def _fn(p: Panel) -> FloatArray:
    m = p["in_universe"]
    return ops.cs_zscore(ops.cs_gauss(Planted().compute(p), m), m)


def _strat() -> OptimizedAlpha:
    return OptimizedAlpha(_fn, OptimizerSpec(max_weight=0.05), ic=0.03, every=5)


def test_synthetic_generator_is_deterministic() -> None:
    a = generate(SyntheticSpec(n_symbols=30, seed=3))
    b = generate(SyntheticSpec(n_symbols=30, seed=3))
    assert array_digest(*a.fields.values()) == array_digest(*b.fields.values())


def test_backtest_is_bit_reproducible() -> None:
    p = generate(SyntheticSpec(n_symbols=40, start="2016-01-04", end="2017-06-30", ic=0.03, seed=4))
    cfg = BacktestConfig(start="2017-01-03")
    r1 = Backtester(p, cfg).run(_strat())
    r2 = Backtester(p, cfg).run(_strat())
    assert array_digest(r1.equity, r1.weights) == array_digest(r2.equity, r2.weights)


def test_forward_engine_resume_equals_continuous_run(tmp_path: Path) -> None:
    p = generate(SyntheticSpec(n_symbols=40, start="2016-01-04", end="2017-06-30", ic=0.03, seed=5))
    cfg = BacktestConfig(kill_switch=False)
    t0, mid, t1 = p.date_index("2017-01-03"), p.date_index("2017-03-31"), p.n_dates - 1

    cont = PaperEngine(Backtester(p, cfg), _strat(), "book", journal=Journal(tmp_path / "a.jsonl"))
    for e in range(t0, t1 + 1):
        cont.step(e)

    first = PaperEngine(Backtester(p, cfg), _strat(), "book", journal=Journal(tmp_path / "b.jsonl"))
    for e in range(t0, mid + 1):
        first.step(e)
    state = json.loads(json.dumps(first.control_state()))  # through a file, as in production
    resumed = PaperEngine(
        Backtester(p, cfg), _strat(), "book", journal=Journal(tmp_path / "b.jsonl")
    )
    resumed.restore(state)
    for e in range(mid + 1, t1 + 1):
        resumed.step(e)

    a = [(e.kind, e.data) for e in Journal(tmp_path / "a.jsonl")]
    b = [(e.kind, e.data) for e in Journal(tmp_path / "b.jsonl")]
    assert a == b
    assert (
        Journal(tmp_path / "a.jsonl").events[-1].hash
        == Journal(tmp_path / "b.jsonl").events[-1].hash
    )
    assert np.isclose(cont.equity_curve[t1], resumed.equity_curve[t1])


def test_restore_refuses_a_tampered_journal(tmp_path: Path) -> None:
    p = generate(SyntheticSpec(n_symbols=20, start="2016-01-04", end="2017-02-28", seed=6))
    cfg = BacktestConfig(kill_switch=False)
    eng = PaperEngine(Backtester(p, cfg), _strat(), "book", journal=Journal(tmp_path / "j.jsonl"))
    t0 = p.date_index("2017-01-03")
    for e in range(t0, t0 + 10):
        eng.step(e)
    state = eng.control_state()
    state["journal_head"] = "0" * 64
    other = PaperEngine(Backtester(p, cfg), _strat(), "book", journal=Journal(tmp_path / "j.jsonl"))
    with pytest.raises(ValueError, match="refusing"):
        other.restore(state)
