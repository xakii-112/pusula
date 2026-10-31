"""The cross-asset family's plumbing, the forward signal record and the robustness tools."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula.alpha.combine import SignalSpec
from pusula.alpha.signals import ConstantForecast
from pusula.core.errors import DataError
from pusula.data import pipeline as P
from pusula.data.panel import Panel
from pusula.data.store import BarStore
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig
from pusula.live.signal_record import render_signal_record, signal_record
from pusula.portfolio.strategies import ForecastSpec
from pusula.research import robustness as R
from pusula.research.families import us_crossasset_v1
from pusula.research.study import StudySpec, Trial, registered_config


def _panel(seed: int = 3) -> Panel:
    return generate(SyntheticSpec(n_symbols=30, end="2017-06-30", seed=seed))


def _bars(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1e6,
            "trades": 100.0,
            "vwap": 10.2,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------- cross-asset family
def test_universe_is_part_of_the_registered_config() -> None:
    spec = us_crossasset_v1()
    cfg = registered_config(spec, spec.trials[0])
    assert cfg["universe"] == list(P.CROSS_ASSET)
    assert "universe" not in registered_config(replace(spec, universe=()), spec.trials[0])
    assert len(set(P.CROSS_ASSET)) == 18


def test_passive_book_is_a_benchmark_and_is_not_counted() -> None:
    spec = us_crossasset_v1()
    assert {t.name for t in spec.trials if t.benchmark} == {"equal_weight", "risk_parity_long"}
    assert sum(not t.benchmark for t in spec.trials) == 3


def test_constant_forecast_exists_only_where_a_price_does() -> None:
    p = _panel()
    f = ConstantForecast().compute(p)
    assert np.all(f[np.isfinite(p["close"])] == 10.0)
    assert np.all(np.isnan(f[~np.isfinite(p["close"])]))


def test_etf_data_is_a_member_from_its_first_bar(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    now = dt.datetime(2026, 10, 30, tzinfo=dt.UTC)
    late = _bars(30).iloc[10:]
    store.put("AAA", "AAA", None, _bars(30), adjustment="all", feed="sip", fetched_at=now)
    store.put("BBB", "BBB", None, late, adjustment="all", feed="sip", fetched_at=now)
    store.flush()
    rd = P.etf_data(("AAA", "BBB"), tmp_path)
    j = rd.panel.symbol_index("BBB")
    assert rd.panel["in_universe"][:10, j].sum() == 0
    assert np.all(rd.panel["in_universe"][10:, j] == 1)
    assert rd.sectors == {"AAA": "ETF", "BBB": "ETF"}


def test_etf_data_refuses_a_missing_fund(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cross-asset"):
        P.etf_data(("NOPE",), tmp_path)


def test_fetch_etfs_asks_only_for_what_is_missing(tmp_path: Path) -> None:
    class Client:
        def __init__(self) -> None:
            self.asked: list[tuple[str, ...]] = []

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def daily_bars(self, req: P.BarRequest) -> dict[str, pd.DataFrame]:
            self.asked.append(tuple(req.symbols))
            return {s: _bars() for s in req.symbols}

    client = Client()
    end = dt.date(2026, 10, 29)
    assert P.fetch_etfs(("AAA", "BBB"), tmp_path, end=end, client=client) == ["AAA", "BBB"]  # type: ignore[arg-type]
    assert P.fetch_etfs(("AAA", "BBB", "CCC"), tmp_path, end=end, client=client) == ["CCC"]  # type: ignore[arg-type]
    assert client.asked == [("AAA", "BBB"), ("CCC",)]


# ---------------------------------------------------------------------------- signal record
def _spec(panel: Panel, *trials: Trial) -> StudySpec:
    d = panel.dates
    return StudySpec(
        family="f",
        trials=trials,
        dev=(str(d[100]), str(d[300])),
        holdout=(str(d[301]), str(d[500])),
        backtest=BacktestConfig(),
    )


def _momentum(horizon: float = 5.0) -> Trial:
    return Trial("m", "optimized", "h", signals=(SignalSpec("momentum"),), horizon_days=horizon)


def test_signal_record_counts_only_matured_labels() -> None:
    p = _panel()
    first = 501
    row = signal_record(_spec(p, _momentum(5.0)), p, first)[0]
    # labels of the last lag + horizon rows have not matured
    assert row["forward_n"] == p.n_dates - first - 6
    assert row["horizon"] == 5


def test_signal_record_does_not_see_the_future_of_a_row() -> None:
    p = _panel()
    spec = _spec(p, _momentum())
    cut = p.rows(0, 560)
    a = signal_record(spec, p, 501)[0]
    b = signal_record(spec, cut, 501)[0]
    assert a["holdout_ic"] == pytest.approx(b["holdout_ic"])
    assert b["forward_n"] < a["forward_n"]


def test_signal_record_lists_each_signal_once_and_skips_benchmarks() -> None:
    p = _panel()
    bench = Trial("eq", "equal_weight", "h", benchmark=True)
    rows = signal_record(_spec(p, _momentum(), replace(_momentum(), name="n"), bench), p, 501)
    assert [r["signal"] for r in rows] == ["momentum"]


def test_signal_record_says_when_the_data_is_not_there() -> None:
    p = _panel()
    t = Trial("iv", "optimized", "h", signals=(SignalSpec("iv_rv_spread"),))
    rows = signal_record(_spec(p, t), p, 501)
    assert "unavailable" in rows[0]
    assert "no data" in "\n".join(render_signal_record(rows))


def test_no_positive_holdout_edge_has_nothing_to_confirm() -> None:
    p = _panel()
    row = signal_record(_spec(p, _momentum()), p, 501)[0]
    needed, ir_positive = row["sessions_needed"], row["holdout_ic"] > 0
    assert math.isinf(needed) != ir_positive


# ---------------------------------------------------------------------------- robustness
def test_signal_neighbours_keep_other_parameters() -> None:
    t = Trial("f", "optimized", "h", signals=(SignalSpec("ear", params={"hold": 63}),))
    pts = R.signal_neighbours(t, {"window": (1, 2), "skip": (3,)})
    assert [label for label, _ in pts] == ["window=1, skip=3", "window=2, skip=3"]
    assert dict(pts[0][1].signals[0].params) == {"hold": 63, "window": 1, "skip": 3}
    with pytest.raises(ValueError, match="single-signal"):
        R.signal_neighbours(replace(t, signals=t.signals * 2), {"window": (1,)})


def test_forecast_neighbours_change_only_the_engine() -> None:
    t = Trial(
        "f", "forecast", "h", signals=(SignalSpec("trend_forecast"),), forecast=ForecastSpec()
    )
    pts = dict(R.forecast_neighbours(t, {"buffer": (0.0, 0.25)}))
    assert pts["buffer=0.25"].forecast == replace(ForecastSpec(), buffer=0.25)
    assert t.forecast == ForecastSpec()
    with pytest.raises(ValueError, match="forecast trial"):
        R.forecast_neighbours(replace(t, forecast=None), {"buffer": (0.0,)})


def test_price_noise_is_seeded_and_keeps_a_bar_a_bar() -> None:
    p = _panel()
    a, b = R.with_price_noise(p, 1e-9, 7), R.with_price_noise(p, 1e-9, 7)
    assert np.array_equal(a["close"], b["close"], equal_nan=True)
    assert not np.array_equal(a["close"], p["close"], equal_nan=True)
    ok = np.isfinite(p["close"])
    assert np.allclose(a["close"][ok], p["close"][ok], rtol=1e-7)
    assert np.all(a["high"][ok] >= a["low"][ok])
    assert np.array_equal(a["volume"], p["volume"], equal_nan=True)


def test_a_lone_peak_is_a_spike_and_a_broad_top_is_a_plateau() -> None:
    spike = R.summarise_grid({"a": 1.2, "b": 0.3, "c": 0.2, "d": 0.4}, "a")
    assert (spike.verdict, spike.base_rank, spike.share_near) == ("spike", 1, 0.0)
    flat = R.summarise_grid({"a": 0.9, "b": 0.85, "c": 0.95, "d": 0.8}, "a")
    assert (flat.verdict, flat.base_rank, flat.share_near) == ("plateau", 2, 1.0)
    with pytest.raises(ValueError, match="other than"):
        R.summarise_grid({"a": 1.0}, "a")


def test_few_neighbours_near_the_registered_value_is_not_a_plateau() -> None:
    """xs_momentum: 0.94 registered, 0.68 at the median neighbour, 3 of 11 within 0.20."""
    holdout = dict(
        zip(
            "abcdefghijkl",
            (0.94, 0.95, 0.79, 0.78, 0.69, 0.69, 0.68, 0.63, 0.61, 0.58, 0.50, 0.39),
            strict=True,
        )
    )
    s = R.summarise_grid(holdout, "a")
    assert s.verdict == "narrow"
    assert s.share_near == pytest.approx(3 / 11)


def test_a_better_neighbour_is_reported_not_chosen() -> None:
    s = R.summarise_grid({"a": 0.5, "b": 1.5}, "a")
    assert s.base == 0.5 and s.best == 1.5
    text = "\n".join(
        R.render_grid(
            "t", "a", {"a": {"dev": 0.1, "holdout": 0.5}, "b": {"dev": 0.2, "holdout": 1.5}}, s
        )
    )
    assert "| a (registered) |" in text and "| b |" in text


def test_noise_floor() -> None:
    assert R.noise_floor([1.0, 1.0, 1.0]) == 0.0
    assert math.isnan(R.noise_floor([1.0]))
