"""The families outside FAMILIES: what runs is what was registered, and their small pieces."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula import cli
from pusula.core.errors import RegistryError
from pusula.data import pipeline as P
from pusula.data.panel import Panel
from pusula.data.store import BarStore
from pusula.research import mm_study, optquote_study
from pusula.research.registry import Registry

NOW = dt.datetime(2026, 10, 30, tzinfo=dt.UTC)


def test_market_making_family_is_registered_as_coded() -> None:
    hashes = mm_study.verify(Registry(cli.LEDGER))
    assert set(hashes) == {t.name for t in mm_study.TRIALS}


@pytest.mark.parametrize("study", [mm_study, optquote_study])
def test_an_edited_trial_is_refused(study: object, tmp_path: Path) -> None:
    reg = Registry(tmp_path / "ledger.jsonl")
    study.register(reg, NOW)  # type: ignore[attr-defined]
    assert study.verify(reg)  # type: ignore[attr-defined]
    trials = study.TRIALS  # type: ignore[attr-defined]
    edited = (
        replace(
            trials[0],
            spec=replace(trials[0].spec, max_names=1)
            if study is optquote_study
            else replace(trials[0].spec, notional=1.0),
        ),
        *trials[1:],
    )
    study.TRIALS = edited  # type: ignore[attr-defined]
    try:
        with pytest.raises(RegistryError, match="differs"):
            study.verify(reg)  # type: ignore[attr-defined]
    finally:
        study.TRIALS = trials  # type: ignore[attr-defined]


def test_an_unregistered_family_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not pre-registered"):
        optquote_study.verify(Registry(tmp_path / "ledger.jsonl"))


def _panel(n: int = 60) -> Panel:
    dates = np.array(pd.bdate_range("2026-01-05", periods=n), dtype="datetime64[D]")
    c = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, (n, 2)), axis=0) * 100.0
    return Panel(
        dates,
        ("AAA@2016-01-01", "BBB@2016-01-01"),
        {
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": np.ones_like(c),
            "in_universe": np.ones_like(c),
        },
    )


def test_realised_vol_does_not_read_ahead() -> None:
    p = _panel()
    a = optquote_study.realised_vol(p)
    b = optquote_study.realised_vol(p.rows(0, 40))
    assert np.allclose(a[:40], b, equal_nan=True)
    assert np.nanmean(a[25:]) == pytest.approx(0.01 * np.sqrt(252), rel=0.3)


def test_member_chain_keeps_member_symbols_and_the_expiry_range() -> None:
    d, near, far = (
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-02-05"),
        pd.Timestamp("2026-09-05"),
    )
    chain = pd.DataFrame(
        {
            "date": [d, d, d],
            "symbol": ["AAA", "ZZZ", "AAA"],
            "expiration": [near, near, far],
            "strike": [100.0, 100.0, 100.0],
        }
    )
    got = optquote_study.member_chain(chain, _panel())
    assert list(got) == [d]
    assert list(got[d]["symbol"]) == ["AAA"] and got[d]["dte"].iloc[0] == 31


def test_a_saturday_snapshot_belongs_to_the_friday_before_it() -> None:
    p = _panel(20)  # business days from Monday 2026-01-05
    friday, saturday = pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-10")
    monday = pd.Timestamp("2026-01-12")
    chain = pd.DataFrame({"date": [saturday, monday], "symbol": ["AAA", "BBB"]})
    got = optquote_study.by_session(chain, p)
    assert set(got) == {friday, monday}
    assert list(got[friday]["symbol"]) == ["AAA"]


def test_a_session_keeps_its_own_snapshot_over_one_carried_onto_it() -> None:
    p = _panel(20)
    friday, saturday = pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-10")
    chain = pd.DataFrame({"date": [friday, saturday], "symbol": ["OWN", "CARRIED"]})
    assert list(optquote_study.by_session(chain, p)[friday]["symbol"]) == ["OWN"]


def test_raw_closes_line_up_with_the_panel(tmp_path: Path) -> None:
    p = _panel(10)
    store = BarStore(tmp_path / "raw")
    index = pd.DatetimeIndex(p.dates.astype("datetime64[ns]"))[2:]
    frame = pd.DataFrame(
        {
            "open": 5.0,
            "high": 5.0,
            "low": 5.0,
            "close": np.arange(8.0) + 50,
            "volume": 1.0,
            "trades": 1.0,
            "vwap": 5.0,
        },
        index=index,
    )
    store.put("AAA@2016-01-01", "AAA", None, frame, adjustment="raw", feed="sip", fetched_at=NOW)
    store.flush()
    raw = P.raw_closes(p, tmp_path)
    assert np.isnan(raw[:2, 0]).all() and raw[2, 0] == 50.0 and raw[-1, 0] == 57.0
    assert np.isnan(raw[:, 1]).all()
