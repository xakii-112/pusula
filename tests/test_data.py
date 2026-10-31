"""Data layer: universe reconstruction, vendor payload parsing, quality rules, stores, IV."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula.data import universe as U
from pusula.data.alpaca import bars_to_frame
from pusula.data.factors import parse_french_csv
from pusula.data.iv import iv_panel
from pusula.data.panel import Panel
from pusula.data.quality import check_and_repair
from pusula.data.store import BarStore

CURRENT_HTML = """<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
{rows}</table>"""
CHANGES_HTML = (
    "<table><tr><th rowspan=2>Effective Date</th><th colspan=2>Added</th>"
    "<th colspan=2>Removed</th><th rowspan=2>Reason</th></tr>"
    "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
    "<tr><td>March 1, 2024</td><td>NEWCO</td><td>x</td><td>OLDCO</td><td>x</td><td>cap</td></tr>"
    "<tr><td>June 5, 2020</td><td>MID</td><td>x</td><td>GONE</td><td>x</td><td>acquired</td></tr>"
    "<tr><td>January 2, 2019</td><td>GONE</td><td>x</td><td>EARLY</td><td>x</td><td>cap</td></tr>"
    "</table>"
)


@pytest.fixture
def history() -> U.IndexHistory:
    base = [f"T{i:03d}" for i in range(470)]
    rows = "".join(f"<tr><td>{t}</td><td>x</td><td>Tech</td></tr>" for t in [*base, "NEWCO", "MID"])
    return U.parse_tables(CURRENT_HTML.format(rows=rows), CHANGES_HTML, dt.date(2026, 10, 23))


def test_membership_rolls_back_changes(history: U.IndexHistory) -> None:
    on = history.members_on
    assert "NEWCO" in on(dt.date(2024, 3, 1)) and "NEWCO" not in on(dt.date(2024, 2, 29))
    assert "OLDCO" in on(dt.date(2024, 2, 29)) and "OLDCO" not in on(dt.date(2024, 3, 1))
    assert "GONE" in on(dt.date(2020, 6, 4)) and "GONE" not in on(dt.date(2020, 6, 5))
    assert "GONE" not in on(dt.date(2018, 12, 31))  # added 2019-01-02
    assert "EARLY" in on(dt.date(2018, 12, 31))
    assert len(on(dt.date(2021, 1, 4))) == len(on(dt.date(2026, 1, 2)))


def test_spells_have_stable_ids_and_asof(history: U.IndexHistory) -> None:
    sp = {s.security_id: s for s in history.spells(dt.date(2016, 1, 1))}
    gone = sp["GONE@2019-01-02"]
    assert gone.end == dt.date(2020, 6, 5)
    assert gone.asof == dt.date(2020, 6, 4)  # prices requested as the entity of its last day
    assert sp["NEWCO@2024-03-01"].end is None and sp["NEWCO@2024-03-01"].asof is None
    assert "T000@2016-01-01" in sp and "OLDCO@2016-01-01" in sp
    dates = np.array(["2020-06-04", "2020-06-05"], dtype="datetime64[D]")
    m = U.membership_mask([gone], dates, ["GONE@2019-01-02"])
    np.testing.assert_array_equal(m[:, 0], [1.0, 0.0])


def test_ticker_normalisation() -> None:
    assert U.normalise_ticker("BRK-B") == "BRK.B"
    assert U.normalise_ticker("AAPL[1]") == "AAPL"
    assert U.normalise_ticker(float("nan")) is None
    assert U.normalise_ticker("not a ticker") is None


def test_bars_payload_to_frame() -> None:
    bars = [
        {
            "t": "2024-03-08T05:00:00Z",
            "o": 10,
            "h": 11,
            "l": 9,
            "c": 10.5,
            "v": 1000,
            "vw": 10.2,
            "n": 50,
        },
        {
            "t": "2024-03-11T04:00:00Z",
            "o": 10.5,
            "h": 12,
            "l": 10,
            "c": 11,
            "v": 900,
            "vw": 11.1,
            "n": 40,
        },
    ]
    df = bars_to_frame(bars)
    # midnight New York (EST then EDT after the DST change) maps to the session date
    assert list(df.index.date) == [dt.date(2024, 3, 8), dt.date(2024, 3, 11)]
    assert df["vwap"].iloc[1] == 11.1


def test_french_parser() -> None:
    text = (
        "header\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20240102,  -0.71,  0.12, 0.30, -0.10, 0.05, 0.02\n"
        "20240103,  0.50, -0.20, 0.10, 0.00, 0.01, 0.02\n\n Annual Factors\n,Mkt-RF\n2023, 21.0\n"
    )
    df = parse_french_csv(text)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert len(df) == 2 and df["mkt_rf"].iloc[0] == pytest.approx(-0.0071)


def _bar_panel(**over: np.ndarray) -> Panel:
    dates = np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    f = {
        "open": np.array([[10.0], [10.0], [10.0]]),
        "high": np.array([[11.0], [9.0], [11.0]]),  # row 1: high below open -> invalid
        "low": np.array([[9.0], [9.0], [9.0]]),
        "close": np.array([[10.5], [9.5], [10.5]]),
        "volume": np.array([[100.0], [100.0], [100.0]]),
        "vwap": np.array([[0.0], [9.2], [10.4]]),  # row 0: zero vwap only
        "in_universe": np.ones((3, 1)),
    }
    f.update(over)
    return Panel(dates, ("A",), f)


def test_quality_repairs_are_pointwise() -> None:
    cleaned, rep = check_and_repair(_bar_panel())
    assert rep.repaired_bars == 1 and rep.invalid_ohlc == 1 and rep.invalid_vwap == 1
    assert np.isnan(cleaned["close"][1, 0]) and np.isnan(cleaned["vwap"][1, 0])
    assert cleaned["close"][0, 0] == 10.5 and np.isnan(cleaned["vwap"][0, 0])
    assert cleaned["in_universe"][1, 0] == 1.0  # non-bar fields untouched


def test_store_round_trip(tmp_path: Path) -> None:
    st = BarStore(tmp_path)
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date")
    df = pd.DataFrame(
        {c: [1.0, 2.0] for c in ("open", "high", "low", "close", "volume", "vwap", "trades")},
        index=idx,
    )
    st.put(
        "A@2016-01-01",
        "A",
        None,
        df,
        adjustment="all",
        feed="sip",
        fetched_at=dt.datetime(2026, 1, 1),
    )
    st.flush()
    st2 = BarStore(tmp_path)
    pd.testing.assert_frame_equal(st2.get("A@2016-01-01"), df, check_freq=False)
    d1 = st2.manifest_digest()
    st2.put(
        "A@2016-01-01",
        "A",
        None,
        df * 2,
        adjustment="all",
        feed="sip",
        fetched_at=dt.datetime(2026, 1, 1),
    )
    assert st2.manifest_digest() != d1


def test_iv_alignment_is_point_in_time(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-20"]),
            "symbol": ["A", "A", "A"],
            "iv": [0.2, 0.3, 0.4],
        }
    ).to_parquet(tmp_path / "iv_history.parquet", index=False)
    dates = np.array(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-12", "2024-01-16", "2024-01-19"],
        dtype="datetime64[D]",
    )
    p = Panel(dates, ("A@2016-01-01", "B@2016-01-01"), {"close": np.ones((6, 2))})
    iv = iv_panel(p, tmp_path, max_stale_days=7)
    np.testing.assert_allclose(iv[:4, 0], [0.2, 0.2, 0.3, 0.3])  # carried at most 7 days
    assert np.isnan(iv[4, 0]) and np.isnan(iv[5, 0])  # 01-16: 11 days stale; 01-20 not yet known
    assert np.isnan(iv[:, 1]).all()
