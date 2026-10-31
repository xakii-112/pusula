"""FINRA short-sale volume: file parsing, the incremental store and its hash chain, the
one-session publication lag, and the short-pressure signal."""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pusula.core.calendar import sessions
from pusula.core.errors import DataError
from pusula.data import shortvol as SV
from pusula.data.panel import Panel


def _file(day: dt.date, rows: list[tuple[str, int, int]]) -> bytes:
    lines = ["Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"]
    lines += [f"{day:%Y%m%d}|{s}|{sv}|0|{tv}|B,Q,N" for s, sv, tv in rows]
    return ("\n".join(lines) + "\n").encode()


def test_parse_day_maps_share_classes() -> None:
    body = _file(dt.date(2024, 1, 2), [("AAA", 40, 100), ("BF/B", 7, 10)])
    body += b"2\n"  # some older files end with a record count
    df = SV.parse_day(body)
    assert df["symbol"].tolist() == ["AAA", "BF.B"]
    assert df["short"].tolist() == [40, 7] and df["total"].tolist() == [100, 10]
    assert (df["date"] == pd.Timestamp("2024-01-02")).all()


def _fetch(missing: set[dt.date]) -> tuple[dict[dt.date, bytes], object]:
    served: dict[dt.date, bytes] = {}

    def get(day: dt.date) -> bytes | None:
        if day in missing:
            return None
        served[day] = _file(day, [("AAA", day.day, 100), ("BBB", 2 * day.day, 100)])
        return served[day]

    return served, get


def test_store_appends_and_chains_per_file(tmp_path: Path) -> None:
    days = [d.item() for d in sessions(SV.FIRST, dt.date(2018, 8, 20))]
    gap, tail = days[3], days[-1]
    served, get = _fetch({gap, tail})
    one, two = tmp_path / "one", tmp_path / "two"
    # built in one run vs in two runs: same rows, same chain
    assert SV.update_shortvol(one, days[-1], fetch=get, log=False) == len(days) - 2  # type: ignore[arg-type]
    SV.update_shortvol(two, days[6], fetch=get, log=False)  # type: ignore[arg-type]
    SV.update_shortvol(two, days[-1], fetch=get, log=False)  # type: ignore[arg-type]
    m1 = json.loads((one / "short_volume.json").read_text(encoding="utf-8"))
    m2 = json.loads((two / "short_volume.json").read_text(encoding="utf-8"))
    chain = b""
    for d in days:
        if d in served:
            chain = hashlib.sha256(chain + hashlib.sha256(served[d]).digest()).digest()
    assert m1["chain_sha256"] == m2["chain_sha256"] == chain.hex()
    # a gap followed by a later file is missing; the newest session may just be unpublished
    assert m1["missing"] == m2["missing"] == [gap.isoformat()]
    assert m1["last"] == days[-2].isoformat() and m1["symbols"] == 2
    pd.testing.assert_frame_equal(
        pd.read_parquet(one / "short_volume.parquet"), pd.read_parquet(two / "short_volume.parquet")
    )
    # the next run asks for the tail session again and finds it
    served2, get2 = _fetch(set())
    assert SV.update_shortvol(one, days[-1], fetch=get2, log=False) == 1  # type: ignore[arg-type]
    assert list(served2) == [tail]
    assert SV.update_shortvol(one, days[-1], fetch=get2, log=False) == 0  # type: ignore[arg-type]


def test_store_refuses_duplicate_rows(tmp_path: Path) -> None:
    day = dt.date(2018, 8, 1)

    def get(d: dt.date) -> bytes:
        return _file(day, [("AAA", 1, 2)])  # every file claims the same date

    with pytest.raises(DataError, match="duplicate"):
        SV.update_shortvol(tmp_path, dt.date(2018, 8, 3), fetch=get, log=False)


def test_fields_are_usable_one_session_after_the_file(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-31"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "short": [40, 50, 60],
            "total": [100, 100, 100],
        }
    ).to_parquet(tmp_path / "short_volume.parquet", index=False)
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 1, 31))
    p = Panel(dates, ("AAA@2016-01-01", "BBB@2016-01-01"), {"close": np.ones((dates.size, 2))})
    f = SV.shortvol_fields(p, tmp_path)
    assert np.isnan(f["short_volume"][0, 0])  # 2 Jan's file is published after 2 Jan's close
    assert f["short_volume"][1, 0] == 40 and f["short_volume"][2, 0] == 50
    assert np.isnan(f["short_volume"][3:, 0]).all()  # 31 Jan's file has no next session here
    assert np.isnan(f["finra_volume"][:, 1]).all() and f["finra_volume"][1, 0] == 100


def test_fields_follow_a_renamed_member_and_ignore_the_tickers_earlier_owner(
    tmp_path: Path,
) -> None:
    """NEW@2016-01-01 traded as OLD until 2024-01-10; before that, NEW was another company's
    ticker. ADD@2024-01-08 listed on 2024-01-08; its ticker had an earlier owner too."""
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 1, 16))
    day = [str(d) for d in dates]
    rows = [(d, "OLD", 1) for d in day[:6]] + [(d, "NEW", 2) for d in day]
    rows += [(d, "ADD", 3) for d in day]
    pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "short": [r[2] for r in rows],
            "total": [10] * len(rows),
        }
    ).to_parquet(tmp_path / "short_volume.parquet", index=False)
    close = np.ones((dates.size, 2))
    close[: day.index("2024-01-08"), 1] = np.nan  # ADD has no price before it listed
    p = Panel(dates, ("NEW@2016-01-01", "ADD@2024-01-08"), {"close": close})
    earlier = {"NEW@2016-01-01": (("OLD", "2024-01-10"),)}
    seg = SV.ticker_segments(p.symbols, earlier)
    assert seg["symbol"].tolist() == ["OLD", "NEW", "ADD"]
    f = SV.shortvol_fields(p, tmp_path, earlier)["short_volume"]
    # row t holds session t-1's file: OLD's rows through 9 Jan, NEW's from 10 Jan
    assert f[1 : day.index("2024-01-10") + 1, 0].tolist() == [1.0] * 6  # 2, 3, 4, 5, 8, 9 Jan
    assert (f[day.index("2024-01-10") + 1 :, 0] == 2.0).all()
    i = day.index("2024-01-08")
    assert np.isnan(f[: i + 1, 1]).all() and (f[i + 1 :, 1] == 3.0).all()


def test_earlier_tickers_are_well_formed() -> None:
    from pusula.data.ticker_history import EARLIER_TICKERS

    for sid, segs in EARLIER_TICKERS.items():
        tick, start = sid.split("@")
        until = [dt.date.fromisoformat(u) for _, u in segs]
        assert until == sorted(until) and len(set(until)) == len(until), sid
        assert until[0] > dt.date.fromisoformat(start) and until[-1] > SV.FIRST, sid
        # the id's own ticker is the last one; an earlier one differs from its successor
        names = [t for t, _ in segs] + [tick]
        assert all(a != b for a, b in itertools.pairwise(names)), sid
        assert all(t == t.strip().upper() and t for t in names), sid


def test_short_pressure_is_causal_and_signed(noise_panel: Panel) -> None:
    from pusula.alpha.lookahead import check_causal, default_cutoffs
    from pusula.alpha.signals import SHORT_SIGNALS, build

    rng = np.random.default_rng(0)
    shape = noise_panel["close"].shape
    total = rng.uniform(1e5, 1e6, shape)
    short = total * rng.uniform(0.3, 0.6, shape)
    short[rng.random(shape) < 0.05] = np.nan
    panel = noise_panel.with_fields(short_volume=short, finra_volume=total)
    for name in SHORT_SIGNALS:
        check_causal(build(name).compute, panel, default_cutoffs(panel.n_dates), name=name)
    # a stock whose short share jumps over the last week scores below its own history
    jump = short.copy()
    jump[-5:, 0] = total[-5:, 0] * 0.9
    score = build("short_pressure").compute(
        noise_panel.with_fields(short_volume=jump, finra_volume=total)
    )
    assert score[-1, 0] < -0.2 and np.isfinite(score[-1, 1:]).mean() > 0.9
    # the level: minus the week's short share, a ratio of sums over the sessions with a file
    flat = total * 0.4
    flat[-5:, 0] = total[-5:, 0] * np.array([0.9, 0.9, 0.9, 0.9, 0.4])
    flat[-2, 1] = np.nan  # one missing file: the other four sessions still count
    level = build("short_share").compute(
        noise_panel.with_fields(short_volume=flat, finra_volume=total)
    )
    t = total[-5:, 0]
    assert np.isclose(level[-1, 0], -(0.9 * t[:4].sum() + 0.4 * t[4]) / t.sum())
    assert np.isclose(level[-1, 1], -0.4) and np.isclose(level[-1, 2], -0.4)


@pytest.mark.parametrize("make", ["us_shortsale_v1", "us_shortsale_v2"])
def test_short_families_are_wired_into_the_nightly_run(make: str) -> None:
    from pusula import cli
    from pusula.research import families

    spec = getattr(families, make)()
    assert cli.needs_short_volume(spec) and not cli.needs_fundamentals(spec)
    assert cli.program_of(spec.family) == "us-" and spec.family in families.FORWARD_FAMILIES
