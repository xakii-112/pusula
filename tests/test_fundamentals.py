"""Point-in-time fundamentals: SEC client, distillation, as-known values, session mapping,
security master verification, the event signals and their alignment to the panel."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from pusula.alpha import ops
from pusula.alpha.lookahead import check_causal, default_cutoffs
from pusula.alpha.signals import FUNDAMENTAL_SIGNALS, build, market_return
from pusula.core.calendar import sessions
from pusula.data import fundamentals as F
from pusula.data import universe as U
from pusula.data.panel import Panel
from pusula.data.sec import USER_AGENT, SecClient, retry_after_seconds

T = pd.Timestamp


# ------------------------------------------------------------------------ SEC client
def test_retry_after_both_forms() -> None:
    now = dt.datetime(2026, 10, 24, 12, 0, tzinfo=dt.UTC)
    assert retry_after_seconds("7", now) == 7.0
    assert retry_after_seconds("Sat, 24 Oct 2026 12:01:30 GMT", now) == pytest.approx(90.0)
    assert retry_after_seconds("Sat, 24 Oct 2026 11:00:00 GMT", now) == 0.0
    assert retry_after_seconds("soon", now) is None and retry_after_seconds(None) is None


def test_client_throttles_retries_and_identifies_without_personal_data() -> None:
    seen: list[httpx.Request] = []
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        calls["n"] += 1
        if req.url.path.endswith("CIK0000000404.json"):
            return httpx.Response(404)
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"ok": True})

    slept: list[float] = []
    clock = {"t": 0.0}

    def sleep(s: float) -> None:
        slept.append(s)
        clock["t"] += s

    c = SecClient(httpx.MockTransport(handler), rate=5.0, sleep=sleep, clock=lambda: clock["t"])
    got = c.companyfacts(320193)
    assert got is not None and got[0] == {"ok": True} and len(got[1]) == 64
    assert 3.0 in slept  # Retry-After honoured
    assert c.companyfacts(404) is None  # no such entity is not an error
    ua = seen[0].headers["User-Agent"]
    assert ua == USER_AGENT and "@" not in ua
    # throttle: consecutive requests are at least 1/rate apart on the injected clock
    assert all(s > 0 for s in slept)


def test_submissions_merges_archive_pages() -> None:
    main = {
        "name": "X",
        "filings": {
            "recent": {"form": ["10-Q"], "accessionNumber": ["a2"], "filingDate": ["2024-05-01"]},
            "files": [{"name": "CIK0000000001-submissions-001.json"}],
        },
    }
    page = {"form": ["10-K"], "accessionNumber": ["a1"], "filingDate": ["2016-02-01"]}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page if "submissions-001" in req.url.path else main)

    sub, _ = SecClient(httpx.MockTransport(handler), sleep=lambda s: None).submissions(1)  # type: ignore[misc]
    assert sub["filings"]["recent"]["accessionNumber"] == ["a2", "a1"]


# ------------------------------------------------------------------------ distillation
def _payload(facts: list[tuple[str, str | None, str, float, str, str]]) -> dict[str, Any]:
    gaap: dict[str, Any] = {}
    for tag, start, end, val, accn, form in facts:
        row: dict[str, Any] = {
            "end": end,
            "val": val,
            "accn": accn,
            "form": form,
            "filed": "2020-01-01",
        }
        if start:
            row["start"] = start
        gaap.setdefault(tag, {"units": {"USD": []}})["units"]["USD"].append(row)
    return {"facts": {"us-gaap": gaap}}


def test_tag_chain_resolves_within_each_filing() -> None:
    p = _payload(
        [
            ("Revenues", "2018-01-01", "2018-12-31", 90.0, "k1", "10-K"),
            (
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "2018-01-01",
                "2018-12-31",
                100.0,
                "k1",
                "10-K",
            ),
            ("Revenues", "2017-01-01", "2017-12-31", 80.0, "k0", "10-K"),
            ("NetIncomeLoss", "2018-01-01", "2018-12-31", 5.0, "e1", "8-K"),  # not a report form
        ]
    )
    d = F.distill_facts(p)
    rev = d[d["concept"] == "revenue"].set_index("accn")["val"]
    assert rev["k1"] == 100.0  # earlier tag in the chain wins inside one filing
    assert rev["k0"] == 80.0  # a filing that only has the later tag still counts
    assert "net_income" not in set(d["concept"])


def test_quarters_derive_q4_from_year_minus_nine_months() -> None:
    latest = {
        (T("2023-01-01"), T("2023-03-31")): 10.0,
        (T("2023-04-01"), T("2023-06-30")): 11.0,
        (T("2023-07-01"), T("2023-09-30")): 12.0,
        (T("2023-01-01"), T("2023-09-30")): 33.0,  # nine-month YTD
        (T("2023-01-01"), T("2023-12-31")): 50.0,  # fiscal year
        (T("2023-01-01"), T("2023-06-30")): 21.0,  # six-month YTD: ignored
    }
    q = F.quarters(latest)
    assert q[T("2023-03-31")] == 10.0 and q[T("2023-09-30")] == 12.0
    assert q[T("2023-12-31")] == pytest.approx(17.0)


def test_surprise_matches_foster_definition() -> None:
    ends = pd.date_range("2018-03-31", periods=13, freq="QE")
    vals = np.array([10, 11, 12, 13, 12, 14, 13, 16, 15, 15, 17, 18, 25], dtype=float)
    q = dict(zip(ends, vals, strict=True))
    d = vals[4:] - vals[:-4]  # seasonal differences for quarters 4..12
    hist = d[:-1][-8:]
    want = (d[-1] - hist.mean()) / hist.std(ddof=1)
    assert F.surprise(q, ends[-1]) == pytest.approx(want)
    assert F.surprise(dict(list(q.items())[:9]), ends[8]) is None  # fewer than six differences


def _filings(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["accn", "form", "accepted", "report"])
    df["accepted"] = pd.to_datetime(df["accepted"], utc=True)
    df["report"] = pd.to_datetime(df["report"])
    df["items"] = ""
    df["filed"] = df["accepted"].dt.tz_localize(None).dt.normalize()
    return df


def test_events_use_values_as_known_at_each_filing() -> None:
    ends = pd.date_range("2018-03-31", periods=13, freq="QE")
    vals = [10, 11, 12, 13, 12, 14, 13, 16, 15, 15, 17, 18, 25]
    facts, fil = [], []
    for k, (e, v) in enumerate(zip(ends, vals, strict=True)):
        s = (e - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
        acc = f"q{k}"
        facts.append(("NetIncomeLoss", s, e.strftime("%Y-%m-%d"), float(v), acc, "10-Q"))
        fil.append(
            (
                acc,
                "10-Q",
                (e + pd.Timedelta(days=30)).strftime("%Y-%m-%dT21:00:00Z"),
                e.strftime("%Y-%m-%d"),
            )
        )
    # a restatement of the last quarter, filed two months later, must not change the SUE that
    # was known at the original filing
    last_s = (ends[-1] - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
    facts.append(("NetIncomeLoss", last_s, ends[-1].strftime("%Y-%m-%d"), 99.0, "r1", "10-Q/A"))
    fil.append(
        (
            "r1",
            "10-Q/A",
            (ends[-1] + pd.Timedelta(days=90)).strftime("%Y-%m-%dT21:00:00Z"),
            ends[-1].strftime("%Y-%m-%d"),
        )
    )
    ev = F.company_events(F.distill_facts(_payload(facts)), _filings(fil), sic=3571)
    sue = ev[ev["field"] == "sue"].set_index("accepted")["value"]
    q = dict(zip(ends, map(float, vals), strict=True))
    assert sue.iloc[-1] == pytest.approx(F.surprise(q, ends[-1]))
    # a quarter needs its own seasonal difference and six earlier ones: quarters 10, 11 and 12
    # (amendments make no event)
    assert len(sue) == 3


def test_annual_ratios_and_financial_exclusion() -> None:
    facts = [
        ("NetIncomeLoss", "2022-01-01", "2022-12-31", 10.0, "k22", "10-K"),
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "2022-01-01",
            "2022-12-31",
            16.0,
            "k22",
            "10-K",
        ),
        ("Assets", None, "2022-12-31", 220.0, "k22", "10-K"),
        ("Assets", None, "2021-12-31", 180.0, "k22", "10-K"),
        ("Revenues", "2022-01-01", "2022-12-31", 150.0, "k22", "10-K"),
        ("CostOfRevenue", "2022-01-01", "2022-12-31", 95.0, "k22", "10-K"),
    ]
    fil = _filings([("k22", "10-K", "2023-02-15T21:30:00Z", "2022-12-31")])
    ev = F.company_events(F.distill_facts(_payload(facts)), fil, sic=2834).set_index("field")[
        "value"
    ]
    assert ev["accruals"] == pytest.approx((10 - 16) / 200.0)
    assert ev["gross_profitability"] == pytest.approx(55 / 220.0)
    bank = F.company_events(F.distill_facts(_payload(facts)), fil, sic=6021)
    assert not set(bank["field"]) & {"accruals", "gross_profitability"}


# ------------------------------------------------------------------------ session mapping
def test_reaction_session_follows_the_close() -> None:
    dates = sessions(dt.date(2024, 7, 1), dt.date(2024, 7, 10))
    clock = F.SessionClock.of(dates)
    at = pd.Series(
        pd.to_datetime(
            [
                "2024-07-01T19:00:00Z",  # 15:00 ET, before the close -> that session
                "2024-07-01T20:30:00Z",  # after the close -> next session
                "2024-07-01T20:00:00Z",  # exactly at the close -> next session
                "2024-07-03T17:30:00Z",  # 13:30 ET after the 13:00 early close; July 4 shut
                "2024-07-06T15:00:00Z",  # Saturday -> Monday
                "2024-06-20T12:00:00Z",  # before the panel's preceding close -> dropped
                "2024-06-28T21:00:00Z",  # after the preceding close -> first session
                "2024-07-10T21:00:00Z",  # after the last close -> dropped
                None,
            ],
            utc=True,
        )
    )
    day = {str(d): i for i, d in enumerate(dates)}
    want = [
        day["2024-07-01"],
        day["2024-07-02"],
        day["2024-07-02"],
        day["2024-07-05"],
        day["2024-07-08"],
        -1,
        0,
        -1,
        -1,
    ]
    assert clock.reaction_index(at).tolist() == want


# ------------------------------------------------------------------------ security master
def test_name_rules() -> None:
    assert F.name_similarity("Kohl's", "KOHLS Corp (KSS)") == 1.0
    assert F.name_similarity("Aimco", "APARTMENT INVESTMENT & MANAGEMENT CO") == 0.0
    assert F.name_similarity("Owens-Illinois", "OWENS ILLINOIS INC /DE/") == 1.0
    # measured against the index name: the false matches of the first resolution run
    assert F.name_similarity("Coca-Cola Enterprises", "COCA COLA CO (KO)") < 1.0
    assert F.name_similarity("Time Warner Cable", "Time Inc.") < 1.0
    qs = F.search_queries("The Goodyear Tire & Rubber Company", "GT")
    assert "GT" in qs and "GOODYEAR TIRE" in qs and "GOODYEAR" in qs
    assert "Macy" in F.search_queries("Macy's", "M")


class FakeSec:
    def __init__(
        self, subs: dict[int, dict[str, Any]], hits: dict[str, list[tuple[int, str]]]
    ) -> None:
        self.subs, self.hits, self.requests = subs, hits, 0

    def submissions(self, cik: int) -> tuple[dict[str, Any], str] | None:
        self.requests += 1
        return (self.subs[cik], "sha") if cik in self.subs else None

    def search_entities(self, q: str) -> list[tuple[int, str]]:
        self.requests += 1
        return self.hits.get(q, [])


def _sub(
    name: str,
    years: range,
    tickers: tuple[str, ...] = (),
    former: tuple[str, ...] = (),
    sic: str = "3221",
) -> dict[str, Any]:
    dates = [f"{y}-{m:02d}-10" for y in years for m in (2, 5, 8, 11)]
    return {
        "name": name,
        "tickers": list(tickers),
        "formerNames": [{"name": f} for f in former],
        "sic": sic,
        "filings": {"recent": {"form": ["10-Q"] * len(dates), "filingDate": dates}},
    }


def test_master_rejects_successors_and_prefers_the_listed_entity() -> None:
    subs = {
        812233: _sub("OWENS-ILLINOIS GROUP INC", range(2015, 2020)),
        812074: _sub("O-I Glass, Inc.", range(2015, 2026), ("OI",), ("OWENS ILLINOIS INC /DE/",)),
        1701605: _sub("Baker Hughes Co", range(2017, 2026), ("BKR",)),
        808362: _sub("BAKER HUGHES INC", range(2015, 2018)),
        99: _sub("Unrelated Corp", range(2015, 2026)),
    }
    hits = {
        "Owens-Illinois": [(812233, "OWENS-ILLINOIS GROUP INC"), (812074, "O-I Glass, Inc. (OI)")],
        "Baker Hughes": [(1701605, "Baker Hughes Co (BKR)"), (808362, "BAKER HUGHES INC")],
    }
    spells = [
        U.Spell("OI", dt.date(2016, 1, 1), dt.date(2019, 6, 3)),
        U.Spell("BHI", dt.date(2016, 1, 1), dt.date(2017, 7, 5)),
        U.Spell("ZZZ", dt.date(2016, 1, 1), dt.date(2018, 1, 2)),
    ]
    removed = {
        (dt.date(2019, 6, 3), "OI"): "Owens-Illinois",
        (dt.date(2017, 7, 5), "BHI"): "Baker Hughes",
        (dt.date(2018, 1, 2), "ZZZ"): "Nowhere",
    }
    rows = F.resolve_master(
        spells, {}, removed, FakeSec(subs, hits), since=dt.date(2016, 1, 1), overrides={}
    )  # type: ignore[arg-type]
    by = {r.ticker: r for r in rows}
    assert by["OI"].cik == 812074 and by["OI"].method == "search"  # ticker breaks the name tie
    assert by["BHI"].cik == 808362  # the 2017 successor did not file during the spell
    assert by["ZZZ"].cik is None and by["ZZZ"].method == "unresolved"
    ov = F.resolve_master(
        spells[1:2],
        {},
        removed,
        FakeSec(subs, hits),
        since=dt.date(2016, 1, 1),  # type: ignore[arg-type]
        overrides={"BHI@2016-01-01": (1701605, "test")},
    )
    assert ov[0].method == "override_unverified"  # a researcher override is checked, not trusted


# ------------------------------------------------------------------------ signals
def _event_panel() -> Panel:
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 3, 29))
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (dates.size, 4)), axis=0))
    ann = np.zeros_like(close)
    ann[10, 0] = 1.0
    return Panel(
        dates,
        ("A", "B", "C", "D"),
        {"close": close, "in_universe": np.ones_like(close), "ann_day0": ann},
    )


def test_ear_is_the_three_day_market_adjusted_reaction_known_at_day_one() -> None:
    p = _event_panel()
    s = build("ear", hold=5).compute(p)
    r = ops.log_return(p["close"])
    ar = r[:, 0] - market_return(p)[:, 0]
    want = ar[9] + ar[10] + ar[11]
    assert np.all(np.isnan(s[:11, 0]))  # nothing before the close of day +1
    np.testing.assert_allclose(s[11:17, 0], want)  # held for `hold` sessions
    assert np.all(np.isnan(s[17:, 0])) and np.all(np.isnan(s[:, 1:]))


def test_new_event_without_a_value_is_not_covered_by_the_old_one() -> None:
    p = _event_panel()
    ann = np.array(p["ann_day0"])
    ann[20, 0] = 1.0
    close = np.array(p["close"])
    close[21, 0] = np.nan  # the second reaction window cannot be measured
    s = build("ear", hold=30).compute(p.with_fields(ann_day0=ann, close=close))
    assert np.isfinite(s[15, 0]) and np.all(np.isnan(s[21:52, 0]))


def test_fundamental_signals_are_causal(noise_panel: Panel) -> None:
    rng = np.random.default_rng(0)
    shape = noise_panel["close"].shape
    sparse = lambda p: np.where(rng.random(shape) < p, rng.normal(size=shape), np.nan)  # noqa: E731
    panel = noise_panel.with_fields(
        ann_day0=(rng.random(shape) < 0.016).astype(float),
        sue=sparse(0.016),
        accruals=sparse(0.004),
        gross_profitability=sparse(0.004),
    )
    for name in FUNDAMENTAL_SIGNALS:
        check_causal(build(name).compute, panel, default_cutoffs(panel.n_dates), name=name)


def test_low_accruals_sign_and_hold() -> None:
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 2, 29))
    acc = np.full((dates.size, 1), np.nan)
    acc[3, 0] = 0.05
    s = build("accruals", hold=10).compute(Panel(dates, ("A",), {"accruals": acc}))
    assert s[3, 0] == -0.05 and s[13, 0] == -0.05 and np.isnan(s[14, 0])


def test_fundamental_family_runs_end_to_end() -> None:
    from dataclasses import replace

    from pusula import cli
    from pusula.data.synthetic import SyntheticSpec, generate
    from pusula.engine.backtest import BacktestConfig
    from pusula.research import study as S
    from pusula.research.families import FORWARD_FAMILIES, us_fundamental_v1
    from pusula.risk.model import RiskModelSpec

    spec = us_fundamental_v1()
    assert cli.needs_fundamentals(spec) and not cli.needs_implied_vol(spec)
    assert cli.program_of(spec.family) == "us-" and spec.family in FORWARD_FAMILIES
    from pusula.research.families import us_equity_v3

    assert not cli.needs_fundamentals(us_equity_v3())
    market = generate(
        SyntheticSpec(n_symbols=50, start="2016-01-04", end="2018-12-31", ic=0.0, seed=5)
    )
    rng = np.random.default_rng(1)
    shape = market["close"].shape
    ann = np.zeros(shape)
    for j in range(shape[1]):
        ann[rng.integers(0, 63) :: 63, j] = 1.0  # one release a quarter
    sparse = lambda p: np.where(rng.random(shape) < p, rng.normal(size=shape), np.nan)  # noqa: E731
    panel = market.with_fields(
        ann_day0=ann,
        sue=sparse(1 / 63),
        accruals=sparse(1 / 252),
        gross_profitability=sparse(1 / 252),
    )
    keep = {"ear_drift", "fundamental_composite", "equal_weight"}
    trials = tuple(
        replace(t, optimizer=replace(t.optimizer, n_factors=4)) if t.optimizer is not None else t
        for t in spec.trials
        if t.name in keep
    )
    small = replace(
        spec,
        trials=trials,
        dev=("2017-01-03", "2017-12-29"),
        holdout=("2018-01-02", "2018-12-31"),
        backtest=BacktestConfig(
            start="2017-01-03",
            end="2018-12-31",
            kill_switch=False,
            risk_spec=RiskModelSpec(n_factors=4, window=200),
        ),
    )
    runs = S.run_backtests(small, lambda: (panel, np.zeros(panel.n_dates)), workers=1)
    assert set(runs) == keep
    for name in ("ear_drift", "fundamental_composite"):
        r = runs[name]["base"]
        assert (
            np.isfinite(r.returns).sum() > 400 and float(np.nanmean(r.gross_exposure[100:])) > 0.5
        )


def test_coverage_counts_live_events_per_member_group() -> None:
    dates = sessions(dt.date(2024, 1, 2), dt.date(2024, 1, 31))  # 21 sessions
    n = dates.size
    member = np.ones((n, 2))
    member[-1, 1] = 0.0  # B left the index: "removed"
    ann = np.zeros((n, 2))
    ann[0, 0] = 1.0  # A: one event, live for 5 sessions -> rows 0..5
    sue = np.full((n, 2), np.nan)
    sue[10, 1] = 0.3  # B: live rows 10..20 while a member (rows 10..19)
    p = Panel(dates, ("A", "B"), {"in_universe": member, "ann_day0": ann, "sue": sue})
    cov = F.coverage(p, {"ann_day0": 5, "sue": 63}).set_index("members")
    assert cov.loc["current", "member_days"] == n and cov.loc[
        "current", "ann_day0"
    ] == pytest.approx(6 / n)
    assert cov.loc["removed", "member_days"] == n - 1
    assert cov.loc["removed", "sue"] == pytest.approx(10 / (n - 1))


# ------------------------------------------------------------------------ panel alignment
def test_fundamental_fields_place_events_on_the_reaction_session(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {
                "security_id": "AAA@2016-01-01",
                "ticker": "AAA",
                "name": "A",
                "cik": 1,
                "sic": 3571,
                "method": "current_table",
                "note": "",
            },
            {
                "security_id": "BBB@2016-01-01",
                "ticker": "BBB",
                "name": "B",
                "cik": None,
                "sic": None,
                "method": "unresolved",
                "note": "",
            },
        ]
    )
    fil = _filings([("e1", "8-K", "2024-02-01T21:05:00Z", None)]).assign(items="2.02,9.01")
    manifest = F._write_store(tmp_path, master, {1: (F.distill_facts({}), fil)}, {"ciks": {}})
    assert (
        manifest["events"] == {"ann_day0": 1}
        and json.loads((F.store_dir(tmp_path) / "manifest.json").read_text())["digest"]
    )
    dates = sessions(dt.date(2024, 1, 29), dt.date(2024, 2, 9))
    panel = Panel(dates, ("AAA@2016-01-01", "BBB@2016-01-01"), {"close": np.ones((dates.size, 2))})
    out = F.fundamental_fields(panel, tmp_path)
    hit = np.argwhere(out["ann_day0"] > 0).tolist()
    assert hit == [
        [int(np.searchsorted(dates, np.datetime64("2024-02-02"))), 0]
    ]  # 16:05 ET -> next day
    assert np.all(np.isnan(out["sue"]))
