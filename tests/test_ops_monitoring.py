"""The dead-man check, alerts, decision log and forward review."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from pusula.alpha.combine import SignalSpec
from pusula.data.alpaca import end_param
from pusula.data.panel import Panel
from pusula.data.synthetic import SyntheticSpec, generate
from pusula.engine.backtest import BacktestConfig, Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.live import health as HL
from pusula.live import notify as N
from pusula.live.decisions import (
    SignalCache,
    explain_security,
    render_decisions,
    session_decisions,
    write_decisions,
)
from pusula.live.forward import last_bar_flags
from pusula.live.report import render_daily, snapshot_warnings
from pusula.live.review import MIN_SESSIONS, book_returns, render_review, review_family
from pusula.portfolio.strategies import EqualWeight
from pusula.research.registry import Registry
from pusula.research.study import StudySpec, Trial
from pusula.stats.bootstrap import sharpe_difference

NOW = dt.datetime(2026, 10, 30, 12, 0, tzinfo=dt.UTC)
EXPECTED = dt.date(2026, 10, 29)


# ----------------------------------------------------------------- the bar request's end
def test_end_of_the_evenings_session_stays_out_of_the_sip_embargo() -> None:
    """01:00 Istanbul on 2026-10-29 is 18:00 in New York: a bare ``2026-10-28`` got HTTP 403."""
    now = dt.datetime(2026, 10, 28, 22, 0, tzinfo=dt.UTC)
    assert end_param(dt.date(2026, 10, 28), now) == "2026-10-28T21:44:00Z"


def test_end_of_a_past_session_stays_a_date() -> None:
    now = dt.datetime(2026, 10, 30, 11, 0, tzinfo=dt.UTC)
    assert end_param(dt.date(2026, 10, 29), now) == "2026-10-29"


# ------------------------------------------------------------------------ health
def _book(**kw: object) -> dict[str, object]:
    return {
        "equity": 1e7,
        "equity_curve": [["2026-10-28", 1e7], ["2026-10-29", 1e7]],
        "gross": 1.0,
        "net": 1.0,
        "halted": False,
        "kill_reason": "",
        **kw,
    }


def _family(root: Path, as_of: str = "2026-10-29", **book: object) -> None:
    root.mkdir(parents=True, exist_ok=True)
    snap = {
        "family": root.name,
        "as_of": as_of,
        "books": {"b": _book(**book)},
        "data": {"quality": "q", "digest": "d"},
    }
    (root / "dashboard.json").write_text(json.dumps(snap), encoding="utf-8")


def _ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    if not path.exists():
        Registry(path).preregister(
            family="f",
            name="t",
            hypothesis="",
            config={},
            gates={},
            windows={"dev": ["a", "b"], "holdout": ["c", "d"]},
            now=NOW,
            code="c",
        )
    return path


def _run(tmp_path: Path, **kw: object) -> HL.HealthReport:
    return HL.run_health(
        ["f"],
        _ledger(tmp_path),
        NOW,
        EXPECTED,
        forward_root=tmp_path / "fwd",
        **kw,  # type: ignore[arg-type]
    )


def test_a_current_family_is_healthy(tmp_path: Path) -> None:
    _family(tmp_path / "fwd" / "f")
    r = _run(tmp_path)
    assert r.healthy
    assert r.alerts == ()
    assert "OK" in HL.render_health(r)
    assert HL.write_health(r, tmp_path).exists()


def test_books_that_stopped_are_critical_but_one_session_is_not(tmp_path: Path) -> None:
    _family(tmp_path / "fwd" / "f", as_of="2026-10-28")
    assert _run(tmp_path).healthy
    _family(tmp_path / "fwd" / "f", as_of="2026-10-26")  # what actually happened
    r = _run(tmp_path)
    assert not r.healthy
    assert "3 sessions behind" in r.alerts[0].message


def test_a_missing_snapshot_is_critical(tmp_path: Path) -> None:
    (tmp_path / "fwd").mkdir()
    assert not _run(tmp_path).healthy


def test_a_halted_book_and_a_failed_night_are_critical(tmp_path: Path) -> None:
    _family(tmp_path / "fwd" / "f", halted=True, kill_reason="drawdown")
    err = tmp_path / "err.log"
    err.write_text("Traceback\n  x\npusula.DataError: HTTP 403 not permitted\n", encoding="utf-8")
    r = _run(tmp_path, forward_exit=1, err_log=err)
    text = " ".join(a.message for a in r.alerts)
    assert "HALTED (drawdown)" in text
    assert "exited 1: pusula.DataError: HTTP 403" in text
    assert not r.healthy


def test_an_edited_journal_or_state_is_critical(tmp_path: Path) -> None:
    fam = tmp_path / "fwd" / "f"
    _family(fam)
    book = fam / "b"
    j = Journal(book / "journal.jsonl")
    j.append("capital", "2026-10-28", {"micros": 1})
    state = book / "state.json"
    state.write_text(json.dumps({"journal_head": "other"}), encoding="utf-8")
    assert "does not match" in _run(tmp_path).alerts[0].message
    state.write_text(json.dumps({"journal_head": j.events[-1].hash}), encoding="utf-8")
    assert _run(tmp_path).healthy
    text = (book / "journal.jsonl").read_text(encoding="utf-8").replace('"micros":1', '"micros":2')
    (book / "journal.jsonl").write_text(text, encoding="utf-8")
    assert "journal unreadable or edited" in _run(tmp_path).alerts[0].message


def test_report_warnings_become_alerts_without_their_markup(tmp_path: Path) -> None:
    _family(
        tmp_path / "fwd" / "f",
        flagged_holdings={"X@2016": "a move of -40%"},
        solver_failures=2,
    )
    r = _run(tmp_path)
    assert r.healthy  # warnings do not fail the check
    msgs = [a.message for a in r.alerts]
    assert any("X@2016: a move of -40% on the last session, held by 1 book" in m for m in msgs)
    assert any("optimiser failed 2" in m for m in msgs)
    assert not any("**" in m for m in msgs)


def test_one_bad_bar_held_by_many_books_is_one_alert(tmp_path: Path) -> None:
    """2026-10-29: a single move in one stock produced 18 lines from 18 books."""
    root = tmp_path / "fwd" / "f"
    root.mkdir(parents=True)
    snap = {
        "family": "f",
        "as_of": "2026-10-29",
        "books": {n: _book(flagged_holdings={"X@2016": "no bar"}) for n in ("a", "b", "c")},
        "data": {"quality": "q", "digest": "d"},
    }
    (root / "dashboard.json").write_text(json.dumps(snap), encoding="utf-8")
    alerts = _run(tmp_path).alerts
    assert len(alerts) == 1
    assert "held by 3 book(s)" in alerts[0].message


def _digest_report(tmp_path: Path) -> HL.HealthReport:
    root = tmp_path / "fwd" / "f"
    root.mkdir(parents=True)
    curve = [["2026-10-28", 100.0], ["2026-10-29", 101.0]]
    snap = {
        "family": "f",
        "as_of": "2026-10-29",
        "books": {
            "up": _book(equity_curve=curve, last_trades={"n_trades": 3}),
            "down": _book(equity_curve=[["2026-10-28", 100.0], ["2026-10-29", 98.0]]),
            "new": _book(equity_curve=[["2026-10-29", 100.0]], halted=True),
        },
        "data": {"quality": "q", "digest": "d"},
    }
    (root / "dashboard.json").write_text(json.dumps(snap), encoding="utf-8")
    return _run(tmp_path)


def test_digest_summarises_the_night_from_its_snapshots(tmp_path: Path) -> None:
    text, session = HL.session_digest(_digest_report(tmp_path), tmp_path / "fwd")
    assert session == "2026-10-29"
    assert text.splitlines() == [
        "pusula 2026-10-29: FAILING",
        "3 books, 1 traded, 1 halted",
        "best f/up +1.00%",
        "worst f/down -2.00%",
        "open alerts: 1 critical, 0 warnings",
    ]


def test_digest_is_sent_once_per_session_and_retried_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUSULA_TELEGRAM_TOKEN", "t")
    monkeypatch.setenv("PUSULA_TELEGRAM_CHAT_ID", "c")
    state = tmp_path / "digest.json"
    sent: list[str] = []
    ok = httpx.MockTransport(lambda r: sent.append(r.content.decode()) or httpx.Response(200))
    down = httpx.MockTransport(lambda r: httpx.Response(500))
    with pytest.raises(N.NotifyError):
        N.deliver_digest("m", "2026-10-29", state, down)
    assert N.deliver_digest("m", "2026-10-29", state, ok) == "digest: sent"
    assert N.deliver_digest("m", "2026-10-29", state, ok) == "digest: already sent"
    assert N.deliver_digest("m", "2026-10-30", state, ok) == "digest: sent"
    assert len(sent) == 2


def test_digest_stays_silent_without_telegram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUSULA_TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("PUSULA_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("PUSULA_ENV_FILE", raising=False)
    out = N.deliver_digest("m", "2026-10-29", tmp_path / "s.json")
    assert out == "digest: telegram not configured"
    assert not (tmp_path / "s.json").exists()


def test_report_renders_new_warnings_and_trades() -> None:
    snap = {
        "family": "f",
        "as_of": "2026-10-29",
        "books": {
            "b": _book(
                flagged_holdings={"X@2016": "no bar"},
                last_trades={
                    "n_trades": 1,
                    "traded": 100.0,
                    "fees": 1.0,
                    "trades": [{"security": "X@2016", "notional": 100.0}],
                },
            )
        },
        "data": {"quality": "q", "digest": "d"},
    }
    text = render_daily(snap)
    assert "holds X@2016: no bar" in text
    assert "| b | 1 | 100 | 1 | X@2016 +100 |" in text
    assert snapshot_warnings(snap) == []  # the health check reports it once, across books


def test_last_bar_flags_finds_a_gap_and_a_jump() -> None:
    p = generate(SyntheticSpec(n_symbols=5, start="2016-01-04", end="2016-03-31", seed=1))
    c = p["close"].copy()
    c[-1, 0] = np.nan
    c[-1, 1] = c[-2, 1] * 0.5
    flags = last_bar_flags(p.with_fields(close=c))
    assert flags[p.symbols[0]] == "no bar"
    assert flags[p.symbols[1]] == "a move of -50%"
    assert len(flags) == 2


# ------------------------------------------------------------------------ notify
def _telegram(status: int = 200) -> tuple[httpx.MockTransport, list[str]]:
    sent: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content)["text"])
        return httpx.Response(status, json={"ok": status == 200})

    return httpx.MockTransport(handler), sent


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUSULA_ENV_FILE", raising=False)
    monkeypatch.delenv("PUSULA_HEARTBEAT_URL", raising=False)
    monkeypatch.setenv("PUSULA_TELEGRAM_TOKEN", "SECRET-TOKEN")
    monkeypatch.setenv("PUSULA_TELEGRAM_CHAT_ID", "42")


A1 = N.Alert("a", "critical", "books stopped")
A2 = N.Alert("b", "warning", "feed stale")


def test_alerts_are_sent_once_reminded_daily_and_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch)
    state = tmp_path / "state.json"
    tr, sent = _telegram()
    N.deliver([A1, A2], state, NOW, tr)
    assert "NEW" in sent[-1]
    assert sent[-1].index("books stopped") < sent[-1].index("feed stale")  # critical first
    N.deliver([A1, A2], state, NOW + dt.timedelta(hours=2), tr)
    assert len(sent) == 1  # nothing changed, nothing sent
    N.deliver([A1, A2], state, NOW + dt.timedelta(hours=25), tr)
    assert "STILL OPEN" in sent[-1]
    N.deliver([A2], state, NOW + dt.timedelta(hours=26), tr)
    assert "RESOLVED" in sent[-1]
    assert "books stopped" in sent[-1]
    N.deliver([A2], state, NOW + dt.timedelta(hours=27), tr)
    assert len(sent) == 3  # the resolution was sent once


def test_a_failed_send_is_retried_and_leaks_no_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch)
    state = tmp_path / "state.json"

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {req.url}")

    with pytest.raises(N.NotifyError) as exc:
        N.deliver([A1], state, NOW, httpx.MockTransport(boom))
    assert "SECRET-TOKEN" not in str(exc.value)
    assert exc.value.__context__ is None  # no chained httpx error carrying the URL
    assert not state.exists()  # not marked as sent
    tr, sent = _telegram()
    N.deliver([A1], state, NOW, tr)
    assert "books stopped" in sent[0]
    assert "SECRET-TOKEN" not in repr(N.telegram_config())


def test_nothing_is_sent_without_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("PUSULA_ENV_FILE", "PUSULA_TELEGRAM_TOKEN", "PUSULA_HEARTBEAT_URL"):
        monkeypatch.delenv(k, raising=False)
    tr, sent = _telegram()
    assert N.deliver([A1], tmp_path / "s.json", NOW, tr) == "telegram not configured"
    assert sent == []


def test_heartbeat_only_when_nothing_is_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("PUSULA_ENV_FILE", "PUSULA_TELEGRAM_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PUSULA_HEARTBEAT_URL", "https://hb.example/x")
    pings: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        pings.append(str(req.url))
        return httpx.Response(200)

    tr = httpx.MockTransport(handler)
    N.deliver([A1], tmp_path / "s.json", NOW, tr)
    assert pings == []
    assert "heartbeat pinged" in N.deliver([A2], tmp_path / "s.json", NOW, tr)
    assert pings == ["https://hb.example/x"]


# ------------------------------------------------------------------------ decisions
@pytest.fixture(scope="module")
def traded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Panel, Journal]:
    p = generate(SyntheticSpec(n_symbols=20, start="2016-01-04", end="2017-06-30", seed=3))
    journal = Journal(tmp_path_factory.mktemp("book") / "journal.jsonl")
    eng = PaperEngine(
        Backtester(p, BacktestConfig(kill_switch=False)),
        EqualWeight("bench", every=5),
        "bench",
        journal=journal,
    )
    for e in range(p.date_index("2017-01-03"), p.n_dates):
        eng.step(e)
    return p, journal


TRIAL = Trial("bench", "equal_weight", "h", signals=(SignalSpec("momentum"),))


def test_session_decisions_reconcile_with_the_journal(traded: tuple[Panel, Journal]) -> None:
    p, journal = traded
    session = next(e.session for e in journal.events if e.kind == "fill")
    d = session_decisions(journal, session, TRIAL, SignalCache(p), lag=1)
    fills = [e for e in journal.events if e.kind == "fill" and e.session == session]
    assert d["n_trades"] == len({e.data["security"] for e in fills})
    assert abs(d["trades"][0]["notional"]) >= abs(d["trades"][-1]["notional"]) > 0
    assert 0.0 <= d["trades"][0]["why"]["momentum"] <= 1.0
    text = render_decisions("f", {"bench": [d]})
    assert "momentum p" in text
    assert session in text


def test_decision_file_is_written_only_when_something_traded(
    traded: tuple[Panel, Journal], tmp_path: Path
) -> None:
    p, journal = traded
    session = next(e.session for e in journal.events if e.kind == "fill")
    d = session_decisions(journal, session, TRIAL, SignalCache(p), lag=1)
    quiet = {**d, "n_trades": 0, "trades": []}
    assert write_decisions(tmp_path, "f", "2017-01-03", {"b": [quiet]}) is None
    path = write_decisions(tmp_path, "f", "2017-01-03", {"b": [d]})
    assert path is not None
    assert path.parent.name == "decisions"


def test_explain_tells_a_books_history_in_one_security(traded: tuple[Panel, Journal]) -> None:
    p, journal = traded
    sid = next(str(e.data["security"]) for e in journal.events if e.kind == "fill")
    spec = StudySpec("f", (TRIAL,), ("a", "b"), ("c", "d"))
    text = explain_security(spec, TRIAL, journal, p, sid)
    assert sid in text
    assert "momentum p" in text
    assert "->" in text
    assert "not a security" in explain_security(spec, TRIAL, journal, p, "NOPE")


# ------------------------------------------------------------------------ review
def test_sharpe_difference_finds_a_real_gap_and_not_a_false_one() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 0.01, 1500)
    good = common + rng.normal(0.0015, 0.003, 1500)
    d = sharpe_difference(good, common, n_boot=300)
    assert d.diff > 0
    assert d.lo > 0
    assert d.p_better > 0.99
    same = sharpe_difference(common, common + rng.normal(0.0, 0.003, 1500), n_boot=300)
    assert same.lo < 0 < same.hi
    assert np.isnan(sharpe_difference([0.1], [0.1]).diff)


def _write_book(root: Path, name: str, rets: np.ndarray, start: str = "2026-01-02") -> None:
    j = Journal(root / name / "journal.jsonl")
    j.append("capital", start, {"micros": 10_000_000_000_000})
    eq = 1e7
    for d, r in zip(pd.bdate_range(start, periods=len(rets)), rets, strict=True):
        session = str(d.date())
        j.append("financing", session, {"micros": round(eq * r * 1e6)})
        j.append("mark", session, {"marks": {}})
        eq *= 1.0 + r


def test_book_returns_come_from_the_journals_marks(tmp_path: Path) -> None:
    j = Journal(tmp_path / "j.jsonl")
    j.append("capital", "2026-01-02", {"micros": 100_000_000})
    j.append("mark", "2026-01-02", {"marks": {}})
    j.append("financing", "2026-01-05", {"micros": 1_000_000})
    j.append("mark", "2026-01-05", {"marks": {}})
    r = book_returns(j)
    assert r.iloc[0] == pytest.approx(0.01)
    assert str(r.index[0])[:10] == "2026-01-05"


def _spec() -> StudySpec:
    trials = (
        Trial("cand", "trend", "h"),
        Trial("cand2", "trend", "h"),
        Trial("equal_weight", "equal_weight", "h", benchmark=True),
    )
    return StudySpec("f-v1", trials, ("2019-01-01", "2022-12-30"), ("2023-01-03", "2026-10-22"))


def _registry(tmp_path: Path) -> Registry:
    reg = Registry(tmp_path / "ledger.jsonl")
    for t in _spec().trials:
        h = reg.preregister(
            family="f-v1",
            name=t.name,
            hypothesis="",
            config={"benchmark": t.benchmark},
            gates={},
            windows={"dev": ["a", "b"], "holdout": ["c", "d"]},
            now=NOW,
            code="c",
        )
        reg.record_result(
            h,
            metrics={
                "holdout": {"sharpe": 1.0, "sr_daily": 0.063, "skew": 0.0, "kurt": 3.0, "n": 900},
                "dev": {},
                "factor": {},
                "summary": {},
            },
            verdict={},
            data_digest="d",
            now=NOW,
        )
    return reg


def _factors(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=n)
    rng = np.random.default_rng(9)
    cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    f = pd.DataFrame(rng.normal(0, 0.005, (n, 6)), index=idx, columns=cols)
    f["rf"] = 0.0001
    return f


def test_a_short_record_gets_no_verdict_but_knows_what_it_needs(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    root = tmp_path / "fwd"
    for n in ("cand", "equal_weight"):
        _write_book(root, n, rng.normal(0.0005, 0.01, 20))
    fac = _factors(40)
    rev = review_family(_spec(), _registry(tmp_path), root, fac["rf"], fac, "f-")
    b = rev["books"]["cand"]
    assert b["status"] == "insufficient"
    assert b["n"] == 19  # returns: one fewer than marks
    assert 500 < b["sessions_needed"] < 1000  # holdout Sharpe 1.0 a year
    assert "vs_benchmark" in b
    assert rev["books"]["equal_weight"]["status"] == "benchmark"
    assert "cand2" not in rev["books"]  # not opened
    assert "| cand | 19 |" in render_review(rev)


def test_a_long_record_goes_through_the_gate_and_missing_stress_fails(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = MIN_SESSIONS + 40
    root = tmp_path / "fwd"
    for name in ("cand", "cand2", "equal_weight"):
        _write_book(root, name, rng.normal(0.002, 0.005, n))
    fac = _factors(n + 5)
    rev = review_family(_spec(), _registry(tmp_path), root, fac["rf"], fac, "f-")
    b = rev["books"]["cand"]
    assert b["status"] == "fails"
    failed = b["verdict"]["failed"]
    assert "lag_stress" in failed  # not run: a missing check fails
    assert "cost_stress" in failed
    assert "t_nw" not in failed  # this record is strong; the statistics did run
    assert "fails: " in render_review(rev)
