"""Dead-man check: is the nightly system actually alive, and are its records intact?

A monitor nothing runs is not a monitor (NihaiAlgoritma BULGU-61-7), and a failure nobody is
told about is not noticed: pusula's books stood still from 2026-10-27 to 2026-10-30, first
because every scheduled run got HTTP 403 and then because the task's script had moved, and
nothing said so. This module only measures; it decides nothing and changes no book.

It reads what the nightly run left behind, so it also works when that run never started:

* every forward family has a snapshot, no more than one completed session old;
* every book's journal chain verifies and its head matches the saved state;
* no book is halted, and the warnings of the daily report (stale feeds, a signal missing from a
  book, a held name with a suspicious last bar, optimiser failures) become alerts;
* the research ledger's chain verifies;
* the nightly run's own exit code, if it reports one.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pusula.core import calendar as cal
from pusula.core.errors import PusulaError
from pusula.data.settings import atomic_write_text, home
from pusula.engine.journal import Journal
from pusula.live.notify import Alert
from pusula.live.report import snapshot_warnings
from pusula.research.registry import Registry

MAX_SESSIONS_BEHIND = 1  # the session that closed since the last nightly run is not a failure


@dataclass(frozen=True, slots=True)
class HealthReport:
    checked_at: str
    expected_session: str
    families: dict[str, str]  # family -> as_of of its snapshot, or "" without one
    alerts: tuple[Alert, ...]

    @property
    def healthy(self) -> bool:
        return not any(a.severity == "critical" for a in self.alerts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "expected_session": self.expected_session,
            "healthy": self.healthy,
            "families": self.families,
            "alerts": [asdict(a) for a in self.alerts],
        }


def sessions_behind(as_of: dt.date, expected: dt.date) -> int:
    """Completed sessions in ``(as_of, expected]``."""
    n, d = 0, expected
    while d > as_of:
        n += 1
        d = cal.previous_session(d)
    return n


def check_book(name: str, path: Path) -> list[Alert]:
    """Journal chain and the state's head. Reading the journal verifies its chain."""
    journal_path, state_path = path / "journal.jsonl", path / "state.json"
    if not journal_path.exists():
        return []  # registered but not opened yet
    key = f"book:{name}"
    try:
        journal = Journal(journal_path)
    except (PusulaError, ValueError, KeyError) as exc:
        return [Alert(key, "critical", f"{name}: journal unreadable or edited: {exc}")]
    if not state_path.exists():
        return [Alert(key, "critical", f"{name}: has a journal but no saved state")]
    head = json.loads(state_path.read_text(encoding="utf-8")).get("journal_head")
    if journal.events and journal.events[-1].hash != head:
        return [Alert(key, "critical", f"{name}: journal head does not match the saved state")]
    return []


def check_family(
    family: str, root: Path, expected: dt.date, flagged: dict[tuple[str, str], list[str]]
) -> tuple[str, list[Alert]]:
    """(as_of of the family's snapshot, alerts); adds the books holding a suspicious last bar to
    ``flagged``, so that one bar held by twenty books is one alert."""
    snap_path = root / "dashboard.json"
    if not snap_path.exists():
        return "", [Alert(f"{family}:snapshot", "critical", f"{family}: no forward snapshot")]
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    as_of = str(snap["as_of"])
    out: list[Alert] = []
    behind = sessions_behind(dt.date.fromisoformat(as_of), expected)
    if behind > MAX_SESSIONS_BEHIND:
        out.append(
            Alert(
                f"{family}:stale",
                "critical",
                f"{family}: books stopped at {as_of}, {behind} sessions behind {expected}: "
                "the nightly run is not working",
            )
        )
    for name, book in sorted(snap.get("books", {}).items()):
        for sid, why in book.get("flagged_holdings", {}).items():
            flagged.setdefault((str(sid), str(why)), []).append(f"{family}/{name}")
        if book.get("halted"):
            out.append(
                Alert(
                    f"{family}:{name}:halted",
                    "critical",
                    f"{family}/{name}: HALTED ({book.get('kill_reason')}); stays halted until "
                    "an operator decides",
                )
            )
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "decisions"):
            out += check_book(f"{family}/{sub.name}", sub)
    out += [
        Alert(f"{family}:warning:{text}", "warning", f"{family}: {text}")
        for text in snapshot_warnings(snap)
    ]
    return as_of, out


def check_mirror(store: Path) -> list[Alert]:
    """Orders the broker refused on the last plan, or a plan that could not be made, if an account
    mirrors the books."""
    refused_path = store / "refused.json"
    if refused_path.exists():
        why = json.loads(refused_path.read_text(encoding="utf-8")).get("why")
        return [Alert("mirror:refused", "warning", f"paper mirror stopped: {why}")]
    plans = sorted(store.glob("plan-*.json"))
    if not plans:
        return []
    plan = json.loads(plans[-1].read_text(encoding="utf-8"))
    outcome = plan.get("outcome", {})
    refused = sum(v == "rejected" for v in outcome.values())
    if not refused:
        return []
    return [
        Alert(
            "mirror:rejected",
            "warning",
            f"paper mirror, {plan.get('session')}: the broker refused {refused} of "
            f"{len(outcome)} orders",
        )
    ]


def run_health(
    families: Sequence[str],
    ledger: Path,
    now: dt.datetime,
    expected: dt.date,
    forward_root: Path | None = None,
    forward_exit: int | None = None,
    err_log: Path | None = None,
) -> HealthReport:
    root = forward_root or home() / "forward"
    alerts: list[Alert] = []
    seen: dict[str, str] = {}
    flagged: dict[tuple[str, str], list[str]] = {}
    for family in families:
        seen[family], found = check_family(family, root / family, expected, flagged)
        alerts += found
    alerts += [
        Alert(
            f"bar:{sid}",
            "warning",
            f"{sid}: {why} on the last session, held by {len(books)} book(s); check it is real",
        )
        for (sid, why), books in sorted(flagged.items())
    ]
    alerts += check_mirror(root.parent / "execution" / "paper")
    try:
        Registry(ledger)  # reading it verifies its chain
    except PusulaError as exc:
        alerts.append(Alert("ledger", "critical", f"research ledger: {exc}"))
    if forward_exit:
        alerts.append(
            Alert(
                "nightly:exit",
                "critical",
                f"nightly run exited {forward_exit}: {_last_line(err_log)}",
            )
        )
    return HealthReport(now.isoformat(), expected.isoformat(), seen, tuple(alerts))


def _last_line(path: Path | None, limit: int = 300) -> str:
    """The end of the failed run's stderr: a Python traceback ends with what went wrong."""
    if path is None or not path.exists():
        return "(no error log)"
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln]
    return lines[-1][:limit] if lines else "(empty error log)"


def write_health(report: HealthReport, root: Path | None = None) -> Path:
    path = (root or home()) / "health.json"
    atomic_write_text(path, json.dumps(report.as_dict(), indent=1))
    return path


def _session_return(book: dict[str, Any]) -> float | None:
    curve = book.get("equity_curve", [])
    return float(curve[-1][1]) / float(curve[-2][1]) - 1.0 if len(curve) >= 2 else None


def session_digest(report: HealthReport, root: Path | None = None) -> tuple[str, str]:
    """(message, session): one short summary of the night, read from what it left behind."""
    forward = root or home() / "forward"
    moves: list[tuple[float, str]] = []
    n_books = n_traded = n_halted = 0
    for family, as_of in report.families.items():
        if not as_of:
            continue
        snap = json.loads((forward / family / "dashboard.json").read_text(encoding="utf-8"))
        for name, book in snap.get("books", {}).items():
            n_books += 1
            n_halted += bool(book.get("halted"))
            n_traded += bool(book.get("last_trades", {}).get("n_trades"))
            r = _session_return(book)
            if r is not None:
                moves.append((r, f"{family}/{name}"))
    session = max(filter(None, report.families.values()), default=report.expected_session)
    lines = [
        f"pusula {session}: {'OK' if report.healthy else 'FAILING'}",
        f"{n_books} books, {n_traded} traded, {n_halted} halted",
    ]
    if moves:
        moves.sort()
        lines += [
            f"best {moves[-1][1]} {moves[-1][0]:+.2%}",
            f"worst {moves[0][1]} {moves[0][0]:+.2%}",
        ]
    n_crit = sum(a.severity == "critical" for a in report.alerts)
    if report.alerts:
        lines.append(f"open alerts: {n_crit} critical, {len(report.alerts) - n_crit} warnings")
    return "\n".join(lines), session


def render_health(report: HealthReport) -> str:
    lines = [
        f"pusula health: {'OK' if report.healthy else 'FAILING'} "
        f"(expected session {report.expected_session})"
    ]
    lines += [f"  {fam}: {as_of or 'no snapshot'}" for fam, as_of in report.families.items()]
    lines += [f"  [{a.severity}] {a.message}" for a in report.alerts]
    return "\n".join(lines)
