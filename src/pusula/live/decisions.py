"""What did a book trade, and why: a daily decision log and a per-security explanation.

A book's journal says what happened; this adds, for each trade, where the security stood on each
of the trial's signals when the decision was taken (its percentile among index members at the
decision close). The log is derived from the journal and the panel, so it can always be rebuilt
and it can never disagree with the record. QuantBot wrote such a file every run and Nihai3 could
explain a single name; neither tied the explanation to a hash-chained record.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from pusula.alpha.combine import SignalSpec
from pusula.alpha.signals import build
from pusula.core.money import to_dollars
from pusula.data.panel import FloatArray, Panel
from pusula.data.settings import atomic_write_text
from pusula.engine.journal import Journal
from pusula.research.study import StudySpec, Trial

MAX_LISTED = 15  # per book and session: the largest trades, the rest are counted


class SignalCache:
    """Raw signal arrays of one panel, shared between the books of a run."""

    def __init__(self, panel: Panel) -> None:
        self.panel = panel
        self._raw: dict[tuple[str, str], FloatArray] = {}

    def raw(self, s: SignalSpec) -> FloatArray:
        key = (s.name, repr(sorted(dict(s.params).items())))
        if key not in self._raw:
            self._raw[key] = build(s.name, **dict(s.params)).compute(self.panel)
        return self._raw[key]

    def percentile(self, s: SignalSpec, row: int, column: int) -> float:
        """Share of index members scoring at or below this security on row ``row``."""
        x = self.raw(s)[row]
        v = x[column]
        members = self.panel["in_universe"][row] > 0
        pool = x[members & np.isfinite(x)]
        if not math.isfinite(v) or pool.size == 0:
            return math.nan
        return float((pool <= v).mean())


def trial_signals(trial: Trial) -> tuple[SignalSpec, ...]:
    return (*trial.signals, *(trial.ml.features if trial.ml is not None else ()))


def _why(trial: Trial, cache: SignalCache, security: str, row: int) -> dict[str, float | None]:
    if row < 0 or security not in cache.panel.symbols:
        return {}
    j = cache.panel.symbol_index(security)
    out: dict[str, float | None] = {}
    for s in trial_signals(trial):
        p = cache.percentile(s, row, j)
        out[s.name] = p if math.isfinite(p) else None
    return out


def session_decisions(
    journal: Journal, session: str, trial: Trial, cache: SignalCache, lag: int
) -> dict[str, Any]:
    """The trades of one session, largest first, with the signal standing of each. A trade on
    ``session`` was decided at the close ``lag`` sessions earlier."""
    panel = cache.panel
    decided = panel.date_index(session) - lag
    by_security: dict[str, dict[str, float]] = {}
    fees = 0.0
    for e in journal.events:
        if e.session != session or e.kind != "fill":
            continue
        d = e.data
        sid = str(d["security"])
        t = by_security.setdefault(sid, {"qty": 0.0, "notional": 0.0})
        t["qty"] += int(d["qty"])
        t["notional"] += int(d["qty"]) * to_dollars(int(d["price_micros"]))
        fees += to_dollars(int(d["fee_micros"]))
    ordered = sorted(by_security.items(), key=lambda kv: (-abs(kv[1]["notional"]), kv[0]))
    breaches: list[str] = [
        str(b)
        for e in journal.events
        if e.session == session and e.kind == "decision"
        for b in e.data.get("breaches", [])
    ]
    return {
        "session": session,
        "n_trades": len(ordered),
        "traded": sum(abs(t["notional"]) for _, t in ordered),
        "fees": fees,
        "breaches": breaches,
        "trades": [
            {
                "security": sid,
                "qty": int(t["qty"]),
                "notional": t["notional"],
                "why": _why(trial, cache, sid, decided),
            }
            for sid, t in ordered[:MAX_LISTED]
        ],
    }


def _why_text(why: dict[str, float | None]) -> str:
    if not why:
        return "rebalance"
    return ", ".join(f"{k} p{100 * v:.0f}" if v is not None else f"{k} n/a" for k, v in why.items())


def render_decisions(family: str, books: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        f"# pusula decisions: {family}",
        "",
        "Each trade with the security's percentile among index members on each of the trial's "
        "signals at the decision close. Derived from the books' journals.",
    ]
    for name, sessions in sorted(books.items()):
        for s in sessions:
            lines += [
                "",
                f"## {name}, {s['session']}: {s['n_trades']} trades, "
                f"{s['traded']:,.0f} traded, {s['fees']:,.0f} fees",
            ]
            lines += [f"> gate: {b}" for b in s["breaches"]]
            if s["trades"]:
                lines += ["", "| security | shares | notional | why |", "|---|---:|---:|---|"]
                lines += [
                    f"| {t['security']} | {t['qty']:+,d} | {t['notional']:+,.0f} | "
                    f"{_why_text(t['why'])} |"
                    for t in s["trades"]
                ]
                rest = s["n_trades"] - len(s["trades"])
                if rest > 0:
                    lines += ["", f"and {rest} smaller trades."]
    return "\n".join(lines) + "\n"


def write_decisions(
    root: Path, family: str, as_of: str, books: dict[str, list[dict[str, Any]]]
) -> Path | None:
    """One file per run that traded anything; a repeated run does not overwrite it with nothing."""
    if not any(s["n_trades"] for sessions in books.values() for s in sessions):
        return None
    path = root / "decisions" / f"{as_of}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, render_decisions(family, books))
    return path


def explain_security(
    spec: StudySpec, trial: Trial, journal: Journal, panel: Panel, security: str, last: int = 10
) -> str:
    """A book's whole history in one security, and where it stands on the trial's signals now."""
    if security not in panel.symbols:
        matches = [s for s in panel.symbols if s.split("@")[0] == security]
        if len(matches) != 1:
            return f"{security}: not a security of the panel (matches: {matches})"
        security = matches[0]
    cache = SignalCache(panel)
    lag = spec.backtest.lag
    held = 0
    fills: list[tuple[str, int, int, str]] = []
    for e in journal.events:
        if e.kind == "fill" and e.data["security"] == security:
            q = int(e.data["qty"])
            held += q
            why = _why_text(_why(trial, cache, security, panel.date_index(e.session) - lag))
            fills.append((e.session, q, held, why))
    lines = [f"{trial.name} / {security}: {held:+,d} shares held"]
    lines += [f"  {sess}  {q:+,d} -> {after:+,d}  ({why})" for sess, q, after, why in fills[-last:]]
    if len(fills) > last:
        lines.insert(1, f"  ... {len(fills) - last} earlier fills")
    if not fills:
        lines.append("  never traded by this book")
    row = panel.n_dates - 1
    lines.append(
        f"  now ({panel.dates[row]}): "
        + _why_text(_why(trial, cache, security, row))
        + (", index member" if panel["in_universe"][row, panel.symbol_index(security)] > 0 else "")
    )
    return "\n".join(lines)
