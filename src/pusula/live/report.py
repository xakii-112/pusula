"""Nightly operations report (markdown) from a forward snapshot.

What an operator needs each morning, in one page: every book's equity and P&L, exposure,
risk, the worst stress scenario, halts, and the data-quality line. Written next to the
dashboard snapshot by ``pusula forward``.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any


def _pct(x: Any, d: int = 2) -> str:
    return (
        "n/a"
        if x is None or (isinstance(x, float) and not math.isfinite(x))
        else f"{100 * x:.{d}f}%"
    )


MIN_COVERAGE = 0.5  # below it a signal is reported as missing for the book


def _iv_warning(snapshot: dict[str, Any], max_age_days: int = 7) -> list[str]:
    """A warning when implied vols are older than the signals accept (option-implied books
    then unwind, or fall back to their price-only half)."""
    return _stale(snapshot, "iv_last", max_age_days, "implied vols", "pusula data iv-update")


def _short_warning(snapshot: dict[str, Any], max_age_days: int = 4) -> list[str]:
    """A warning when FINRA short volume stopped (normally the last row holds the file of the
    session before; four calendar days cover a weekend and a holiday)."""
    return _stale(snapshot, "short_last", max_age_days, "short volume", "pusula data short-volume")


def _stale(snapshot: dict[str, Any], key: str, max_age: int, what: str, cmd: str) -> list[str]:
    last = snapshot.get("data", {}).get(key)
    if not last:
        return []
    age = (dt.date.fromisoformat(str(snapshot["as_of"])) - dt.date.fromisoformat(str(last))).days
    if age <= max_age:
        return []
    return [
        f"> **WARNING**: {what} last observed {last} ({age} days old): the signals that read "
        f"them are missing; run `{cmd}`.",
        "",
    ]


def _coverage_warnings(snapshot: dict[str, Any]) -> list[str]:
    """A warning per book trading without one of its registered signals (a composite then runs
    on the others; a single-signal book unwinds)."""
    out = []
    for name, b in sorted(snapshot["books"].items()):
        for sig, share in b.get("signal_coverage", {}).items():
            if share is None or not math.isfinite(share) or share < MIN_COVERAGE:
                out.append(
                    f"> **WARNING**: `{name}`: signal `{sig}` scores {_pct(share, 0)} of index "
                    "members on the last session; the book is trading without it."
                )
    return [*out, ""] if out else []


def _holding_warnings(snapshot: dict[str, Any]) -> list[str]:
    """A warning per book holding a name whose last bar looks wrong (reported, not acted on: the
    registered strategy decides what it decides)."""
    out = [
        f"> **WARNING**: `{name}` holds {sid}: {why} on the last session."
        for name, b in sorted(snapshot["books"].items())
        for sid, why in sorted(b.get("flagged_holdings", {}).items())
    ]
    return [*out, ""] if out else []


def _solver_warnings(snapshot: dict[str, Any]) -> list[str]:
    out = []
    for name, b in sorted(snapshot["books"].items()):
        if b.get("solver_failures"):
            out.append(
                f"> **WARNING**: `{name}`: the optimiser failed {b['solver_failures']} time(s) "
                "this run and the book kept its portfolio."
            )
    return [*out, ""] if out else []


def snapshot_warnings(snapshot: dict[str, Any]) -> list[str]:
    """The report's warnings as plain sentences, for the health check and its alerts. Flagged
    holdings are left out: one bad bar is held by many books, and the check reports it once."""
    lines = [
        *_iv_warning(snapshot),
        *_short_warning(snapshot),
        *_coverage_warnings(snapshot),
        *_solver_warnings(snapshot),
    ]
    return [ln.removeprefix("> **WARNING**: ") for ln in lines if ln]


def _coverage_table(snapshot: dict[str, Any]) -> list[str]:
    rows = [(n, b.get("signal_coverage")) for n, b in sorted(snapshot["books"].items())]
    rows = [(n, c) for n, c in rows if c]
    if not rows:
        return []
    lines = ["", "Signal coverage (index members with a score, last session):", ""]
    lines += ["| book | signals |", "|---|---|"]
    for n, c in rows:
        lines.append(f"| {n} | " + ", ".join(f"{k} {_pct(v, 1)}" for k, v in c.items()) + " |")
    return lines


def _trades_table(snapshot: dict[str, Any]) -> list[str]:
    rows = [
        (n, b["last_trades"])
        for n, b in sorted(snapshot["books"].items())
        if b.get("last_trades", {}).get("n_trades")
    ]
    if not rows:
        return []
    lines = ["", "Last session's trades (why each one: `decisions/`):", ""]
    lines += ["| book | trades | traded | fees | largest |", "|---|---:|---:|---:|---|"]
    for n, t in rows:
        top = ", ".join(f"{x['security']} {x['notional']:+,.0f}" for x in t["trades"][:3])
        lines.append(f"| {n} | {t['n_trades']} | {t['traded']:,.0f} | {t['fees']:,.0f} | {top} |")
    return lines


def render_daily(snapshot: dict[str, Any]) -> str:
    lines = [
        f"# pusula forward books: {snapshot['family']} as of {snapshot['as_of']}",
        "",
        *(
            [
                "> **REHEARSAL**: replayed from an earlier date into a separate directory; not a "
                "forward record.",
                "",
            ]
            if snapshot.get("rehearsal")
            else []
        ),
        *_iv_warning(snapshot),
        *_short_warning(snapshot),
        *_coverage_warnings(snapshot),
        *_holding_warnings(snapshot),
        *_solver_warnings(snapshot),
        f"Data: {snapshot['data']['quality']} (digest `{str(snapshot['data']['digest'])[:12]}`).",
        "",
        "| book | equity | day | since open | gross | net | ex-ante vol | VaR99 | worst stress "
        "| status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, b in sorted(snapshot["books"].items()):
        curve = b["equity_curve"]
        day = curve[-1][1] / curve[-2][1] - 1.0 if len(curve) > 1 else 0.0
        since = curve[-1][1] / curve[0][1] - 1.0 if curve else 0.0
        risk = b.get("risk", {})
        stress = sorted(b.get("stress", []), key=lambda s: s["pnl"])
        worst = f"{stress[0]['scenario']} {_pct(stress[0]['pnl'], 1)}" if stress else "n/a"
        status = f"HALTED ({b['kill_reason']})" if b.get("halted") else "running"
        lines.append(
            f"| {name} | {b['equity']:,.0f} | {_pct(day)} | {_pct(since)} | {b['gross']:.2f} | "
            f"{b['net']:+.2f} | {_pct(risk.get('ex_ante_vol'), 1)} | {_pct(risk.get('var99'))} | "
            f"{worst} | {status} |"
        )
    lines += _coverage_table(snapshot)
    lines += _trades_table(snapshot)
    return "\n".join(lines) + "\n"
