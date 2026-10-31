"""Render a family's results from the ledger (the ledger is the source of truth; this file is
generated and must not be edited by hand)."""

from __future__ import annotations

import math
from typing import Any

from pusula.research.registry import Registry


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return fmt.format(x)


NOTES = {
    "us-equity-v1": "All returns are net of the calibrated cost model (spread, square-root impact, fees, "
    "borrow, financing) with execution at the next session's VWAP. Research backtests run with "
    "the latching production kill switch (see us-equity-v2).",
    "us-equity-v2": "All returns are net of the calibrated cost model (spread, square-root impact, fees, "
    "borrow, financing) with execution at the next session's VWAP. Identical trials to "
    "us-equity-v1; research backtests without the latching kill switch.",
    "us-options-v1": "Calls are bought at the close one session after the signal, priced with "
    "Black-Scholes-Merton at the observed 30-day ATM implied volatility plus a 2.5% half-spread on "
    "premium, and held to expiry (payoff exact). The stock mirror pays 5 bp per side.",
}


def _programme(family: str) -> str:
    from pusula.research.families import PROGRAMME_OVERRIDE

    if family in PROGRAMME_OVERRIDE:
        return PROGRAMME_OVERRIDE[family]
    head, sep, tail = family.rpartition("-")
    return head + sep if sep and tail.startswith("v") else family


def render(registry: Registry, family: str) -> str:
    pre = {e.body["name"]: e for e in registry.preregs(family)}
    res = registry.latest_results(family)
    lines = [
        f"# Results: `{family}`",
        "",
        "> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.",
        f"> Registered trials in this family: **{registry.family_trials(family)}**; deflated-Sharpe "
        f"trial count (whole programme `{_programme(family)}`): "
        f"**{registry.program_trials(_programme(family))}**. Sharpe, t and DSR use excess returns.",
        "",
    ]
    if not res:
        return "\n".join([*lines, "No results recorded yet."]) + "\n"
    any_res = next(iter(res.values()))
    w = pre[any_res.body["name"]].body["windows"]
    lines += [
        f"Development window {w['dev'][0]} .. {w['dev'][1]}; holdout {w['holdout'][0]} .. {w['holdout'][1]}. "
        + NOTES.get(family, ""),
        "",
        "## Holdout summary",
        "",
        "| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | "
        "turnover/yr | promoted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, e in res.items():
        m = e.body["metrics"]
        h = m["holdout"]
        fa = m["factor"]
        v = e.body["verdict"]
        prom = (
            "benchmark"
            if v.get("benchmark")
            else ("**yes**" if v.get("promoted") else "no: " + ", ".join(v.get("failed", [])))
        )
        lines.append(
            f"| {name} | {_f(h.get('sharpe'))} | {_f(h.get('ann_return'), '{:.1%}')} | {_f(h.get('ann_vol'), '{:.1%}')} | "
            f"{_f(h.get('max_drawdown'), '{:.1%}')} | {_f(h.get('t_nw'))} | {_f(fa.get('alpha_annual'), '{:.1%}')} "
            f"({_f(fa.get('alpha_t'))}) | {_f(m.get('dsr'))} | {_f(m.get('lag2_holdout', {}).get('sharpe'))} | "
            f"{_f(m.get('cost2_holdout', {}).get('t_nw'))} | {_f(m['summary'].get('turnover_annual'), '{:.1f}')} | {prom} |"
        )
    lines += [
        "",
        "## Development window",
        "",
        "| trial | Sharpe | ann. ret | vol | max DD | NW t |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, e in res.items():
        d = e.body["metrics"]["dev"]
        lines.append(
            f"| {name} | {_f(d.get('sharpe'))} | {_f(d.get('ann_return'), '{:.1%}')} | "
            f"{_f(d.get('ann_vol'), '{:.1%}')} | {_f(d.get('max_drawdown'), '{:.1%}')} | {_f(d.get('t_nw'))} |"
        )
    # the factors the family's alpha check used; FF5+MOM unless it says otherwise
    measured = [list(e.body["metrics"]["factor"].get("betas", {})) for e in res.values()]
    ff = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    names = next((m for m in measured if m), ff)
    names = ff if set(names) == set(ff) else names  # the ledger's canonical form sorts its keys
    lines += [
        "",
        "## Factor exposures (holdout, HAC)",
        "",
        "| trial | " + " | ".join(n.removesuffix("_rf") for n in names) + " | R2 |",
        "|---|" + "---:|" * (len(names) + 1),
    ]
    for name, e in res.items():
        fa = e.body["metrics"]["factor"]
        b, t = fa.get("betas", {}), fa.get("tstats", {})
        cells = " | ".join(f"{_f(b.get(k))} ({_f(t.get(k), '{:.1f}')})" for k in names)
        lines.append(f"| {name} | {cells} | {_f(fa.get('r2'))} |")
    lines += ["", "## Calendar years (net)", ""]
    years = sorted({y for e in res.values() for y in e.body["metrics"]["yearly"]})
    lines += ["| trial | " + " | ".join(years) + " |", "|---|" + "---:|" * len(years)]
    for name, e in res.items():
        yt = e.body["metrics"]["yearly"]
        lines.append(
            f"| {name} | "
            + " | ".join(_f(yt.get(y, {}).get("net"), "{:.1%}") for y in years)
            + " |"
        )
    lines += ["", "## Gate detail", ""]
    for name, e in res.items():
        v = e.body["verdict"]
        if v.get("benchmark"):
            continue
        checks = ", ".join(
            f"{c['name']} {_f(c['value'])}{'✓' if c['passed'] else '✗'}"
            for c in v.get("checks", [])
        )
        lines.append(f"* **{name}**: {checks}")
    lines += ["", "## Provenance", ""]
    for name, e in res.items():
        b = e.body
        lines.append(
            f"* {name}: prereg `{b['prereg'][:12]}`, result `{e.hash[:12]}`, data `{b['data_digest'][:12]}`, "
            f"code `{b['code_digest'][:12]}`, {b['created_at']}"
            + (f", rerun: {b['rerun_reason']}" if b.get("rerun_reason") else "")
        )
    return "\n".join(lines) + "\n"
