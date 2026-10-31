"""Is the signal still there? The information coefficient of each signal since registration.

QuantForge kept a ledger of predictions and scored them as their outcomes matured. A forward book
mixes the signal with its costs and its optimiser, so a poor record does not say which of them
failed; this scores each signal on its own, against the holdout it was registered on. Signals are
causal by test (truncate the future, the past must not move), so the record is recomputed from the
panel, and it needs no state of its own. A row counts only once its label has matured.

It is not a shortcut to a verdict: an IC of 0.02 takes years to tell from zero, as a Sharpe does,
and ``sessions_needed`` says how many. No gate reads this.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pusula.alpha.evaluate import ICSummary, daily_ic, forward_returns, summarize_ic
from pusula.data.panel import FloatArray, Panel
from pusula.live.decisions import SignalCache, trial_signals
from pusula.research.study import StudySpec

MIN_NAMES = 10  # a family on 18 funds has no more than that
T_TARGET = 2.0


def _z(forward: ICSummary, holdout: ICSummary) -> float:
    """Forward IC against the holdout's, in standard errors of the forward mean."""
    if not (math.isfinite(forward.t_nw) and forward.t_nw != 0.0):
        return math.nan
    return (forward.mean - holdout.mean) / abs(forward.mean / forward.t_nw)


def _needed(holdout: ICSummary) -> float:
    """Sessions until the mean is ``T_TARGET`` standard errors from zero, if the holdout is true.
    Infinite for a signal with no positive holdout edge: there is nothing to confirm."""
    ir = holdout.ir_annual
    return 252.0 * (T_TARGET / ir) ** 2 if math.isfinite(ir) and ir > 0.0 else math.inf


def _summary_row(
    label: str, h: int, ic: FloatArray, n_names: FloatArray, holdout: tuple[int, int], first: int
) -> dict[str, Any]:
    past = summarize_ic(ic[holdout[0] : holdout[1] + 1], n_names[holdout[0] : holdout[1] + 1], h)
    now = summarize_ic(ic[first:], n_names[first:], h)
    return {
        "signal": label,
        "horizon": h,
        "holdout_ic": past.mean,
        "holdout_t": past.t_nw,
        "forward_n": now.n_days,
        "forward_ic": now.mean,
        "forward_t": now.t_nw,
        "z_vs_holdout": _z(now, past),
        "sessions_needed": _needed(past),
    }


def signal_record(
    spec: StudySpec, panel: Panel, first: int, cache: SignalCache | None = None
) -> list[dict[str, Any]]:
    """One row per distinct (signal, horizon) of the family's candidates. ``first`` is the first
    row after registration. A signal whose data the panel lacks is listed as unavailable."""
    cache = cache or SignalCache(panel)
    lag = spec.backtest.lag
    price = "vwap" if "vwap" in panel else "open"
    mask = panel["in_universe"]
    holdout = (panel.date_index(spec.holdout[0]), panel.asof_index(spec.holdout[1]))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for t in spec.trials:
        if t.benchmark:
            continue
        h = max(1, round(t.horizon_days))
        for s in trial_signals(t):
            params = dict(s.params)
            label = s.name + (f" {params}" if params else "")
            if (label, h) in seen:
                continue
            seen.add((label, h))
            try:
                raw = cache.raw(s)
            except KeyError as exc:
                rows.append({"signal": label, "horizon": h, "unavailable": str(exc)})
                continue
            ic = daily_ic(raw, forward_returns(panel, h, lag, price), mask, min_names=MIN_NAMES)
            n_names = (np.isfinite(raw) & (mask > 0)).sum(axis=1).astype(np.float64)
            rows.append(_summary_row(label, h, ic, n_names, holdout, first))
    return rows


def _f(x: Any, fmt: str = "{:.3f}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render_signal_record(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "Signals on their own, rank IC against the return over the trial's horizon (no costs, no "
        "optimiser; not a gate input). Only matured labels count.",
        "",
        "| signal | horizon | holdout IC | holdout t | matured | forward IC | forward t "
        "| z vs holdout | sessions for t = 2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "unavailable" in r:
            lines.append(f"| {r['signal']} | {r['horizon']} | no data: {r['unavailable']} |||||||")
            continue
        lines.append(
            f"| {r['signal']} | {r['horizon']} | {_f(r['holdout_ic'])} "
            f"| {_f(r['holdout_t'], '{:.2f}')} "
            f"| {r['forward_n']} | {_f(r['forward_ic'])} | {_f(r['forward_t'], '{:.2f}')} "
            f"| {_f(r['z_vs_holdout'], '{:.2f}')} | {_f(r['sessions_needed'], '{:.0f}')} |"
        )
    return lines
