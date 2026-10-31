"""The forward record against the gate: the check RESEARCH_PROTOCOL §4 promises.

A strategy is eligible for capital only after its forward record passes the same gate on its own.
Nothing evaluated that until now. For each book of a family, from its journal:

* the record's statistics (excess-return Sharpe, Newey-West t, PSR, drawdown);
* how many sessions the record needs to mean anything: MinTRL at the Sharpe the holdout claimed;
* whether the record is consistent with that claim (z of the difference), QuantBot's
  "deviation from the model" on the quantity that matters;
* the Sharpe difference against the family's equal-weight book (paired stationary bootstrap);
* below ``MIN_SESSIONS`` no verdict at all; above it the registered gate, with the deflated
  Sharpe counting every trial of the programme. Lag and cost stresses need the research engine,
  so they come from ``--stress``; without it they are missing, and a missing check fails.

``--stress`` also re-runs the research engine over the forward window and reports how far the
paper book drifted from it (the parity test's measure, on real data).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.core.money import to_dollars
from pusula.data.settings import atomic_write_text
from pusula.engine.attribution import FACTORS, factor_attribution
from pusula.engine.backtest import BacktestResult
from pusula.engine.journal import Journal
from pusula.live.signal_record import render_signal_record
from pusula.research.gates import GateSpec, evaluate
from pusula.research.registry import Registry
from pusula.research.study import StudySpec, json_safe, metrics, programme_srs
from pusula.stats import sharpe as S
from pusula.stats.bootstrap import sharpe_difference
from pusula.stats.pbo import pbo_cscv

MIN_SESSIONS = 63  # about a quarter: below it the gate's statistics are noise
BENCHMARK_BOOK = "equal_weight"


def book_returns(journal: Journal) -> pd.Series:
    """Daily returns of a book from its journal's marks (a book takes capital once, at open)."""
    curve = journal.equity_by_session()
    if len(curve) < 2:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([s for s, _ in curve])
    eq = np.array([to_dollars(v) for _, v in curve])
    return pd.Series(eq[1:] / eq[:-1] - 1.0, index=idx[1:])


def _positive_share(r: pd.Series) -> float:
    if r.empty:
        return math.nan
    years = (1.0 + r).groupby(pd.DatetimeIndex(r.index).year).prod() - 1.0
    return float((years > 0).mean())


def _sessions_needed(holdout: Mapping[str, Any]) -> float:
    """MinTRL for PSR(0) >= 0.95 if the true Sharpe is what the holdout measured."""
    sr, sk, ku = (holdout.get(k) for k in ("sr_daily", "skew", "kurt"))
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (sr, sk, ku)):
        return math.inf
    return S.min_track_record(float(sr), float(sk), float(ku))  # type: ignore[arg-type]


def review_family(
    spec: StudySpec,
    registry: Registry,
    root: Path,
    rf: pd.Series,
    factors: pd.DataFrame,
    programme: str,
    research: dict[str, dict[str, BacktestResult]] | None = None,
) -> dict[str, Any]:
    """Statistics and verdict of each book of the family; ``research`` is the engine's re-run
    over the forward window (``--stress``)."""
    results = registry.latest_results(spec.family)
    returns: dict[str, pd.Series] = {}
    for t in spec.trials:
        path = root / t.name / "journal.jsonl"
        if path.exists():
            returns[t.name] = book_returns(Journal(path))
    bench = returns.get(BENCHMARK_BOOK)
    n_trials = registry.program_trials(programme)
    srs = programme_srs(registry, programme, "")
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0

    pbo = math.nan
    usable = [
        t.name
        for t in spec.trials
        if not t.benchmark and len(returns.get(t.name, ())) >= MIN_SESSIONS
    ]
    if len(usable) >= 2:
        M = pd.concat([returns[n] for n in usable], axis=1).dropna()
        pbo = pbo_cscv(M.to_numpy(), n_blocks=10).pbo

    books: dict[str, Any] = {}
    for t in spec.trials:
        if t.name not in returns:
            continue
        r = returns[t.name]
        x = r - rf.reindex(r.index).ffill().fillna(0.0)
        m = metrics(r.to_numpy(), x.to_numpy()) if len(r) >= 4 else {"n": float(len(r))}
        holdout = results[t.name].body["metrics"].get("holdout", {}) if t.name in results else {}
        needed = _sessions_needed(holdout)
        book: dict[str, Any] = {
            "benchmark": t.benchmark,
            "first": str(r.index[0].date()) if len(r) else None,
            "last": str(r.index[-1].date()) if len(r) else None,
            "n": len(r),
            "record": m,
            "holdout_sharpe": holdout.get("sharpe"),
            "sessions_needed": needed,
        }
        sr_h, sr_f = holdout.get("sr_daily"), m.get("sr_daily")
        if isinstance(sr_h, (int, float)) and isinstance(sr_f, (int, float)) and len(r) >= 4:
            se = S.sr_std(float(sr_f), len(r), float(m["skew"]), float(m["kurt"]))
            book["vs_holdout_z"] = (float(sr_f) - float(sr_h)) / se if se > 0 else math.nan
        if bench is not None and not t.benchmark and len(r) >= 4:
            a, b = r.align(bench, join="inner")
            d = sharpe_difference(a.to_numpy(), b.to_numpy())
            book["vs_benchmark"] = {
                "diff": d.diff,
                "lo": d.lo,
                "hi": d.hi,
                "p_better": d.p_better,
                "n": d.n,
            }
        if research is not None and t.name in research:
            book["drift"] = _drift(r, research[t.name]["base"])
        if t.benchmark:
            book["status"] = "benchmark"
        elif len(r) < MIN_SESSIONS:
            book["status"] = "insufficient"
        else:
            book.update(
                _verdict(
                    spec.gates,
                    r,
                    x,
                    m,
                    factors,
                    n_trials,
                    var_sr,
                    pbo,
                    research,
                    t.name,
                    spec.factor_names or FACTORS,
                )
            )
        books[t.name] = book
    return {
        "family": spec.family,
        "min_sessions": MIN_SESSIONS,
        "stress_run": research is not None,
        "programme_trials": n_trials,
        "books": books,
    }


def _verdict(
    gates: GateSpec,
    r: pd.Series,
    x: pd.Series,
    m: Mapping[str, float],
    factors: pd.DataFrame,
    n_trials: int,
    var_sr: float,
    pbo: float,
    research: dict[str, dict[str, BacktestResult]] | None,
    name: str,
    factor_names: tuple[str, ...] = FACTORS,
) -> dict[str, Any]:
    published = r[r.index.isin(factors.dropna().index)]  # the French library lags by months
    index = pd.DatetimeIndex(published.index)
    fa = factor_attribution(published.to_numpy(), index.to_numpy(), factors, factor_names)
    stressed = research.get(name, {}) if research is not None else {}

    def stress(variant: str, key: str) -> float:
        res = stressed.get(variant)
        if res is None:
            return math.nan
        return float(metrics(res.returns, res.excess_returns)[key])

    v = evaluate(
        gates,
        dsr=S.dsr(x.to_numpy(), max(n_trials, 1), var_sr),
        t_nw=float(m["t_nw"]),
        alpha_t=fa.alpha_t,
        sharpe=float(m["sharpe"]),
        sharpe_lag2=stress("lag2", "sharpe"),
        t_nw_cost2=stress("cost2", "t_nw"),
        pbo=pbo,
        max_dd=float(m["max_drawdown"]),
        positive_year_share=_positive_share(r),
    )
    return {"status": "passes" if v.promoted else "fails", "verdict": v.as_dict()}


def _drift(book: pd.Series, res: BacktestResult) -> dict[str, float]:
    """How far the paper book drifted from the research engine, day by day."""
    eng = pd.Series(res.returns, index=pd.DatetimeIndex(res.dates.astype("datetime64[ns]")))
    a, b = book.align(eng, join="inner")
    if len(a) < 4:
        return {"n": float(len(a))}
    d = (a - b).to_numpy()
    return {
        "n": float(len(a)),
        "mean_bp": float(np.mean(d) * 1e4),
        "max_abs_bp": float(np.max(np.abs(d)) * 1e4),
        "corr": float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
        if np.std(a.to_numpy()) > 0 and np.std(b.to_numpy()) > 0
        else math.nan,
    }


def stress_spec(spec: StudySpec, first_session: str) -> StudySpec:
    """The family's backtest over the forward window, as the books run it (kill switch on)."""
    return replace(
        spec,
        backtest=replace(spec.backtest, start=first_session, end=None, kill_switch=True),
    )


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render_review(rev: dict[str, Any]) -> str:
    lines = [
        f"# pusula forward record: {rev['family']}",
        "",
        f"A verdict needs {rev['min_sessions']} sessions, and capital needs a pass of every check "
        f"(deflated Sharpe over {rev['programme_trials']} trials). Lag and cost stresses "
        + ("were run." if rev["stress_run"] else "were not run (`--stress`), so they fail."),
        "",
        "| book | sessions | needed | Sharpe | NW t | max DD | holdout Sharpe | z vs holdout "
        "| vs equal weight (95% CI) | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, b in sorted(rev["books"].items()):
        m = b["record"]
        vb = b.get("vs_benchmark")
        versus = f"{_f(vb['diff'])} [{_f(vb['lo'])}, {_f(vb['hi'])}]" if vb else ""
        status = b["status"]
        if status == "fails":
            status = "fails: " + ", ".join(b["verdict"]["failed"])
        lines.append(
            f"| {name} | {b['n']} | {_f(b['sessions_needed'], '{:.0f}')} | {_f(m.get('sharpe'))} "
            f"| {_f(m.get('t_nw'))} | {_f(m.get('max_drawdown'), '{:.1%}')} "
            f"| {_f(b.get('holdout_sharpe'))} | {_f(b.get('vs_holdout_z'))} | {versus} | {status} |"
        )
    drift = [(n, b["drift"]) for n, b in sorted(rev["books"].items()) if b.get("drift")]
    if drift:
        lines += [
            "",
            "Paper book against the research engine over the forward window:",
            "",
            "| book | sessions | mean bp/day | max bp/day | correlation |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {n} | {d['n']:.0f} | {_f(d.get('mean_bp'))} | {_f(d.get('max_abs_bp'))} "
            f"| {_f(d.get('corr'), '{:.4f}')} |"
            for n, d in drift
        ]
    if rev.get("signals"):
        lines += render_signal_record(rev["signals"])
    return "\n".join(lines) + "\n"


def write_review(rev: dict[str, Any], root: Path) -> None:
    atomic_write_text(root / "review.md", render_review(rev))
    atomic_write_text(root / "review.json", json.dumps(json_safe(rev), indent=1))
