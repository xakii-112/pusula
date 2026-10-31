"""Forward (out-of-time) paper trading of every registered trial: the shadow books.

NihaiAlgoritma ran a separate paper "kasa" per mechanism; here every trial of a registered
family gets one, opened on the first session after its pre-registration, so the forward record
is out of sample by construction and cannot be back-filled. Each night:

1. the bar store is updated with the newly completed session(s);
2. the research panel is rebuilt;
3. each book resumes from ``state.json`` + its journal (the ledger is rebuilt by replay and the
   journal head must match the saved state) and processes every new session with the paper
   engine (same strategy object, same risk gate, same cost model as the research backtest);
4. a dashboard snapshot (equity, positions, risk, stress) is written for the web view.

Nothing here sends orders to a broker: ``pusula execute`` mirrors one book in an account
(``live.execute``), and the book stays the record whatever the account does.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.core.money import to_dollars
from pusula.data import factors as french
from pusula.data.panel import Panel
from pusula.data.pipeline import ResearchData, is_crypto, research_panel
from pusula.data.settings import atomic_write_text, data_dir, home
from pusula.engine.backtest import Backtester
from pusula.engine.journal import Journal
from pusula.engine.paper import PaperEngine
from pusula.live.decisions import SignalCache, session_decisions, trial_signals, write_decisions
from pusula.live.report import render_daily
from pusula.research.study import StudySpec, make_strategy
from pusula.risk.measures import HISTORICAL_SCENARIOS, historical_var, scenario_pnl


@dataclass(frozen=True, slots=True)
class BookPaths:
    root: Path

    @property
    def journal(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "state.json"


def books_root(family: str) -> Path:
    p = home() / "forward" / family
    p.mkdir(parents=True, exist_ok=True)
    return p


def rf_series(panel: Panel) -> np.ndarray:
    """Daily risk-free rate on the panel's sessions; sessions after the last published French
    print carry the last published rate (known at the time), never a future one."""
    f = french.fetch(data_dir() / "factors")
    rf = f["rf"].reindex(pd.DatetimeIndex(panel.dates.astype("datetime64[ns]"))).ffill()
    return rf.fillna(0.0).to_numpy(dtype=np.float64)


def first_session_after(panel: Panel, when: dt.datetime) -> int:
    d = np.datetime64(when.date(), "D")
    return int(np.searchsorted(panel.dates, d, side="right"))


def step_books(
    spec: StudySpec,
    registered_at: dict[str, dt.datetime],
    rd: ResearchData | None = None,
    log: bool = True,
    root: Path | None = None,
    rehearsal: bool = False,
) -> dict[str, Any]:
    rd = rd or research_panel()
    panel = rd.panel
    rf = rf_series(panel)
    root = root or books_root(spec.family)
    snapshot: dict[str, Any] = {
        "rehearsal": rehearsal,
        "family": spec.family,
        "as_of": str(panel.dates[-1]),
        "books": {},
        "data": {
            "quality": rd.quality.summary(),
            "digest": rd.data_digest,
            "iv_last": iv_last_date(panel),
            "short_last": field_last_date(panel, "short_volume") if _reads_short(spec) else None,
        },
    }
    cache = SignalCache(panel)
    coverage = signal_coverage(spec, panel, cache)
    # a 25% day is a data question for a stock and an ordinary day for an altcoin
    flags = last_bar_flags(panel, 1.0 if is_crypto(spec.universe) else 0.25)
    decisions: dict[str, list[dict[str, Any]]] = {}
    for trial in spec.trials:
        paths = BookPaths(root / trial.name)
        paths.root.mkdir(parents=True, exist_ok=True)
        cfg = spec.backtest
        from dataclasses import replace

        # production books always carry the latching kill switch, whatever the research config
        cfg = replace(cfg, end=None, start=None, kill_switch=True)
        bt = Backtester(panel, cfg, rf)
        journal = Journal(paths.journal)
        eng = PaperEngine(
            bt,
            make_strategy(trial, cfg.cost_model),
            trial.name,
            journal=journal,
            initial_capital=cfg.initial_equity,
        )
        first = first_session_after(panel, registered_at[trial.name])
        if paths.state.exists():
            eng.restore(json.loads(paths.state.read_text(encoding="utf-8")))
            nxt = eng.last_step + 1
        else:
            nxt = first
        last = panel.n_dates - 1
        for e in range(nxt, last + 1):
            eng.step(e)
        if eng.opened:
            atomic_write_text(paths.state, json.dumps(eng.control_state(), indent=1))
            snapshot["books"][trial.name] = book_snapshot(eng, panel, rd)
            book = snapshot["books"][trial.name]
            if trial.name in coverage:
                book["signal_coverage"] = coverage[trial.name]
            book["flagged_holdings"] = {s: flags[s] for s in eng.ledger.positions if s in flags}
            book["solver_failures"] = int(getattr(eng.strategy, "solver_failures", 0))
            book["last_trades"] = session_decisions(
                journal, str(panel.dates[eng.last_step]), trial, cache, cfg.lag
            )
            decisions[trial.name] = [
                session_decisions(journal, str(panel.dates[e]), trial, cache, cfg.lag)
                for e in range(nxt, last + 1)
            ]
        if log:
            print(
                f"[forward] {trial.name}: processed {max(0, last - nxt + 1)} new session(s)",
                flush=True,
            )
    atomic_write_text(
        root / "dashboard.json", json.dumps(snapshot, indent=1, default=_json_default)
    )
    if snapshot["books"]:
        atomic_write_text(root / "daily_report.md", render_daily(snapshot))
        write_decisions(root, spec.family, snapshot["as_of"], decisions)
    return snapshot


def _json_default(o: object) -> object:
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def book_snapshot(eng: PaperEngine, panel: Panel, rd: ResearchData) -> dict[str, Any]:
    bt = eng.bt
    e = eng.last_step
    led = eng.ledger
    marks = {s: round(float(bt.close_ff[e, panel.symbol_index(s)]) * 1e6) for s in led.positions}
    eq = to_dollars(led.equity(marks))
    curve = [[str(panel.dates[k]), to_dollars(v)] for k, v in sorted(eng.equity_curve.items())]
    w = np.zeros(panel.n_symbols)
    for s, pos in led.positions.items():
        w[panel.symbol_index(s)] = pos.qty * to_dollars(marks[s]) / eq if eq else 0.0
    risk = bt.risk_at(e)
    out: dict[str, Any] = {
        "equity": eq,
        "equity_curve": curve,
        "gross": float(np.abs(w).sum()),
        "net": float(w.sum()),
        "n_positions": len(led.positions),
        "halted": eng.halted,
        "kill_reason": eng.kill.reason,
        "realised": to_dollars(led.realised),
        "fees": to_dollars(led.fees),
        "financing": to_dollars(led.financing),
    }
    top = np.argsort(-np.abs(w))[:25]
    out["top_positions"] = [
        {
            "security": panel.symbols[j],
            "weight": float(w[j]),
            "sector": rd.sectors.get(panel.symbols[j], "Unknown"),
        }
        for j in top
        if w[j] != 0
    ]
    if risk is not None and np.any(w != 0):
        wr = w[risk.cols]
        mw = np.full(risk.n, 1.0 / risk.n)
        beta = float(wr @ risk.beta(mw))
        R = bt.returns[max(0, e - 499) : e + 1, risk.cols]
        var = historical_var(wr, R, risk)
        out["risk"] = {
            "ex_ante_vol": risk.vol(wr),
            "beta": beta,
            "var95": var.var95,
            "var99": var.var99,
            "es975": var.es975,
            "parametric_var99": var.parametric_var99,
            "outside_model_weight": float(np.abs(w).sum() - np.abs(wr).sum()),
        }
        stress = []
        mret = bt.returns[:, :]
        for sc in HISTORICAL_SCENARIOS:
            lo = int(np.searchsorted(panel.dates, np.datetime64(sc.start, "D")))
            hi = int(np.searchsorted(panel.dates, np.datetime64(sc.end, "D"), side="right")) - 1
            if hi <= lo:
                continue
            with np.errstate(invalid="ignore"):
                cum = (
                    np.prod(
                        1.0
                        + np.where(np.isfinite(mret[lo + 1 : hi + 1]), mret[lo + 1 : hi + 1], 0.0),
                        axis=0,
                    )
                    - 1.0
                )
            seen = np.isfinite(bt.close[lo]) & np.isfinite(bt.close[hi])
            cum = np.where(seen, cum, np.nan)
            mkt = float(np.nanmean(cum[seen])) if seen.any() else 0.0
            betas = np.full(panel.n_symbols, 1.0)
            betas[risk.cols] = risk.beta(mw)
            stress.append({"scenario": sc.name, "pnl": scenario_pnl(w, cum, betas, mkt)})
        out["stress"] = stress
    return out


def iv_last_date(panel: Panel) -> str | None:
    """Last session with any implied-vol observation (None without the field)."""
    return field_last_date(panel, "iv")


def field_last_date(panel: Panel, name: str) -> str | None:
    """Last session on which the field has any value (None without the field)."""
    if name not in panel:
        return None
    rows = np.nonzero(np.isfinite(panel[name]).any(axis=1))[0]
    return str(panel.dates[rows[-1]]) if rows.size else None


def _reads_short(spec: StudySpec) -> bool:
    from pusula.alpha.signals import SHORT_SIGNALS

    return any(s.name in SHORT_SIGNALS for t in spec.trials for s in t.signals)


def signal_coverage(
    spec: StudySpec, panel: Panel, cache: SignalCache | None = None
) -> dict[str, dict[str, float]]:
    """Share of index members with a score on the panel's last session, per optimised trial and
    signal. A composite averages the signals that exist, so a stopped feed does not fail a book:
    it quietly trades on the remaining signals. This makes that visible in the daily report."""
    cache = cache or SignalCache(panel)
    member = panel["in_universe"][-1] > 0
    out: dict[str, dict[str, float]] = {}
    for t in spec.trials:
        cov: dict[str, float] = {}
        for s in trial_signals(t):
            x = cache.raw(s)[-1]
            share = float(np.isfinite(x[member]).mean()) if member.any() else math.nan
            label = s.name if s.name not in cov else f"{s.name} {dict(s.params)}"
            cov[label] = share
        if cov:
            out[t.name] = cov
    return out


def last_bar_flags(panel: Panel, max_move: float = 0.25) -> dict[str, str]:
    """Securities whose last session looks wrong: no bar, or a move beyond ``max_move``. QuantBot
    refused to trade such a name; a forward book must keep running its registered strategy, so
    here they are reported for the names a book holds."""
    if panel.n_dates < 2:
        return {}
    c = panel["close"]
    out: dict[str, str] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        move = c[-1] / c[-2] - 1.0
    for j, sid in enumerate(panel.symbols):
        if math.isfinite(c[-2, j]) and not math.isfinite(c[-1, j]):
            out[sid] = "no bar"
        elif math.isfinite(move[j]) and abs(move[j]) > max_move:
            out[sid] = f"a move of {move[j]:+.0%}"
    return out
