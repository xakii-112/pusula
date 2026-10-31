"""``us-options-v1``: is the trend signal better expressed with listed calls than with stock?

Trials share one selection rule (positive multi-horizon trend; ties broken by 12-month return,
never by column order: NihaiAlgoritma's live book was once filled alphabetically, BULGU-101-1)
and differ only in how the view is expressed. The stock mirror holds the same names with the same
delta-dollar exposure on the same schedule and is the benchmark.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from pusula.alpha import ops
from pusula.alpha.signals import Momentum, TimeSeriesMomentum
from pusula.data.panel import FloatArray, Panel
from pusula.engine.attribution import factor_attribution
from pusula.options.overlay import CallOverlaySpec, OverlayResult, run_call_overlay
from pusula.research.gates import GateSpec, evaluate
from pusula.research.registry import Registry
from pusula.research.study import json_safe, metrics
from pusula.stats import sharpe as S
from pusula.stats.pbo import pbo_cscv

FAMILY = "us-options-v1"
DEV = ("2019-03-01", "2022-12-30")
HOLDOUT = ("2023-01-03", "2026-09-25")
START, END = "2019-03-01", "2026-09-25"
STOCK_COST = 0.0005  # one-way, stock legs (spread + impact for an S&P name, monthly size)


@dataclass(frozen=True, slots=True)
class OptionTrial:
    name: str
    hypothesis: str
    overlay: CallOverlaySpec
    stock_mirror: bool = False

    @property
    def benchmark(self) -> bool:
        return self.stock_mirror


TRIALS = (
    OptionTrial(
        "calls_d50_t21",
        "ATM (delta 0.5) one-month calls on positive-trend names beat the stock "
        "mirror on risk-adjusted terms (convexity outweighs premium decay and spread).",
        CallOverlaySpec(tenor=21, target_delta=0.5),
    ),
    OptionTrial(
        "calls_d30_t21",
        "Out-of-the-money (delta 0.3) one-month calls do better than ATM.",
        CallOverlaySpec(tenor=21, target_delta=0.3),
    ),
    OptionTrial(
        "calls_d50_t42",
        "Two-month ATM calls (less theta per day, fewer spreads paid) do better.",
        CallOverlaySpec(tenor=42, target_delta=0.5),
    ),
    OptionTrial(
        "stock_mirror_t21",
        "Benchmark: the same picks held as stock, monthly.",
        CallOverlaySpec(tenor=21),
        stock_mirror=True,
    ),
)


def selection_score(panel: Panel) -> FloatArray:
    trend = TimeSeriesMomentum().compute(panel)
    tie = ops.cs_rank(Momentum(252, 0).compute(panel), panel.get("in_universe"))
    return trend + 1e-3 * np.where(np.isfinite(tie), tie, 0.0)


def run_stock_mirror(
    panel: Panel,
    score: FloatArray,
    iv: FloatArray,
    rf: FloatArray,
    start: int,
    end: int,
    spec: CallOverlaySpec,
    initial_equity: float = 10_000_000.0,
) -> OverlayResult:
    """Same names, same schedule as the call overlay, held as stock at ``gross_delta`` exposure."""
    raw_close = panel["close"]
    close = ops.ffill(raw_close)
    member = panel["in_universe"] > 0
    eq = np.full(end - start + 1, np.nan)
    npos = np.zeros(eq.size, dtype=np.int64)
    cash = initial_equity
    shares: dict[int, float] = {}
    next_roll = start + 1
    for e in range(start, end + 1):
        if e > start and math.isfinite(rf[e]):
            cash *= 1.0 + float(rf[e])
        if e == next_roll:
            # liquidate, then re-select exactly like the overlay (signal at e-1, IV must exist)
            for j, q in shares.items():
                cash += q * float(close[e, j]) * (1 - STOCK_COST)
            shares = {}
            if e + spec.tenor <= end:
                d = e - 1
                s = np.where(
                    member[d]
                    & np.isfinite(score[d])
                    & np.isfinite(iv[e])
                    & np.isfinite(raw_close[e]),
                    score[d],
                    -np.inf,
                )
                pick = [int(j) for j in np.argsort(-s)[: spec.max_names] if s[j] > 0]
                if pick:
                    per = spec.gross_delta * cash / len(pick)
                    for j in pick:
                        q = per / float(close[e, j])
                        shares[j] = q
                        cash -= q * float(close[e, j]) * (1 + STOCK_COST)
                next_roll = e + spec.tenor
            else:
                next_roll = end + 1
        mv = sum(q * float(close[e, j]) for j, q in shares.items())
        eq[e - start] = cash + mv
        npos[e - start] = len(shares)
    rets = np.zeros_like(eq)
    rets[1:] = eq[1:] / eq[:-1] - 1.0
    return OverlayResult(panel.dates[start : end + 1], eq, rets, np.zeros_like(eq), npos, [])


def register(registry: Registry, now: dt.datetime) -> dict[str, str]:
    return {
        t.name: registry.preregister(
            family=FAMILY,
            name=t.name,
            hypothesis=t.hypothesis,
            config={
                "overlay": asdict(t.overlay),
                "stock_mirror": t.stock_mirror,
                "benchmark": t.benchmark,
                "stock_cost": STOCK_COST,
                "start": START,
                "end": END,
            },
            gates=GateSpec().as_dict(),
            windows={"dev": list(DEV), "holdout": list(HOLDOUT)},
            now=now,
        )
        for t in TRIALS
    }


def _window(res: OverlayResult, lo: str, hi: str) -> np.ndarray:
    m = (res.dates >= np.datetime64(lo)) & (res.dates <= np.datetime64(hi))
    return res.returns[m]


def run(
    registry: Registry,
    panel: Panel,
    iv: FloatArray,
    rf: FloatArray,
    factors: pd.DataFrame,
    data_digest: str,
    now: dt.datetime,
    rerun_reason: str | None = None,
    code_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    hashes = {e.body["name"]: e.hash for e in registry.preregs(FAMILY)}
    if any(t.name not in hashes for t in TRIALS):
        raise RuntimeError("options family not pre-registered")
    score = selection_score(panel)
    t0 = int(np.searchsorted(panel.dates, np.datetime64(START)))
    t1 = int(np.searchsorted(panel.dates, np.datetime64(END), side="right")) - 1
    out: dict[str, Any] = {}
    runs: dict[str, dict[str, OverlayResult]] = {}
    rf_seg = np.where(np.isfinite(rf[t0 : t1 + 1]), rf[t0 : t1 + 1], 0.0)

    def xw(res: OverlayResult, lo: str, hi: str) -> np.ndarray:
        """Excess returns (over the T-bill rate) in the window: the basis of Sharpe/t/DSR."""
        x = res.returns - rf_seg
        x[0] = 0.0
        return x[(res.dates >= np.datetime64(lo)) & (res.dates <= np.datetime64(hi))]

    for t in TRIALS:
        fn = run_stock_mirror if t.stock_mirror else run_call_overlay
        base = fn(panel, score, iv, rf, t0, t1, t.overlay)
        lag = fn(panel, ops.lag(score, 1), iv, rf, t0, t1, t.overlay)
        dear = fn(
            panel,
            score,
            iv,
            rf,
            t0,
            t1,
            CallOverlaySpec(
                **{**asdict(t.overlay), "option_half_spread": 2 * t.overlay.option_half_spread}
            ),
        )
        runs[t.name] = {"base": base, "lag2": lag, "cost2": dear}
    cands = [t for t in TRIALS if not t.benchmark]
    M = np.column_stack([xw(runs[t.name]["base"], DEV[0], HOLDOUT[1]) for t in cands])
    pbo = pbo_cscv(M, n_blocks=10).pbo
    srs = np.array([S.sharpe(xw(runs[t.name]["base"], *HOLDOUT)) for t in cands])
    var_sr = float(np.var(srs, ddof=1))
    n_trials = registry.family_trials(FAMILY)
    for t in TRIALS:
        rr = runs[t.name]
        h = _window(rr["base"], *HOLDOUT)
        hd = rr["base"].dates[
            (rr["base"].dates >= np.datetime64(HOLDOUT[0]))
            & (rr["base"].dates <= np.datetime64(HOLDOUT[1]))
        ]
        fa = factor_attribution(h, hd, factors)
        years = (
            pd.Series(rr["base"].returns, index=pd.DatetimeIndex(rr["base"].dates))
            .groupby(lambda d: d.year)
            .apply(lambda x: float(np.prod(1 + x) - 1))
        )
        m: dict[str, Any] = {
            "dev": metrics(_window(rr["base"], *DEV), xw(rr["base"], *DEV)),
            "holdout": metrics(h, xw(rr["base"], *HOLDOUT)),
            "lag2_holdout": metrics(_window(rr["lag2"], *HOLDOUT), xw(rr["lag2"], *HOLDOUT)),
            "cost2_holdout": metrics(_window(rr["cost2"], *HOLDOUT), xw(rr["cost2"], *HOLDOUT)),
            "factor": {
                "alpha_annual": fa.alpha_annual,
                "alpha_t": fa.alpha_t,
                "betas": fa.betas,
                "tstats": fa.tstats,
                "r2": fa.r2,
            },
            "summary": {
                "avg_positions": float(np.mean(rr["base"].n_positions)),
                "premium_share_of_equity": float(
                    np.sum(rr["base"].premium_spent)
                    / np.mean(rr["base"].equity)
                    / (len(h) / 252 + 1e-9)
                )
                if not t.stock_mirror
                else 0.0,
                "mean_payoff_ratio": float(np.mean(rr["base"].payoff_ratio))
                if rr["base"].payoff_ratio
                else None,
                "turnover_annual": None,
            },
            "yearly": {str(k): {"net": float(v)} for k, v in years.items()},
            "dsr": None,
            "pbo": pbo,
        }
        if t.benchmark:
            verdict: dict[str, Any] = {"promoted": False, "benchmark": True}
        else:
            m["dsr"] = S.dsr(xw(rr["base"], *HOLDOUT), n_trials, var_sr)
            v = evaluate(
                GateSpec(),
                dsr=m["dsr"],
                t_nw=m["holdout"]["t_nw"],
                alpha_t=fa.alpha_t,
                sharpe=m["holdout"]["sharpe"],
                sharpe_lag2=m["lag2_holdout"]["sharpe"],
                t_nw_cost2=m["cost2_holdout"]["t_nw"],
                pbo=pbo,
                max_dd=m["holdout"]["max_drawdown"],
                positive_year_share=float((years > 0).mean()),
            )
            verdict = v.as_dict()
        registry.record_result(
            hashes[t.name],
            metrics=json_safe(m),
            verdict=json_safe(verdict),
            data_digest=data_digest,
            now=now,
            rerun_reason=rerun_reason,
            code_files=code_files,
        )
        out[t.name] = {"holdout": m["holdout"], "verdict": verdict}
    return out
