"""``us-optquote-v1``: option strategies at the market's own quotes.

``us-options-v1`` asked whether calls beat stock, at one ATM volatility and an assumed spread. This
asks it again where those assumptions are gone (``options.quoted``), and adds what a full chain
makes possible to test at all: selling the variance risk premium (Carr & Wu 2009), writing
cash-secured puts and covered calls, and Goyal & Saretto's (2009) straddles on the gap between
implied and realised volatility, which needs a strike-level book on both sides. Written and
registered before any of them was run on these quotes. It pays through the "us-" programme.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.alpha import ops
from pusula.core.errors import RegistryError
from pusula.core.hashing import content_hash
from pusula.data.optchain import read_member_chain
from pusula.data.panel import FloatArray, Panel
from pusula.engine.attribution import factor_attribution
from pusula.options.quoted import QuotedResult, QuotedSpec, run_quoted, stressed
from pusula.research.gates import GateSpec, evaluate
from pusula.research.options_study import selection_score
from pusula.research.registry import Registry
from pusula.research.study import json_safe, metrics, programme_srs
from pusula.stats import sharpe as S
from pusula.stats.pbo import pbo_cscv

FAMILY = "us-optquote-v1"
DEV = ("2019-06-03", "2022-12-30")
HOLDOUT = ("2023-01-03", "2026-10-22")
RV_WINDOW = 21
CHAIN_DTE = (5, 60)  # what the runs read of the chain


@dataclass(frozen=True, slots=True)
class QuotedTrial:
    name: str
    hypothesis: str
    spec: QuotedSpec


TRIALS = (
    QuotedTrial(
        "calls_quoted",
        "One-month calls near delta 0.5 on positive-trend names, bought at the ask, earn net alpha "
        "beyond FF5+MOM: us-options-v1's question at real prices.",
        QuotedSpec("long_call", target_delta=0.5),
    ),
    QuotedTrial(
        "put_write",
        "Writing cash-secured one-month 20-delta puts at the bid, on the 50 tightest markets among "
        "index members, earns the variance risk premium net of spreads and fees, with alpha beyond "
        "FF5+MOM (Carr & Wu 2009; the option-writing indices).",
        QuotedSpec("short_put", target_delta=0.2),
    ),
    QuotedTrial(
        "covered_call",
        "Holding the stock and writing a one-month 30-delta call against it earns more per unit of "
        "risk than holding the stock (Whaley 2002).",
        QuotedSpec("covered_call", target_delta=0.3),
    ),
    QuotedTrial(
        "ivrv_straddles",
        "Long one-month ATM straddles where implied volatility is lowest against the last month's "
        "realised, short where it is highest, earns net alpha (Goyal & Saretto 2009), at a premium "
        "of 10% of equity a side and without delta hedging.",
        QuotedSpec("straddle_ls"),
    ),
)


def realised_vol(panel: Panel) -> FloatArray:
    """Annualised realised volatility over ``RV_WINDOW`` sessions, known at each close."""
    r = ops.log_return(panel["close"])
    var = ops.rolling_mean(r * r, RV_WINDOW, RV_WINDOW // 2)
    return np.sqrt(var * 252.0)


def config_of(t: QuotedTrial) -> dict[str, Any]:
    return {"spec": asdict(t.spec), "benchmark": False, "chain_dte": list(CHAIN_DTE)}


def verify(registry: Registry) -> dict[str, str]:
    """The registration hashes, after checking that the code is what was registered."""
    pre = {e.body["name"]: e for e in registry.preregs(FAMILY)}
    out = {}
    for t in TRIALS:
        if t.name not in pre:
            raise RegistryError(f"{FAMILY}/{t.name} is not pre-registered")
        if content_hash(config_of(t)) != pre[t.name].body["config_hash"]:
            raise RegistryError(f"{FAMILY}/{t.name}: the code differs from what was registered")
        out[t.name] = pre[t.name].hash
    return out


def register(registry: Registry, now: dt.datetime) -> dict[str, str]:
    return {
        t.name: registry.preregister(
            family=FAMILY,
            name=t.name,
            hypothesis=t.hypothesis,
            config=config_of(t),
            gates=GateSpec().as_dict(),
            windows={"dev": list(DEV), "holdout": list(HOLDOUT)},
            now=now,
        )
        for t in TRIALS
    }


def member_chain(chain: pd.DataFrame, panel: Panel) -> dict[pd.Timestamp, pd.DataFrame]:
    """Quotes of index members by date, each with its days to expiry. A symbol counts on the dates
    its security was a member (the run checks again against the previous close)."""
    symbols = {s.split("@")[0] for s in panel.symbols}
    c = chain[chain["symbol"].isin(symbols)].copy()
    c["dte"] = (c["expiration"] - c["date"]).dt.days
    c = c[c["dte"].between(*CHAIN_DTE)]
    return {pd.Timestamp(str(d)): g for d, g in c.groupby("date", observed=True)}


def load_chain(store: Path, panel: Panel) -> dict[pd.Timestamp, pd.DataFrame]:
    """What the family reads of the store: index members, in the expiry range and its windows."""
    symbols = sorted({s.split("@")[0] for s in panel.symbols})
    c = read_member_chain(store, symbols, DEV[0], HOLDOUT[1], CHAIN_DTE)
    return by_session(c, panel)


def by_session(chain: pd.DataFrame, panel: Panel) -> dict[pd.Timestamp, pd.DataFrame]:
    """Quotes by the session they belong to. Until 2020 the table dates its weekly snapshot on
    Saturdays: those are Friday's closing quotes. A session's own snapshot wins over one carried
    onto it. (The first run of this family looked only at dates that were sessions, so it did not
    trade at all in 2019.)"""
    sessions = pd.DatetimeIndex(panel.dates.astype("datetime64[ns]"))
    days = pd.DatetimeIndex(sorted(chain["date"].unique()))
    pos = sessions.searchsorted(days, side="right") - 1
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    groups: dict[Any, pd.DataFrame] = dict(list(chain.groupby("date", observed=True)))
    for day, p in zip(days, pos, strict=True):
        if p < 0:
            continue
        session = sessions[int(p)]
        if day == session or session not in groups:
            out[session] = groups[day]
    return out


def _windows(res: QuotedResult, rf: FloatArray, lo: str, hi: str) -> tuple[np.ndarray, np.ndarray]:
    m = (res.dates >= np.datetime64(lo)) & (res.dates <= np.datetime64(hi))
    return res.returns[m], (res.returns - rf)[m]


def run(
    registry: Registry,
    panel: Panel,
    close_raw: FloatArray,
    chain: dict[pd.Timestamp, pd.DataFrame],
    rf: FloatArray,
    factors: pd.DataFrame,
    data_digest: str,
    now: dt.datetime,
    rerun_reason: str | None = None,
    code_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    hashes = verify(registry)
    score = selection_score(panel)
    rv = realised_vol(panel)
    t0 = int(np.searchsorted(panel.dates, np.datetime64(DEV[0])))
    t1 = int(np.searchsorted(panel.dates, np.datetime64(HOLDOUT[1]), side="right")) - 1
    rf_seg = np.where(np.isfinite(rf[t0 : t1 + 1]), rf[t0 : t1 + 1], 0.0)
    runs: dict[str, dict[str, QuotedResult]] = {}
    for t in TRIALS:
        runs[t.name] = {
            v: run_quoted(s, panel, close_raw, chain, score, rv, rf, t0, t1)
            for v, s in stressed(replace(t.spec)).items()
        }
    M = np.column_stack(
        [_windows(runs[t.name]["base"], rf_seg, DEV[0], HOLDOUT[1])[1] for t in TRIALS]
    )
    pbo = pbo_cscv(M, n_blocks=10).pbo
    own = [S.sharpe(_windows(runs[t.name]["base"], rf_seg, *HOLDOUT)[1]) for t in TRIALS]
    srs = np.array([*own, *programme_srs(registry, "us-", FAMILY)])
    var_sr = float(np.var(srs[np.isfinite(srs)], ddof=1))
    n_trials = registry.program_trials("us-")
    out: dict[str, Any] = {}
    for t in TRIALS:
        rr = runs[t.name]
        base = rr["base"]
        r_h, x_h = _windows(base, rf_seg, *HOLDOUT)
        mask = (base.dates >= np.datetime64(HOLDOUT[0])) & (base.dates <= np.datetime64(HOLDOUT[1]))
        fa = factor_attribution(r_h, base.dates[mask], factors)
        years = (
            pd.Series(base.returns, index=pd.DatetimeIndex(base.dates))
            .groupby(lambda d: d.year)
            .apply(lambda x: float(np.prod(1 + x) - 1))
        )
        dsr = S.dsr(x_h, n_trials, var_sr)
        m: dict[str, Any] = {
            "dev": metrics(*_windows(base, rf_seg, *DEV)),
            "holdout": metrics(r_h, x_h),
            "lag2_holdout": metrics(*_windows(rr["lag2"], rf_seg, *HOLDOUT)),
            "cost2_holdout": metrics(*_windows(rr["cost2"], rf_seg, *HOLDOUT)),
            "factor": {
                "alpha_annual": fa.alpha_annual,
                "alpha_t": fa.alpha_t,
                "betas": fa.betas,
                "tstats": fa.tstats,
                "r2": fa.r2,
            },
            "summary": {
                "avg_positions": float(np.mean(base.n_positions)),
                "rolls": base.rolls,
                "fees_annual": float(
                    np.sum(base.fees) / np.nanmean(base.equity) / (len(base.dates) / 252)
                ),
                "turnover_annual": None,
            },
            "yearly": {str(k): {"net": float(v)} for k, v in years.items()},
            "dsr": dsr,
            "pbo": pbo,
        }
        v = evaluate(
            GateSpec(),
            dsr=dsr,
            t_nw=m["holdout"]["t_nw"],
            alpha_t=fa.alpha_t,
            sharpe=m["holdout"]["sharpe"],
            sharpe_lag2=m["lag2_holdout"]["sharpe"],
            t_nw_cost2=m["cost2_holdout"]["t_nw"],
            pbo=pbo,
            max_dd=m["holdout"]["max_drawdown"],
            positive_year_share=float((years > 0).mean()),
        )
        registry.record_result(
            hashes[t.name],
            metrics=json_safe(m),
            verdict=json_safe(v.as_dict()),
            data_digest=data_digest,
            now=now,
            rerun_reason=rerun_reason,
            code_files=code_files,
        )
        out[t.name] = {"holdout": m["holdout"], "dev": m["dev"], "verdict": v.as_dict()}
    return out
