"""Pre-registered study runner.

Two commands, deliberately separate in time:

* ``register`` writes every trial of a family to the ledger (config hash, hypothesis, gate
  thresholds, windows, code digest) **before** any of them is run on real data;
* ``run`` executes each registered trial three times (base, execution one session later,
  costs doubled), measures dev and holdout windows, computes the family's PBO and each trial's
  deflated Sharpe ratio with the registered trial count, applies the gate and appends the
  results to the ledger.
"""

from __future__ import annotations

import datetime as dt
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from pusula.alpha.combine import CompositeSpec, SignalSpec, composite
from pusula.alpha.signals import build
from pusula.data.panel import FloatArray, Panel
from pusula.engine.attribution import FACTORS, factor_attribution, yearly_table
from pusula.engine.backtest import BacktestConfig, Backtester, BacktestResult
from pusula.execution.costs import CostModel
from pusula.ml.walkforward import WalkForwardSpec, ml_alpha_fn
from pusula.portfolio.context import Strategy
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import (
    EqualWeight,
    ForecastRiskParity,
    ForecastSpec,
    HoldOne,
    OptimizedAlpha,
    TrendFollowing,
)
from pusula.research.gates import GateSpec, evaluate
from pusula.research.registry import Registry
from pusula.stats import sharpe as S
from pusula.stats.inference import nw_mean_test
from pusula.stats.pbo import pbo_cscv


@dataclass(frozen=True, slots=True)
class Trial:
    name: str
    kind: str  # optimized | trend | forecast | equal_weight
    hypothesis: str
    signals: tuple[SignalSpec, ...] = ()
    ml: WalkForwardSpec | None = None
    optimizer: OptimizerSpec | None = None
    ic: float = 0.02
    horizon_days: float = 5.0
    rebalance_every: int = 5
    long_only: bool = True
    vol_target_annual: float = 0.15
    benchmark: bool = False
    # added for us-equity-v3; omitted from the registered form at their defaults so that the
    # hashes of trials registered before these fields existed do not change
    ic_mode: str = "fixed"  # fixed | adaptive (trailing realised IC, shrunk, floored at 0)
    ic_window: int = 252
    ic_prior_days: int = 252
    vol_feedback: bool = False  # rescale the vol target by realised/ex-ante vol of the held book
    forecast: ForecastSpec | None = None  # added for us-trend-v1 (kind "forecast")
    hold: str | None = None  # added for us-crypto-v1 (kind "hold")


_LATER_FIELDS = {
    "ic_mode": "fixed",
    "ic_window": 252,
    "ic_prior_days": 252,
    "vol_feedback": False,
    "forecast": None,
    "hold": None,
}


def trial_config(t: Trial) -> dict[str, Any]:
    d = asdict(t)
    for k, default in _LATER_FIELDS.items():
        if d.get(k) == default:
            d.pop(k)
    return d


# Risk-model fields added after families were registered (CALIBRATION.md §3); omitted from the
# registered form at their defaults for the same reason as _LATER_FIELDS.
_LATER_RISK_FIELDS: dict[str, object] = {
    "eigen_adjust": None,
    "eigen_sims": 100,
    "industry_min_size": None,
}


def backtest_config(b: BacktestConfig) -> dict[str, Any]:
    d = asdict(b)
    rs = d.get("risk_spec")
    if isinstance(rs, dict):
        for k, default in _LATER_RISK_FIELDS.items():
            if k in rs and rs[k] == default:
                rs.pop(k)
    return d


def registered_config(spec: StudySpec, t: Trial) -> dict[str, Any]:
    """The exact object whose hash is pre-registered for trial ``t``."""
    cfg: dict[str, Any] = {
        "trial": trial_config(t),
        "backtest": backtest_config(spec.backtest),
        "benchmark": t.benchmark,
    }
    if spec.universe:  # absent for the index universe, so that earlier hashes do not change
        cfg["universe"] = list(spec.universe)
    if spec.factor_names:  # likewise for FF5+MOM
        cfg["factors"] = list(spec.factor_names)
    return cfg


@dataclass(frozen=True, slots=True)
class StudySpec:
    family: str
    trials: tuple[Trial, ...]
    dev: tuple[str, str]
    holdout: tuple[str, str]
    gates: GateSpec = field(default_factory=GateSpec)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    universe: tuple[str, ...] = ()  # a fixed list of ETFs; empty: the point-in-time S&P 500
    factor_names: tuple[str, ...] = ()  # the alpha check's factors; empty: FF5+MOM

    def windows(self) -> dict[str, list[str]]:
        return {"dev": list(self.dev), "holdout": list(self.holdout)}


# ---------------------------------------------------------------------------- strategies
def _composite_fn(signals: tuple[SignalSpec, ...]) -> Callable[[Panel], FloatArray]:
    spec = CompositeSpec(signals)

    def fn(p: Panel) -> FloatArray:
        return composite(spec, p)

    return fn


def _trend_fn(signals: tuple[SignalSpec, ...]) -> Callable[[Panel], FloatArray]:
    def fn(p: Panel) -> FloatArray:
        s = signals[0]
        return build(s.name, **dict(s.params)).compute(p)

    return fn


def make_strategy(trial: Trial, cost_model: CostModel) -> Strategy:
    if trial.kind == "equal_weight":
        return EqualWeight(label=trial.name, every=trial.rebalance_every)
    if trial.kind == "trend":
        return TrendFollowing(
            _trend_fn(trial.signals),
            label=trial.name,
            every=trial.rebalance_every,
            long_only=trial.long_only,
            vol_target_annual=trial.vol_target_annual,
        )
    if trial.kind == "hold":
        assert trial.hold is not None
        return HoldOne(trial.hold, label=trial.name, every=trial.rebalance_every)
    if trial.kind == "forecast":
        assert trial.forecast is not None
        return ForecastRiskParity(
            _trend_fn(trial.signals),
            trial.forecast,
            label=trial.name,
            every=trial.rebalance_every,
        )
    if trial.kind == "optimized":
        assert trial.optimizer is not None
        fn = ml_alpha_fn(trial.ml) if trial.ml is not None else _composite_fn(trial.signals)
        return OptimizedAlpha(
            fn,
            trial.optimizer,
            label=trial.name,
            ic=trial.ic,
            horizon_days=trial.horizon_days,
            every=trial.rebalance_every,
            cost_model=cost_model,
            ic_mode=trial.ic_mode,
            ic_window=trial.ic_window,
            ic_prior_days=trial.ic_prior_days,
            vol_feedback=trial.vol_feedback,
        )
    raise KeyError(trial.kind)


# ---------------------------------------------------------------------------- execution
_WORKER: dict[str, Any] = {}


def _init_worker(loader: Callable[[], tuple[Panel, FloatArray]]) -> None:
    panel, rf = loader()
    _WORKER["panel"], _WORKER["rf"] = panel, rf


def _run_one(
    args: tuple[Trial, BacktestConfig, str, CostModel],
) -> tuple[str, str, BacktestResult]:
    trial, cfg, variant, believed = args
    bt = Backtester(_WORKER["panel"], cfg, _WORKER["rf"])
    # The strategy decides with the *registered* cost model; the backtester charges ``cfg``'s.
    # In the cost stress the market charges twice what the optimiser believes (model risk). Had
    # the optimiser been told, it would trade less and the stress would test nothing (an earlier
    # version did exactly that: the stressed t-stat came out *higher*).
    res = bt.run(make_strategy(trial, believed))
    res.weights = res.weights[:, :0]  # drop the T x N weight history before pickling back
    return trial.name, variant, res


def variants(cfg: BacktestConfig) -> dict[str, BacktestConfig]:
    return {
        "base": cfg,
        "lag2": replace(cfg, lag=cfg.lag + 1),
        "cost2": replace(cfg, cost_model=cfg.cost_model.stressed(2.0)),
    }


def run_backtests(
    spec: StudySpec, loader: Callable[[], tuple[Panel, FloatArray]], workers: int = 4
) -> dict[str, dict[str, BacktestResult]]:
    jobs = []
    for t in spec.trials:
        cfg = spec.backtest if not t.benchmark else replace(spec.backtest, kill_switch=False)
        for v, c in variants(cfg).items():
            if t.benchmark and v != "base":
                continue
            jobs.append((t, c, v, cfg.cost_model))
    out: dict[str, dict[str, BacktestResult]] = {}
    results: list[tuple[str, str, BacktestResult]] = []
    if workers > 1:
        # One BLAS thread per worker: N processes x all-cores BLAS oversubscribed 16 cores and made
        # a 10 ms SVD take 750 ms (measured 2026-10-24). Children read these at numpy import.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
    if workers <= 1:
        _init_worker(loader)
        for j in jobs:
            results.append(_run_one(j))
            print(f"[study] {len(results)}/{len(jobs)} {j[0].name}/{j[2]}", flush=True)
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(loader,)) as ex:
            futs = [ex.submit(_run_one, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
                name, v, _ = results[-1]
                print(f"[study] {len(results)}/{len(jobs)} {name}/{v}", flush=True)
    for name, v, res in results:
        out.setdefault(name, {})[v] = res
    return out


# ---------------------------------------------------------------------------- metrics
def _window(res: BacktestResult, lo: str, hi: str) -> np.ndarray:
    d = res.dates
    return (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))


def window_returns(res: BacktestResult, lo: str, hi: str) -> np.ndarray:
    return res.returns[_window(res, lo, hi)]


def window_excess(res: BacktestResult, lo: str, hi: str) -> np.ndarray:
    return res.excess_returns[_window(res, lo, hi)]


def metrics(r: np.ndarray, x: np.ndarray | None = None) -> dict[str, float]:
    """Window statistics. ``r`` raw net returns, ``x`` excess returns (``r`` if omitted).

    Sharpe, t-statistic and PSR use **excess** returns; return, volatility and drawdown are
    reported on raw returns.
    """
    x = r if x is None else x
    m = S.moments(x)
    t = nw_mean_test(x)
    return {
        "n": float(m.n),
        "ann_return": S.annualised_return(r),
        "ann_excess_return": float(np.mean(x) * 252) if x.size else math.nan,
        "ann_vol": S.annualised_vol(r),
        "sharpe": S.annualised_sharpe(x),
        "sr_daily": m.sr,
        "skew": m.skew,
        "kurt": m.kurt,
        "t_nw": t.t,
        "psr0": S.psr(m.sr, m.n, m.skew, m.kurt) if m.n > 3 else math.nan,
        "max_drawdown": S.max_drawdown(r),
        "worst_day": float(np.min(r)) if r.size else math.nan,
    }


def window_metrics(res: BacktestResult, lo: str, hi: str) -> dict[str, float]:
    return metrics(window_returns(res, lo, hi), window_excess(res, lo, hi))


def positive_year_share(res: BacktestResult) -> float:
    yt = yearly_table(res)
    return float((yt["net"] > 0).mean()) if len(yt) else math.nan


@dataclass
class TrialOutcome:
    name: str
    benchmark: bool
    dev: dict[str, float]
    holdout: dict[str, float]
    full: dict[str, float]
    lag2_holdout: dict[str, float]
    cost2_holdout: dict[str, float]
    factor: dict[str, Any]
    summary: dict[str, Any]
    yearly: dict[str, dict[str, float]]
    dsr: float = math.nan
    pbo: float = math.nan
    verdict: dict[str, Any] = field(default_factory=dict)
    events: int = 0


def evaluate_study(
    spec: StudySpec,
    runs: dict[str, dict[str, BacktestResult]],
    factors: pd.DataFrame,
    n_trials: int,
    other_srs: Sequence[float] = (),
) -> list[TrialOutcome]:
    """``n_trials`` and ``other_srs`` describe the whole programme: the deflated Sharpe ratio
    uses the programme's trial count *and* the dispersion of all its trials' Sharpe ratios
    (a family of three similar trials must not get a low hurdle from its own small spread)."""
    lo_d, hi_d = spec.dev
    lo_h, hi_h = spec.holdout
    outs: list[TrialOutcome] = []
    for t in spec.trials:
        rr = runs[t.name]
        base = rr["base"]
        hold = window_returns(base, lo_h, hi_h)
        fa = factor_attribution(
            hold,
            base.dates[(base.dates >= np.datetime64(lo_h)) & (base.dates <= np.datetime64(hi_h))],
            factors,
            spec.factor_names or FACTORS,
        )
        yt = yearly_table(base)
        outs.append(
            TrialOutcome(
                name=t.name,
                benchmark=t.benchmark,
                dev=window_metrics(base, lo_d, hi_d),
                holdout=window_metrics(base, lo_h, hi_h),
                full=window_metrics(base, lo_d, hi_h),
                lag2_holdout=window_metrics(rr["lag2"], lo_h, hi_h) if "lag2" in rr else {},
                cost2_holdout=window_metrics(rr["cost2"], lo_h, hi_h) if "cost2" in rr else {},
                factor={
                    "alpha_annual": fa.alpha_annual,
                    "alpha_t": fa.alpha_t,
                    "betas": fa.betas,
                    "tstats": fa.tstats,
                    "r2": fa.r2,
                },
                summary={**base.summary(), "halted_on": base.halted_on},
                yearly={
                    str(k): {str(c): float(v) for c, v in row.items()} for k, row in yt.iterrows()
                },
                events=len(base.events),
            )
        )
    cands = [o for o in outs if not o.benchmark]
    # family PBO on the full-sample daily returns of all candidate configurations
    if len(cands) >= 2:
        names = [o.name for o in cands]
        M = np.column_stack([window_excess(runs[n]["base"], lo_d, hi_h) for n in names])
        pbo = pbo_cscv(M, n_blocks=10).pbo
    else:
        pbo = math.nan
    srs = np.array([o.holdout["sr_daily"] for o in cands] + list(other_srs), dtype=float)
    var_sr = float(np.var(srs[np.isfinite(srs)], ddof=1)) if np.isfinite(srs).sum() > 1 else 0.0
    for o in cands:
        r = window_excess(runs[o.name]["base"], lo_h, hi_h)
        o.dsr = S.dsr(r, max(n_trials, 1), var_sr)
        o.pbo = pbo
        v = evaluate(
            spec.gates,
            dsr=o.dsr,
            t_nw=o.holdout["t_nw"],
            alpha_t=o.factor["alpha_t"],
            sharpe=o.holdout["sharpe"],
            sharpe_lag2=o.lag2_holdout.get("sharpe", math.nan),
            t_nw_cost2=o.cost2_holdout.get("t_nw", math.nan),
            pbo=pbo,
            max_dd=o.holdout["max_drawdown"],
            positive_year_share=positive_year_share(runs[o.name]["base"]),
        )
        o.verdict = v.as_dict()
    return outs


# ---------------------------------------------------------------------------- ledger glue
def register(spec: StudySpec, registry: Registry, now: dt.datetime) -> dict[str, str]:
    out = {}
    for t in spec.trials:
        cfg = registered_config(spec, t)
        out[t.name] = registry.preregister(
            family=spec.family,
            name=t.name,
            hypothesis=t.hypothesis,
            config=cfg,
            gates=spec.gates.as_dict(),
            windows=spec.windows(),
            now=now,
        )
    return out


def verify_registration(spec: StudySpec, registry: Registry) -> dict[str, str]:
    """Pre-registration hashes of ``spec``'s trials, after checking that the code about to run
    is the configuration that was registered (a trial edited after registration must not run
    under the old registration)."""
    from pusula.core.errors import RegistryError
    from pusula.core.hashing import content_hash

    pre = {e.body["name"]: e for e in registry.preregs(spec.family)}
    out = {}
    for t in spec.trials:
        if t.name not in pre:
            raise RegistryError(f"{spec.family}/{t.name} is not pre-registered")
        if content_hash(registered_config(spec, t)) != pre[t.name].body["config_hash"]:
            raise RegistryError(
                f"{spec.family}/{t.name}: the configuration in the code differs from the "
                "registered one; register a new trial instead of running an edited one"
            )
        out[t.name] = pre[t.name].hash
    return out


def json_safe(x: Any) -> Any:
    """Replace non-finite floats by None (the ledger's canonical form has no NaN)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def record(
    spec: StudySpec,
    registry: Registry,
    outcomes: list[TrialOutcome],
    hashes: dict[str, str],
    data_digest: str,
    now: dt.datetime,
    rerun_reason: str | None = None,
    code_files: dict[str, str] | None = None,
) -> None:
    for o in outcomes:
        registry.record_result(
            hashes[o.name],
            metrics=json_safe(
                {
                    "dev": o.dev,
                    "holdout": o.holdout,
                    "full": o.full,
                    "lag2_holdout": o.lag2_holdout,
                    "cost2_holdout": o.cost2_holdout,
                    "factor": o.factor,
                    "summary": o.summary,
                    "dsr": o.dsr,
                    "pbo": o.pbo,
                    "yearly": o.yearly,
                }
            ),
            verdict=json_safe(o.verdict or {"promoted": False, "benchmark": True}),
            data_digest=data_digest,
            now=now,
            rerun_reason=rerun_reason,
            code_files=code_files,
        )


def programme_srs(registry: Registry, prefix: str, exclude_family: str) -> list[float]:
    """Holdout per-period Sharpe ratios of the latest result of every non-benchmark trial of
    the programme ``prefix``, other families than ``exclude_family``."""
    fams = {str(e.body["family"]) for e in registry.entries if e.kind == "prereg"}
    out: list[float] = []
    for fam in sorted(f for f in fams if f.startswith(prefix) and f != exclude_family):
        bench = {e.body["name"] for e in registry.preregs(fam) if e.body["config"].get("benchmark")}
        for name, e in registry.latest_results(fam).items():
            sr = e.body["metrics"].get("holdout", {}).get("sr_daily")
            if name not in bench and sr is not None and math.isfinite(sr):
                out.append(float(sr))
    return out


def reevaluate(
    registry: Registry,
    family: str,
    prefix: str,
    now: dt.datetime,
    reason: str,
    code_files: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Recompute the deflated Sharpe ratio and the gate verdict of a family's latest results
    from the statistics stored in the ledger (no backtest is re-run), and record them."""
    from pusula.research.gates import GateSpec

    pre = {e.body["name"]: e for e in registry.preregs(family)}
    latest = registry.latest_results(family)
    cands = [n for n in latest if not pre[n].body["config"].get("benchmark")]
    own = [latest[n].body["metrics"]["holdout"].get("sr_daily") for n in cands]
    srs = [float(x) for x in own if x is not None and math.isfinite(x)]
    srs += programme_srs(registry, prefix, family)
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    n_trials = registry.program_trials(prefix)
    out: dict[str, dict[str, Any]] = {}
    for name in cands:
        e = latest[name]
        m = dict(e.body["metrics"])
        h = m["holdout"]
        sr, n, sk, ku = (h.get(k) for k in ("sr_daily", "n", "skew", "kurt"))
        ok = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (sr, n, sk, ku))
        dsr = (
            S.psr(float(sr), int(n), float(sk), float(ku), S.expected_max_sr(n_trials, var_sr))
            if ok
            else math.nan
        )
        m["dsr"] = dsr
        m["dsr_inputs"] = {"n_trials": n_trials, "var_sr": var_sr, "programme": prefix}
        years = [v.get("net") for v in m.get("yearly", {}).values()]
        pos = [y for y in years if y is not None]
        gates = GateSpec(**pre[name].body["gates"])

        def g(d: dict[str, Any], k: str) -> float:
            v = d.get(k)
            return float(v) if isinstance(v, (int, float)) else math.nan

        v = evaluate(
            gates,
            dsr=dsr,
            t_nw=g(h, "t_nw"),
            alpha_t=g(m.get("factor", {}), "alpha_t"),
            sharpe=g(h, "sharpe"),
            sharpe_lag2=g(m.get("lag2_holdout", {}), "sharpe"),
            t_nw_cost2=g(m.get("cost2_holdout", {}), "t_nw"),
            pbo=g(m, "pbo"),
            max_dd=g(h, "max_drawdown"),
            positive_year_share=float(np.mean([y > 0 for y in pos])) if pos else math.nan,
        )
        registry.record_result(
            pre[name].hash,
            metrics=json_safe(m),
            verdict=json_safe(v.as_dict()),
            data_digest=e.body["data_digest"],
            now=now,
            rerun_reason=reason,
            code_files=code_files,
        )
        out[name] = {"dsr": dsr, "verdict": v.as_dict()}
    return out
