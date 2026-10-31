"""``mm-crypto-v1``: is market making worth its adverse selection, on a real book?

Its own programme: the question, the data and the unit of observation (an hour of one symbol) have
nothing in common with the daily books. The four makers were written and registered before any of
them saw a recording. Each is replayed on every recorded hour of every symbol; the first half of the
hours is development, the second half the holdout, as in the daily families.

An hour's PnL at the fee a retail account can get (7.5 bp maker with the BNB discount) is the
observation, and the gate is meridian's, with its flaw fixed: stresses must keep a positive t, not
merely a positive mean. The result is also reported at every other fee tier, which is what decides
whether the strategy is reachable at all: the fee is the first thing a maker has to beat.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pusula.core.errors import RegistryError
from pusula.core.hashing import content_hash
from pusula.data.binance import read_recording
from pusula.micro.book import Instrument, LocalBook, ReplayStats, replay
from pusula.micro.maker import MARKOUTS_NS, MakerResult, MakerSpec, run_makers
from pusula.research.registry import Registry
from pusula.research.study import json_safe
from pusula.stats import sharpe as S
from pusula.stats.inference import nw_mean_test

FAMILY = "mm-crypto-v1"
RETAIL_FEE_BPS = 7.5
FEE_TIERS = (10.0, 7.5, 2.0, 0.0, -0.5)
LATENCY_STRESS_NS = 250_000_000
MIN_HOURS = 24 * 20  # below this the gate does not speak: an hour's PnL is too noisy
MIN_DSR = 0.95
MIN_T = 2.0
MIN_STRESS_T = 1.0


@dataclass(frozen=True, slots=True)
class MakerTrial:
    name: str
    hypothesis: str
    spec: MakerSpec


TRIALS = (
    MakerTrial(
        "touch",
        "Quoting at the touch on both sides earns more in spread than it loses to adverse selection "
        "and fees.",
        MakerSpec("symmetric"),
    ),
    MakerTrial(
        "wide_2bp",
        "Quoting 2 bp from the mid gets fewer but better fills: being picked off costs less than the "
        "volume given up.",
        MakerSpec("symmetric", half_spread_bps=2.0),
    ),
    MakerTrial(
        "stoikov",
        "Skewing quotes against inventory (Avellaneda & Stoikov 2008) reduces inventory losses by more "
        "than the spread it gives up: meridian's result, on a real market.",
        MakerSpec("stoikov", gamma=0.5),
    ),
    MakerTrial(
        "imbalance",
        "Leaning quotes towards the side the touch's imbalance predicts (Cont et al. 2014; Stoikov "
        "2018) avoids the fills that adverse selection is made of.",
        MakerSpec("imbalance", imbalance_bps=0.5),
    ),
)


def variants(spec: MakerSpec) -> dict[str, MakerSpec]:
    return {
        "base": spec,
        "latency": replace(spec, latency_ns=LATENCY_STRESS_NS),
        "queue": replace(spec, cancel_ahead=0.0),
    }


def config_of(t: MakerTrial) -> dict[str, Any]:
    return {"spec": asdict(t.spec), "benchmark": False, "fee_bps": RETAIL_FEE_BPS}


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
    gates = {
        "min_dsr": MIN_DSR,
        "min_t": MIN_T,
        "min_stress_t": MIN_STRESS_T,
        "min_hours": MIN_HOURS,
    }
    return {
        t.name: registry.preregister(
            family=FAMILY,
            name=t.name,
            hypothesis=t.hypothesis,
            config=config_of(t),
            gates=gates,
            windows={"dev": ["first half of recorded hours"], "holdout": ["second half"]},
            now=now,
        )
        for t in TRIALS
    }


def run_symbol(
    path: Path, instrument: Instrument, specs: Sequence[MakerSpec]
) -> tuple[list[MakerResult], ReplayStats]:
    book = LocalBook(instrument)
    stats = ReplayStats()
    results = run_makers(specs, replay(read_recording(path), book, stats), book)
    return results, stats


def _stat(hours: Sequence[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(hours, dtype=np.float64)
    if x.size < 4:
        return {"n": float(x.size), "mean": math.nan, "t": math.nan, "sr": math.nan}
    return {
        "n": float(x.size),
        "mean": float(x.mean()),
        "t": nw_mean_test(x).t,
        "sr": S.sharpe(x),
    }


def evaluate_trial(
    hours: Mapping[str, Sequence[float]], n_trials: int, var_sr: float
) -> dict[str, Any]:
    """``hours``: variant -> pooled hourly net PnL at the retail fee, in time order per symbol."""
    base = np.asarray(hours["base"], dtype=np.float64)
    half = base.size // 2
    holdout = base[half:]
    failed = []
    dsr = S.dsr(holdout, max(n_trials, 1), var_sr) if holdout.size > 3 else math.nan
    if base.size < MIN_HOURS:
        failed.append(f"sample: {base.size} symbol-hours, {MIN_HOURS} needed")
    stat = _stat(holdout)
    if not dsr >= MIN_DSR:
        failed.append("dsr")
    if not stat["t"] >= MIN_T:
        failed.append("t")
    for v in ("latency", "queue"):
        stressed = np.asarray(hours[v], dtype=np.float64)[half:]
        if not _stat(stressed)["t"] >= MIN_STRESS_T:
            failed.append(f"{v}_stress")
    return {
        "promoted": not failed,
        "failed": failed,
        "dsr": dsr,
        "dev": _stat(base[:half]),
        "holdout": stat,
    }


def summarise(result: MakerResult, tick: float, lot: float) -> dict[str, Any]:
    hours = len(result.blocks)
    notional = result.traded_notional * tick * lot
    gross = result.pnl * tick * lot / 2.0
    return {
        "hours": hours,
        "fills": len(result.fills),
        "notional": notional,
        "gross_pnl": gross,
        "edge": result.edge * tick * lot / 2.0,
        "inventory": result.inventory * tick * lot / 2.0,
        "gross_bps": gross / notional * 1e4 if notional else math.nan,
        "breakeven_fee_bps": result.breakeven_fee_bps(),
        "net_by_fee": {str(f): result.net(f, tick, lot) for f in FEE_TIERS},
        "max_abs_position": result.max_abs_position,
        "size_lots": result.size_lots,
        "markout_bps": {f"{h // 1_000_000_000}s": result.markout_bps(h, tick) for h in MARKOUTS_NS},
    }


def record(
    registry: Registry,
    outcomes: Mapping[str, dict[str, Any]],
    digest: str,
    now: dt.datetime,
    rerun_reason: str | None = None,
    code_files: dict[str, str] | None = None,
) -> None:
    hashes = verify(registry)
    for name, out in outcomes.items():
        registry.record_result(
            hashes[name],
            metrics=json_safe(
                {
                    "dev": out["verdict"]["dev"],
                    "holdout": out["verdict"]["holdout"],
                    "dsr": out["verdict"]["dsr"],
                    "by_symbol": out["by_symbol"],
                    "factor": {},
                    "summary": {},
                    "yearly": {},
                }
            ),
            verdict=json_safe(
                {"promoted": out["verdict"]["promoted"], "failed": out["verdict"]["failed"]}
            ),
            data_digest=digest,
            now=now,
            rerun_reason=rerun_reason,
            code_files=code_files,
        )


def render_results(
    outcomes: Mapping[str, dict[str, Any]], replay_stats: Mapping[str, dict[str, float]]
) -> str:
    def f(x: Any, fmt: str = "{:.2f}") -> str:
        return fmt.format(x) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"

    lines = [
        f"# Results: `{FAMILY}`",
        "",
        "> Generated by `pusula micro run`. Market making on recorded Binance books, with queue "
        f"position tracked. One observation is an hour of one symbol at a {RETAIL_FEE_BPS} bp maker "
        f"fee. Below {MIN_HOURS} symbol-hours the gate refuses on sample size whatever the numbers.",
        "",
        "## Gate",
        "",
        "| maker | symbol-hours | dev mean, $/h | holdout mean, $/h | holdout t | DSR | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, out in outcomes.items():
        v = out["verdict"]
        verdict = "promoted" if v["promoted"] else "no: " + ", ".join(v["failed"])
        n = v["dev"]["n"] + v["holdout"]["n"]
        lines.append(
            f"| {name} | {n:.0f} | {f(v['dev']['mean'])} | {f(v['holdout']['mean'])} "
            f"| {f(v['holdout']['t'])} | {f(v['dsr'])} | {verdict} |"
        )
    lines += [
        "",
        "## Where the money goes, all recorded hours",
        "",
        "Gross of fees. `breakeven` is the maker fee at which the run's PnL is zero: a retail account "
        f"pays {FEE_TIERS[0]} bp, or {FEE_TIERS[1]} with the discount. A negative one needs a rebate.",
        "",
        "| maker | pair | fills | traded, $ | edge, $ | inventory, $ | gross, bp | breakeven, bp "
        "| markout 1s / 10s / 60s, bp | net at 7.5 bp, $ | net at 0 bp, $ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name, out in outcomes.items():
        for symbol, s in sorted(out["by_symbol"].items()):
            m = s["markout_bps"]
            lines.append(
                f"| {name} | {symbol} | {s['fills']} | {s['notional']:,.0f} | {f(s['edge'])} "
                f"| {f(s['inventory'])} | {f(s['gross_bps'])} | {f(s['breakeven_fee_bps'])} "
                f"| {f(m['1s'])} / {f(m['10s'])} / {f(m['60s'])} "
                f"| {f(s['net_by_fee']['7.5'])} | {f(s['net_by_fee']['0.0'])} |"
            )
    lines += [
        "",
        "## Is the reconstructed book the real one?",
        "",
        "Later snapshots against the book maintained up to them, on the best 20 levels a side.",
        "",
        "| pair | diffs | trades | chain breaks | snapshots | levels checked | agreeing | crossed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, st in sorted(replay_stats.items()):
        share = st["level_matches"] / st["level_checks"] if st["level_checks"] else math.nan
        lines.append(
            f"| {symbol} | {st['diffs']:,.0f} | {st['trades']:,.0f} | {st['gaps']:.0f} "
            f"| {st['snapshots']:.0f} | {st['level_checks']:.0f} | {f(share, '{:.1%}')} "
            f"| {st['crossed']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def load_instruments(path: Path) -> dict[str, Instrument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {s: Instrument(s, float(v["tick"]), float(v["lot"])) for s, v in raw.items()}
