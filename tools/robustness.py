"""Exploratory robustness of registered trials (NOT a gate input, nothing here may be selected):
parameter neighbours and a noise floor. Writes docs/ROBUSTNESS.md. A grid point is not a trial; a
neighbour that looks better becomes a new registered trial and pays for itself.

    uv run python tools/robustness.py --workers 12
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pusula.data.pipeline import etf_data, research_panel
from pusula.engine.backtest import Backtester
from pusula.live.forward import rf_series
from pusula.research.families import us_crossasset_v1, us_equity_v2, us_trend_v1
from pusula.research.robustness import (
    forecast_neighbours,
    noise_floor,
    render_grid,
    signal_neighbours,
    summarise_grid,
    with_price_noise,
)
from pusula.research.study import StudySpec, Trial, make_strategy, window_excess
from pusula.stats import sharpe as S

ROOT = Path(__file__).resolve().parents[1]
NOISE = 1e-9
SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class Target:
    title: str
    spec: StudySpec
    base: Trial
    base_label: str
    points: tuple[tuple[str, Trial], ...]
    noise: bool


def _trial(spec: StudySpec, name: str) -> Trial:
    return next(t for t in spec.trials if t.name == name)


def targets() -> list[Target]:
    v2, trend, cross = us_equity_v2(), us_trend_v1(), us_crossasset_v1()
    momentum = _trial(v2, "xs_momentum")
    heritage = _trial(v2, "trend_heritage")
    hrp = _trial(trend, "forecast_hrp")
    long_trend = _trial(cross, "ca_trend_long")
    engine = {"target_vol": (0.05, 0.10, 0.15), "buffer": (0.0, 0.10, 0.25)}
    stocks = {"target_vol": (0.10, 0.15, 0.20), "buffer": (0.0, 0.10, 0.25)}
    return [
        Target(
            "`xs_momentum` (us-equity-v2): formation and skip",
            v2,
            momentum,
            "window=252, skip=21",
            tuple(
                signal_neighbours(momentum, {"window": (126, 189, 252, 315), "skip": (5, 21, 42)})
            ),
            True,
        ),
        Target(
            "`trend_heritage` (us-equity-v2): trend horizons",
            v2,
            heritage,
            "horizons=(21, 63, 252)",
            tuple(
                signal_neighbours(
                    heritage,
                    {
                        "horizons": (
                            (21, 63, 252),
                            (10, 40, 160),
                            (21, 63, 126),
                            (42, 126, 252),
                            (63, 126, 252),
                        )
                    },
                )
            ),
            False,
        ),
        Target(
            "`forecast_hrp` (us-trend-v1): volatility target and buffer",
            trend,
            hrp,
            "target_vol=0.15, buffer=0.1",
            tuple(forecast_neighbours(hrp, stocks)),
            True,
        ),
        Target(
            "`ca_trend_long` (us-crossasset-v1): volatility target and buffer",
            cross,
            long_trend,
            "target_vol=0.1, buffer=0.1",
            tuple(forecast_neighbours(long_trend, engine)),
            False,
        ),
    ]


def _run(job: tuple[int, str, Trial, StudySpec, int | None]) -> tuple[int, str, dict[str, object]]:
    index, label, trial, spec, seed = job
    panel = (etf_data(spec.universe) if spec.universe else research_panel()).panel
    if seed is not None:
        panel = with_price_noise(panel, NOISE, seed)
    cfg = replace(spec.backtest, kill_switch=False)
    res = Backtester(panel, cfg, rf_series(panel)).run(make_strategy(trial, cfg.cost_model))
    return (
        index,
        label,
        {
            "dev": S.annualised_sharpe(window_excess(res, *spec.dev)),
            "holdout": S.annualised_sharpe(window_excess(res, *spec.holdout)),
            "daily": window_excess(res, spec.dev[0], spec.holdout[1]),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    # one BLAS thread per worker (study.run_backtests explains why)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    tg = targets()
    jobs: list[tuple[int, str, Trial, StudySpec, int | None]] = []
    for i, t in enumerate(tg):
        jobs += [(i, label, trial, t.spec, None) for label, trial in t.points]
        if t.noise:
            jobs += [(i, f"noise:{s}", t.base, t.spec, s) for s in SEEDS]
    results: dict[int, dict[str, dict[str, object]]] = {i: {} for i in range(len(tg))}
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for done, f in enumerate(as_completed(futs), 1):
            i, label, r = f.result()
            results[i][label] = r
            print(
                f"[robustness] {done}/{len(jobs)} {tg[i].base.name} {label} "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    lines = [
        "# Robustness of registered trials (exploratory, not a gate input)",
        "",
        "Generated by `tools/robustness.py`. A neighbour of a registered trial is not a trial: "
        "nothing here may be selected. A neighbour that looks better becomes a new registered "
        "trial and pays for itself in the deflated Sharpe. Sharpe on excess returns, net of costs, "
        "without the kill switch, as the research runs.",
        "",
    ]
    for i, t in enumerate(tg):
        got = results[i]
        labels = [label for label, _ in t.points]
        points = {
            k: {"dev": float(got[k]["dev"]), "holdout": float(got[k]["holdout"])} for k in labels
        }  # type: ignore[arg-type]
        daily = np.column_stack([np.asarray(got[k]["daily"]) for k in labels])
        summary = summarise_grid({k: p["holdout"] for k, p in points.items()}, t.base_label, daily)
        noisy = [float(got[f"noise:{s}"]["holdout"]) for s in SEEDS] if t.noise else []  # type: ignore[arg-type]
        lines += render_grid(t.title, t.base_label, points, summary, noise_floor(noisy))
    out = ROOT / "docs" / "ROBUSTNESS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
