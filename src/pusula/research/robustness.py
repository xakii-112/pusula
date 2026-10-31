"""Exploratory robustness: is a registered configuration a plateau or a lucky spike?

QuantBot re-ran its bot over parameter grids and with 1e-9 noise on prices. Both questions apply
here, and the protocol decides what the answers may be used for: a neighbour of a registered trial
is not a trial, so nothing here may be selected. A neighbour that looks better becomes a new
registered trial and pays for itself in the deflated Sharpe. No gate reads this.

* **Neighbours** of a trial: how much of its result depends on the exact values registered.
* **Noise floor**: how far the result moves when the prices move by nothing that matters. A
  difference between two trials below it is the solver's, not the strategy's.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pusula.alpha.combine import SignalSpec
from pusula.data.panel import Panel
from pusula.research.study import Trial
from pusula.stats.pbo import pbo_cscv

NEAR = 0.20  # Sharpe distance that counts as "the same result" (about the holdout's standard error)
SPIKE = 0.30  # the registered value this far above the neighbours' median is a spike
PLATEAU = 0.50  # share of neighbours within NEAR for the result to be a plateau
PRICE_FIELDS = ("open", "high", "low", "close", "vwap")


def _label(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in values.items())


def signal_neighbours(trial: Trial, grid: Mapping[str, Sequence[Any]]) -> list[tuple[str, Trial]]:
    """The trial with its single signal's parameters over the grid's product."""
    if len(trial.signals) != 1:
        raise ValueError(f"{trial.name}: a signal grid needs a single-signal trial")
    s = trial.signals[0]
    out = []
    for combo in itertools.product(*grid.values()):
        values = dict(zip(grid, combo, strict=True))
        spec = SignalSpec(s.name, params={**dict(s.params), **values})
        out.append((_label(values), replace(trial, signals=(spec,))))
    return out


def forecast_neighbours(trial: Trial, grid: Mapping[str, Sequence[Any]]) -> list[tuple[str, Trial]]:
    """The trial with its portfolio engine's parameters over the grid's product."""
    if trial.forecast is None:
        raise ValueError(f"{trial.name}: a forecast grid needs a forecast trial")
    out = []
    for combo in itertools.product(*grid.values()):
        values = dict(zip(grid, combo, strict=True))
        out.append((_label(values), replace(trial, forecast=replace(trial.forecast, **values))))
    return out


def with_price_noise(panel: Panel, eps: float, seed: int) -> Panel:
    """Prices times ``1 + eps * z``, one z per security and session, so a bar stays a valid bar."""
    z = np.random.default_rng(seed).standard_normal(panel["close"].shape)
    noisy = {f: panel[f] * (1.0 + eps * z) for f in PRICE_FIELDS if f in panel}
    return panel.with_fields(**noisy)


@dataclass(frozen=True, slots=True)
class GridSummary:
    n: int
    base: float
    median: float
    best: float
    worst: float
    share_near: float  # of the other points, within NEAR of the registered value
    base_rank: int  # 1 = the registered value is the best of the grid
    pbo: float  # of choosing the grid's best point
    verdict: str


def summarise_grid(
    holdout: Mapping[str, float], base_label: str, daily: np.ndarray | None = None
) -> GridSummary:
    """``holdout``: Sharpe per point. ``daily``: T x points of excess returns, for the PBO."""
    base = holdout[base_label]
    others = np.array([v for k, v in holdout.items() if k != base_label], dtype=float)
    if others.size == 0:
        raise ValueError("a grid needs a point other than the registered one")
    every = np.array(list(holdout.values()), dtype=float)
    median = float(np.median(others))
    share_near = float((np.abs(others - base) <= NEAR).mean())
    verdict = "plateau" if share_near >= PLATEAU else "narrow"
    if base - median > SPIKE:
        verdict = "spike"
    pbo = math.nan
    if daily is not None and daily.shape[1] >= 2:
        pbo = pbo_cscv(daily, n_blocks=10).pbo
    return GridSummary(
        n=len(holdout),
        base=base,
        median=median,
        best=float(every.max()),
        worst=float(every.min()),
        share_near=share_near,
        base_rank=1 + int((others > base).sum()),
        pbo=pbo,
        verdict=verdict,
    )


def noise_floor(sharpes: Sequence[float]) -> float:
    """Sample standard deviation of the Sharpe ratios of the noisy runs."""
    x = np.asarray(sharpes, dtype=float)
    return float(x.std(ddof=1)) if x.size > 1 else math.nan


def render_grid(
    title: str,
    base_label: str,
    points: Mapping[str, Mapping[str, float]],
    summary: GridSummary,
    floor: float = math.nan,
) -> list[str]:
    """``points``: label -> {"dev": Sharpe, "holdout": Sharpe}."""
    lines = [
        f"## {title}",
        "",
        f"Holdout Sharpe {summary.base:.2f} at the registered values, {summary.median:.2f} at the "
        f"median neighbour, range [{summary.worst:.2f}, {summary.best:.2f}]; "
        f"{summary.share_near:.0%} of neighbours within {NEAR:.2f}, rank {summary.base_rank} of "
        f"{summary.n}: **{summary.verdict}**."
        + (
            f" PBO of picking the best point: {summary.pbo:.2f}."
            if math.isfinite(summary.pbo)
            else ""
        )
        + (
            f" Noise floor (SD of Sharpe at 1e-9 price noise): {floor:.3f}."
            if math.isfinite(floor)
            else ""
        ),
        "",
        "| values | development | holdout |",
        "|---|---:|---:|",
    ]
    for label, p in points.items():
        mark = " (registered)" if label == base_label else ""
        lines.append(f"| {label}{mark} | {p['dev']:.2f} | {p['holdout']:.2f} |")
    lines.append("")
    return lines
