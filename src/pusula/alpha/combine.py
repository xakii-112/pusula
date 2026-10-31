"""From raw scores to one composite alpha score per security and date.

Pipeline (each stage causal): mask to the point-in-time universe -> normal scores (robust to
outliers and to each signal's units) -> optional neutralisation against market beta ->
optional time smoothing (cuts turnover; costs are paid on changes) -> weighted sum ->
re-standardise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from pusula.alpha import ops
from pusula.alpha.signals import Signal, build, rolling_beta, universe_mask
from pusula.data.panel import FloatArray, Panel


@dataclass(frozen=True, slots=True)
class SignalSpec:
    name: str
    weight: float = 1.0
    params: Mapping[str, object] = field(default_factory=dict)
    smooth_halflife: float = 0.0

    def signal(self) -> Signal:
        return build(self.name, **dict(self.params))


@dataclass(frozen=True, slots=True)
class CompositeSpec:
    signals: tuple[SignalSpec, ...]
    beta_neutral: bool = True

    @property
    def lookback(self) -> int:
        return max(s.signal().lookback for s in self.signals) + (252 if self.beta_neutral else 0)


def standardized(
    sig: Signal, panel: Panel, smooth_halflife: float = 0.0, beta: FloatArray | None = None
) -> FloatArray:
    mask = universe_mask(panel)
    z = ops.cs_gauss(sig.compute(panel), mask)
    if beta is not None:
        z = ops.cs_neutralize(z, [beta], mask)
        z = ops.cs_zscore(z, mask)
    if smooth_halflife > 0:
        z = ops.ewm_mean(z, smooth_halflife, min_periods=1)
        if mask is not None:
            z = np.where(mask > 0, z, np.nan)
        z = ops.cs_zscore(z, mask)
    return z


def composite(spec: CompositeSpec, panel: Panel) -> FloatArray:
    mask = universe_mask(panel)
    beta = rolling_beta(panel) if spec.beta_neutral else None
    total = np.zeros((panel.n_dates, panel.n_symbols))
    wsum = np.zeros_like(total)
    for s in spec.signals:
        z = standardized(s.signal(), panel, s.smooth_halflife, beta)
        ok = np.isfinite(z)
        total += np.where(ok, s.weight * z, 0.0)
        wsum += np.where(ok, abs(s.weight), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(wsum > 0, total / wsum, np.nan)
    return ops.cs_zscore(raw, mask)


def feature_stack(
    specs: Sequence[SignalSpec], panel: Panel, beta_neutral: bool = True
) -> FloatArray:
    """T x N x K array of standardised signals (ML input)."""
    beta = rolling_beta(panel) if beta_neutral else None
    return np.stack(
        [standardized(s.signal(), panel, s.smooth_halflife, beta) for s in specs], axis=2
    )
