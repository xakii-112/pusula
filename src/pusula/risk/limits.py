"""Pre-trade risk gate: limits enforced independently of the strategy.

The optimiser already respects these limits as constraints. The gate checks them again on its
output, with its own code, because a strategy bug, a solver tolerance or a stale input must
not be able to reach the market (defence in depth). Breaches are repaired conservatively
(scale or clip towards the current portfolio) and always reported; a target that cannot be
repaired (non-finite values) is rejected and the current portfolio is kept.

Rules carried over from NihaiAlgoritma: Reg T overnight leverage (gross <= 2x equity),
minimum price, restricted list, no shorts in names flagged hard-to-borrow, daily-loss and
drawdown kill switch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_gross: float = 2.0  # Reg T overnight: 50% initial margin
    max_net: float = 1.0
    min_net: float = -1.0
    max_weight: float = 0.05
    min_price: float = 5.0
    max_participation: float = 0.05  # traded shares / ADV shares, per day
    max_turnover: float = 2.0  # sum |dw| per rebalance
    max_sector_net: float | None = None
    max_beta_abs: float | None = None
    max_vol_annual: float | None = None
    restricted: frozenset[str] = frozenset()
    hard_to_borrow: frozenset[str] = frozenset()


@dataclass(slots=True)
class GateResult:
    weights: FloatArray
    accepted: bool
    breaches: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.accepted and not self.breaches


def gate_targets(
    target: FloatArray,
    current: FloatArray,
    symbols: tuple[str, ...],
    limits: RiskLimits,
    *,
    price: FloatArray,
    adv_dollars: FloatArray,
    equity: float,
    sector_of: Mapping[str, str] | None = None,
    beta: FloatArray | None = None,
    vol_fn: Callable[[FloatArray], float] | None = None,
) -> GateResult:
    """Validate and repair target weights. ``current`` are today's (drifted) weights."""
    if not np.all(np.isfinite(target)):
        return GateResult(current.copy(), False, ["non-finite target weights: rejected"])
    w = target.astype(np.float64, copy=True)
    br: list[str] = []
    names = np.asarray(symbols)

    # Per-name eligibility (restricted, price floor, borrow): can only reduce towards zero.
    # A blocked name may only move from its current weight towards zero.
    blocked = np.isin(names, list(limits.restricted)) | ~(price >= limits.min_price)
    lo, hi = np.minimum(current, 0.0), np.maximum(current, 0.0)
    repaired = np.where(blocked, np.clip(w, lo, hi), w)
    n_fixed = int((np.abs(repaired - w) > 1e-12).sum())
    if n_fixed:
        br.append(
            f"{n_fixed} blocked names (restricted / below ${limits.min_price:g}) held or reduced"
        )
    w = repaired
    # Hard-to-borrow: an existing short may be covered, never increased.
    htb = np.isin(names, list(limits.hard_to_borrow))
    floor = np.where(htb, np.minimum(current, 0.0), -np.inf)
    if np.any(w < floor - 1e-12):
        br.append(f"{int((w < floor - 1e-12).sum())} short increases in hard-to-borrow names")
        w = np.maximum(w, floor)

    over = np.abs(w) > limits.max_weight + 1e-12
    if over.any():
        br.append(f"{int(over.sum())} names above max weight {limits.max_weight:.3f}")
        w = np.clip(w, -limits.max_weight, limits.max_weight)

    # Liquidity: one day's trade limited to max_participation of ADV.
    cap = limits.max_participation * adv_dollars / max(equity, 1e-9)
    cap = np.where(np.isfinite(cap), cap, 0.0)
    dw = w - current
    too_big = np.abs(dw) > cap + 1e-12
    if too_big.any():
        br.append(f"{int(too_big.sum())} trades above {limits.max_participation:.0%} ADV")
        w = current + np.clip(dw, -cap, cap)

    gross = float(np.abs(w).sum())
    if gross > limits.max_gross + 1e-9:
        br.append(f"gross {gross:.3f} > {limits.max_gross:.3f}")
        w *= limits.max_gross / gross
    net = float(w.sum())
    if net > limits.max_net + 1e-9 or net < limits.min_net - 1e-9:
        br.append(f"net {net:.3f} outside [{limits.min_net:.3f}, {limits.max_net:.3f}]")
        bound = limits.max_net if net > limits.max_net else limits.min_net
        side = w > 0 if net > bound else w < 0
        excess = net - bound
        side_sum = float(w[side].sum())
        if side_sum != 0:
            w[side] *= 1.0 - excess / side_sum

    if limits.max_sector_net is not None and sector_of is not None:
        secs = np.array([sector_of.get(s, "Unknown") for s in symbols])
        for sec in np.unique(secs):
            if sec == "Unknown":
                continue
            m = secs == sec
            sn = float(w[m].sum())
            if abs(sn) > limits.max_sector_net + 1e-9:
                br.append(f"sector {sec} net {sn:.3f}")
                # shrink the side that causes the excess (magnitudes only fall, so the
                # per-name limits enforced above stay satisfied; an equal shift did not)
                side = m & ((w > 0) if sn > 0 else (w < 0))
                excess = sn - np.sign(sn) * limits.max_sector_net
                w[side] *= 1.0 - excess / float(w[side].sum())

    if limits.max_beta_abs is not None and beta is not None:
        b = float(np.nansum(w * beta))
        if abs(b) > limits.max_beta_abs + 1e-9:
            br.append(
                f"beta {b:.3f} beyond +/-{limits.max_beta_abs:.3f} (reported; optimiser owns beta)"
            )

    turnover = float(np.abs(w - current).sum())
    if turnover > limits.max_turnover + 1e-9:
        br.append(f"turnover {turnover:.3f} > {limits.max_turnover:.3f}")
        w = current + (w - current) * (limits.max_turnover / turnover)

    if vol_fn is not None and limits.max_vol_annual is not None:
        v = float(vol_fn(w))
        if v > limits.max_vol_annual:
            br.append(f"ex-ante vol {v:.3f} > {limits.max_vol_annual:.3f}")
            w *= limits.max_vol_annual / v
    return GateResult(w, True, br)


@dataclass(slots=True)
class KillSwitch:
    """Drawdown / daily-loss circuit breaker. Once tripped it stays tripped until reset by an
    operator (never automatically): a strategy that has just lost its limit is the least
    trustworthy judge of whether to resume."""

    max_drawdown: float = 0.15
    max_daily_loss: float = 0.04
    derisk_drawdown: float = 0.10
    derisk_scale: float = 0.5
    peak: float = 0.0
    tripped: bool = False
    reason: str = ""

    def update(self, equity: float, prev_equity: float) -> float:
        """Record the new equity and return the gross scale to apply (0 = flatten)."""
        self.peak = max(self.peak, equity)
        if self.tripped:
            return 0.0
        dd = 1.0 - equity / self.peak if self.peak > 0 else 0.0
        day = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        if dd >= self.max_drawdown:
            self.tripped, self.reason = True, f"drawdown {dd:.1%} >= {self.max_drawdown:.0%}"
            return 0.0
        if day <= -self.max_daily_loss:
            self.tripped, self.reason = True, f"daily loss {day:.1%} <= -{self.max_daily_loss:.0%}"
            return 0.0
        return self.derisk_scale if dd >= self.derisk_drawdown else 1.0

    def reset(self, equity: float) -> None:
        self.tripped, self.reason, self.peak = False, "", equity
