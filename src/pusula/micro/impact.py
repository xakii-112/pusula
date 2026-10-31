"""What a trade costs, measured instead of assumed.

``execution.costs`` charges ``half_spread + Y sigma sqrt(Q / ADV)``, with Y = 0.7 from the
literature because daily bars cannot fit it. Two measurements from tick data bear on that number:

* **Flow impact.** Sign each trade (the exchange says who the aggressor was; for stocks, the tick
  rule), sum over buckets, and regress the bucket's return in daily volatilities on the signed
  square root of its share of daily volume: the coefficient is a Y. It measures aggregate order
  flow, not one desk's metaorder, whose impact studies find similar in form; read it as the order
  of magnitude a registered Y should have, not as that Y.
* **Sweep.** What an immediate order of a given size pays walking a real book, against what the
  model charges for it. It bounds the immediate cost; a patient order pays less and waits for it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pusula.execution.costs import CostModel
from pusula.stats.inference import nw_mean_test

FloatArray = npt.NDArray[np.float64]


def tick_rule(prices: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Lee & Ready's fallback: an uptick is a buy, a downtick a sell, a repeat keeps the last sign.
    Trades before the first price change are unsigned (0)."""
    p = np.asarray(prices, dtype=np.float64)
    out = np.zeros(p.size, dtype=np.int64)
    last = 0
    for i in range(1, p.size):
        if p[i] > p[i - 1]:
            last = 1
        elif p[i] < p[i - 1]:
            last = -1
        out[i] = last
    return out


@dataclass(frozen=True, slots=True)
class Buckets:
    signed_dollars: FloatArray
    returns: FloatArray  # log return from the bucket's first price to its last


def bucketise(
    time_ns: npt.ArrayLike,
    prices: npt.ArrayLike,
    sizes: npt.ArrayLike,
    signs: npt.ArrayLike,
    bucket_ns: int,
) -> Buckets:
    """Signed dollar flow and return per bucket; a bucket needs two trades to have a return."""
    t = np.asarray(time_ns, dtype=np.int64)
    p = np.asarray(prices, dtype=np.float64)
    q = np.asarray(sizes, dtype=np.float64)
    s = np.asarray(signs, dtype=np.float64)
    keys = (t - t[0]) // bucket_ns
    flows, rets = [], []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < 2:
            continue
        flows.append(float(np.sum(s[m] * q[m] * p[m])))
        rets.append(float(np.log(p[m][-1] / p[m][0])))
    return Buckets(np.array(flows), np.array(rets))


@dataclass(frozen=True, slots=True)
class ImpactFit:
    y: float  # the coefficient of sigma_daily x sign(Q) x sqrt(|Q| / ADV)
    t: float
    n: int
    r2: float


def fit_square_root(
    signed_dollars: npt.ArrayLike, returns: npt.ArrayLike, sigma_daily: float, adv: float
) -> ImpactFit:
    """Slope through the origin of ``return / sigma`` on ``sign(Q) sqrt(|Q| / ADV)``, with a
    Newey-West t: consecutive buckets of one stock are not independent."""
    q = np.asarray(signed_dollars, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if q.size < 3 or sigma_daily <= 0 or adv <= 0:
        return ImpactFit(math.nan, math.nan, int(q.size), math.nan)
    x = np.sign(q) * np.sqrt(np.abs(q) / adv)
    y = r / sigma_daily
    xx = float(x @ x)
    if xx <= 0:
        return ImpactFit(math.nan, math.nan, int(q.size), math.nan)
    b = float(x @ y) / xx
    # the slope is the mean of x * y scaled, so its t-statistic is that mean's
    t = nw_mean_test(x * y).t
    resid = y - b * x
    r2 = 1.0 - float(resid @ resid) / float(y @ y) if float(y @ y) > 0 else math.nan
    return ImpactFit(b, t, int(q.size), r2)


def sweep_cost_bps(levels: Sequence[tuple[float, float]], notional: float) -> float:
    """Average price paid against the best, in bps, for an immediate order of ``notional`` walking
    ``levels`` (price, size; best first). NaN if the visible book cannot take it."""
    if not levels or notional <= 0:
        return math.nan
    best = levels[0][0]
    left = notional
    units = 0.0
    for price, size in levels:
        take = min(left, price * size)
        units += take / price
        left -= take
        if left <= 0:
            return abs(notional / units / best - 1.0) * 1e4
    return math.nan


def model_cost_bps(
    notional: float, half_spread: float, sigma_daily: float, adv: float, model: CostModel
) -> float:
    """What ``execution.costs`` charges beyond commissions: half-spread plus square-root impact."""
    impact = model.impact_y * sigma_daily * math.sqrt(notional / adv)
    return (half_spread + impact) * 1e4 * model.multiplier
