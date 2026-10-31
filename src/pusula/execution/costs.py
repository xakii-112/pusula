"""Transaction-cost model.

Cost of trading Q dollars of a name, as a fraction of Q::

    half_spread + Y * sigma_daily * sqrt(|Q| / ADV$) + commission (+ SEC fee on sells)

* **Spread.** Daily bars carry no quotes, so the half-spread is a log-linear model of
  volatility, dollar volume and price, ``ln hs = a_year + b ln sigma + c ln ADV$ + d ln P``,
  calibrated on 439 consolidated NBBO samples (11 sessions, 2016-2026; docs/CALIBRATION.md).
  The year intercept tracks the spread environment, which is observable in real time and
  widened after 2021 relative to daily-bar proxies (the trades-count model ``sigma/sqrt(N)``
  under-predicted 2022-26 spreads by 2.2x out of sample and is kept only as an alternative).
  Floor and cap guard against data errors.
* **Impact.** Square-root law (Almgren et al. 2005; Toth et al. 2011), ``Y ~ 0.5-1``.
* **Capacity.** A day's trade in one name is capped at ``max_participation`` of its ADV;
  the rest is not filled. Strategies that need more liquidity than exists show it as tracking
  error, not as free return.
* **Financing.** Cash earns the risk-free rate; shorts pay a borrow fee.

All inputs are as of the decision close (ADV and volatility through t), so the model never
uses the execution day's own volume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from pusula.alpha import ops
from pusula.data.panel import Panel

FloatArray = npt.NDArray[np.float64]
BPS = 1e-4


SPREAD_SLOPES_V1 = (0.7163, -0.3614, 0.4248)  # sigma, ADV$, price
SPREAD_YEAR_INTERCEPTS_V1 = (
    (2016, -0.5531),
    (2017, -0.6255),
    (2018, -0.2983),
    (2019, -0.6549),
    (2020, -0.4283),
    (2021, -0.2804),
    (2022, -0.1249),
    (2023, -0.1111),
    (2024, 0.1566),
    (2025, -0.0067),
    (2026, -0.1641),
)


@dataclass(frozen=True, slots=True)
class CostModel:
    spread_model: str = "loglinear"  # loglinear | sqrt_trades
    spread_slopes: tuple[float, float, float] = SPREAD_SLOPES_V1
    spread_year_intercepts: tuple[tuple[int, float], ...] = SPREAD_YEAR_INTERCEPTS_V1
    spread_k: float = 2.3
    spread_floor_bps: float = 0.5
    spread_cap_bps: float = 75.0
    fallback_half_spread_bps: float = 5.0
    impact_y: float = 0.7
    commission_bps: float = 0.2
    sec_fee_rate: float = 27.8e-6  # Section 31 fee, sells only (conservative recent rate)
    borrow_annual: float = 0.0030  # general collateral
    financing_spread_annual: float = 0.0050  # debit balances pay rf + spread
    adv_window: int = 21
    vol_halflife: float = 21.0
    max_participation: float = 0.05
    multiplier: float = 1.0  # cost stress (x2 in the gate)

    def stressed(self, factor: float) -> CostModel:
        return replace(self, multiplier=self.multiplier * factor)


@dataclass(frozen=True, slots=True)
class CostInputs:
    """Per-date, per-name inputs known at each decision close (all T x N)."""

    half_spread: FloatArray  # fraction of price
    adv_dollars: FloatArray
    sigma: FloatArray  # daily return vol


def cost_inputs(panel: Panel, model: CostModel) -> CostInputs:
    close = panel["close"]
    vol = panel["volume"]
    dollar_vol = close * vol
    adv = ops.rolling_mean(dollar_vol, model.adv_window, max(5, model.adv_window // 2))
    r = ops.log_return(close)
    sigma = np.sqrt(ops.ewm_var(r, model.vol_halflife, min_periods=10))
    if model.spread_model == "loglinear":
        years = panel.dates.astype("datetime64[Y]").astype(int) + 1970
        table = dict(model.spread_year_intercepts)
        lo_y, hi_y = min(table), max(table)
        a = np.array([table[min(max(int(y), lo_y), hi_y)] for y in years])[:, None]
        b, c, d = model.spread_slopes
        with np.errstate(invalid="ignore", divide="ignore"):
            hs = np.exp(a + b * np.log(sigma) + c * np.log(adv) + d * np.log(close))
    elif "trades" in panel:
        n_tr = ops.rolling_mean(panel["trades"], model.adv_window, max(5, model.adv_window // 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            hs = model.spread_k * sigma / np.sqrt(n_tr)
    else:
        hs = np.full_like(close, model.fallback_half_spread_bps * BPS)
    hs = np.where(np.isfinite(hs), hs, model.fallback_half_spread_bps * BPS)
    hs = np.clip(hs, model.spread_floor_bps * BPS, model.spread_cap_bps * BPS)
    return CostInputs(hs, adv, sigma)


def trade_cost(
    trade_dollars: FloatArray,
    half_spread: FloatArray,
    adv_dollars: FloatArray,
    sigma: FloatArray,
    model: CostModel,
) -> FloatArray:
    """Dollar cost of signed trades (>= 0). Names with unknown ADV pay the spread cap."""
    q = np.abs(trade_dollars)
    with np.errstate(invalid="ignore", divide="ignore"):
        part = np.where(adv_dollars > 0, q / adv_dollars, np.nan)
    impact = (
        model.impact_y
        * np.where(np.isfinite(sigma), sigma, 0.02)
        * np.sqrt(np.where(np.isfinite(part), part, 1.0))
    )
    hs = np.where(np.isfinite(half_spread), half_spread, model.spread_cap_bps * BPS)
    unit = hs + impact + model.commission_bps * BPS
    sells = np.where(trade_dollars < 0, q, 0.0)
    return model.multiplier * (q * unit + sells * model.sec_fee_rate)


def linear_cost_rate(half_spread: FloatArray, model: CostModel) -> FloatArray:
    """Per-unit-of-weight linear cost used by the optimiser (spread + commission)."""
    hs = np.where(np.isfinite(half_spread), half_spread, model.spread_cap_bps * BPS)
    return model.multiplier * (hs + model.commission_bps * BPS + 0.5 * model.sec_fee_rate)


def impact_coef(
    sigma: FloatArray, adv_dollars: FloatArray, equity: float, model: CostModel
) -> FloatArray:
    """k_i such that impact cost / equity = k_i * |dw_i|^1.5."""
    with np.errstate(invalid="ignore", divide="ignore"):
        k = model.impact_y * sigma * np.sqrt(equity / adv_dollars)
    return model.multiplier * np.where(np.isfinite(k), k, 0.0)


def participation_cap(adv_dollars: FloatArray, equity: float, model: CostModel) -> FloatArray:
    """Largest tradable |dw| per name for one session."""
    cap = model.max_participation * adv_dollars / equity
    return np.where(np.isfinite(cap), cap, 0.0)
