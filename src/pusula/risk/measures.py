"""Portfolio risk measures: ex-ante volatility, VaR / expected shortfall, stress scenarios."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

from pusula.risk.model import FactorRisk

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class VaRReport:
    var95: float  # loss as a positive fraction of NAV, one day
    var99: float
    es975: float
    parametric_var99: float
    n_scenarios: int


def historical_var(
    weights: FloatArray, returns_window: FloatArray, risk: FactorRisk | None = None
) -> VaRReport:
    """Full-revaluation historical simulation: today's weights against each past day's returns.

    Missing returns (name not yet listed on that past day) are replaced by the model-implied
    zero-mean proxy 0, which understates tails for young names; the parametric figure from the
    factor model is reported next to it for that reason.
    """
    R = np.where(np.isfinite(returns_window), returns_window, 0.0)
    pnl = R @ weights
    losses = -pnl
    var95 = float(np.quantile(losses, 0.95))
    var99 = float(np.quantile(losses, 0.99))
    tail = losses[losses >= np.quantile(losses, 0.975)]
    es = float(tail.mean()) if tail.size else var99
    pv = (
        float(sps.norm.ppf(0.99) * risk.vol(weights, annualise=False))
        if risk is not None
        else float("nan")
    )
    return VaRReport(var95, var99, es, pv, int(R.shape[0]))


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    start: dt.date
    end: dt.date


HISTORICAL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("2018 Feb vol shock (Volmageddon)", dt.date(2018, 1, 26), dt.date(2018, 2, 8)),
    Scenario("2018 Q4 sell-off", dt.date(2018, 10, 1), dt.date(2018, 12, 24)),
    Scenario("2020 COVID crash", dt.date(2020, 2, 19), dt.date(2020, 3, 23)),
    Scenario("2020 Nov vaccine rotation", dt.date(2020, 11, 6), dt.date(2020, 11, 13)),
    Scenario("2022 H1 rates shock", dt.date(2022, 1, 3), dt.date(2022, 6, 16)),
    Scenario("2023 Mar regional banks", dt.date(2023, 3, 8), dt.date(2023, 3, 17)),
    Scenario("2025 Apr tariff shock", dt.date(2025, 4, 2), dt.date(2025, 4, 8)),
)


def scenario_pnl(
    weights: FloatArray, cum_returns: FloatArray, betas: FloatArray, market_cum: float
) -> float:
    """P&L (fraction of NAV) of ``weights`` under one historical window.

    ``cum_returns`` are each name's realised cumulative returns over the window; names without
    history in the window are proxied by ``beta * market_cum``.
    """
    proxy = np.where(np.isfinite(cum_returns), cum_returns, betas * market_cum)
    proxy = np.where(np.isfinite(proxy), proxy, market_cum)
    return float(weights @ proxy)


def realised_to_model_vol(
    returns_window: FloatArray, weights: FloatArray, risk: FactorRisk, cols_weights: FloatArray
) -> float:
    """Realised volatility of today's holdings over a trailing window divided by the factor
    model's forecast for the same holdings (``nan`` if either is degenerate).

    Optimised portfolios concentrate in directions the model under-estimates, so this ratio
    for the *held* book is a direct, causal measure of the optimisation bias.
    """
    R = np.where(np.isfinite(returns_window), returns_window, 0.0)
    realised = float(np.std(R @ weights, ddof=1)) if R.shape[0] > 2 else float("nan")
    model = risk.vol(cols_weights, annualise=False)
    return realised / model if model > 0 and np.isfinite(realised) else float("nan")
