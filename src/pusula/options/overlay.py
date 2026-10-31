"""Expressing a directional equity view with listed calls (NihaiAlgoritma's OPTION_MODE).

Each roll date the strategy picks names with a positive trend score, and for each buys calls
expiring ``tenor`` sessions later at the listed strike closest to ``target_delta``, sized to
the same delta-dollar exposure the stock sleeve would hold. Cash not spent on premium earns the
risk-free rate. Positions are marked daily with Black-Scholes at the name's **observed** ATM
implied volatility; at expiry the payoff is exact (``max(S - K, 0)``), so the model only
prices entry and interim marks, never the outcome.

Honest limitations, kept in the results: one IV per name (no smile; OTM/ITM strikes are priced
at ATM vol), options spread is a fixed fraction of premium, and dividends are approximated by a
constant yield because the price series are total-return adjusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pusula.alpha import ops
from pusula.data.panel import FloatArray, Panel
from pusula.options import black

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class CallOverlaySpec:
    tenor: int = 21  # sessions to expiry
    target_delta: float = 0.5
    gross_delta: float = (
        1.0  # delta-dollar exposure / equity (1.0 = like a fully invested stock book)
    )
    max_names: int = 50
    option_half_spread: float = 0.025  # fraction of premium paid on entry (and on early exit)
    dividend_yield: float = 0.015
    max_premium_frac: float = 0.25  # never spend more than this share of equity on premium


@dataclass
class OverlayResult:
    dates: npt.NDArray[np.datetime64]
    equity: FloatArray
    returns: FloatArray
    premium_spent: FloatArray
    n_positions: npt.NDArray[np.int64]
    payoff_ratio: list[float]  # per expired contract: payoff / premium


@dataclass
class _Leg:
    j: int
    K: float
    n: float  # number of "share-equivalent" call units
    expiry_row: int
    entry_px: float
    premium: float  # paid, including the spread
    last_iv: float


def run_call_overlay(
    panel: Panel,
    score: FloatArray,
    iv: FloatArray,
    rf_daily: FloatArray,
    start: int,
    end: int,
    spec: CallOverlaySpec,
    initial_equity: float = 10_000_000.0,
) -> OverlayResult:
    raw_close = panel["close"]
    close = ops.ffill(raw_close)  # a delisted name settles at its last price, not at entry
    member = panel["in_universe"] > 0
    dates = panel.dates
    cash = initial_equity
    legs: list[_Leg] = []
    eq = np.full(end - start + 1, np.nan)
    prem = np.zeros_like(eq)
    npos = np.zeros(eq.size, dtype=np.int64)
    ratios: list[float] = []
    r_ann = 252.0 * np.where(np.isfinite(rf_daily), rf_daily, 0.0)

    def t_years(a: int, b: int) -> float:
        return max((dates[b] - dates[a]).astype("timedelta64[D]").astype(int), 0) / DAYS_PER_YEAR

    def mark(e: int) -> float:
        tot = 0.0
        for lg in legs:
            S = close[e, lg.j]
            if not math.isfinite(S):
                S = lg.entry_px
            T = t_years(e, lg.expiry_row)
            if math.isfinite(iv[e, lg.j]):
                lg.last_iv = float(iv[e, lg.j])  # stale IV is carried, never invented
            tot += lg.n * float(black.price(S, lg.K, T, r_ann[e], spec.dividend_yield, lg.last_iv))
        return tot

    next_roll = start + 1  # decision at start's close, entry at the next close (one-session lag)
    for e in range(start, end + 1):
        if e > start:
            cash *= 1.0 + float(rf_daily[e]) if math.isfinite(rf_daily[e]) else 1.0
        # expiries
        keep = []
        for lg in legs:
            if e >= lg.expiry_row:
                S = close[e, lg.j]
                if not math.isfinite(S):
                    S = lg.entry_px
                payoff = lg.n * max(
                    S * (1.0 - spec.dividend_yield * spec.tenor / 252.0) - lg.K, 0.0
                )
                cash += payoff
                ratios.append(payoff / lg.premium if lg.premium > 0 else math.nan)
            else:
                keep.append(lg)
        legs = keep
        if e == next_roll and e + spec.tenor <= end and not legs:
            d = e - 1  # signal known at the previous close
            s = np.where(
                member[d] & np.isfinite(score[d]) & np.isfinite(iv[e]) & np.isfinite(raw_close[e]),
                score[d],
                -np.inf,
            )
            pick = [j for j in np.argsort(-s)[: spec.max_names] if s[j] > 0]
            equity_now = cash
            if pick:
                per_name = spec.gross_delta * equity_now / len(pick)
                T = t_years(e, e + spec.tenor)
                plan = []
                for j in pick:
                    S, vol = float(close[e, j]), float(iv[e, j])
                    K = black.strike_for_delta(
                        S, T, r_ann[e], spec.dividend_yield, vol, spec.target_delta
                    )
                    px = float(black.price(S, K, T, r_ann[e], spec.dividend_yield, vol))
                    dl = float(black.delta(S, K, T, r_ann[e], spec.dividend_yield, vol))
                    if px <= 0 or dl <= 0.05:
                        continue
                    n = per_name / (dl * S)
                    plan.append((j, K, n, px, vol))
                cost = sum(n * px * (1 + spec.option_half_spread) for _, _, n, px, _ in plan)
                scale = min(1.0, spec.max_premium_frac * equity_now / cost) if cost > 0 else 0.0
                for j, K, n, px, vol in plan:
                    paid = n * scale * px * (1 + spec.option_half_spread)
                    legs.append(
                        _Leg(int(j), K, n * scale, e + spec.tenor, float(close[e, j]), paid, vol)
                    )
                    cash -= paid
                    prem[e - start] += paid
            next_roll = e + spec.tenor
        elif e == next_roll:
            next_roll = e + 1
        eq[e - start] = cash + mark(e)
        npos[e - start] = len(legs)
    rets = np.zeros_like(eq)
    rets[1:] = eq[1:] / eq[:-1] - 1.0
    return OverlayResult(dates[start : end + 1], eq, rets, prem, npos, ratios)
