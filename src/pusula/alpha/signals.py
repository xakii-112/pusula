"""Alpha library: causal scores computed from a point-in-time panel.

A signal maps a :class:`Panel` to a T x N array of scores; row t uses rows <= t only (checked
for every registered signal by ``tests/test_lookahead.py``). Higher score = higher expected
return. Scores are raw; cross-sectional normalisation and combination happen downstream.

Each signal cites the published effect it implements. None of them is claimed to work here
until the research protocol says so; most published anomalies decay after publication
(McLean & Pontiff 2016), which is exactly why the protocol exists.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from pusula.alpha import ops
from pusula.core.hashing import content_hash
from pusula.data.panel import FloatArray, Panel


@runtime_checkable
class Signal(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def lookback(self) -> int: ...

    def compute(self, panel: Panel) -> FloatArray: ...


def signal_id(sig: Signal) -> str:
    return f"{sig.name}:{content_hash(sig)[:12]}"


def universe_mask(panel: Panel) -> FloatArray | None:
    return panel.get("in_universe")


def market_return(panel: Panel) -> FloatArray:
    """Equal-weighted mean daily log return of universe members (T x 1), causal."""
    r = ops.log_return(panel["close"])
    m = universe_mask(panel)
    if m is not None:
        r = np.where(m > 0, r, np.nan)
    return ops.cs_mean(r)


def rolling_beta(panel: Panel, window: int = 252, min_periods: int = 126) -> FloatArray:
    r = ops.log_return(panel["close"])
    rm = market_return(panel)
    cov = ops.rolling_cov(r, rm, window, min_periods)
    # market variance over exactly the sessions on which the stock has a return
    rmb = np.where(np.isfinite(r), np.broadcast_to(rm, r.shape), np.nan)
    var = ops.rolling_cov(rmb, rmb, window, min_periods)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / var


# ---------------------------------------------------------------------------- library
@dataclass(frozen=True, slots=True)
class Momentum:
    """Cross-sectional price momentum, 12-1 (Jegadeesh & Titman 1993)."""

    window: int = 252
    skip: int = 21
    name: str = "momentum"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        c = panel["close"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(ops.lag(c, self.skip) / ops.lag(c, self.window))


@dataclass(frozen=True, slots=True)
class TimeSeriesMomentum:
    """Multi-horizon trend sign score (Moskowitz, Ooi & Pedersen 2012; Hurst et al. 2017).

    The live strategy of NihaiAlgoritma (``quant_trend``), kept as the heritage baseline.
    """

    horizons: tuple[int, ...] = (21, 63, 252)
    name: str = "ts_momentum"

    @property
    def lookback(self) -> int:
        return max(self.horizons) + 1

    def compute(self, panel: Panel) -> FloatArray:
        c = panel["close"]
        signs = [np.sign(ops.log_return(c, h)) for h in self.horizons]
        return np.mean(np.stack(signs), axis=0)


@dataclass(frozen=True, slots=True)
class ShortTermReversal:
    """One-week reversal (Lehmann 1990; Lo & MacKinlay 1990), market-adjusted."""

    window: int = 5
    name: str = "reversal"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        r = ops.log_return(panel["close"], self.window)
        mkt = ops.rolling_sum(market_return(panel), self.window)
        return -(r - mkt)


@dataclass(frozen=True, slots=True)
class LowVolatility:
    """Low-volatility anomaly (Ang, Hodrick, Xing & Zhang 2006; Blitz & van Vliet 2007)."""

    window: int = 63
    name: str = "low_vol"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        r = ops.log_return(panel["close"])
        return -ops.rolling_std(r, self.window, int(self.window * 0.8))


@dataclass(frozen=True, slots=True)
class ResidualMomentum:
    """Momentum of market-residual returns, scaled by residual vol (Blitz, Huij & Martens 2011)."""

    window: int = 252
    skip: int = 21
    beta_window: int = 252
    name: str = "residual_momentum"

    @property
    def lookback(self) -> int:
        return self.window + self.beta_window

    def compute(self, panel: Panel) -> FloatArray:
        r = ops.log_return(panel["close"])
        rm = market_return(panel)
        beta = ops.lag(rolling_beta(panel, self.beta_window, self.beta_window // 2), 1)
        resid = r - beta * rm
        span = self.window - self.skip
        cum = ops.lag(ops.rolling_sum(resid, span, int(span * 0.8)), self.skip)
        vol = ops.lag(ops.rolling_std(resid, span, int(span * 0.8)), self.skip)
        with np.errstate(invalid="ignore", divide="ignore"):
            return cum / (vol * np.sqrt(span))


@dataclass(frozen=True, slots=True)
class FiftyTwoWeekHigh:
    """Nearness to the 52-week high (George & Hwang 2004)."""

    window: int = 252
    name: str = "high_52w"

    @property
    def lookback(self) -> int:
        return self.window

    def compute(self, panel: Panel) -> FloatArray:
        c = panel["close"]
        high = panel.get("high")
        hi = ops.rolling_max(c if high is None else high, self.window, int(self.window * 0.9))
        with np.errstate(invalid="ignore", divide="ignore"):
            return c / hi - 1.0


@dataclass(frozen=True, slots=True)
class BettingAgainstBeta:
    """Low-beta premium (Frazzini & Pedersen 2014): score = -beta."""

    window: int = 252
    name: str = "bab"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        return -rolling_beta(panel, self.window, self.window // 2)


@dataclass(frozen=True, slots=True)
class OvernightMomentum:
    """Persistence of overnight returns (Lou, Polk & Skouras 2019): cumulative close-to-open
    return over the window, net of the market's."""

    window: int = 21
    name: str = "overnight"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            on = np.log(panel["open"] / ops.lag(panel["close"], 1))
        m = universe_mask(panel)
        onm = np.where(m > 0, on, np.nan) if m is not None else on
        mkt = ops.cs_mean(onm)
        return ops.rolling_sum(on - mkt, self.window, int(self.window * 0.8))


@dataclass(frozen=True, slots=True)
class ImpliedRealizedSpread:
    """Expensive options predict low stock returns (Bali & Hovakimian 2009; Goyal & Saretto
    2009 for option returns): score = -(30-day ATM implied vol - realised vol).

    Needs the panel field ``iv`` (annualised IV known at the close, ``data.iv``).
    """

    window: int = 21
    name: str = "iv_rv_spread"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        r = ops.log_return(panel["close"])
        rv = ops.rolling_std(r, self.window, int(self.window * 0.8)) * np.sqrt(252.0)
        return -(panel["iv"] - rv)


@dataclass(frozen=True, slots=True)
class EarningsAnnouncementReturn:
    """Earnings-announcement drift measured by the market's own reaction (Chan, Jegadeesh &
    Lakonishok 1996; Brandt, Kishore, Santa-Clara & Venkatachalam 2008): the market-adjusted log
    return over sessions -1..+1 around the release, day 0 being the first session that closes
    after the 8-K Item 2.02 is accepted. Known at the close of day +1, held ``hold`` sessions.

    Needs the panel field ``ann_day0`` (``data.fundamentals``)."""

    hold: int = 63
    name: str = "ear"

    @property
    def lookback(self) -> int:
        return self.hold + 4

    def compute(self, panel: Panel) -> FloatArray:
        r = ops.log_return(panel["close"])
        car = ops.rolling_sum(r - market_return(panel), 3, 3)  # sessions t-2 .. t
        day1 = ops.lag(panel["ann_day0"], 1) > 0  # day 0 was the previous session
        return ops.hold_last_event(np.where(day1, car, np.nan), day1, self.hold)


@dataclass(frozen=True, slots=True)
class EarningsSurprise:
    """Post-earnings-announcement drift on standardised unexpected earnings (Bernard & Thomas
    1989), quarterly net income as first known from the 10-Q/10-K, seasonal random walk with
    drift (Foster 1977). Held ``hold`` sessions after the report. Needs the field ``sue``."""

    hold: int = 63
    name: str = "sue"

    @property
    def lookback(self) -> int:
        return self.hold + 1

    def compute(self, panel: Panel) -> FloatArray:
        s = panel["sue"]
        return ops.hold_last_event(s, np.isfinite(s), self.hold)


@dataclass(frozen=True, slots=True)
class LowAccruals:
    """Accrual anomaly (Sloan 1996): score = -(net income - operating cash flow) / average total
    assets of the last fiscal year, from the 10-K, held up to ``hold`` sessions (a year plus the
    filing lag). Financials excluded. Needs the field ``accruals``."""

    hold: int = 315
    name: str = "accruals"

    @property
    def lookback(self) -> int:
        return self.hold + 1

    def compute(self, panel: Panel) -> FloatArray:
        a = panel["accruals"]
        return -ops.hold_last_event(a, np.isfinite(a), self.hold)


@dataclass(frozen=True, slots=True)
class GrossProfitability:
    """Gross profitability (Novy-Marx 2013): (revenue - cost of revenue) / total assets of the
    last fiscal year, from the 10-K, held up to ``hold`` sessions. Financials excluded. Needs
    the field ``gross_profitability``."""

    hold: int = 315
    name: str = "gross_profitability"

    @property
    def lookback(self) -> int:
        return self.hold + 1

    def compute(self, panel: Panel) -> FloatArray:
        g = panel["gross_profitability"]
        return ops.hold_last_event(g, np.isfinite(g), self.hold)


@dataclass(frozen=True, slots=True)
class VolatilitySmirk:
    """Steep implied-volatility smirks predict low returns (Xing, Zhang & Zhao 2010): score =
    -(IV of the ~25-delta put - IV of the ATM call), 30-day expiration. Needs the panel field
    ``smirk`` (``data.optskew``; at most a week old)."""

    name: str = "iv_smirk"

    @property
    def lookback(self) -> int:
        return 1

    def compute(self, panel: Panel) -> FloatArray:
        return -np.asarray(panel["smirk"], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolatilitySpread:
    """Call-minus-put implied volatility at matched strikes predicts returns (Cremers &
    Weinbaum 2010; options trade on information first). Needs the field ``vol_spread``."""

    name: str = "iv_spread"

    @property
    def lookback(self) -> int:
        return 1

    def compute(self, panel: Panel) -> FloatArray:
        return np.asarray(panel["vol_spread"], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolatilityTermSlope:
    """An upward-sloping implied-volatility term structure (90-day ATM IV above 30-day)
    predicts high returns (Vasquez 2017). Needs the field ``term_slope``."""

    name: str = "iv_term"

    @property
    def lookback(self) -> int:
        return 1

    def compute(self, panel: Panel) -> FloatArray:
        return np.asarray(panel["term_slope"], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ShortPressure:
    """Abnormal short selling predicts low returns (Diether, Lee & Werner 2009): score =
    -(short share of FINRA-reported volume over the last ``fast`` sessions - the same share over
    the last ``slow``). Measured against the stock's own history because the level is set by
    market makers' routine shorting. Needs ``short_volume`` and ``finra_volume``
    (``data.shortvol``, already lagged one session)."""

    fast: int = 5
    slow: int = 60
    name: str = "short_pressure"

    @property
    def lookback(self) -> int:
        return self.slow + 1

    def compute(self, panel: Panel) -> FloatArray:
        return -(_short_share(panel, self.fast) - _short_share(panel, self.slow))


def _short_share(panel: Panel, w: int) -> FloatArray:
    """Short share of FINRA-reported volume over the last ``w`` sessions; 80% of them must have
    a file."""
    s = panel["short_volume"]
    v = np.where(np.isnan(s), np.nan, panel["finra_volume"])  # the same sessions in both
    num, den = ops.rolling_sum(s, w, int(w * 0.8)), ops.rolling_sum(v, w, int(w * 0.8))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclass(frozen=True, slots=True)
class ShortShare:
    """Heavily shorted stocks underperform (Boehmer, Jones & Zhang 2008): score = -(short share
    of FINRA-reported volume over the last ``window`` sessions). Added after us-shortsale-v1,
    whose exploratory diagnostics showed the level, not the change, related to next-month
    returns (FINDINGS §3e)."""

    window: int = 5
    name: str = "short_share"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def compute(self, panel: Panel) -> FloatArray:
        return -_short_share(panel, self.window)


@dataclass(frozen=True, slots=True)
class Planted:
    """The synthetic market's planted score (tests only)."""

    name: str = "planted"

    @property
    def lookback(self) -> int:
        return 1

    def compute(self, panel: Panel) -> FloatArray:
        return np.array(panel["planted"], dtype=np.float64)


# ---------------------------------------------------------------------- continuous trend forecasts
def _pooled_mean_abs(x: FloatArray, member: FloatArray | None) -> FloatArray:
    """Expanding mean of |x| over all index members up to each row (T), causal and point in
    time: a security's bars from before it joined the index do not set the scale."""
    a = np.abs(x)
    ok = np.isfinite(a) if member is None else np.isfinite(a) & (member > 0)
    total = np.cumsum(np.where(ok, a, 0.0).sum(axis=1))
    count = np.cumsum(ok.sum(axis=1)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _scaled(raw: FloatArray, member: FloatArray | None, target: float, cap: float) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        scalar = target / _pooled_mean_abs(raw, member)
    return np.clip(raw * scalar[:, None], -cap, cap)


def ewma_daily_vol(close: FloatArray, lam: float = 0.94, floor_annual: float = 0.05) -> FloatArray:
    """RiskMetrics EWMA of daily log returns blended with its own long-run mean (Carver's 70/30:
    less noise and less turnover than the EWMA alone), floored. Causal."""
    r = ops.log_return(close)
    v = pd.DataFrame(r * r).ewm(alpha=1.0 - lam, adjust=False, ignore_na=True).mean()
    s = pd.DataFrame(np.sqrt(v.to_numpy()))
    long_run = s.rolling(2520, min_periods=1).mean()
    blended = (0.7 * s + 0.3 * long_run).to_numpy()
    floor = floor_annual / math.sqrt(252.0)
    with np.errstate(invalid="ignore"):
        return np.where(blended < floor, floor, blended)


@dataclass(frozen=True, slots=True)
class TrendForecast:
    """Continuous multi-rule trend forecast with average magnitude 10, capped at 20 (Carver 2015).

    EWMA crossovers, momentum measured in units of its own volatility (Moskowitz, Ooi & Pedersen
    2012) and range breakouts (Hurst, Ooi & Pedersen 2017), weighted equally between families and
    within them. Each rule is scaled by its pooled expanding mean; the combination is multiplied
    by the forecast diversification multiplier of the rules' pooled expanding correlation (no
    credit until ``MIN_PAIR_OBS`` pairs are seen). Nihai3's rules with its registered values.
    """

    ewmac_pairs: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))
    mom_horizons: tuple[int, ...] = (21, 63, 252)
    breakout_windows: tuple[int, ...] = (40, 80, 160, 320)
    target_abs: float = 10.0
    cap: float = 20.0
    fdm_cap: float = 2.5
    name: str = "trend_forecast"

    MIN_PAIR_OBS: ClassVar[int] = 500

    @property
    def lookback(self) -> int:
        slow = max(s for _, s in self.ewmac_pairs)
        return max(slow, *self.mom_horizons, *self.breakout_windows) + 1

    def _rules(self, panel: Panel) -> tuple[list[FloatArray], list[float]]:
        c = panel["close"]
        lc = np.log(c)
        sigma = ewma_daily_vol(c)
        hist = np.cumsum(np.isfinite(c), axis=0)
        df = pd.DataFrame(c)
        rules: list[FloatArray] = []
        weights: list[float] = []
        for fast, slow in self.ewmac_pairs:
            ef = df.ewm(span=fast, adjust=False, ignore_na=True).mean().to_numpy()
            es = df.ewm(span=slow, adjust=False, ignore_na=True).mean().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = (ef - es) / (sigma * c)
            rules.append(np.where(hist > slow, raw, np.nan))
            weights.append(1.0 / 3.0 / len(self.ewmac_pairs))
        for h in self.mom_horizons:
            with np.errstate(invalid="ignore", divide="ignore"):
                rules.append((lc - ops.lag(lc, h)) / (sigma * math.sqrt(h)))
            weights.append(1.0 / 3.0 / len(self.mom_horizons))
        for n in self.breakout_windows:
            hi, lo = ops.rolling_max(c, n, n), ops.rolling_min(c, n, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = (c - (hi + lo) / 2.0) / np.where(hi > lo, hi - lo, np.nan)
            smooth = (
                pd.DataFrame(raw).ewm(span=max(n // 4, 1), adjust=False, ignore_na=True).mean()
            ).to_numpy()
            rules.append(np.where(np.isfinite(hi) & np.isfinite(c), smooth, np.nan))
            weights.append(1.0 / 3.0 / len(self.breakout_windows))
        return rules, weights

    def compute(self, panel: Panel) -> FloatArray:
        member = universe_mask(panel)
        rules, weights = self._rules(panel)
        f = [_scaled(r, member, self.target_abs, self.cap) for r in rules]
        T, N = f[0].shape
        R = len(f)
        base = np.array(weights)
        out = np.full((T, N), np.nan)
        # pooled expanding pairwise moments, updated row by row (R x R)
        n_ab = np.zeros((R, R))
        s_a = np.zeros((R, R))
        s_aa = np.zeros((R, R))
        s_ab = np.zeros((R, R))
        for t in range(T):
            ft = np.stack([x[t] for x in f], axis=1)  # N x R
            avail = np.isfinite(ft)
            pooled = avail if member is None else avail & (member[t] > 0)[:, None]
            m = pooled.astype(np.float64)
            x = np.where(pooled, ft, 0.0)
            n_ab += m.T @ m
            s_a += x.T @ m
            s_aa += (x * x).T @ m
            s_ab += x.T @ x
            with np.errstate(invalid="ignore", divide="ignore"):
                ma, mb = s_a / n_ab, s_a.T / n_ab
                va, vb = s_aa / n_ab - ma * ma, s_aa.T / n_ab - mb * mb
                rho = (s_ab / n_ab - ma * mb) / np.sqrt(va * vb)
            # an unknown correlation counts as 1 (no diversification credit), a negative one as 0
            known = np.isfinite(rho) & (n_ab >= self.MIN_PAIR_OBS)
            rho = np.where(known, np.maximum(rho, 0.0), 1.0)
            np.fill_diagonal(rho, 1.0)
            W = np.where(avail, base, 0.0)
            tot = W.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                W = np.where(tot > 0, W / tot, 0.0)
            q = np.einsum("ia,ab,ib->i", W, rho, W)
            fdm = np.minimum(1.0 / np.sqrt(np.maximum(q, 1e-12)), self.fdm_cap)
            comb = (W * np.where(avail, ft, 0.0)).sum(axis=1) * fdm
            out[t] = np.where(tot[:, 0] > 0, np.clip(comb, -self.cap, self.cap), np.nan)
        return out


@dataclass(frozen=True, slots=True)
class TrendSignForecast:
    """NihaiAlgoritma's ``quant_trend`` score as a forecast: the mean sign over the horizons,
    scaled to average magnitude 10. The heritage signal, for the portfolio engine of
    :class:`pusula.portfolio.strategies.ForecastRiskParity`."""

    horizons: tuple[int, ...] = (21, 63, 252)
    target_abs: float = 10.0
    cap: float = 20.0
    name: str = "trend_sign_forecast"

    @property
    def lookback(self) -> int:
        return max(self.horizons) + 1

    def compute(self, panel: Panel) -> FloatArray:
        raw = TimeSeriesMomentum(self.horizons).compute(panel)
        return _scaled(raw, universe_mask(panel), self.target_abs, self.cap)


@dataclass(frozen=True, slots=True)
class ConstantForecast:
    """Always long at the average forecast: the portfolio engine with no view at all, the passive
    book a forecast has to beat (QuantBot's "passive" expert)."""

    value: float = 10.0
    name: str = "constant_forecast"

    @property
    def lookback(self) -> int:
        return 1

    def compute(self, panel: Panel) -> FloatArray:
        return np.where(np.isfinite(panel["close"]), self.value, np.nan)


REGISTRY: dict[str, Callable[[], Signal]] = {
    "momentum": Momentum,
    "constant_forecast": ConstantForecast,
    "ts_momentum": TimeSeriesMomentum,
    "reversal": ShortTermReversal,
    "low_vol": LowVolatility,
    "residual_momentum": ResidualMomentum,
    "high_52w": FiftyTwoWeekHigh,
    "bab": BettingAgainstBeta,
    "overnight": OvernightMomentum,
    "trend_forecast": TrendForecast,
    "trend_sign_forecast": TrendSignForecast,
}


# Signals that need option-market data (panel field ``iv``); kept apart from REGISTRY, whose
# members need prices only.
OPTION_SIGNALS: dict[str, Callable[[], Signal]] = {"iv_rv_spread": ImpliedRealizedSpread}

# Signals that need company filings (panel fields from ``data.fundamentals``).
FUNDAMENTAL_SIGNALS: dict[str, Callable[[], Signal]] = {
    "ear": EarningsAnnouncementReturn,
    "sue": EarningsSurprise,
    "accruals": LowAccruals,
    "gross_profitability": GrossProfitability,
}


# Signals that need the option surface (strike-level chains reduced by ``data.optskew``).
SURFACE_SIGNALS: dict[str, Callable[[], Signal]] = {
    "iv_smirk": VolatilitySmirk,
    "iv_spread": VolatilitySpread,
    "iv_term": VolatilityTermSlope,
}


# Signals that need FINRA short-sale volume (``data.shortvol``).
SHORT_SIGNALS: dict[str, Callable[[], Signal]] = {
    "short_pressure": ShortPressure,
    "short_share": ShortShare,
}


def build(name: str, **params: object) -> Signal:
    if name == "planted":
        return Planted()
    if name in SHORT_SIGNALS:
        return SHORT_SIGNALS[name](**params)
    if name in OPTION_SIGNALS:
        return OPTION_SIGNALS[name](**params)
    if name in FUNDAMENTAL_SIGNALS:
        return FUNDAMENTAL_SIGNALS[name](**params)
    if name in SURFACE_SIGNALS:
        return SURFACE_SIGNALS[name](**params)
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown signal {name!r}; known: {sorted(REGISTRY)}") from None
    return cls(**params)
