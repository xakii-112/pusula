"""Registered study families. Editing a registered trial changes its hash and the ledger
refuses it: a changed idea is a new trial with a new name, counted in the deflated Sharpe.

``us-equity-v1`` was written on 2026-10-23 before any strategy was run on real data. Its only
real-data inputs were the data-quality report and the spread calibration against SIP quotes
(``docs/CALIBRATION.md``), which set ``CostModel.spread_k``; neither looks at returns.
"""

from __future__ import annotations

import math
from dataclasses import replace

from pusula.alpha.combine import SignalSpec
from pusula.data.crypto import BTC, CRYPTO_FACTORS, UNIVERSE
from pusula.data.pipeline import CROSS_ASSET
from pusula.engine.backtest import BacktestConfig
from pusula.execution.costs import CostModel
from pusula.ml.walkforward import WalkForwardSpec
from pusula.portfolio.optimizer import OptimizerSpec
from pusula.portfolio.strategies import ForecastSpec
from pusula.research.gates import GateSpec
from pusula.research.study import StudySpec, Trial
from pusula.risk.limits import RiskLimits
from pusula.risk.model import RiskModelSpec

DAILY = 1.0 / math.sqrt(252.0)

# Spread model calibrated 2026-10-23 on 439 SIP NBBO samples (11 sessions 2016-2026); see
# docs/CALIBRATION.md. Impact (Y = 0.7) is the literature value: no trade-level data to fit it,
# which is why the gate doubles all costs as a stress.
COSTS_V1 = CostModel()

BACKTEST_V1 = BacktestConfig(
    initial_equity=10_000_000.0,
    lag=1,
    exec_price="vwap",
    start="2019-01-02",
    end="2026-10-22",
    cost_model=COSTS_V1,
    risk_spec=RiskModelSpec(n_factors=12, window=252),
    limits=RiskLimits(
        max_gross=2.0,
        max_net=1.0,
        min_net=-1.0,
        max_weight=0.05,
        min_price=5.0,
        max_participation=0.05,
        max_turnover=2.0,
    ),
    kill_switch=True,
    max_drawdown=0.25,
    max_daily_loss=0.08,
)

LS = OptimizerSpec(
    risk_aversion=1.0,
    max_gross=2.0,
    net_lo=0.0,
    net_hi=0.0,
    max_weight=0.015,
    beta_max=0.05,
    vol_target_daily=0.10 * DAILY,
    n_factors=12,
)
LONG_ONLY = OptimizerSpec(
    risk_aversion=4.0,
    max_gross=1.0,
    net_lo=1.0,
    net_hi=1.0,
    long_only=True,
    max_weight=0.02,
    beta_max=None,
    vol_target_daily=None,
    n_factors=12,
)

COMPOSITE = (
    SignalSpec("momentum"),
    SignalSpec("residual_momentum"),
    SignalSpec("reversal", smooth_halflife=3.0),
    SignalSpec("low_vol"),
    SignalSpec("high_52w"),
    SignalSpec("overnight"),
)
ML_FEATURES = (*COMPOSITE, SignalSpec("bab"))


def us_equity_v1() -> StudySpec:
    trials = (
        Trial(
            "trend_heritage",
            "trend",
            "NihaiAlgoritma's live quant_trend (21/63/252-day trend sign, inverse-vol sizing, 15% vol "
            "target, long-only) run as a portfolio earns a positive net Sharpe with alpha beyond "
            "FF5+MOM.",
            signals=(SignalSpec("ts_momentum"),),
            rebalance_every=5,
            long_only=True,
            vol_target_annual=0.15,
        ),
        Trial(
            "xs_momentum",
            "optimized",
            "12-1 cross-sectional momentum, dollar- and beta-neutral at 10% ex-ante vol, earns net alpha.",
            signals=(SignalSpec("momentum"),),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_residual_momentum",
            "optimized",
            "Momentum of market-residual returns (Blitz-Huij-Martens) earns net alpha, market neutral.",
            signals=(SignalSpec("residual_momentum"),),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_reversal",
            "optimized",
            "One-week reversal survives realistic spread + impact costs when traded through a "
            "cost-aware optimiser.",
            signals=(SignalSpec("reversal"),),
            optimizer=LS,
            ic=0.02,
            horizon_days=5.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_low_vol",
            "optimized",
            "Low-volatility names outperform high-volatility names at equal beta (BAB-style).",
            signals=(SignalSpec("low_vol"),),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_composite",
            "optimized",
            "An equal-weight composite of six published anomalies is more robust than any one of them.",
            signals=COMPOSITE,
            optimizer=LS,
            ic=0.02,
            horizon_days=10.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_ml_ridge",
            "optimized",
            "A walk-forward ridge combination of the anomaly signals beats the equal-weight composite.",
            ml=WalkForwardSpec(
                ML_FEATURES, model="ridge", horizon=5, refit_every=63, min_train=504
            ),
            optimizer=LS,
            ic=0.02,
            horizon_days=5.0,
            rebalance_every=5,
        ),
        Trial(
            "xs_ml_gbm",
            "optimized",
            "Shallow gradient-boosted trees find non-linear signal interactions that ridge misses.",
            ml=WalkForwardSpec(ML_FEATURES, model="gbm", horizon=5, refit_every=63, min_train=504),
            optimizer=LS,
            ic=0.02,
            horizon_days=5.0,
            rebalance_every=5,
        ),
        Trial(
            "composite_long_only",
            "optimized",
            "A long-only, fully invested composite tilt earns alpha beyond FF5+MOM after costs.",
            signals=COMPOSITE,
            optimizer=LONG_ONLY,
            ic=0.02,
            horizon_days=10.0,
            rebalance_every=5,
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-equity-v1",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_V1,
    )


# --------------------------------------------------------------------------------------------
# us-equity-v2 (registered 2026-10-24, after v1's results, before any v2 result).
#
# v1 ran research backtests with the production kill switch, which latches: three trials were
# flattened in 2020 (the March crash; the 9 Nov 2020 momentum crash hit the 8% daily-loss
# limit) and spent the rest of the sample in T-bills, so their holdout measured cash, not the
# strategy. The trials below are v1's, unchanged; only the research configuration drops the
# latch (drawdown is still judged by the gate's 25% limit and reported with the worst day).
# Forward paper books keep the kill switch. Because v2 exists only because v1 was seen, its
# deflated Sharpe counts v1's and v2's trials together (programme "us-equity-").
BACKTEST_V2 = replace(BACKTEST_V1, kill_switch=False)


def us_equity_v2() -> StudySpec:
    return replace(us_equity_v1(), family="us-equity-v2", backtest=BACKTEST_V2)


# --------------------------------------------------------------------------------------------
# us-optsignal-v1 (registered 2026-10-24 before it was run): equity signals from the option
# market, NihaiAlgoritma's "hisse & opsiyon" idea on the alpha side. Implied vol history starts
# 2019-02 and ends 2026-08-26, which sets the windows. Same research configuration as v2.
BACKTEST_OPT = replace(BACKTEST_V2, start="2019-06-03", end="2026-08-26")


def us_optsignal_v1() -> StudySpec:
    trials = (
        Trial(
            "iv_rv_spread",
            "optimized",
            "Stocks whose 30-day ATM implied vol is high relative to realised vol underperform "
            "(Bali & Hovakimian 2009); a dollar- and beta-neutral book on -(IV - RV) earns net alpha.",
            signals=(SignalSpec("iv_rv_spread"),),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        Trial(
            "iv_rv_plus_momentum",
            "optimized",
            "The implied-realised spread adds to 12-1 momentum: their equal-weight composite beats "
            "momentum alone (momentum is the strongest single sleeve of us-equity-v2).",
            signals=(SignalSpec("iv_rv_spread"), SignalSpec("momentum")),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-optsignal-v1",
        trials=trials,
        dev=("2019-06-03", "2022-12-30"),
        holdout=("2023-01-03", "2026-08-26"),
        gates=GateSpec(),
        backtest=BACKTEST_OPT,
    )


# --------------------------------------------------------------------------------------------
# us-equity-v3 (registered 2026-10-24, after v2's results). Two mechanisms found in v2, not new
# signals: (1) the optimiser over-traded weak signals (a fixed IC of 0.02 against realised ICs of
# 0.00-0.023; capacity diagnostic: Sharpe rose as size forced slower trading), so alpha is scaled
# by each score's trailing realised IC (causal, shrunk, floored at 0); (2) market-neutral books
# realised 12.3-12.5% against a 10% ex-ante target, so the target is divided by the held book's
# realised/model volatility ratio. Same signals, optimiser and costs as v2. v2's holdout has been
# seen, so v3's holdout is NOT a clean out-of-sample test: its deflated Sharpe pays for all 21
# trials of the programme, and any promotion is provisional until its forward record passes.
def _v3(t: Trial) -> Trial:
    return replace(t, ic_mode="adaptive", ic_window=252, ic_prior_days=252, vol_feedback=True)


def us_equity_v3() -> StudySpec:
    trials = (
        _v3(
            Trial(
                "xs_momentum_adaptive",
                "optimized",
                "12-1 momentum with alpha scaled by its trailing realised IC and a realised-vol "
                "feedback on the risk target beats v2's fixed-IC momentum sleeve after costs, with "
                "realised volatility close to the 10% target.",
                signals=(SignalSpec("momentum"),),
                optimizer=LS,
                ic=0.02,
                horizon_days=21.0,
                rebalance_every=5,
            )
        ),
        _v3(
            Trial(
                "xs_residual_momentum_adaptive",
                "optimized",
                "Residual momentum with adaptive IC and vol feedback improves on its v2 "
                "counterpart.",
                signals=(SignalSpec("residual_momentum"),),
                optimizer=LS,
                ic=0.02,
                horizon_days=21.0,
                rebalance_every=5,
            )
        ),
        _v3(
            Trial(
                "xs_composite_adaptive",
                "optimized",
                "Scaling the six-anomaly composite by its own realised IC stops the optimiser "
                "paying spread for signals that are not working, and beats v2's composite.",
                signals=COMPOSITE,
                optimizer=LS,
                ic=0.02,
                horizon_days=10.0,
                rebalance_every=5,
            )
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-equity-v3",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_V2,
    )


# --------------------------------------------------------------------------------------------
# us-fundamental-v1 (written 2026-10-24, registered before any of its signals was evaluated on
# real returns). FINDINGS §5's first hypothesis: different information, not new combinations of
# price anomalies. Company filings from SEC EDGAR, point in time to the second of acceptance
# (data/fundamentals.py). Two signals use the earnings release itself (the market's 3-day
# reaction and the standardised surprise), two use the annual accounts. Same windows, costs,
# optimiser and research configuration as v2, plus v3's realised-vol feedback (the one v3
# mechanism that worked); fixed IC, because v3's adaptive IC did not help. The holdout is clean
# for these signals but not for the platform: it pays, through the "us-" programme, for every
# trial run on this universe before it.
FUND_HOLD = {"ear": 63, "sue": 63, "accruals": 315, "gross_profitability": 315}


def _fund(name: str, hypothesis: str, *signals: str) -> Trial:
    return Trial(
        name,
        "optimized",
        hypothesis,
        signals=tuple(SignalSpec(s, params={"hold": FUND_HOLD[s]}) for s in signals),
        optimizer=LS,
        ic=0.02,
        horizon_days=21.0,
        rebalance_every=5,
        vol_feedback=True,
    )


def us_fundamental_v1() -> StudySpec:
    trials = (
        _fund(
            "ear_drift",
            "Stocks with a high market-adjusted return over the three sessions around their "
            "earnings release (8-K Item 2.02) keep outperforming over the next quarter (Chan, "
            "Jegadeesh & Lakonishok 1996; Brandt et al. 2008): a dollar- and beta-neutral book "
            "on that reaction earns net alpha beyond FF5+MOM.",
            "ear",
        ),
        _fund(
            "sue_drift",
            "Standardised unexpected quarterly earnings, as first reported in the 10-Q/10-K "
            "(seasonal random walk with drift), predict returns over the following quarter "
            "(Bernard & Thomas 1989).",
            "sue",
        ),
        _fund(
            "low_accruals",
            "Firms whose earnings exceed their operating cash flow (high accruals, scaled by "
            "assets) underperform over the next year (Sloan 1996); financials excluded.",
            "accruals",
        ),
        _fund(
            "earnings_composite",
            "The price reaction and the accounting surprise are two noisy measures of the same "
            "news; their equal-weight composite beats either alone.",
            "ear",
            "sue",
        ),
        _fund(
            "fundamental_composite",
            "Announcement drift, low accruals and gross profitability (Novy-Marx 2013) are "
            "different information; their equal-weight composite earns net alpha beyond "
            "FF5+MOM.",
            "ear",
            "sue",
            "accruals",
            "gross_profitability",
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-fundamental-v1",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_V2,
    )


# --------------------------------------------------------------------------------------------
# us-optsurface-v1 (written 2026-10-24, registered before any of its signals was evaluated on
# returns). FINDINGS §5: the option market's *shape*, not the ATM level that us-optsignal-v1
# tested. Features from DoltHub's strike-level option chains (data/optskew.py), delta-based.
# Rule fixed before the data was built: a feature gets a trial only if it covers at least 50%
# of index-member sessions in both windows. Coverage (FUNDAMENTALS-style check, no returns):
# smirk 91% / 94%, volatility spread 96% / 98%, term slope 17% / 20% -> no term-slope trial.
# Horizons are the papers' (smirk: a month; Cremers-Weinbaum spread: a week). Same research
# configuration as us-fundamental-v1. Not a forward family: the store is a one-off stream.
BACKTEST_SURFACE = replace(BACKTEST_V2, start="2019-06-03", end="2026-10-22")


def _surface(name: str, hypothesis: str, horizon: float, *signals: str) -> Trial:
    return Trial(
        name,
        "optimized",
        hypothesis,
        signals=tuple(SignalSpec(s) for s in signals),
        optimizer=LS,
        ic=0.02,
        horizon_days=horizon,
        rebalance_every=5,
        vol_feedback=True,
    )


def us_optsurface_v1() -> StudySpec:
    trials = (
        _surface(
            "iv_smirk_ls",
            "Stocks whose 30-day out-of-the-money put implied vol is high relative to the ATM "
            "call's (a steep smirk: informed demand for downside protection) underperform over "
            "the next month (Xing, Zhang & Zhao 2010); a dollar- and beta-neutral book on the "
            "negative smirk earns net alpha beyond FF5+MOM.",
            21.0,
            "iv_smirk",
        ),
        _surface(
            "iv_spread_ls",
            "Call-minus-put implied vol at matched strikes (options trade on information before "
            "the stock) predicts next-week returns (Cremers & Weinbaum 2010), net of costs.",
            5.0,
            "iv_spread",
        ),
        _surface(
            "surface_composite",
            "Smirk and volatility spread are different reads of option-market information; "
            "their equal-weight composite beats either alone.",
            10.0,
            "iv_smirk",
            "iv_spread",
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-optsurface-v1",
        trials=trials,
        dev=("2019-06-03", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_SURFACE,
    )


# --------------------------------------------------------------------------------------------
# us-shortsale-v1 (written 2026-10-24, registered before any of its signals was evaluated on
# returns). FINDINGS §5's "different information", from the one free daily source of trading
# intent not yet used: FINRA's Reg SHO short-sale volume (data/shortvol.py; files since
# 2018-08-01, lagged one session). The level of a stock's short share is mostly market makers
# filling buy orders, so the signal is the change against the stock's own history. Rule fixed
# before the store was built: a variant gets a trial only if it covers at least 50% of
# index-member sessions in both windows. Coverage (SHORTVOL_DATA.md, no returns read), after
# members that traded under earlier tickers were mapped (data/ticker_history.py: 26 members,
# 7 of whose current tickers carried another company's volume before): 5/60 99.9% / 99.9%,
# 20/120 99.7% / 99.8% -> both qualify. Two horizons, not a grid: a week (Diether, Lee &
# Werner 2009) and a month, with windows scaled to match. Same research configuration as
# us-fundamental-v1. A forward family: one small file a night keeps the store current.
def _short(name: str, hypothesis: str, horizon: float, fast: int, slow: int) -> Trial:
    return Trial(
        name,
        "optimized",
        hypothesis,
        signals=(SignalSpec("short_pressure", params={"fast": fast, "slow": slow}),),
        optimizer=LS,
        ic=0.02,
        horizon_days=horizon,
        rebalance_every=5,
        vol_feedback=True,
    )


def us_shortsale_v1() -> StudySpec:
    trials = (
        _short(
            "short_pressure_week",
            "Stocks whose short share of FINRA-reported volume over the last week is high "
            "relative to their own last three months underperform over the next week (Diether, "
            "Lee & Werner 2009: short sellers are informed and trade against short-term "
            "overreaction); a dollar- and beta-neutral book on the negative change earns net "
            "alpha beyond FF5+MOM.",
            5.0,
            5,
            60,
        ),
        _short(
            "short_pressure_month",
            "The same change measured over a month against the last six months predicts the "
            "next month's returns, at a turnover low enough to survive costs.",
            21.0,
            20,
            120,
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-shortsale-v1",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_V2,
    )


# --------------------------------------------------------------------------------------------
# us-shortsale-v2 (registered 2026-10-24, AFTER us-shortsale-v1's results and an exploratory
# look at the short share's level over both windows: FINDINGS §3e item 3). One trial, exactly
# the measure that was looked at, not tuned further: the negative 5-day short share, a month's
# horizon (the IC was largest at 21 sessions, which is part of what was seen). Its holdout is NOT
# a clean out-of-sample test; any promotion waits for its forward record, as for us-equity-v3.
def us_shortsale_v2() -> StudySpec:
    trials = (
        Trial(
            "short_share_level",
            "optimized",
            "Stocks with a low share of short sales in their FINRA-reported volume over the last "
            "week outperform over the next month (Boehmer, Jones & Zhang 2008: heavily shorted "
            "stocks underperform), beyond liquidity, momentum and volatility; a dollar- and "
            "beta-neutral book earns net alpha beyond FF5+MOM in its forward record.",
            signals=(SignalSpec("short_share", params={"window": 5}),),
            optimizer=LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
            vol_feedback=True,
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return replace(us_shortsale_v1(), family="us-shortsale-v2", trials=trials)


# --------------------------------------------------------------------------------------------
# us-trend-v1 (registered 2026-10-30, before it was run on this data): the portfolio engine of
# Nihai3 (Desktop/algo). There the old sign score through this engine doubled its Sharpe against
# NihaiAlgoritma's own portfolio, on today's survivors at a flat 10 bp a side, and it passed four
# pre-registered gates; pusula never tested it. Long-only, unlevered, 15% volatility target, as
# trend_heritage, so that a difference is the portfolio construction. Nihai3's and trend_heritage's
# results were known at registration, so its holdout is not a clean out-of-sample test: any
# promotion waits for its forward record, as for us-equity-v3.
def _forecast(name: str, hypothesis: str, signal: str, spec: ForecastSpec) -> Trial:
    return Trial(
        name,
        "forecast",
        hypothesis,
        signals=(SignalSpec(signal),),
        rebalance_every=1,
        forecast=spec,
    )


def us_trend_v1() -> StudySpec:
    trials = (
        _forecast(
            "forecast_hrp",
            "A continuous multi-rule trend forecast sized by hierarchical risk parity, a volatility "
            "target and a no-trade buffer earns net alpha beyond FF5+MOM on the point-in-time "
            "S&P 500 and beats both trend_heritage and the equal-weight benchmark on Sharpe.",
            "trend_forecast",
            ForecastSpec(),
        ),
        _forecast(
            "signonly_hrp",
            "NihaiAlgoritma's trend sign score through the same engine beats trend_heritage (the "
            "same signal, inverse-vol weights, weekly full rebalance) on Sharpe: the gain Nihai3 "
            "measured comes from portfolio construction and survives a survivorship-free universe.",
            "trend_sign_forecast",
            ForecastSpec(),
        ),
        _forecast(
            "forecast_equal_risk",
            "Equal risk between securities instead of hierarchical risk parity does no worse: on "
            "500 stocks, correlation clusters add nothing beyond the volatility scaling.",
            "trend_forecast",
            ForecastSpec(weighting="equal"),
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal-weight point-in-time S&P 500.",
            rebalance_every=21,
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-trend-v1",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_V2,
    )


# --------------------------------------------------------------------------------------------
# us-crossasset-v1 (registered 2026-10-30, before it was run on this data; its machinery was only
# run on 2017-2018, outside both windows). FINDINGS §3g: Nihai3's engine does not help on 500
# stocks, which share one market factor. Time-series trend has its evidence across asset classes
# (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017), and that is where correlation
# clusters are real. QuantBot ran such a universe and ended level with holding it; it never paid
# for its trials. The universe is QuantBot's, fixed by its return-blind rule. 10% volatility, as in
# that literature; Reg T gross with shorts, unlevered without. Three factors for the risk model:
# 18 funds cannot support the index's twelve. FF5+MOM are US equity factors, so the alpha check is
# lenient here; the second benchmark, the same engine always long, is the stricter comparison.
# Nihai3's and QuantBot's results were known at registration, so the holdout is not a clean test.
BACKTEST_CROSS = replace(
    BACKTEST_V2,
    risk_spec=RiskModelSpec(n_factors=3, window=252),
    limits=replace(BACKTEST_V2.limits, max_weight=0.30),
)
CROSS_LONG_SHORT = ForecastSpec(target_vol=0.10, max_weight=0.30, max_gross=2.0, long_only=False)
CROSS_LONG = ForecastSpec(target_vol=0.10, max_weight=0.30, max_gross=1.0)


def us_crossasset_v1() -> StudySpec:
    trials = (
        _forecast(
            "ca_trend_ls",
            "A continuous multi-rule trend forecast, long and short, across 18 asset-class ETFs, "
            "sized by hierarchical risk parity at 10% volatility, earns net alpha beyond FF5+MOM "
            "and beats both benchmarks on Sharpe.",
            "trend_forecast",
            CROSS_LONG_SHORT,
        ),
        _forecast(
            "ca_trend_long",
            "The same forecast without shorts, standing aside instead, keeps most of the return "
            "at a lower cost: most of these funds rise over time, and shorting them pays for it.",
            "trend_forecast",
            CROSS_LONG,
        ),
        _forecast(
            "ca_trend_equal_risk",
            "Equal risk between funds instead of hierarchical risk parity does worse: between "
            "asset classes correlation clusters are real, unlike on 500 stocks (FINDINGS §3g).",
            "trend_forecast",
            replace(CROSS_LONG_SHORT, weighting="equal"),
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal weight of the 18 ETFs.",
            rebalance_every=21,
            benchmark=True,
        ),
        replace(
            _forecast(
                "risk_parity_long",
                "Benchmark: the same engine and volatility target, always long, with no forecast.",
                "constant_forecast",
                CROSS_LONG,
            ),
            benchmark=True,
        ),
    )
    return StudySpec(
        family="us-crossasset-v1",
        trials=trials,
        dev=("2019-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_CROSS,
        universe=CROSS_ASSET,
    )


# --------------------------------------------------------------------------------------------
# us-crypto-v1 (registered 2026-10-30, before it was run on this data; its machinery was only run on
# 2019, outside both windows). FINDINGS §5's wider universe, the one that is free: every USDT spot
# pair Binance has listed, dead ones included, top 100 by dollar volume, reconstituted monthly
# (data/crypto.py). Costs are crypto's: 10 bp taker fee, a spread model fitted on 100 live books
# (tools/crypto_spreads.py, one snapshot), impact Y = 1.0 until a calibration says otherwise, 10% a
# year to be short (spot cannot short at all: a short here stands for a perpetual, and the trial says
# so). FF5+MOM would call any crypto holding alpha, so the alpha check is against the US market,
# Bitcoin and the equal-weight crypto market. A latching kill switch at the stock books' 25% would stop
# every crypto book within weeks, so the forward books get 60%; the gate's 25% drawdown limit stands
# and the long-only books are expected to struggle with it. It pays through the "us-" programme.
COSTS_CRYPTO = replace(
    COSTS_V1,
    spread_slopes=(0.4496, -0.6814, 0.0),
    spread_year_intercepts=((2018, 3.7259),),
    commission_bps=10.0,
    sec_fee_rate=0.0,
    borrow_annual=0.10,
    financing_spread_annual=0.02,
    impact_y=1.0,
)
BACKTEST_CRYPTO = replace(
    BACKTEST_V2,
    start="2020-01-02",
    cost_model=COSTS_CRYPTO,
    risk_spec=RiskModelSpec(n_factors=5, window=252),
    limits=replace(BACKTEST_V2.limits, max_weight=1.0, min_price=0.01),
    max_drawdown=0.60,
    max_daily_loss=0.25,
    min_price=0.01,
)
CRYPTO_LS = replace(LS, max_weight=0.05, vol_target_daily=0.25 * DAILY, n_factors=5)


def us_crypto_v1() -> StudySpec:
    trials = (
        _forecast(
            "crypto_trend_long",
            "A continuous multi-rule trend forecast on the 100 most traded coins, long or in cash, "
            "at 25% volatility, earns net alpha beyond the US market, Bitcoin and the equal-weight "
            "crypto market (Liu & Tsyvinski 2021 and Hu et al. 2019 find momentum in crypto).",
            "trend_forecast",
            ForecastSpec(target_vol=0.25, max_weight=0.15, max_gross=1.0),
        ),
        _forecast(
            "crypto_trend_ls",
            "The same forecast long and short (a short stands for a perpetual future at 10% a year) "
            "adds what the falling coins pay, net of that cost.",
            "trend_forecast",
            ForecastSpec(target_vol=0.25, max_weight=0.15, max_gross=2.0, long_only=False),
        ),
        Trial(
            "crypto_momentum_ls",
            "optimized",
            "One-month cross-sectional momentum among coins, dollar- and beta-neutral at 25% ex-ante "
            "volatility, earns net alpha (Liu, Tsyvinski & Wu 2022: momentum at one to four weeks).",
            signals=(SignalSpec("momentum", params={"window": 21, "skip": 0}),),
            optimizer=CRYPTO_LS,
            ic=0.02,
            horizon_days=21.0,
            rebalance_every=5,
            vol_feedback=True,
        ),
        Trial(
            "equal_weight",
            "equal_weight",
            "Benchmark: monthly equal weight of the 100 most traded coins.",
            rebalance_every=21,
            benchmark=True,
        ),
        Trial(
            "btc_hold",
            "hold",
            "Benchmark: holding Bitcoin.",
            rebalance_every=21,
            benchmark=True,
            hold=BTC,
        ),
    )
    return StudySpec(
        family="us-crypto-v1",
        trials=trials,
        dev=("2020-01-02", "2022-12-30"),
        holdout=("2023-01-03", "2026-10-22"),
        gates=GateSpec(),
        backtest=BACKTEST_CRYPTO,
        universe=(UNIVERSE,),
        factor_names=CRYPTO_FACTORS,
    )


# Deflated-Sharpe programmes that are wider than "versions of one family": us-optsignal-v1,
# us-fundamental-v1, us-optsurface-v1 and the short-sale families add trials on the same
# universe and data as everything before them, so they pay for all of them (decided at
# registration, before they were run).
PROGRAMME_OVERRIDE = {
    "us-optsignal-v1": "us-",
    "us-fundamental-v1": "us-",
    "us-optsurface-v1": "us-",
    "us-shortsale-v1": "us-",
    "us-shortsale-v2": "us-",
    "us-trend-v1": "us-",
    "us-crossasset-v1": "us-",
    "us-crypto-v1": "us-",
}

FAMILIES = {
    "us-equity-v1": us_equity_v1,
    "us-equity-v2": us_equity_v2,
    "us-optsignal-v1": us_optsignal_v1,
    "us-equity-v3": us_equity_v3,
    "us-fundamental-v1": us_fundamental_v1,
    "us-optsurface-v1": us_optsurface_v1,
    "us-shortsale-v1": us_shortsale_v1,
    "us-shortsale-v2": us_shortsale_v2,
    "us-trend-v1": us_trend_v1,
    "us-crossasset-v1": us_crossasset_v1,
    "us-crypto-v1": us_crypto_v1,
}

# Families whose trials run as forward paper books every night. us-equity-v1 is left out: its
# books would be identical to v2's (production books always carry the kill switch). The
# option-implied family runs on the implied-vol store kept current by `data.iv.update_iv`, the
# fundamentals family on the filings store kept current by `data.fundamentals.update_fundamentals`,
# the short-sale family on the FINRA store kept current by `data.shortvol.update_shortvol`.
FORWARD_FAMILIES = (
    "us-equity-v2",
    "us-equity-v3",
    "us-optsignal-v1",
    "us-fundamental-v1",
    "us-shortsale-v1",
    "us-shortsale-v2",
    "us-trend-v1",
    "us-crossasset-v1",
    "us-crypto-v1",
)
