# Findings (interpretation of the registered results)

> Written after the results, by hand. The numbers come from the generated reports
> (`RESULTS_*.md`), which come from the ledger. Exploratory numbers are marked as such.

## 1. The answer

**No trial passed the promotion gate.** Across 50 registered trials (three equity versions, three
option expressions, two option-implied signals, five signals from company filings, three from
the option surface, three from daily short-sale volume, three of Nihai3's portfolio engine on the
point-in-time S&P 500, three of it across asset classes, three on crypto, four option strategies
at real quotes), 2019-2026, and four market makers on a real book,
net of calibrated costs, nothing
delivered alpha beyond the Fama-French five factors plus momentum with the statistical strength
the gate requires. The platform's job was to be able to say this with confidence, and to say
where each idea failed.

| programme | best trial (holdout 2023-01 → 2026-10, excess-return Sharpe) | why it failed the gate |
|---|---|---|
| us-equity-v2 | `xs_momentum` **0.94** (FF alpha 7.0%/yr, t 1.81; DSR 0.37) | the return is the momentum premium: loading 0.59 on MOM (t 21); DSR over the programme's 21 trials |
| us-equity-v3 | `xs_momentum_adaptive` 0.93 (DSR 0.36) | adaptive IC changed turnover, not skill; also fails PBO and stability |
| us-equity-v2 | `composite_long_only` 0.86 vs equal weight 0.71 | alpha t 0.18: a beta/factor tilt, not alpha |
| us-equity-v2 | `trend_heritage` (NihaiAlgoritma's live `quant_trend`) 0.51 | below the equal-weight benchmark (0.71); fails t, alpha, 2× cost |
| us-options-v1 | `calls_d50_t21` 0.46 vs stock mirror 0.81 | expressing the trend view in calls *loses* to holding the same names in stock |
| us-optsignal-v1 | `iv_rv_plus_momentum` 0.32 | implied-minus-realised vol spread alone: −0.81 |
| us-fundamental-v1 | `earnings_composite` 0.50 (development −0.58) | the sign flips between windows; alpha t 0.55 (§3c) |
| us-optsurface-v1 | `iv_smirk_ls` −1.05 (development −1.44) | no information (smirk IC of the wrong sign) and no persistence: ~90 turns a year (§3d) |
| us-shortsale-v1 | `short_pressure_month` −0.15 (development −0.24) | the *change* in a stock's short share carries no information (ICs within ±0.005, sign flips); the weekly version turns 57 times a year (§3e) |
| us-shortsale-v2 | `short_share_level` 0.82 (development −0.66; holdout already seen) | sign flips between windows; alpha t 1.30, fails 2× cost (56 turns a year); forward record only (§3f) |
| us-trend-v1 | `forecast_equal_risk` 0.32, `forecast_hrp` 0.31 vs `trend_heritage` 0.51 | Nihai3's engine does not beat the strategy it improved there once survivors are gone; lower drawdown and turnover, lower return (§3g) |
| us-crypto-v1 | `crypto_trend_long` 0.29 vs Bitcoin 1.02 (development 1.39 vs 0.75) | trend on the top 100 coins paid in 2020-2021 and not after; no alpha beyond Bitcoin and the crypto market (§3i) |
| us-optquote-v1 | `put_write` 0.91 (alpha t 0.32, DSR 0.09) | writing puts at real quotes is 0.37 of the market with less risk, not alpha; calls at real quotes earn nothing (§3j) |
| mm-crypto-v1 | every maker's breakeven fee is negative on every pair, −0.6 to −12 bp | adverse selection exceeds the spread on the books recorded, before any fee; a very short sample (§3k) |
| us-crossasset-v1 | `ca_trend_long` 0.25, `ca_trend_ls` −0.05 vs equal weight 0.84, always long 0.56 (development 0.95, 0.74 vs 0.57, 0.32) | trend across 18 asset-class ETFs paid in 2019-2022 and not after; over the whole window it is level with holding them (§3h) |

## 2. What the measurements say

1. **Momentum is the only robust signal, and it is a known factor.** Both momentum sleeves
   survive one-day execution delay (Sharpe 0.94 → 0.91) and doubled costs (NW t 2.06 → 1.90),
   PBO is 0.25 and every holdout year is positive (2023-2026: +9.7%, +26.9%, +10.3%,
   +17.7%). But it loads 0.59 on the Fama-French momentum factor (t 21; R² of the six-factor
   regression 0.62) and its remaining alpha (7.0%/yr) has t = 1.81. Buying
   that premium does not need a platform; beating it would. The development window
   (2019-2022) was *negative* for momentum (−0.12): a researcher looking at 2019-2022 alone would
   have dropped it, one looking at 2023-2026 alone would have over-sized it. The gate's DSR is
   what refuses to promote on 3.8 good years after 4 bad ones.
2. **Short-term reversal has a positive IC and loses money.** Its rank IC is positive in both
   windows (0.017 / 0.008, exploratory), but it turns the book over 87 times a year; at the
   calibrated spread + impact it loses 7.1% a year in the holdout. This is the cost model doing
   its job: NihaiAlgoritma's bar-close fills would have shown a profit.
3. **Machine learning did not beat the equal-weight composite.** Ridge 0.56, boosted trees 0.36,
   composite 0.71. The trees turned over 46 times a year and their edge vanished under 2× costs
   (t 0.04). Consistent with NihaiAlgoritma's 282 M-parameter network: with weak, noisy inputs,
   more flexible models fit more noise.
4. **The heritage strategy is a beta strategy.** `trend_heritage` loads 0.63 on the market
   (t 20) and has negative FF alpha (−3.5%/yr). As a long-only, vol-targeted book it earned
   10.2% a year at 11% vol in the holdout, below simply holding the equal-weight index.
5. **Options added cost, not convexity.** With premiums priced at observed ATM IV and a 2.5%
   half-spread, one- and two-month calls under-performed the stock mirror in every
   configuration and almost every year (2021: −12.0% vs +10.9%). Options are priced fairly
   enough that buying them on a weak directional signal transfers the signal's edge to the
   option seller.
6. **Low volatility / betting-against-beta failed in this window** (2023-2026 rewarded high
   beta): holdout Sharpe −0.15, IC negative (exploratory).

## 3. What the platform caught (before any of the above could be trusted)

* A first run reported holdout Sharpe ratios of **12, 12 and 83**. Two real defects: statistics
  on raw instead of excess returns, and a latching kill switch in research that parked three
  books in T-bills from 2020. Neither produced a false promotion (the factor-alpha check and PBO
  held), but every reported Sharpe and DSR was wrong until fixed (`RESEARCH_PROTOCOL.md` §3b).
* The **cost stress was too lenient**: the optimiser was told costs had doubled and traded less,
  so the stressed t-stat came out *above* the base one. The stress now charges double while the
  strategy believes the registered model.
* **Optimisation amplifies risk-model error.** The momentum and composite market-neutral sleeves
  targeted 10% ex-ante volatility and realised 12.3-12.5%. Measured on every session, the risk
  model's bias on *random* long-short portfolios is 1.03, on optimised noise-alpha portfolios
  1.21 (`CALIBRATION.md` §2-3; an earlier one-in-ten-sessions estimate of 1.08 was mostly
  noise): optimised portfolios concentrate in exactly the directions the model under-estimates.
  Seven model variants were measured against it, including the eigen-adjusted covariance and
  industry factors; none removed it (§3 there). The remedy that works is v3's realised-vol
  feedback (§3b).
* Data and infrastructure defects found by tests or measurements are listed in
  `00_INCELEME.md` §4 (13 items, from a vendor dropping a symbol in multi-symbol requests to
  solver warm starts breaking determinism).

## 3b. us-equity-v3: two fixes, one worked

| | v2 → v3 momentum | v2 → v3 residual momentum | v2 → v3 composite |
|---|---|---|---|
| holdout Sharpe | 0.94 → 0.93 | 0.79 → 0.77 | 0.71 → 0.55 |
| realised vol (target 10%) | 12.5% → 11.0% | 12.4% → 11.0% | 12.3% → 10.6% |
| development-window vol | 12.9% → 9.9% | 12.4% → 9.6% | 12.9% → 10.0% |
| turnover / year | 13.9 → 11.1 | 13.8 → 10.9 | 16.2 → 11.0 |

The realised-vol feedback (target divided by the held book's realised/model volatility ratio)
does what the risk model alone could not: a 10% target now gives about 10% (development
window) to 11% (holdout). It is a risk-management improvement worth keeping in every optimised
book. The adaptive IC traded less but no better; for the composite it was worse. A plausible
reason, *not verified*: a one-year trailing IC lags regime changes, so it scales a signal down
after it has stopped working and back up after it has recovered.

**A DSR inconsistency, fixed from the ledger.** v3's momentum sleeve first showed a DSR of
0.86: the trial *count* came from the whole programme (21) but the *dispersion* of trial Sharpe
ratios from the family's own three, very similar trials, which made the hurdle low. The
dispersion is now taken over the programme as well (`study reevaluate`, recorded with its reason
for every family; no backtest re-run): 0.36, in line with v2's 0.37.

## 3c. us-fundamental-v1: different information, same answer

§5's first hypothesis was that price-based anomalies are crowded and that information with
economic content might not be. `us-fundamental-v1` (registered 2026-10-24, before any of its
signals was evaluated on returns) tested it with company filings from SEC EDGAR, point in time
to the second of EDGAR acceptance ([FUNDAMENTALS_DATA.md](FUNDAMENTALS_DATA.md): 729 of 731
securities matched to a CIK, removed members as well covered as current ones, so no
survivorship gap hides in the coverage).

| trial | development Sharpe | holdout Sharpe | holdout alpha (t) | MOM loading (t) |
|---|---:|---:|---:|---:|
| `ear_drift` (3-day reaction to the 8-K earnings release) | −0.90 | 0.47 | 2.1% (0.69) | 0.16 (7.6) |
| `sue_drift` (standardised unexpected earnings, 10-Q/10-K) | 0.31 | −0.16 | −2.4% (−0.76) | 0.10 (5.2) |
| `low_accruals` (Sloan) | 0.32 | −0.14 | −1.7% (−0.50) | −0.09 (−4.7) |
| `earnings_composite` | −0.58 | 0.50 | 1.7% (0.55) | 0.18 (9.1) |
| `fundamental_composite` (+ gross profitability) | −0.41 | 0.23 | −0.5% (−0.15) | 0.10 (4.8) |

1. **No stable information, before costs too.** Rank ICs (exploratory, not a gate input) are
   below 0.025 in absolute value at 5, 21 and 63 sessions, and every holdout IC is
   insignificant (largest |t| 1.57). The announcement signals change sign between the windows:
   the 63-session IC of the earnings reaction is −0.024 (t −2.15) in 2019-2022 and +0.014
   (t 1.53) in 2023-2026, the surprise's −0.021 and +0.018. A 2× cost stress was not what
   failed them; there is no edge for costs to eat. PBO is 0.82 for every trial: the better
   development trial is usually the worse holdout trial.
2. **What is left loads on momentum.** The earnings-reaction books load 0.16-0.18 on the
   momentum factor (t 7.6-9.1): a three-day abnormal return is a short-horizon price-momentum
   score with an event clock. Their FF5+MOM alpha is 1.7-2.1% a year with t below 0.7.
3. **Consistent with NihaiAlgoritma and the literature.** NihaiAlgoritma found
   post-earnings drift break-even on a point-in-time S&P 500 (ACI DERS #1) and no
   cross-sectional edge from surprise features (#6); post-earnings-announcement drift and the
   accrual anomaly are documented to have decayed in large caps since their publication.
   This family measured it again with a different data source (as-reported XBRL, acceptance
   timestamps) and a market-neutral, cost-aware book, and found the same.
4. **The books are under-used.** All five realised 6.4-7.2% volatility against the 10% ex-ante
   target: with a fixed IC of 0.02 and calibrated costs, the optimiser does not find it worth
   spending the risk budget on these scores. That is the cost model and the risk aversion
   working as intended on a weak signal, not a defect; scaling them up would scale a zero.

Provenance note: during the run, `cli.py` was edited (the nightly path only, to keep an SEC
outage from stopping the other books); the code manifest recorded at the start of the run
describes the code before that edit. The computation the workers ran (`_loader`, the study,
the signals) did not change.

## 3d. us-optsurface-v1: the option market's shape

`us-optsignal-v1` had tested the ATM implied-vol *level*; this family tested the *shape*, from
DoltHub's strike-level chains (11.1 GB streamed once and reduced to daily per-stock features;
2019-02 to 2026-10, 2,331 symbols). Registered 2026-10-24 before any of its signals was
evaluated on returns, with a coverage rule fixed before the data was built (a feature needs 50%
of index-member sessions in both windows): smirk 91% / 94% and volatility spread 96% / 98%
qualified, the term slope (17% / 20%; most chains carry no 60-180-day expiration) did not.

| trial | development Sharpe | holdout Sharpe | holdout alpha (t) | turnover / year |
|---|---:|---:|---:|---:|
| `iv_smirk_ls` (Xing, Zhang & Zhao 2010) | −1.44 | −1.05 | −7.1% (−2.24) | 97 |
| `iv_spread_ls` (Cremers & Weinbaum 2010) | −1.49 | −1.96 | −10.8% (−3.42) | 86 |
| `surface_composite` | −1.66 | −1.70 | −9.9% (−3.66) | 94 |

1. **No information in this universe.** Rank ICs (exploratory) are insignificant in both
   windows, and the smirk's has the *wrong* sign: stocks with steeper smirks did slightly
   better, not worse (21-session holdout IC of the negative smirk −0.014, t −1.50). The
   volatility spread's IC is +0.002 to +0.006, fading with the horizon (largest t 1.51). Both
   papers measured the full cross-section of optionable stocks in the 1990s-2000s; in the
   S&P 500 since 2019 neither effect is there.
2. **No persistence, so costs decide.** Day-to-day the features are noisy (one snapshot of
   each contract's quote): the 5-session rank autocorrelation is 0.27 for the smirk and 0.07
   for the spread, so every weekly rebalance meets a nearly new ranking and the books turn
   over ~90 times a year. On a near-zero signal that is a reliable way to lose 2-6% a year;
   the negative alpha t-statistics measure the costs, not a reversed effect. Smoothing the
   features would cut the costs but cannot create the missing information, so no smoothed
   variant was registered.
3. **The data route is itself a finding for operations.** DoltHub cuts its per-table CSV
   stream after about an hour (the first attempt died at the end of 2022); its whole-database
   ZIP export carries the same table compressed 4×, which fits. The reducer streams it
   without storing the 11 GB (`OPERATIONS.md`).

## 3e. us-shortsale-v1: daily short-sale volume

FINRA publishes, for every stock and session, the shares sold short and the total shares in
trades reported to its facilities (off-exchange). Free, daily since 2018-08 (2,069 files, none
missing), used from the next session's close ([SHORTVOL_DATA.md](SHORTVOL_DATA.md)). Registered
2026-10-24 before any of its signals was evaluated on returns, with the coverage rule of §3d
(both variants cover 99.7-99.9% of member sessions).

**The data had to be fixed first.** FINRA names a stock by the ticker it had that day; a security
id carries today's. The first store kept today's tickers only, so 26 renamed members had no data
before their rename (FB→META, BK→BNY, FLT→CPAY, ...), and for 7 of them today's ticker had
belonged to another company: BNY was a municipal-bond fund, GEN Genesis Healthcare, DOC
Physicians Realty, COR CoreSite, CTRA Contura, ECHO Echo Global Logistics, and META a
metaverse ETF in 2021. Their short-sale volume would have entered those members' signals. The
check that found it compares, for every member year, the daily changes of the FINRA volume with
those of the security's own bars (same company ~0.72; another company below 0.33); the earlier
tickers and their change dates were then read off the data (`ticker_history.py`). Caught before
registration, so no registered number ever used the wrong data.

| trial | development Sharpe | holdout Sharpe | holdout alpha (t) | turnover / year |
|---|---:|---:|---:|---:|
| `short_pressure_week` (5-day share against the last 60) | −1.19 | −0.77 | −4.2% (−1.38) | 58 |
| `short_pressure_month` (20-day share against the last 120) | −0.24 | −0.15 | −0.4% (−0.14) | 37 |

1. **The change in short selling carries no information here.** Rank ICs (exploratory) of both
   variants are within ±0.005 at 1, 5 and 21 sessions, negative in 2019-2022 and positive in
   2023-2026, never significant (largest |t| 1.55). Diether, Lee & Werner measured the effect
   on 2005 Reg SHO pilot data, where every short sale was flagged. A plausible reason it is
   absent here, *not verified*: FINRA's files cover off-exchange trades only, where most short
   sales are wholesalers filling retail buy orders, so week-to-week changes in the share track
   retail flow more than informed shorting.
2. **The weekly book paid for nothing.** It turned over 58 times a year and fails the lag and
   cost stresses (2× cost t −2.49): with no information, trading is all cost. Both books
   realised 5-6% volatility against the 10% target: on a near-zero signal the cost-aware
   optimiser does not spend the risk budget, as with the filing signals (§3c, item 4).
3. **Exploratory, post-hoc, seen in the holdout: the *level* might carry information.** The
   registered trials used the change in the short share, not its level, on the argument that
   market makers' routine shorting sets the level. The data did not support that argument:
   stocks with a *low* 5-day short share earned more over the next month, a rank IC of +0.012
   (t 1.68) in development and +0.026 (t 2.98) in the holdout, +0.008 (t 1.47) / +0.022
   (t 3.46) after neutralising liquidity, momentum, low volatility and reversal (it is not
   one of those in disguise; its strongest overlap, with liquidity, is 0.23). Boehmer, Jones
   & Zhang (2008) found heavily shorted stocks underperforming over the next month on NYSE
   order data. This was looked at after the holdout had been seen, and only the development
   window is clean for it, where it is weak; it cannot be promoted from these numbers. It is
   registered as `us-shortsale-v2` for its forward record (§3f).

## 3f. us-shortsale-v2: the short share's level (forward-only)

One trial, exactly the measure seen in §3e item 3 (the negative 5-day short share, a month's
horizon, weekly rebalancing, v1's configuration otherwise), registered before its backtest ran
but after the level's IC had been looked at over both windows. Its holdout is therefore not a
clean test, and its deflated Sharpe pays for all 37 trials of the programme.

| | development | holdout |
|---|---:|---:|
| Sharpe | −0.66 | 0.82 |
| FF5+MOM alpha (t) | | 4.5% (1.30) |
| turnover / year | | 56 |
| calendar years | 2019 +0.5%, 2020 −4.0%, 2021 −6.9%, 2022 −3.0% | 2023 +11.9%, 2024 +13.3%, 2025 +22.7%, 2026 −5.6% |

1. **Not promoted, even on a holdout it was partly chosen from.** It fails the deflated Sharpe
   (0.06), the Newey-West t (1.59), the alpha t (1.30), the 2× cost stress (t 0.62) and
   stability. Delaying execution by a day barely matters (0.80): the information is slow; the
   costs are not.
2. **Same pattern as the announcement signals (§3c): a sign that flips between windows.** Three
   losing development years out of four, then three winning ones and a losing 2026. The development IC was positive
   but small (0.012), and at 56 turns a year the book lost it to costs; a 5-day share is noisy
   (5-session rank autocorrelation 0.63). A longer window would trade less; testing one now
   would be another look at the same holdout, so it waits for this trial's forward record.
3. **What it loads on.** Large caps (SMB −0.17, t −6.4) and weak profitability (RMW −0.12,
   t −3.4): low short shares are in large, less profitable stocks; the remaining alpha is not
   significant.
4. **Forward record.** Its books run every night with the other forward families
   (`FORWARD_FAMILIES`); that record, not this table, is the only test it can pass.

## 3g. Nihai3's portfolio engine does not survive the honest universe (`us-trend-v1`)

Nihai3 (Desktop/algo) passed four pre-registered gates against NihaiAlgoritma, and measured that
the gain came from portfolio construction: the old sign score through its engine took a Sharpe of
0.52 to 1.06. Its own write-up warned that its universe was today's survivors and that its alpha
vanished once the same universe's buy-and-hold was a factor. This family asked the question on
the point-in-time S&P 500 with calibrated costs: continuous forecast, hierarchical risk parity,
a diversification multiplier, a volatility target and a no-trade buffer instead of stops;
long-only and unlevered, 15% target, as `trend_heritage`. Registered 2026-10-30, three trials
(the programme now counts 40). Its holdout is not clean (the earlier results were known).

| holdout, net | Sharpe | return | vol | max DD | gross | turnover/yr | FF5+MOM alpha (t) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `forecast_hrp` | 0.31 | 7.1% | 8.0% | 10.1% | 0.75 | 8 | −3.2% (−1.32) |
| `signonly_hrp` | 0.30 | 7.0% | 8.0% | 10.5% | 0.75 | 25 | −3.2% (−1.36) |
| `forecast_equal_risk` | 0.32 | 7.2% | 8.1% | 10.1% | 0.73 | 7 | −3.3% (−1.36) |
| `trend_heritage` (v2) | 0.51 | 10.2% | 11.0% | 14.0% | 0.96 | 22 | −3.5% (−1.22) |
| *equal weight* | *0.71* | *14.6%* | *14.0%* | *17.5%* | *1.00* | *1* | *−1.4%* |

All three rejected (DSR 0.01, t, alpha t, cost stress, PBO).

1. **The gain did not transfer.** The engine does not beat `trend_heritage`: Sharpe difference
   −0.20, 95% interval [−0.50, +0.12], paired stationary bootstrap
   ([TREND_COMPARISON.md](TREND_COMPARISON.md), not a gate input). The likeliest reading of the two
   results together, not proven here: a good part of what Nihai3 measured was exposure to a
   universe of winners, whose worst decade it had removed, which is what its own factor
   decomposition suggested.
2. **What it does do is lower risk and trading.** Maximum drawdown 10% against 14% and 17.5%, and
   a third of `trend_heritage`'s turnover for the continuous forecast (8 against 22 a year; the
   sign score through the same buffer trades as much as before, 25). That is real and it costs
   return: the book is not investable as an alpha, it is a less volatile, less exposed long book.
3. **The volatility target is not reached.** 15% was the target, 8% the outcome. A long-only book
   holds nothing on a negative forecast and the gross cap binds; the target was not retuned, since
   that would be a new trial.
4. **Hierarchical risk parity adds nothing on 500 stocks** (−0.01, [−0.12, +0.10] against equal
   risk), as its trial hypothesised. The engine's value here is the buffer and the forecast's
   smoothness, not the clusters.
5. **Forward record.** Its books run with the others from the first session after registration.

## 3h. Trend across asset classes: level with holding them (`us-crossasset-v1`)

§3g left one honest way out for Nihai3's engine: 500 stocks share one market factor, whereas
time-series trend has its evidence across asset classes (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi
& Pedersen 2017), and that is where correlation clusters are real. QuantBot ran such a universe
and ended level with holding it, but it never paid for its trials. This family takes QuantBot's
universe as its return-blind rule picked it, without crypto and Borsa Istanbul: 18 ETFs (equities,
bonds, credit, commodities, currencies, real estate). Registered 2026-10-30, three trials (the
programme now counts 43), with a second benchmark: the same engine always long, with no forecast.
Its machinery was only run on 2017-2018, outside both windows, and the stock-calibrated spread model
was first checked against real quotes on these funds; it over-states them by a median of 1.7 times
(`CALIBRATION.md` §1), so costs are not what sinks it. Nihai3's and QuantBot's results were known at
registration, so the holdout is not a clean test.

| net | Sharpe, development | Sharpe, holdout | Sharpe, whole | return, whole | max DD, whole | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| `ca_trend_ls` | 0.74 | −0.05 | 0.42 | 6.0% | 16.3% | 18 |
| `ca_trend_long` | 0.95 | 0.25 | 0.61 | 6.1% | 7.8% | 8 |
| `ca_trend_equal_risk` | 0.75 | −0.32 | 0.33 | 5.2% | 13.3% | 16 |
| *equal weight* | *0.57* | *0.84* | *0.67* | *9.0%* | *19.6%* | *0.5* |
| *always long, same engine* | *0.32* | *0.56* | *0.43* | *5.1%* | *11.5%* | *3* |

All three rejected (DSR, t, alpha t, 2× cost; the two with shorts also fail the lag stress).

1. **It reproduces QuantBot.** Over the whole window the best book is level with holding the funds:
   `ca_trend_long` minus equal weight −0.06, 95% interval [−0.69, +0.53]; minus always long +0.18,
   [−0.37, +0.70] ([CROSSASSET_COMPARISON.md](CROSSASSET_COMPARISON.md), not a gate input).
2. **The windows disagree, and the gate looks at one.** Development favours trend, the holdout
   favours holding. On the holdout `ca_trend_long` minus equal weight is −0.59, [−1.24, +0.02].
   Eight years hold one regime change, so this cannot say which window is typical.
3. **It did what trend is for, in the one year it was for.** In 2022 equal weight lost 8.3% and
   always long 6.1%; `ca_trend_ls` made 10.7% and `ca_trend_long` 3.0%, the latter with a maximum
   drawdown of 3.7%. That is one episode (the interval on the Sharpe difference that year is
   [−2.4, +5.4]) and not a statistical result, but the drawdown is the reason a less exposed book
   might be held at all: 7.8% against 19.6% over the whole window, for a similar Sharpe.
4. **Shorts cost more than they earned.** The long-short book is behind the long-only one in the
   development, holdout and whole windows (2022 alone is the exception), and turns over twice as much.
5. **Clusters did not earn their keep here either.** Hierarchical risk parity against equal risk:
   +0.08, [−0.14, +0.32] over the whole window, +0.27, [−0.18, +0.72] on the holdout. The hypothesis
   of `ca_trend_equal_risk` (that it does worse) is not supported.
6. **The volatility target is not reached again:** 5-9% against 10% over these windows, not retuned.
7. **The alpha check is lenient here,** since FF5+MOM are US equity factors; it fails anyway.
8. **Forward record.** Its books run with the others from the first session after registration.

## 3i. Crypto: the universe widened, and the answer did not change (`us-crypto-v1`)

Every USDT pair Binance has listed, dead ones included (702 pairs; 517 have been in the top 100, 162
of those are gone), top 100 by dollar volume, reconstituted monthly, on a 10 bp taker fee, a spread
model fitted on 100 live books, and an alpha check against the US market, Bitcoin and the equal-weight
crypto market: on FF5+MOM any coin would be "alpha". Registered 2026-10-30, three trials.

| net | development | holdout | whole | return, whole | max DD, whole | mean gross |
|---|---:|---:|---:|---:|---:|---:|
| `crypto_trend_long` | 1.39 | 0.29 | 0.91 | 14.1% | 13.9% | 0.12 |
| `crypto_trend_ls` | 0.78 | −0.41 | 0.19 | 4.8% | 28.7% | 0.30 |
| `crypto_momentum_ls` | −0.55 | −1.31 | −0.99 | −20.6% | 83.3% | 1.43 |
| *equal weight* | *0.83* | *−0.18* | *0.33* | *−5.4%* | *97.8%* | *0.99* |
| *Bitcoin* | *0.75* | *1.02* | *0.85* | *42.9%* | *76.6%* | *1.00* |

All three rejected.

1. **The first run was wrong, and the numbers showed it.** It had `crypto_momentum_ls` at +17,576% in
   2022. Binance reuses symbols: LUNAUSDT came back on 2022-05-31 as another coin at 177,400 times the
   last price, and token swaps do the same. A gap of three days now ends a security, by rule, whatever
   the coin. Both runs are in the ledger, with the reason.
2. **Holding Bitcoin beat everything in the holdout** (`crypto_trend_long` minus Bitcoin −0.73, 95%
   interval [−1.63, +0.19], [CRYPTO_COMPARISON.md](CRYPTO_COMPARISON.md)). The other coins lost: equal
   weight fell 76% in 2025 alone, and that is what a survivorship-free universe is for.
3. **`crypto_trend_long` is the one to watch, not a discovery.** Over the whole window a Sharpe of 0.91
   with a 13.9% drawdown against Bitcoin's 76.6%, since it holds 12% of its capital on average and
   stands aside in falls. But most of it is 2020-2021 (development 1.39, holdout 0.29, alpha t 1.01).
   Its forward book will say more than another backtest could.
4. **One-month momentum loses in crypto, significantly** (holdout t −2.73), and the literature says it
   should win. The opposite sign would be a new trial chosen from this result; costs of 13% a year
   mean it would have to earn that twice over.
5. **Limits.** A book trades on US sessions only; a short stands for a perpetual at 10% a year; the
   spreads are one snapshot of today's market; Binance is one exchange.

## 3j. Options at the market's own quotes (`us-optquote-v1`)

A full chain changes what can be asked: 120 million quotes, with bid, ask and the vendor's greeks
(`data/optchain.py`), three dates a week since 2020 and one before. Each trade at the bid or ask of the
listed contract, $0.65 a contract, settled on the unadjusted close (`options/quoted.py`).

| net | development | holdout | return | max DD | market beta | FF5+MOM alpha (t) |
|---|---:|---:|---:|---:|---:|---:|
| `put_write` (20-delta, cash-secured) | 0.05 | 0.91 | 11.4% | 9.7% | 0.37 | 0.7% (0.32) |
| `covered_call` (30-delta) | 0.36 | 0.45 | 10.7% | 19.9% | 0.86 | −7.0% (−2.44) |
| `calls_quoted` (50-delta, on trend) | −0.20 | 0.00 | 3.0% | 21.8% | 0.63 | −12.5% (−2.08) |
| `ivrv_straddles` | −0.76 | −0.51 | −8.3% | 33.2% | −0.03 | −10.0% (−1.67) |

All rejected. Holdout and costs as quoted; the development column was re-run after a defect (the
table dates its weekly snapshots on Saturdays before 2020, and the first run held cash through 2019).

1. **Writing puts passes everything but the two checks that matter here.** It survives the lag and
   doubled-cost stresses and every holdout year is positive, but alpha t 0.32 and DSR 0.09: the return
   is 0.37 of the market with a smaller drawdown, and 2020 cost it 11.4%. That is a way to hold equity
   risk, not an edge, and assignment and margin are not modelled in its favour.
2. **Goyal & Saretto's straddles lose at real prices.** Unhedged, at a premium of 10% of equity a side:
   the spreads are a large part of a straddle's value (2× cost t −4.13).
3. **The old engine's flaw was real, and it did not hide an edge.** Pricing a wing at one ATM volatility
   is off by a median 49% on puts and 19% on calls against the mid, 9% with the fitted smile; quoted
   half-spreads are 20-22% ([SURFACE_REPORT.md](SURFACE_REPORT.md)). `calls_d50_t21` earned 0.46 at
   model prices, `calls_quoted` 0.00 at real ones.
4. **The smile fit is good where it is quoted and not a licence to extrapolate.** After the arbitrage
   penalty 90% of slices are free of butterfly arbitrage and 97% of surfaces of calendar arbitrage, over
   the quoted range; 74% out to log-moneyness 1.5. The penalty costs fit: median error 0.94 volatility
   points against 0.66 without it on a similar sample, and the 90th percentile rises from 2.2 to 12.1.

## 3k. Market making on a real book (`mm-crypto-v1`)

meridian (Desktop/jane) promoted a market maker on a synthetic market and said that was a statement
about that market. This is the same question on Binance's book, recorded live: 100 ms diffs and every
trade for six pairs, rebuilt into a local book (later snapshots agree on 97-100% of the best 40 levels;
the only chain break is a lost connection), with the order's place in the queue tracked. Four makers,
registered before any of them saw a recording. **The sample is 18 symbol-hours, against the 480 the
gate requires, and it says so in place of a verdict; `pusula micro run` repeats it as the recording
grows.** The recording ran for eight hours and holds 3.4 of them, in two stretches of one evening: the
connection was lost for 4.6 hours.

| maker | traded, $ | gross, bp | spread earned, $ | inventory, $ | breakeven fee, by pair |
|---|---:|---:|---:|---:|---|
| `touch` | 17.9M | −1.79 | +159 | −3,367 | −1.3 to −3.6 bp |
| `wide_2bp` | 3.7M | −1.68 | +332 | −955 | −0.6 to −3.6 bp |
| `stoikov` | 14.9M | −1.68 | +345 | −2,846 | −1.4 to −2.3 bp |
| `imbalance` | 12.5M | −1.76 | +316 | −2,509 | −1.5 to −11.8 bp |

1. **Adverse selection is larger than the spread, before any fee.** Quoted at the touch, the mid moves
   1-4 bp against the maker within a second of a fill and stays there. On pairs quoted a tick wide the
   spread itself is about 0.001 bp (Bitcoin).
2. **The fee settles it for an ordinary account.** A retail maker pays 7.5-10 bp. Even a rebate of 0.5 bp
   would not close the gap measured here. Market making at this level needs a fee tier and a latency
   this project does not have, which is the reason `jane` was a separate project and this result is
   worth having.
3. **The replay was wrong once more before this.** A connection loss piled up empty hours and carried
   positions across the gap as inventory PnL. A break now closes a stretch and the next starts flat;
   the runs before the fix are in the ledger, with the reason.
4. **Limits.** Three and a half hours of one evening, so one market regime; a one-second decision loop;
   no footprint assumed; 100 ms diffs hide what happens inside each interval, so the queue is an
   estimate. The t-statistics in the results (−6 to −14) measure these few hours, not market making.

## 3l. The impact coefficient, measured (`docs/IMPACT_CALIBRATION.md`)

The cost model's Y = 0.7 came from the literature, which is why the gate doubles costs. On 5.5 million
consolidated trades (40 members, six sessions, tick-rule signs, five-minute buckets) the flow-impact
coefficient is **0.91**, median 0.96 across windows, and tick-rule noise biases it down. That is above
0.7 and below the 1.4 of the stress: the registered results stand, the base case was optimistic, and a
new family should register a higher Y. On crypto 252 buckets give 0.56 against the 1.0 registered,
with pairs from 0.13 to 1.91 (54 buckets had given 1.26): too unstable to act on. Against real books the
crypto model is conservative for immediate orders up to $100,000 on every pair, and a million dollars
swept through a thin pair's book costs far more than it says (ARB about 1,500 bp against 154): it
describes patient execution, and capacity limits apply.

## 4. Exploratory diagnostics (not gate inputs, `DIAGNOSTICS.md`)

* Information coefficients are small everywhere: the best is about 0.023 (momentum, residual
  momentum, holdout, t ≈ 1.9). The strategies scale alpha with an assumed IC of 0.02, so the
  optimiser's belief was roughly right for momentum and too optimistic for the others, which
  makes it trade too much for the weaker signals.
* **Capacity runs backwards.** `xs_composite` (v2 configuration, full window) earns Sharpe 0.52
  at $10M, 0.56 at $100M, 0.61 at $1B and 0.71 at $5B, while its cost drag *falls* from 1.01% to
  0.56% a year. The square-root impact term in the optimiser grows with size, so a bigger book
  trades more slowly; that the slower book does better means the small book over-trades a weak
  signal. (No order went unfilled at any size: the optimiser itself respects the participation
  cap.) This, with the IC figures above, motivated `us-equity-v3` (adaptive IC, §5).
* The flooring of the adaptive IC estimate at zero biases a pure-noise score's estimate to about
  0.01 instead of 0 (`tests/test_adaptive.py`); it halves, not removes, trading in noise.

* **The rejections are not bad luck in the parameters** (`ROBUSTNESS.md`, 45 runs). No neighbour of
  `forecast_hrp` reaches `trend_heritage` (best 0.43 against 0.51); no neighbour of `ca_trend_long`
  reaches always long or equal weight (best 0.46 against 0.56 and 0.84); no horizon of
  `trend_heritage` reaches equal weight (best 0.62 against 0.71).
* **`xs_momentum`'s 0.94 flatters its neighbourhood.** The published 12-1 values were registered,
  not chosen, yet the median of 11 neighbours is 0.68 and only three are within 0.20 of it, with a
  range of [0.39, 0.95]. The development window is negative or near zero for all of them.
* **Picking from these grids would be overfitting, and on the funds it would be measurably so.**
  The probability of backtest overfitting of taking the best point is 0.83 on `ca_trend_long`, 0.63
  on `trend_heritage`, 0.47 on `xs_momentum`. On `ca_trend_long` a wider no-trade buffer helps the
  holdout and hurts development. On `forecast_hrp` (PBO 0.03) a wider buffer helps in both windows, the
  one consistent effect found; it would be a new registered trial, and at 0.43 it would not change
  §3g's answer.
* **The noise floor is negligible.** At 1e-9 noise on prices the Sharpe of `xs_momentum` moves by
  0.006 and that of `forecast_hrp` by less than 0.001, so differences between trials reported here are
  not the solver's.
* **A signal's IC is no shortcut to a verdict.** At its holdout IC, 12-1 momentum over a month needs
  about 2,900 sessions to reach t = 2, against 784 for the book's Sharpe to reach its MinTRL (the
  two targets differ, and the point stands: this takes years). `pusula review` reports it to tell a
  dead signal from a costly book, not to decide earlier.

## 5. What would have to be true for a next family to pass

These are hypotheses for future, separately registered families, not conclusions:

* **Different information, not different combinations.** Every price-based anomaly here is
  public and crowded. Announcement drift and accruals on point-in-time filings were tested in
  `us-fundamental-v1` (§3c) and failed the same way; so did the option surface's smirk and
  volatility spread (`us-optsurface-v1`, §3d), and the change in daily off-exchange short
  selling (`us-shortsale-v1`, §3e); the short share's level is in its forward record
  (`us-shortsale-v2`, §3f). What remains needs paid data: analyst estimate revisions
  (consensus history), short interest and borrow fees, or a wider and less efficient universe
  than the S&P 500 (next bullet).
* **A wider universe.** The S&P 500 is the most efficient segment; anomalies are stronger in
  the Russell 2000, and so are costs: the calibrated cost model is what makes that trade-off
  measurable instead of imagined. Point-in-time membership of a small-cap index is not free.
  Wider across asset classes was tested (§3h).
* **Adaptive IC.** Scaling alpha by the trailing realised IC of each signal (causal) instead of a
  fixed 0.02, so the optimiser stops paying spread for signals it should not believe.
* ~~**Optimisation-bias-adjusted risk model**~~ (measured 2026-10-24, `CALIBRATION.md` §3: the
  eigen-adjustment, industry factors, more factors and stronger shrinkage do not remove the
  optimisation bias; the realised-vol feedback does the job instead).
