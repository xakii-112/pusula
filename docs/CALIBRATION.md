# Calibration (done before any strategy was run on real data)

Two pieces of infrastructure were fitted to real data before the `us-equity-v1` family was
pre-registered: the spread model and the risk model. Neither looks at strategy returns. Both
fits are reproducible from the commands below; the numbers are frozen into
`research/families.py` (spread) and `RiskModelSpec` defaults (risk) and are part of every
registered configuration hash.

## 1. Spread model (2026-10-23)

**Data.** 439 measurements of the consolidated (SIP) NBBO from Alpaca historical quotes:
11 sessions (mid-month, one per year 2016 to 2026) × 40 index members drawn at random from
the point-in-time S&P 500 of that day. Each measurement is the median quoted half-spread
`(ask - bid) / (2 mid)` over three 2-minute windows (10:30, 12:30, 14:30 ET, away from the
auctions). One sample (2017) had too few quotes and was dropped. Samples are stored in
`$PUSULA_HOME/data/spread_samples.json`.

**Predictors** are daily-bar quantities known at the previous close: EWMA daily volatility
(half-life 21), 21-day average dollar volume, previous close, 21-day average trade count.

**What failed.** The microstructure relation `half-spread = k * sigma / sqrt(N_trades)`
(Wyart, Bouchaud et al. 2008) fitted 2016-2021 under-predicts 2022-2026 spreads by a median
factor of **2.2** out of sample (OOS R² < 0). The fitted `k` drifts from 1.5 (2016-2020) to
3.8-4.4 (2024-2026): trade counts exploded after 2020 (odd lots, retail, fragmentation) while
quoted spreads did not tighten with them. Every model with only time-invariant coefficients
under-predicted the later years (best: factor 1.54).

**Adopted model.**

```
ln(half_spread) = a_year + 0.7163 ln(sigma) - 0.3614 ln(ADV$) + 0.4248 ln(price)
```

R² = 0.51, median absolute log error 0.30 (a factor of 1.35). The slopes are stable under
leave-one-year-out refits (sigma 0.67-0.77, ADV −0.35 to −0.40, price 0.40-0.45). The year
intercepts:

| year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | −0.553 | −0.626 | −0.298 | −0.655 | −0.428 | −0.280 | −0.125 | −0.111 | +0.157 | −0.007 | −0.164 |

Using each year's measured spread level is **not** look-ahead: quoted spreads are observable at
the moment of trading. The calibration only recovers that observable quantity from a sample.
Live runs use the latest year's intercept.

Sample statistics: median quoted half-spread 2.6 bp, 90th percentile 7.1 bp, 99th 15 bp.

**Not calibrated:** the square-root impact coefficient (`Y = 0.7`, literature value; no
trade-level data to fit it) and the 5% ADV participation cap. This is why the promotion gate
re-runs every trial with **all costs doubled**.

Reproduce: `execution/calibration.py` (`sample_quotes`, `calibrate`); the year-effects fit is
an OLS of the log half-spread on year dummies and the three logs.

**Impact, measured (2026-10-30, `IMPACT_CALIBRATION.md`).** Y = 0.91 on 5.5 million consolidated
trades, against the 0.7 assumed above and the 1.4 of the gate's stress. Registered cost models keep
their value: a new family should register a higher one.

**Crypto (2026-10-30, before `us-crypto-v1` was registered; `tools/crypto_spreads.py`).** Fitted on the
live books of the 100 members, one snapshot: ln half-spread = 3.7259 + 0.4496 ln sigma − 0.6814 ln ADV
(price is left out, a coin's unit being arbitrary), R² 0.54, median quote 2.43 bp, median model/quote
0.91. Earlier years were probably wider, which the gate's doubling has to cover.

**ETFs (2026-10-30, before `us-crossasset-v1` was registered; no returns read).** The model was fitted
on index members. Against 144 quote samples on its 18 ETFs (eight mid-year sessions, 2019-2026,
`tools/etf_spreads.py`) the median quoted half-spread is 0.60 bp and the model's 1.14 bp: it
over-states them by a median of 1.7 times and is at or above the quote in 74% of samples. The largest
under-statements are HYG (0.63 quoted, 0.50 model), UUP (1.82, 1.12) and EEM (1.20, 0.94); the thinnest
fund, DBB, is 4.99 against 5.93. Kept as it is: conservative for funds, and the gate doubles it anyway.

## 2. Risk model (2026-10-23)

Statistical factor model (`risk/model.py`): 12 principal components of volatility-scaled
returns over 252 sessions, EWMA factor covariance, EWMA specific variance shrunk 10% towards
the cross-sectional median, volatility-regime adjustment (VRA) with a 21-session half-life.

**Test.** Bias statistic `std(r_{t+1} / sigma_t)` for portfolios re-forecast every 10 sessions,
**development window only** (2017-2022), on the point-in-time index members. A correct model
gives 1 (95% band about ±0.05 here).

| configuration | equal weight | random long | random long-short | single name |
|---|---:|---:|---:|---:|
| no VRA, 12 factors | 1.127 | 1.122 | 1.187 | 1.082 |
| no VRA, 8 / 20 factors | 1.127 | 1.122 | 1.184 / 1.191 | 1.078 / 1.087 |
| shorter half-lives (21/42/21), no VRA | 1.076 | 1.072 | 1.179 | 1.087 |
| VRA 21 | 0.980 | 0.976 | 1.092 | 1.009 |
| VRA 21, split factor/specific | 0.899 | 0.895 | 1.065 | 0.983 |
| **VRA 21, specific shrinkage 0.10 (adopted)** | **0.980** | **0.976** | **1.081** | **1.008** |

Without the regime adjustment the model under-forecast risk by 8-19% (it reacts to volatility
shifts only through a 42-day half-life). VRA fixes long-only and single-name forecasts. A
residual under-forecast of about **8% remains for random long-short portfolios**; the number of
factors does not change it. Market-neutral sleeves therefore run somewhat above their ex-ante
volatility target, and optimisation can amplify this (optimised portfolios load on exactly
the directions the model under-estimates). This is a known limitation, recorded here before the
results; the drawdown check of the gate is the backstop.

**Correction (2026-10-24): the table above is one draw of a noisy sample.** It re-forecast on
every 10th session from row 260. Re-run on 2026-10-24 (same code and data, the original script
reproduces it to the third decimal), the equal-weight bias over the ten possible start rows
ranges from **0.86 to 1.17** and the random long-short one from 0.96 to 1.09: with one forecast
in ten sessions, a handful of crash days decide the statistic. Using **every session**
(2017-2022, adopted model):

| portfolio | bias | 95% band |
|---|---:|---:|
| equal weight | **1.006** | ±0.036 |
| random long-only | **1.005** | ±0.016 |
| random long-short | **1.033** | ±0.016 |
| single name | **1.060** | ±0.016 |
| optimised, noise alphas (dollar-neutral Σ⁻¹α) | **1.213** | ±0.016 |
| minimum variance (Σ⁻¹1) | **2.629** | ±0.036 |

So the model is close to unbiased where nobody optimises against it (the "8% long-short
under-forecast" was mostly noise; the truth is about 3%), and clearly biased where an optimiser
does: §3. The comparisons between configurations in the table above stay valid as comparisons
(every configuration was measured on the same sessions), their levels do not. Calibration runs
are now scripted (`tools/risk_bias.py`), not ad hoc.

## 3. Optimisation bias (2026-10-24)

**Question.** The market-neutral momentum books of `us-equity-v2` realised 12.3-12.5% against
a 10% ex-ante target (FINDINGS §3), a ratio of 1.23-1.25, while the model is close to unbiased
on random portfolios (§2, corrected). An optimiser loads on the directions whose risk the model
under-states; can a different model configuration remove that?

**Test** (`tools/risk_bias.py`, no return forecast is used): bias statistic of optimised
portfolios built from pure-noise alphas, re-forecast every 10th session, for configurations
fixed before the run, on 2017-2022 (to choose) and 2023-2026 (to confirm). Two industry
configurations were added after the first six were measured, and fixed before they ran.

| configuration (2017-2022 / 2023-2026) | unconstrained Σ⁻¹α | minimum variance | production LS optimiser |
|---|---:|---:|---:|
| adopted (12 PCs, VRA 21, shrinkage 0.10) | 1.18 / 1.21 | 2.32 / 2.37 | 1.06 / 1.16 |
| specific shrinkage 0.30 | 1.15 / 1.17 | 2.29 / 2.32 | 1.11 / 1.23 |
| specific shrinkage 0.50 | 1.15 / 1.17 | 2.31 / 2.32 | 1.14 / 1.29 |
| 20 statistical factors | 1.28 / 1.30 | 2.47 / 2.42 | 1.09 / 1.20 |
| eigen-adjusted factor covariance, a = 1.2 (Menchero, Wang & Orr 2011) | 1.18 / 1.21 | 2.32 / 2.37 | 1.06 / 1.16 |
| eigen a = 1.2 + shrinkage 0.30 | 1.15 / 1.17 | 2.29 / 2.32 | 1.10 / 1.22 |
| industry factors, 2-digit SIC (≥ 5 names) | 1.20 / 1.22 | 2.30 / 2.27 | 1.03 / 1.02 ¹ |
| industry factors, 3-digit SIC (≥ 5 names) | 1.21 / 1.21 | 2.22 / 2.30 | 1.04 / 1.01 ¹ |

¹ Measured with an equivalent per-date formulation of the optimiser (the production one is
compiled once with a fixed factor count; padded to the industry factors' count its cvxpy
parameter mapping needed 13.5 GB). Re-run through that formulation with different noise
alphas, the adopted model gave **1.04 / 1.01** instead of 1.06 / 1.16: with one rebalance in
ten sessions and five alphas a day sharing that day's returns, the production-optimiser column
moves by up to ±0.15 on noise alone, more than any configuration moves it.

**Findings.**

1. **The bias is real where the optimiser is free.** On every session (§2), unconstrained
   Σ⁻¹α portfolios realise 21% more volatility than forecast, the same ratio as the momentum
   books (1.23-1.25). Minimum-variance portfolios, which concentrate in names with the lowest
   estimated risk, realise 2.3-2.6 times their forecast.
2. **No configuration removes it.** The eigen-adjustment changes nothing (12 PCA factors
   estimated from about a year of data have almost no small-sample eigenvalue bias left to
   correct; its effect is in the third decimal). Industry factors capture a real but small
   residual correlation (same 3-digit SIC: +0.08 to +0.095 against −0.004 across industries)
   without moving the optimised biases beyond noise. More statistical factors make it worse
   (more noise to exploit). Stronger specific shrinkage helps the unconstrained portfolios by
   0.03-0.04 and hurts the constrained ones.
3. **Decision: the risk model is unchanged.** The options stay available (`RiskModelSpec`:
   `eigen_adjust`, `industry_min_size`; off by default, so no registered configuration
   changed) because they are tested and measured, not because they help here. The working
   remedy for a held book's excess risk remains the realised-volatility feedback of
   `us-equity-v3` (target divided by the book's realised/model volatility ratio), which brought
   the momentum books' realised/target ratio from 1.25 to 0.96-1.00 (development) and 1.10
   (holdout). The production constraints (1.5% names,
   gross 2, beta and dollar neutral) already keep noise-alpha books close to their forecast.
