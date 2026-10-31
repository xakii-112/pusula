# Results: `us-equity-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **9**; deflated-Sharpe trial count (whole programme `us-equity-`): **21**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. All returns are net of the calibrated cost model (spread, square-root impact, fees, borrow, financing) with execution at the next session's VWAP. Research backtests run with the latching production kill switch (see us-equity-v2).

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| trend_heritage | 0.51 | 10.2% | 11.0% | 14.0% | 1.04 | -3.5% (-1.22) | 0.12 | 0.36 | 0.83 | 21.0 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| xs_momentum | 0.94 | 16.9% | 12.5% | 9.9% | 2.06 | 7.0% (1.81) | 0.37 | 0.91 | 1.90 | 13.9 | no: dsr, alpha_t, pbo |
| xs_residual_momentum | 0.09 | 4.8% | 0.4% | 0.5% | 0.19 | -0.2% (-0.89) | 0.03 | 0.09 | 0.19 | 4.2 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| xs_reversal | -0.98 | 1.9% | 2.8% | 10.9% | -1.36 | -3.1% (-1.40) | 0.00 | -0.65 | -0.20 | 48.2 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, pbo, stability |
| xs_low_vol | -0.15 | 2.8% | 9.7% | 22.8% | -0.30 | -0.9% (-0.20) | 0.01 | -0.25 | -0.47 | 14.0 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, pbo |
| xs_composite | 0.09 | 4.8% | 0.4% | 0.5% | 0.19 | -0.1% (-0.89) | 0.03 | 0.09 | 0.19 | 4.7 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| xs_ml_ridge | 0.56 | 10.5% | 10.4% | 7.9% | 1.21 | 2.8% (0.75) | 0.14 | 0.57 | 0.98 | 23.2 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| xs_ml_gbm | 0.36 | 7.6% | 8.4% | 7.0% | 0.76 | 0.4% (0.12) | 0.07 | 0.29 | -0.21 | 45.6 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| composite_long_only | n/a | 4.8% | 0.1% | 0.0% | -62.73 | -0.0% (-70.60) | n/a | n/a | -62.73 | 1.2 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, pbo |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| trend_heritage | 0.38 | 5.5% | 13.6% | 22.6% | 0.82 |
| xs_momentum | -0.12 | -1.3% | 12.9% | 22.7% | -0.25 |
| xs_residual_momentum | -0.05 | 0.2% | 9.2% | 13.2% | -0.10 |
| xs_reversal | -0.29 | -2.8% | 11.3% | 22.7% | -0.61 |
| xs_low_vol | 0.25 | 3.0% | 9.2% | 15.4% | 0.51 |
| xs_composite | 0.10 | 1.6% | 9.6% | 15.3% | 0.20 |
| xs_ml_ridge | -0.14 | -0.7% | 9.7% | 12.6% | -0.30 |
| xs_ml_gbm | -0.61 | -3.6% | 7.4% | 19.9% | -1.28 |
| composite_long_only | 0.07 | 1.3% | 8.8% | 21.3% | 0.14 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trend_heritage | 0.63 (20.3) | 0.07 (2.0) | 0.25 (8.6) | 0.05 (1.5) | 0.09 (2.4) | 0.02 (1.2) | 0.71 |
| xs_momentum | 0.04 (1.5) | -0.01 (-0.4) | -0.04 (-1.2) | 0.10 (3.1) | -0.09 (-2.1) | 0.59 (21.4) | 0.62 |
| xs_residual_momentum | -0.00 (-1.5) | -0.00 (-0.8) | 0.00 (1.4) | -0.00 (-1.5) | -0.00 (-1.4) | 0.00 (2.1) | 0.02 |
| xs_reversal | 0.02 (2.2) | 0.00 (0.2) | 0.03 (1.1) | 0.00 (0.0) | -0.03 (-1.7) | -0.01 (-1.2) | 0.03 |
| xs_low_vol | -0.04 (-1.3) | -0.01 (-0.3) | 0.10 (2.2) | 0.34 (7.1) | -0.03 (-0.7) | 0.08 (2.7) | 0.18 |
| xs_composite | -0.00 (-1.5) | -0.00 (-0.8) | 0.00 (1.4) | -0.00 (-1.5) | -0.00 (-1.4) | 0.00 (2.1) | 0.02 |
| xs_ml_ridge | 0.07 (3.1) | 0.03 (0.8) | -0.01 (-0.3) | 0.11 (3.1) | -0.02 (-0.4) | 0.45 (19.9) | 0.48 |
| xs_ml_gbm | 0.07 (2.9) | -0.02 (-0.8) | -0.00 (-0.1) | -0.01 (-0.2) | -0.07 (-1.8) | 0.28 (14.6) | 0.35 |
| composite_long_only | -0.00 (-1.0) | 0.00 (0.6) | 0.00 (0.6) | -0.00 (-1.9) | 0.00 (1.1) | 0.00 (1.2) | 0.01 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trend_heritage | 23.0% | -2.8% | 21.8% | -14.9% | 8.4% | 14.4% | 6.0% | 9.8% |
| xs_momentum | -7.2% | 6.4% | -5.4% | 1.7% | 9.7% | 26.9% | 10.3% | 17.7% |
| xs_residual_momentum | -6.2% | 6.2% | -0.0% | 1.3% | 5.2% | 5.2% | 5.2% | 2.6% |
| xs_reversal | 6.4% | -3.4% | -8.6% | -5.0% | -5.2% | 5.1% | 5.0% | 2.7% |
| xs_low_vol | 3.2% | 8.7% | 3.5% | -2.9% | 15.4% | 10.2% | -7.8% | -5.3% |
| xs_composite | -8.0% | 14.3% | -0.0% | 1.3% | 5.2% | 5.2% | 5.2% | 2.6% |
| xs_ml_ridge | 1.7% | 3.4% | -2.6% | -5.0% | 11.3% | 17.1% | 3.8% | 7.8% |
| xs_ml_gbm | 2.6% | -8.8% | -4.6% | -3.4% | 3.2% | 10.9% | 7.6% | 7.3% |
| composite_long_only | 24.5% | -16.4% | 0.0% | 1.3% | 5.1% | 5.2% | 5.1% | 2.7% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **trend_heritage**: dsr 0.12✗, t_nw 1.04✗, alpha_t -1.22✗, lag_stress 0.70✓, cost_stress 0.83✗, pbo 0.57✗, drawdown 0.14✓, stability 0.75✓
* **xs_momentum**: dsr 0.37✗, t_nw 2.06✓, alpha_t 1.81✗, lag_stress 0.96✓, cost_stress 1.90✓, pbo 0.57✗, drawdown 0.10✓, stability 0.75✓
* **xs_residual_momentum**: dsr 0.03✗, t_nw 0.19✗, alpha_t -0.89✗, lag_stress 1.00✓, cost_stress 0.19✗, pbo 0.57✗, drawdown 0.00✓, stability 0.75✓
* **xs_reversal**: dsr 0.00✗, t_nw -1.36✗, alpha_t -1.40✗, lag_stress n/a✗, cost_stress -0.20✗, pbo 0.57✗, drawdown 0.11✓, stability 0.50✗
* **xs_low_vol**: dsr 0.01✗, t_nw -0.30✗, alpha_t -0.20✗, lag_stress n/a✗, cost_stress -0.47✗, pbo 0.57✗, drawdown 0.23✓, stability 0.62✓
* **xs_composite**: dsr 0.03✗, t_nw 0.19✗, alpha_t -0.89✗, lag_stress 1.00✓, cost_stress 0.19✗, pbo 0.57✗, drawdown 0.00✓, stability 0.75✓
* **xs_ml_ridge**: dsr 0.14✗, t_nw 1.21✗, alpha_t 0.75✗, lag_stress 1.01✓, cost_stress 0.98✗, pbo 0.57✗, drawdown 0.08✓, stability 0.75✓
* **xs_ml_gbm**: dsr 0.07✗, t_nw 0.76✗, alpha_t 0.12✗, lag_stress 0.81✓, cost_stress -0.21✗, pbo 0.57✗, drawdown 0.07✓, stability 0.62✓
* **composite_long_only**: dsr n/a✗, t_nw -62.73✗, alpha_t -70.60✗, lag_stress n/a✗, cost_stress -62.73✗, pbo 0.57✗, drawdown 0.00✓, stability 0.75✓

## Provenance

* trend_heritage: prereg `6626dae39d34`, result `f9766a0366b1`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_momentum: prereg `c15a92b38331`, result `d301530bdd3d`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_residual_momentum: prereg `9d61e9c67bc7`, result `ca5439d24ba9`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_reversal: prereg `7332438fa4cf`, result `e35f440194a9`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_low_vol: prereg `27a4d60b924c`, result `be578ccfb231`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_composite: prereg `ec20dd61c104`, result `fa20b7bdb2dd`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_ml_ridge: prereg `378bd2215121`, result `d437711fa8fc`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_ml_gbm: prereg `6868342f4d61`, result `7e10368f1478`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* composite_long_only: prereg `d3b93bba9224`, result `d30f24212d61`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:00:56.426307+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* equal_weight: prereg `8b6909219a57`, result `ea3bdbee3d20`, data `ea498d776e31`, code `7bded353247a`, 2026-10-23T22:07:06.806183+00:00, rerun: cost-stress design bug: in the 2x-cost variant the optimiser was also told costs doubled and traded less (stressed t-stat came out above base); now it decides on the registered cost model while the backtester charges 2x. Config unchanged.
