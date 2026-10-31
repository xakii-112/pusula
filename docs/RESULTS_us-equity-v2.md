# Results: `us-equity-v2`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **9**; deflated-Sharpe trial count (whole programme `us-equity-`): **21**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. All returns are net of the calibrated cost model (spread, square-root impact, fees, borrow, financing) with execution at the next session's VWAP. Identical trials to us-equity-v1; research backtests without the latching kill switch.

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| trend_heritage | 0.51 | 10.2% | 11.0% | 14.0% | 1.04 | -3.5% (-1.22) | 0.12 | 0.36 | 0.83 | 21.0 | no: dsr, t_nw, alpha_t, cost_stress |
| xs_momentum | 0.94 | 16.9% | 12.5% | 9.9% | 2.06 | 7.0% (1.81) | 0.37 | 0.91 | 1.90 | 13.9 | no: dsr, alpha_t |
| xs_residual_momentum | 0.79 | 14.7% | 12.4% | 9.2% | 1.77 | 5.1% (1.45) | 0.26 | 0.77 | 1.63 | 13.8 | no: dsr, t_nw, alpha_t |
| xs_reversal | -1.09 | -7.1% | 10.6% | 26.2% | -2.09 | -10.4% (-1.92) | 0.00 | -1.18 | -3.10 | 86.9 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, drawdown, stability |
| xs_low_vol | -0.15 | 2.8% | 9.7% | 22.8% | -0.30 | -0.9% (-0.20) | 0.01 | -0.25 | -0.47 | 14.0 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress |
| xs_composite | 0.71 | 13.4% | 12.3% | 10.0% | 1.49 | 5.1% (1.25) | 0.21 | 0.66 | 1.31 | 16.2 | no: dsr, t_nw, alpha_t |
| xs_ml_ridge | 0.56 | 10.5% | 10.4% | 7.9% | 1.21 | 2.8% (0.75) | 0.14 | 0.57 | 0.98 | 23.2 | no: dsr, t_nw, alpha_t, cost_stress |
| xs_ml_gbm | 0.36 | 7.6% | 8.4% | 7.0% | 0.76 | 0.4% (0.12) | 0.07 | 0.29 | 0.04 | 45.6 | no: dsr, t_nw, alpha_t, cost_stress |
| composite_long_only | 0.86 | 13.4% | 9.8% | 11.1% | 1.76 | 0.6% (0.18) | 0.31 | 0.85 | 1.71 | 5.6 | no: dsr, t_nw, alpha_t |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| trend_heritage | 0.38 | 5.5% | 13.6% | 22.6% | 0.82 |
| xs_momentum | -0.12 | -1.3% | 12.9% | 22.7% | -0.25 |
| xs_residual_momentum | -0.06 | -0.4% | 12.4% | 17.9% | -0.13 |
| xs_reversal | -0.29 | -2.8% | 11.3% | 22.7% | -0.61 |
| xs_low_vol | 0.25 | 3.0% | 9.2% | 15.4% | 0.51 |
| xs_composite | -0.10 | -1.0% | 12.9% | 21.7% | -0.21 |
| xs_ml_ridge | -0.14 | -0.7% | 9.7% | 12.6% | -0.30 |
| xs_ml_gbm | -0.61 | -3.6% | 7.4% | 19.9% | -1.28 |
| composite_long_only | 0.69 | 11.9% | 16.8% | 29.4% | 1.47 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trend_heritage | 0.63 (20.3) | 0.07 (2.0) | 0.25 (8.6) | 0.05 (1.5) | 0.09 (2.4) | 0.02 (1.2) | 0.71 |
| xs_momentum | 0.04 (1.5) | -0.01 (-0.4) | -0.04 (-1.2) | 0.10 (3.1) | -0.09 (-2.1) | 0.59 (21.4) | 0.62 |
| xs_residual_momentum | 0.03 (1.3) | -0.00 (-0.0) | -0.02 (-0.6) | 0.09 (2.6) | -0.08 (-2.1) | 0.61 (23.8) | 0.65 |
| xs_reversal | 0.14 (5.2) | 0.10 (2.3) | -0.03 (-0.6) | 0.09 (1.6) | 0.08 (1.2) | -0.14 (-4.3) | 0.12 |
| xs_low_vol | -0.04 (-1.3) | -0.01 (-0.3) | 0.10 (2.2) | 0.34 (7.1) | -0.03 (-0.7) | 0.08 (2.7) | 0.18 |
| xs_composite | 0.02 (0.7) | -0.03 (-0.9) | 0.06 (1.5) | 0.17 (4.7) | -0.10 (-2.3) | 0.59 (21.9) | 0.61 |
| xs_ml_ridge | 0.07 (3.1) | 0.03 (0.8) | -0.01 (-0.3) | 0.11 (3.1) | -0.02 (-0.4) | 0.45 (19.9) | 0.48 |
| xs_ml_gbm | 0.07 (2.9) | -0.02 (-0.8) | -0.00 (-0.1) | -0.01 (-0.2) | -0.07 (-1.8) | 0.28 (14.6) | 0.35 |
| composite_long_only | 0.51 (20.5) | -0.09 (-2.5) | 0.17 (4.8) | 0.07 (2.1) | 0.18 (4.5) | 0.06 (2.9) | 0.53 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trend_heritage | 23.0% | -2.8% | 21.8% | -14.9% | 8.4% | 14.4% | 6.0% | 9.8% |
| xs_momentum | -7.2% | 6.4% | -5.4% | 1.7% | 9.7% | 26.9% | 10.3% | 17.7% |
| xs_residual_momentum | -6.2% | 10.5% | -4.6% | -0.5% | 10.7% | 19.6% | 10.4% | 14.9% |
| xs_reversal | 6.4% | -3.4% | -8.6% | -5.0% | -1.8% | -6.0% | 0.3% | -18.3% |
| xs_low_vol | 3.2% | 8.7% | 3.5% | -2.9% | 15.4% | 10.2% | -7.8% | -5.3% |
| xs_composite | -8.0% | 11.4% | -2.3% | -4.0% | 10.8% | 29.8% | 3.9% | 7.9% |
| xs_ml_ridge | 1.7% | 3.4% | -2.6% | -5.0% | 11.3% | 17.1% | 3.8% | 7.8% |
| xs_ml_gbm | 2.6% | -8.8% | -4.6% | -3.4% | 3.2% | 10.9% | 7.6% | 7.3% |
| composite_long_only | 24.5% | 9.5% | 19.5% | -3.9% | 2.1% | 21.0% | 11.3% | 17.1% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **trend_heritage**: dsr 0.12✗, t_nw 1.04✗, alpha_t -1.22✗, lag_stress 0.70✓, cost_stress 0.83✗, pbo 0.25✓, drawdown 0.14✓, stability 0.75✓
* **xs_momentum**: dsr 0.37✗, t_nw 2.06✓, alpha_t 1.81✗, lag_stress 0.96✓, cost_stress 1.90✓, pbo 0.25✓, drawdown 0.10✓, stability 0.75✓
* **xs_residual_momentum**: dsr 0.26✗, t_nw 1.77✗, alpha_t 1.45✗, lag_stress 0.97✓, cost_stress 1.63✓, pbo 0.25✓, drawdown 0.09✓, stability 0.62✓
* **xs_reversal**: dsr 0.00✗, t_nw -2.09✗, alpha_t -1.92✗, lag_stress n/a✗, cost_stress -3.10✗, pbo 0.25✓, drawdown 0.26✗, stability 0.25✗
* **xs_low_vol**: dsr 0.01✗, t_nw -0.30✗, alpha_t -0.20✗, lag_stress n/a✗, cost_stress -0.47✗, pbo 0.25✓, drawdown 0.23✓, stability 0.62✓
* **xs_composite**: dsr 0.21✗, t_nw 1.49✗, alpha_t 1.25✗, lag_stress 0.94✓, cost_stress 1.31✓, pbo 0.25✓, drawdown 0.10✓, stability 0.62✓
* **xs_ml_ridge**: dsr 0.14✗, t_nw 1.21✗, alpha_t 0.75✗, lag_stress 1.01✓, cost_stress 0.98✗, pbo 0.25✓, drawdown 0.08✓, stability 0.75✓
* **xs_ml_gbm**: dsr 0.07✗, t_nw 0.76✗, alpha_t 0.12✗, lag_stress 0.81✓, cost_stress 0.04✗, pbo 0.25✓, drawdown 0.07✓, stability 0.62✓
* **composite_long_only**: dsr 0.31✗, t_nw 1.76✗, alpha_t 0.18✗, lag_stress 0.98✓, cost_stress 1.71✓, pbo 0.25✓, drawdown 0.11✓, stability 0.88✓

## Provenance

* trend_heritage: prereg `e96f6347f60a`, result `d1d80e8eb19b`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_momentum: prereg `b9f578ea65e3`, result `470efdc6c5e6`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_residual_momentum: prereg `641bab6c17d7`, result `bf4a592091e0`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_reversal: prereg `f133b979b67c`, result `f54788a96fa9`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_low_vol: prereg `3f590172c921`, result `1ec6aa28aedb`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_composite: prereg `2a06b35d46a8`, result `c699edbfa0ad`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_ml_ridge: prereg `8f8b4a11103b`, result `b009dd5e3aeb`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_ml_gbm: prereg `ea7f124c5942`, result `564daf23e70d`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* composite_long_only: prereg `2a1f477bc9d2`, result `e24e19d9eb33`, data `ea498d776e31`, code `c77188739cd6`, 2026-10-24T05:01:00.231123+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* equal_weight: prereg `993cf81cf201`, result `4a1100912fdc`, data `ea498d776e31`, code `7bded353247a`, 2026-10-23T22:09:22.114137+00:00, rerun: cost-stress design bug: in the 2x-cost variant the optimiser was also told costs doubled and traded less (stressed t-stat came out above base); now it decides on the registered cost model while the backtester charges 2x. Config unchanged.
