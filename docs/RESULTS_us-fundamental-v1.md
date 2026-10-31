# Results: `us-fundamental-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **5**; deflated-Sharpe trial count (whole programme `us-`): **31**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ear_drift | 0.47 | 8.0% | 6.9% | 4.7% | 0.97 | 2.1% (0.69) | 0.11 | 0.40 | 0.58 | 19.2 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| sue_drift | -0.16 | 3.5% | 6.4% | 9.7% | -0.32 | -2.4% (-0.76) | 0.01 | -0.20 | -0.69 | 18.7 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, pbo |
| low_accruals | -0.14 | 3.6% | 6.7% | 6.9% | -0.28 | -1.7% (-0.50) | 0.01 | -0.16 | -0.40 | 7.0 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, pbo |
| earnings_composite | 0.50 | 8.4% | 7.2% | 6.8% | 1.03 | 1.7% (0.55) | 0.12 | 0.42 | 0.65 | 19.6 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| fundamental_composite | 0.23 | 6.2% | 6.9% | 7.6% | 0.46 | -0.5% (-0.15) | 0.04 | 0.22 | 0.17 | 15.2 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| ear_drift | -0.90 | -5.1% | 6.8% | 29.7% | -1.93 |
| sue_drift | 0.31 | 3.0% | 6.5% | 10.4% | 0.67 |
| low_accruals | 0.32 | 3.0% | 6.6% | 7.9% | 0.64 |
| earnings_composite | -0.58 | -3.2% | 7.0% | 24.4% | -1.25 |
| fundamental_composite | -0.41 | -1.9% | 6.9% | 19.0% | -0.88 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ear_drift | -0.00 (-0.0) | -0.06 (-2.4) | -0.00 (-0.1) | 0.05 (1.5) | -0.02 (-0.4) | 0.16 (7.6) | 0.18 |
| sue_drift | -0.02 (-1.6) | -0.06 (-2.4) | 0.02 (0.7) | -0.09 (-2.9) | -0.04 (-1.0) | 0.10 (5.2) | 0.12 |
| low_accruals | 0.01 (0.7) | -0.11 (-4.1) | 0.04 (1.5) | -0.15 (-5.3) | -0.01 (-0.2) | -0.09 (-4.7) | 0.06 |
| earnings_composite | 0.00 (0.3) | -0.08 (-3.0) | -0.01 (-0.2) | 0.01 (0.3) | -0.03 (-0.8) | 0.18 (9.1) | 0.22 |
| fundamental_composite | 0.00 (0.1) | -0.10 (-3.2) | -0.05 (-1.9) | 0.01 (0.3) | -0.04 (-1.2) | 0.10 (4.8) | 0.14 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ear_drift | -10.1% | -10.0% | -10.4% | 11.8% | 4.2% | 16.2% | 3.2% | 7.1% |
| sue_drift | 2.7% | -4.6% | 5.8% | 8.5% | -0.9% | -2.9% | 8.5% | 9.1% |
| low_accruals | 0.8% | -3.2% | 7.0% | 7.9% | 4.3% | 3.8% | -0.1% | 5.7% |
| earnings_composite | -10.4% | -9.7% | -2.2% | 11.0% | 7.5% | 8.9% | 7.2% | 8.0% |
| fundamental_composite | -6.3% | -10.1% | 1.2% | 8.5% | 0.6% | 9.8% | 9.9% | 3.6% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **ear_drift**: dsr 0.11✗, t_nw 0.97✗, alpha_t 0.69✗, lag_stress 0.85✓, cost_stress 0.58✗, pbo 0.82✗, drawdown 0.05✓, stability 0.62✓
* **sue_drift**: dsr 0.01✗, t_nw -0.32✗, alpha_t -0.76✗, lag_stress n/a✗, cost_stress -0.69✗, pbo 0.82✗, drawdown 0.10✓, stability 0.62✓
* **low_accruals**: dsr 0.01✗, t_nw -0.28✗, alpha_t -0.50✗, lag_stress n/a✗, cost_stress -0.40✗, pbo 0.82✗, drawdown 0.07✓, stability 0.75✓
* **earnings_composite**: dsr 0.12✗, t_nw 1.03✗, alpha_t 0.55✗, lag_stress 0.82✓, cost_stress 0.65✗, pbo 0.82✗, drawdown 0.07✓, stability 0.62✓
* **fundamental_composite**: dsr 0.04✗, t_nw 0.46✗, alpha_t -0.15✗, lag_stress 0.96✓, cost_stress 0.17✗, pbo 0.82✗, drawdown 0.08✓, stability 0.75✓

## Provenance

* ear_drift: prereg `4cccc33c1da7`, result `fbfcbf36c219`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
* sue_drift: prereg `fbc9dd2151c7`, result `463f48a8d237`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
* low_accruals: prereg `657d1f0ed6a6`, result `58d6b968f327`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
* earnings_composite: prereg `2d8b05c8f4e0`, result `6fc45df4563b`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
* fundamental_composite: prereg `570a08497af4`, result `4c5a4e7ec81e`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
* equal_weight: prereg `183620c8f7de`, result `bca1e4ac9d12`, data `1c8e182eb273`, code `3f0307342ca9`, 2026-10-24T15:46:50.619825+00:00
