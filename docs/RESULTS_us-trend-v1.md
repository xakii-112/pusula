# Results: `us-trend-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-`): **40**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| forecast_hrp | 0.31 | 7.1% | 8.0% | 10.1% | 0.63 | -3.2% (-1.32) | 0.01 | 0.32 | 0.53 | 7.9 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| signonly_hrp | 0.30 | 7.0% | 8.0% | 10.5% | 0.61 | -3.2% (-1.36) | 0.01 | 0.32 | 0.28 | 24.5 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| forecast_equal_risk | 0.32 | 7.2% | 8.1% | 10.1% | 0.66 | -3.3% (-1.36) | 0.01 | 0.33 | 0.58 | 6.4 | no: dsr, t_nw, alpha_t, cost_stress, pbo |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| forecast_hrp | 0.36 | 4.3% | 10.1% | 14.1% | 0.74 |
| signonly_hrp | 0.30 | 3.6% | 9.4% | 14.1% | 0.61 |
| forecast_equal_risk | 0.40 | 4.7% | 10.1% | 14.1% | 0.82 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| forecast_hrp | 0.40 (9.9) | 0.10 (3.9) | 0.14 (5.4) | 0.02 (0.9) | 0.08 (2.1) | 0.07 (3.9) | 0.61 |
| signonly_hrp | 0.40 (9.7) | 0.11 (3.9) | 0.14 (5.0) | 0.03 (1.2) | 0.09 (2.3) | 0.01 (0.7) | 0.61 |
| forecast_equal_risk | 0.41 (9.9) | 0.11 (4.4) | 0.17 (6.9) | 0.03 (1.1) | 0.06 (1.7) | 0.07 (4.0) | 0.63 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forecast_hrp | 14.8% | -1.7% | 18.5% | -11.4% | 4.5% | 12.2% | 3.9% | 6.3% |
| signonly_hrp | 15.4% | -2.6% | 16.4% | -12.0% | 6.9% | 10.9% | 0.8% | 8.1% |
| forecast_equal_risk | 14.4% | -1.7% | 19.8% | -10.9% | 4.5% | 13.6% | 3.4% | 6.0% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **forecast_hrp**: dsr 0.01✗, t_nw 0.63✗, alpha_t -1.32✗, lag_stress 1.04✓, cost_stress 0.53✗, pbo 0.68✗, drawdown 0.10✓, stability 0.75✓
* **signonly_hrp**: dsr 0.01✗, t_nw 0.61✗, alpha_t -1.36✗, lag_stress 1.04✓, cost_stress 0.28✗, pbo 0.68✗, drawdown 0.11✓, stability 0.75✓
* **forecast_equal_risk**: dsr 0.01✗, t_nw 0.66✗, alpha_t -1.36✗, lag_stress 1.02✓, cost_stress 0.58✗, pbo 0.68✗, drawdown 0.10✓, stability 0.75✓

## Provenance

* forecast_hrp: prereg `c85f6847779b`, result `bf65f845eaee`, data `6535432b7d0a`, code `94c2d9bccb97`, 2026-10-30T11:50:53.158522+00:00
* signonly_hrp: prereg `70093308cc16`, result `08ede889b031`, data `6535432b7d0a`, code `94c2d9bccb97`, 2026-10-30T11:50:53.158522+00:00
* forecast_equal_risk: prereg `66292888ca65`, result `1848c81f6db3`, data `6535432b7d0a`, code `94c2d9bccb97`, 2026-10-30T11:50:53.158522+00:00
* equal_weight: prereg `e32a9c13e8eb`, result `8b5d3e7ace86`, data `6535432b7d0a`, code `94c2d9bccb97`, 2026-10-30T11:50:53.158522+00:00
