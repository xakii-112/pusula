# Results: `us-optsurface-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-`): **34**. Sharpe, t and DSR use excess returns.

Development window 2019-06-03 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iv_smirk_ls | -1.05 | -2.2% | 6.3% | 17.8% | -2.11 | -7.1% (-2.24) | 0.00 | -1.14 | -3.97 | 97.4 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, stability |
| iv_spread_ls | -1.96 | -6.1% | 5.5% | 25.0% | -3.50 | -10.8% (-3.42) | 0.00 | -2.05 | -5.19 | 85.7 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, drawdown, stability |
| surface_composite | -1.70 | -5.0% | 5.7% | 19.6% | -3.61 | -9.9% (-3.66) | 0.00 | -1.77 | -5.95 | 93.7 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, stability |
| equal_weight | 0.71 | 14.7% | 14.0% | 17.5% | 1.50 | -1.4% (-0.71) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| iv_smirk_ls | -1.44 | -7.1% | 5.7% | 26.9% | -3.11 |
| iv_spread_ls | -1.49 | -7.3% | 5.6% | 29.0% | -3.12 |
| surface_composite | -1.66 | -8.3% | 5.7% | 29.6% | -3.58 |
| equal_weight | 0.52 | 11.4% | 25.1% | 39.6% | 1.07 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| iv_smirk_ls | -0.01 (-0.8) | -0.10 (-3.5) | -0.07 (-2.3) | -0.05 (-1.2) | -0.05 (-1.6) | 0.02 (1.0) | 0.09 |
| iv_spread_ls | 0.03 (2.4) | 0.01 (0.5) | 0.02 (1.1) | 0.04 (1.4) | -0.03 (-1.0) | -0.05 (-3.6) | 0.04 |
| surface_composite | 0.01 (0.5) | -0.07 (-3.0) | -0.06 (-2.3) | -0.02 (-0.7) | -0.04 (-1.4) | -0.01 (-0.7) | 0.05 |
| equal_weight | 0.87 (68.6) | 0.12 (6.2) | 0.22 (10.2) | -0.01 (-0.3) | 0.20 (7.1) | -0.16 (-14.9) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iv_smirk_ls | -1.9% | -8.7% | -12.1% | -2.5% | -6.5% | -4.6% | 4.1% | -0.8% |
| iv_spread_ls | 3.2% | -10.5% | -6.2% | -12.2% | -1.8% | 0.8% | -6.6% | -14.8% |
| surface_composite | -0.4% | -11.5% | -10.6% | -7.2% | -7.0% | -2.6% | -3.5% | -5.9% |
| equal_weight | 14.5% | 11.4% | 30.1% | -11.3% | 15.3% | 13.3% | 12.5% | 14.3% |

## Gate detail

* **iv_smirk_ls**: dsr 0.00✗, t_nw -2.11✗, alpha_t -2.24✗, lag_stress n/a✗, cost_stress -3.97✗, pbo 0.40✓, drawdown 0.18✓, stability 0.12✗
* **iv_spread_ls**: dsr 0.00✗, t_nw -3.50✗, alpha_t -3.42✗, lag_stress n/a✗, cost_stress -5.19✗, pbo 0.40✓, drawdown 0.25✗, stability 0.25✗
* **surface_composite**: dsr 0.00✗, t_nw -3.61✗, alpha_t -3.66✗, lag_stress n/a✗, cost_stress -5.95✗, pbo 0.40✓, drawdown 0.20✓, stability 0.00✗

## Provenance

* iv_smirk_ls: prereg `a0f4c47da616`, result `fdec565e2fdc`, data `e32191f74ca8`, code `1ae4195eca51`, 2026-10-24T19:30:28.045054+00:00
* iv_spread_ls: prereg `3d6f8d403e21`, result `85ea169b01f5`, data `e32191f74ca8`, code `1ae4195eca51`, 2026-10-24T19:30:28.045054+00:00
* surface_composite: prereg `5085b04a1712`, result `584395f2d83c`, data `e32191f74ca8`, code `1ae4195eca51`, 2026-10-24T19:30:28.045054+00:00
* equal_weight: prereg `4a89c9a7f459`, result `9baf5afcd3d2`, data `e32191f74ca8`, code `1ae4195eca51`, 2026-10-24T19:30:28.045054+00:00
