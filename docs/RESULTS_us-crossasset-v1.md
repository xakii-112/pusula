# Results: `us-crossasset-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-`): **43**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ca_trend_ls | -0.05 | 4.2% | 6.5% | 8.2% | -0.11 | -2.2% (-0.73) | 0.00 | -0.00 | -0.43 | 18.0 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress |
| ca_trend_long | 0.25 | 6.0% | 5.3% | 4.7% | 0.52 | -1.2% (-0.52) | 0.01 | 0.28 | 0.35 | 8.1 | no: dsr, t_nw, alpha_t, cost_stress |
| ca_trend_equal_risk | -0.32 | 2.7% | 5.8% | 6.4% | -0.67 | -4.0% (-1.61) | 0.00 | -0.32 | -0.89 | 15.8 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress |
| equal_weight | 0.84 | 11.3% | 7.6% | 8.3% | 1.70 | 0.6% (0.23) | n/a | n/a | n/a | 0.5 | benchmark |
| risk_parity_long | 0.56 | 7.5% | 4.7% | 4.7% | 1.11 | -0.1% (-0.05) | n/a | n/a | n/a | 3.1 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| ca_trend_ls | 0.74 | 7.7% | 9.0% | 10.6% | 1.51 |
| ca_trend_long | 0.95 | 6.1% | 5.3% | 7.8% | 1.82 |
| ca_trend_equal_risk | 0.75 | 7.7% | 9.0% | 9.4% | 1.54 |
| equal_weight | 0.57 | 6.8% | 10.7% | 19.6% | 1.09 |
| risk_parity_long | 0.32 | 2.8% | 5.8% | 11.5% | 0.61 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | 0.12 (4.8) | 0.01 (0.3) | 0.03 (1.2) | -0.04 (-1.4) | 0.00 (0.2) | 0.06 (3.6) | 0.12 |
| ca_trend_long | 0.16 (8.2) | 0.01 (0.4) | 0.04 (1.8) | -0.05 (-2.4) | 0.01 (0.4) | 0.03 (2.6) | 0.26 |
| ca_trend_equal_risk | 0.15 (5.0) | 0.03 (1.4) | 0.01 (0.3) | -0.03 (-1.3) | 0.00 (0.1) | 0.05 (3.8) | 0.22 |
| equal_weight | 0.39 (19.9) | -0.01 (-0.5) | 0.08 (2.5) | -0.08 (-3.7) | 0.06 (2.0) | -0.01 (-1.0) | 0.63 |
| risk_parity_long | 0.20 (14.0) | 0.01 (0.5) | 0.04 (1.7) | -0.03 (-2.0) | 0.04 (1.6) | -0.01 (-0.9) | 0.44 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | 6.7% | 6.7% | 6.8% | 10.7% | -5.3% | 10.6% | 8.4% | 3.1% |
| ca_trend_long | 10.4% | 4.0% | 7.3% | 3.0% | -0.3% | 10.0% | 9.3% | 4.1% |
| ca_trend_equal_risk | 5.4% | 5.9% | 9.3% | 10.2% | -1.8% | 7.4% | 3.9% | 1.0% |
| equal_weight | 15.1% | 8.7% | 13.5% | -8.3% | 8.9% | 10.0% | 14.0% | 9.9% |
| risk_parity_long | 10.3% | 2.2% | 5.6% | -6.1% | 5.6% | 9.5% | 8.6% | 4.7% |

## Gate detail

* **ca_trend_ls**: dsr 0.00✗, t_nw -0.11✗, alpha_t -0.73✗, lag_stress n/a✗, cost_stress -0.43✗, pbo 0.21✓, drawdown 0.08✓, stability 0.88✓
* **ca_trend_long**: dsr 0.01✗, t_nw 0.52✗, alpha_t -0.52✗, lag_stress 1.10✓, cost_stress 0.35✗, pbo 0.21✓, drawdown 0.05✓, stability 0.88✓
* **ca_trend_equal_risk**: dsr 0.00✗, t_nw -0.67✗, alpha_t -1.61✗, lag_stress n/a✗, cost_stress -0.89✗, pbo 0.21✓, drawdown 0.06✓, stability 0.88✓

## Provenance

* ca_trend_ls: prereg `d74a6aee3492`, result `c230e758257a`, data `a2eb2e1c0151`, code `83f82c760b06`, 2026-10-30T13:20:34.955820+00:00
* ca_trend_long: prereg `10c8ffaaa5dd`, result `528d257eb538`, data `a2eb2e1c0151`, code `83f82c760b06`, 2026-10-30T13:20:34.955820+00:00
* ca_trend_equal_risk: prereg `9525c457a66b`, result `9aa28b37fdd2`, data `a2eb2e1c0151`, code `83f82c760b06`, 2026-10-30T13:20:34.955820+00:00
* equal_weight: prereg `09afff7459be`, result `d27e0d7e2304`, data `a2eb2e1c0151`, code `83f82c760b06`, 2026-10-30T13:20:34.955820+00:00
* risk_parity_long: prereg `bb8592edf6e9`, result `bfcf0e6c0162`, data `a2eb2e1c0151`, code `83f82c760b06`, 2026-10-30T13:20:34.955820+00:00
