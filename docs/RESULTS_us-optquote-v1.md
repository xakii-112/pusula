# Results: `us-optquote-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **4**; deflated-Sharpe trial count (whole programme `us-optquote-`): **4**. Sharpe, t and DSR use excess returns.

Development window 2019-06-03 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| calls_quoted | 0.00 | 3.0% | 18.5% | 21.8% | 0.00 | -12.5% (-2.08) | 0.00 | 0.10 | -0.54 | n/a | no: dsr, t_nw, alpha_t, cost_stress, stability |
| put_write | 0.91 | 11.4% | 7.0% | 9.7% | 2.17 | 0.7% (0.32) | 0.09 | 0.53 | 1.97 | n/a | no: dsr, alpha_t |
| covered_call | 0.45 | 10.7% | 14.6% | 19.9% | 0.99 | -7.0% (-2.44) | 0.01 | 0.54 | 0.90 | n/a | no: dsr, t_nw, alpha_t, cost_stress |
| ivrv_straddles | -0.51 | -8.3% | 21.5% | 33.2% | -1.94 | -10.0% (-1.67) | 0.00 | -0.68 | -4.13 | n/a | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, drawdown, stability |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| calls_quoted | -0.20 | -4.9% | 20.0% | 27.7% | -0.42 |
| put_write | 0.05 | 0.7% | 14.1% | 31.2% | 0.11 |
| covered_call | 0.36 | 6.5% | 21.8% | 34.3% | 0.74 |
| ivrv_straddles | -0.76 | -12.7% | 17.1% | 41.4% | -1.97 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| calls_quoted | 0.63 (7.5) | 0.16 (2.4) | 0.22 (4.0) | -0.18 (-2.0) | -0.11 (-1.3) | 0.49 (9.8) | 0.53 |
| put_write | 0.37 (4.7) | -0.08 (-1.7) | 0.03 (1.5) | -0.06 (-3.0) | 0.11 (2.3) | -0.03 (-1.0) | 0.61 |
| covered_call | 0.86 (17.1) | -0.00 (-0.1) | 0.07 (2.4) | -0.12 (-3.6) | 0.10 (2.4) | -0.06 (-2.6) | 0.84 |
| ivrv_straddles | -0.03 (-0.4) | -0.02 (-0.2) | 0.06 (0.7) | 0.07 (0.9) | -0.03 (-0.3) | 0.02 (0.3) | 0.00 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calls_quoted | -5.3% | 12.8% | -3.6% | -18.8% | -2.2% | 10.7% | -7.7% | 12.1% |
| put_write | 3.3% | -11.4% | 9.6% | 2.0% | 11.8% | 11.0% | 11.1% | 9.3% |
| covered_call | 11.1% | 7.1% | 22.5% | -13.9% | 14.6% | 7.4% | 12.6% | 6.2% |
| ivrv_straddles | -18.3% | 3.1% | -6.6% | -21.9% | -9.1% | -15.1% | -6.5% | -0.3% |

## Gate detail

* **calls_quoted**: dsr 0.00✗, t_nw 0.00✗, alpha_t -2.08✗, lag_stress 53.70✓, cost_stress -0.54✗, pbo 0.16✓, drawdown 0.22✓, stability 0.38✗
* **put_write**: dsr 0.09✗, t_nw 2.17✓, alpha_t 0.32✗, lag_stress 0.59✓, cost_stress 1.97✓, pbo 0.16✓, drawdown 0.10✓, stability 0.88✓
* **covered_call**: dsr 0.01✗, t_nw 0.99✗, alpha_t -2.44✗, lag_stress 1.20✓, cost_stress 0.90✗, pbo 0.16✓, drawdown 0.20✓, stability 0.88✓
* **ivrv_straddles**: dsr 0.00✗, t_nw -1.94✗, alpha_t -1.67✗, lag_stress n/a✗, cost_stress -4.13✗, pbo 0.16✓, drawdown 0.33✗, stability 0.12✗

## Provenance

* calls_quoted: prereg `de1a6cadb555`, result `4e560e7a9a07`, data `9372f6cadf02`, code `8e473eb0f932`, 2026-10-30T15:50:07.003023+00:00, rerun: Data defect in the first run, found from its 2019 column: until 2020 the chain dates its weekly snapshot on Saturdays (Friday's closing quotes), and the run matched sessions only, so it held cash from 2019-06 to 2019-12. Saturday snapshots now belong to the session before them. Configuration unchanged.
* put_write: prereg `2716740501f7`, result `1b92e41994f1`, data `9372f6cadf02`, code `8e473eb0f932`, 2026-10-30T15:50:07.003023+00:00, rerun: Data defect in the first run, found from its 2019 column: until 2020 the chain dates its weekly snapshot on Saturdays (Friday's closing quotes), and the run matched sessions only, so it held cash from 2019-06 to 2019-12. Saturday snapshots now belong to the session before them. Configuration unchanged.
* covered_call: prereg `ebae932e61ce`, result `f1129217a870`, data `9372f6cadf02`, code `8e473eb0f932`, 2026-10-30T15:50:07.003023+00:00, rerun: Data defect in the first run, found from its 2019 column: until 2020 the chain dates its weekly snapshot on Saturdays (Friday's closing quotes), and the run matched sessions only, so it held cash from 2019-06 to 2019-12. Saturday snapshots now belong to the session before them. Configuration unchanged.
* ivrv_straddles: prereg `a077af7b43b3`, result `42fdebf55a6a`, data `9372f6cadf02`, code `8e473eb0f932`, 2026-10-30T15:50:07.003023+00:00, rerun: Data defect in the first run, found from its 2019 column: until 2020 the chain dates its weekly snapshot on Saturdays (Friday's closing quotes), and the run matched sessions only, so it held cash from 2019-06 to 2019-12. Saturday snapshots now belong to the session before them. Configuration unchanged.
