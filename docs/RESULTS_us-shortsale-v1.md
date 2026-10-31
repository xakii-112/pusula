# Results: `us-shortsale-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **2**; deflated-Sharpe trial count (whole programme `us-`): **36**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short_pressure_week | -0.77 | 0.1% | 5.7% | 10.7% | -1.44 | -4.2% (-1.38) | 0.00 | -0.54 | -2.49 | 57.5 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, stability |
| short_pressure_month | -0.15 | 3.8% | 5.3% | 5.8% | -0.30 | -0.4% (-0.14) | 0.00 | -0.24 | -1.13 | 36.8 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| short_pressure_week | -1.19 | -4.8% | 5.0% | 22.2% | -2.64 |
| short_pressure_month | -0.24 | -0.3% | 5.5% | 12.1% | -0.47 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_pressure_week | 0.01 (0.8) | -0.07 (-2.9) | -0.01 (-0.7) | -0.01 (-0.3) | 0.05 (1.6) | 0.04 (2.4) | 0.04 |
| short_pressure_month | -0.02 (-1.8) | -0.08 (-3.4) | 0.01 (0.5) | -0.03 (-1.1) | 0.02 (0.7) | 0.01 (0.6) | 0.03 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_pressure_week | -2.0% | -2.5% | -10.5% | -4.1% | 6.8% | -3.6% | 5.7% | -7.7% |
| short_pressure_month | 2.7% | -1.0% | -7.3% | 4.6% | 7.9% | -1.0% | 5.8% | 1.9% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **short_pressure_week**: dsr 0.00✗, t_nw -1.44✗, alpha_t -1.38✗, lag_stress n/a✗, cost_stress -2.49✗, pbo 0.00✓, drawdown 0.11✓, stability 0.25✗
* **short_pressure_month**: dsr 0.00✗, t_nw -0.30✗, alpha_t -0.14✗, lag_stress n/a✗, cost_stress -1.13✗, pbo 0.00✓, drawdown 0.06✓, stability 0.62✓

## Provenance

* short_pressure_week: prereg `ad57a0a5af3b`, result `779d73483ec9`, data `08e3b07582dd`, code `fffd2abfc009`, 2026-10-24T20:27:40.958864+00:00
* short_pressure_month: prereg `85240fc4496a`, result `c3494bb10ebe`, data `08e3b07582dd`, code `fffd2abfc009`, 2026-10-24T20:27:40.958864+00:00
* equal_weight: prereg `02a5c537ce47`, result `364a862bb8c6`, data `08e3b07582dd`, code `fffd2abfc009`, 2026-10-24T20:27:40.958864+00:00
