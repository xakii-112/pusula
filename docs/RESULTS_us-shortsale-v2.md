# Results: `us-shortsale-v2`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **1**; deflated-Sharpe trial count (whole programme `us-`): **37**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short_share_level | 0.82 | 10.7% | 6.9% | 9.8% | 1.59 | 4.5% (1.30) | 0.06 | 0.80 | 0.62 | 56.3 | no: dsr, t_nw, alpha_t, cost_stress, pbo, stability |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| short_share_level | -0.66 | -3.4% | 6.5% | 18.7% | -1.30 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_share_level | -0.01 (-0.9) | -0.17 (-6.4) | 0.00 (0.1) | -0.12 (-3.4) | -0.02 (-0.5) | 0.06 (2.9) | 0.13 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_share_level | 0.5% | -4.0% | -6.9% | -3.0% | 11.9% | 13.3% | 22.7% | -5.6% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **short_share_level**: dsr 0.06✗, t_nw 1.59✗, alpha_t 1.30✗, lag_stress 0.97✓, cost_stress 0.62✗, pbo n/a✗, drawdown 0.10✓, stability 0.50✗

## Provenance

* short_share_level: prereg `a5b9dbba6c7c`, result `de492f72c22a`, data `08e3b07582dd`, code `c6b6c2547a4c`, 2026-10-24T21:05:08.046244+00:00
* equal_weight: prereg `cca6dab3ee4e`, result `ac922145de36`, data `08e3b07582dd`, code `c6b6c2547a4c`, 2026-10-24T21:05:08.046244+00:00
