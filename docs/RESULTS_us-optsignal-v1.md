# Results: `us-optsignal-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **2**; deflated-Sharpe trial count (whole programme `us-`): **26**. Sharpe, t and DSR use excess returns.

Development window 2019-06-03 .. 2022-12-30; holdout 2023-01-03 .. 2026-08-26. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iv_rv_spread | -0.81 | -1.5% | 7.4% | 14.9% | -1.59 | -5.2% (-1.33) | 0.00 | -0.82 | -2.66 | 55.7 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress |
| iv_rv_plus_momentum | 0.32 | 7.9% | 10.7% | 9.6% | 0.72 | 0.3% (0.08) | 0.06 | 0.36 | 0.07 | 39.1 | no: dsr, t_nw, alpha_t, cost_stress |
| equal_weight | 0.77 | 15.9% | 14.1% | 17.5% | 1.60 | -1.0% (-0.55) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| iv_rv_spread | 0.12 | 1.5% | 7.5% | 22.3% | 0.22 |
| iv_rv_plus_momentum | 0.07 | 1.1% | 11.1% | 22.1% | 0.14 |
| equal_weight | 0.52 | 11.4% | 25.1% | 39.6% | 1.07 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| iv_rv_spread | -0.05 (-2.2) | -0.02 (-0.5) | 0.04 (1.2) | -0.02 (-0.7) | 0.05 (1.3) | 0.05 (2.3) | 0.03 |
| iv_rv_plus_momentum | -0.00 (-0.1) | -0.03 (-1.0) | 0.02 (0.4) | 0.03 (1.1) | -0.06 (-1.2) | 0.46 (19.4) | 0.50 |
| equal_weight | 0.87 (68.8) | 0.12 (6.1) | 0.22 (10.1) | -0.01 (-0.3) | 0.20 (7.0) | -0.16 (-15.0) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iv_rv_spread | 6.5% | -12.2% | 5.3% | 7.4% | 5.4% | -10.0% | 4.7% | -4.7% |
| iv_rv_plus_momentum | 0.9% | -4.5% | 0.1% | 7.9% | 5.6% | 9.8% | 12.1% | 1.4% |
| equal_weight | 14.5% | 11.4% | 30.1% | -11.3% | 15.3% | 13.3% | 12.5% | 16.1% |

## Gate detail

* **iv_rv_spread**: dsr 0.00✗, t_nw -1.59✗, alpha_t -1.33✗, lag_stress n/a✗, cost_stress -2.66✗, pbo 0.16✓, drawdown 0.15✓, stability 0.62✓
* **iv_rv_plus_momentum**: dsr 0.06✗, t_nw 0.72✗, alpha_t 0.08✗, lag_stress 1.13✓, cost_stress 0.07✗, pbo 0.16✓, drawdown 0.10✓, stability 0.88✓

## Provenance

* iv_rv_spread: prereg `04de68fa1839`, result `bb7c8d997f62`, data `1ea4f768f1ca`, code `c77188739cd6`, 2026-10-24T05:01:10.369885+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* iv_rv_plus_momentum: prereg `43f423151978`, result `eca0d260199d`, data `1ea4f768f1ca`, code `c77188739cd6`, 2026-10-24T05:01:10.369885+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* equal_weight: prereg `0a7fc1555e7c`, result `0d9f94f21a00`, data `1ea4f768f1ca`, code `c4a24724faff`, 2026-10-23T22:17:11.238130+00:00
