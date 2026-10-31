# Results: `us-equity-v3`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-equity-`): **21**. Sharpe, t and DSR use excess returns.

Development window 2019-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| xs_momentum_adaptive | 0.93 | 15.4% | 11.0% | 8.8% | 2.01 | 6.3% (1.70) | 0.36 | 0.92 | 1.83 | 11.1 | no: dsr, alpha_t, pbo, stability |
| xs_residual_momentum_adaptive | 0.77 | 13.4% | 11.0% | 7.5% | 1.72 | 4.5% (1.28) | 0.25 | 0.78 | 1.57 | 10.9 | no: dsr, t_nw, alpha_t, pbo |
| xs_composite_adaptive | 0.55 | 10.5% | 10.6% | 9.0% | 1.14 | 2.7% (0.73) | 0.14 | 0.52 | 1.00 | 11.0 | no: dsr, t_nw, alpha_t, pbo |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.49 | -1.4% (-0.75) | n/a | n/a | n/a | 0.9 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| xs_momentum_adaptive | -0.56 | -4.8% | 9.9% | 20.7% | -1.13 |
| xs_residual_momentum_adaptive | -0.31 | -2.3% | 9.6% | 15.6% | -0.64 |
| xs_composite_adaptive | -0.31 | -2.4% | 10.0% | 21.7% | -0.63 |
| equal_weight | 0.63 | 14.4% | 24.1% | 39.6% | 1.37 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xs_momentum_adaptive | 0.03 (1.1) | -0.02 (-0.7) | -0.02 (-0.6) | 0.07 (2.3) | -0.05 (-1.2) | 0.50 (20.5) | 0.56 |
| xs_residual_momentum_adaptive | 0.03 (1.2) | -0.01 (-0.3) | 0.00 (0.1) | 0.06 (2.1) | -0.08 (-1.9) | 0.51 (19.1) | 0.58 |
| xs_composite_adaptive | 0.03 (1.1) | -0.03 (-1.0) | 0.08 (2.6) | 0.17 (4.8) | -0.10 (-2.4) | 0.49 (20.7) | 0.56 |
| equal_weight | 0.87 (68.0) | 0.12 (6.1) | 0.22 (10.2) | -0.01 (-0.4) | 0.20 (7.0) | -0.16 (-14.7) | 0.92 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xs_momentum_adaptive | -8.3% | -0.7% | -5.5% | -4.5% | 9.7% | 26.1% | 6.8% | 16.3% |
| xs_residual_momentum_adaptive | -6.4% | 5.9% | -2.7% | -5.4% | 12.7% | 14.2% | 9.0% | 14.7% |
| xs_composite_adaptive | -8.7% | 8.1% | -1.2% | -7.1% | 7.5% | 24.8% | 1.8% | 6.8% |
| equal_weight | 31.9% | 12.0% | 30.4% | -11.1% | 15.3% | 13.3% | 12.3% | 14.3% |

## Gate detail

* **xs_momentum_adaptive**: dsr 0.36✗, t_nw 2.01✓, alpha_t 1.70✗, lag_stress 0.99✓, cost_stress 1.83✓, pbo 0.83✗, drawdown 0.09✓, stability 0.50✗
* **xs_residual_momentum_adaptive**: dsr 0.25✗, t_nw 1.72✗, alpha_t 1.28✗, lag_stress 1.01✓, cost_stress 1.57✓, pbo 0.83✗, drawdown 0.07✓, stability 0.62✓
* **xs_composite_adaptive**: dsr 0.14✗, t_nw 1.14✗, alpha_t 0.73✗, lag_stress 0.94✓, cost_stress 1.00✓, pbo 0.83✗, drawdown 0.09✓, stability 0.62✓

## Provenance

* xs_momentum_adaptive: prereg `d2bd80b61a19`, result `20e6bbc0fac7`, data `1ea4f768f1ca`, code `c77188739cd6`, 2026-10-24T05:01:04.139965+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_residual_momentum_adaptive: prereg `134cbdc4734b`, result `c37fd5262059`, data `1ea4f768f1ca`, code `c77188739cd6`, 2026-10-24T05:01:04.139965+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* xs_composite_adaptive: prereg `80edf5121ab4`, result `497a35c76c74`, data `1ea4f768f1ca`, code `c77188739cd6`, 2026-10-24T05:01:04.139965+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* equal_weight: prereg `91acd23f3fca`, result `3f4bb7726f68`, data `1ea4f768f1ca`, code `d67b1398d0e0`, 2026-10-24T04:58:46.917710+00:00
