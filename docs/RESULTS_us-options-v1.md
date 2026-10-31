# Results: `us-options-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-options-`): **3**. Sharpe, t and DSR use excess returns.

Development window 2019-03-01 .. 2022-12-30; holdout 2023-01-03 .. 2026-09-25. Calls are bought at the close one session after the signal, priced with Black-Scholes-Merton at the observed 30-day ATM implied volatility plus a 2.5% half-spread on premium, and held to expiry (payoff exact). The stock mirror pays 5 bp per side.

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| calls_d50_t21 | 0.46 | 12.8% | 20.7% | 20.5% | 0.99 | -5.7% (-0.89) | 0.76 | 0.48 | 0.77 | n/a | no: dsr, t_nw, alpha_t, cost_stress |
| calls_d30_t21 | 0.24 | 7.8% | 22.5% | 23.6% | 0.51 | -9.0% (-1.14) | 0.61 | 0.25 | 0.35 | n/a | no: dsr, t_nw, alpha_t, cost_stress, stability |
| calls_d50_t42 | 0.36 | 10.7% | 21.1% | 20.0% | 0.83 | -7.5% (-1.14) | 0.70 | 0.30 | 0.68 | n/a | no: dsr, t_nw, alpha_t, cost_stress |
| stock_mirror_t21 | 0.81 | 20.2% | 19.0% | 19.3% | 1.76 | -1.7% (-0.39) | n/a | 0.86 | 1.76 | n/a | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| calls_d50_t21 | -0.16 | -4.6% | 21.4% | 31.8% | -0.37 |
| calls_d30_t21 | -0.68 | -14.7% | 21.6% | 52.4% | -1.60 |
| calls_d50_t42 | -0.22 | -5.4% | 20.5% | 33.1% | -0.52 |
| stock_mirror_t21 | 0.48 | 9.8% | 23.1% | 30.0% | 1.09 |

## Factor exposures (holdout, HAC)

| trial | mkt | smb | hml | rmw | cma | mom | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| calls_d50_t21 | 0.67 (8.1) | 0.19 (2.5) | 0.31 (3.9) | -0.30 (-3.1) | -0.14 (-1.4) | 0.58 (9.4) | 0.57 |
| calls_d30_t21 | 0.60 (6.0) | 0.22 (2.4) | 0.32 (3.2) | -0.33 (-2.7) | -0.17 (-1.4) | 0.59 (7.5) | 0.45 |
| calls_d50_t42 | 0.66 (7.0) | 0.17 (2.2) | 0.31 (4.3) | -0.38 (-3.9) | -0.12 (-1.3) | 0.61 (9.4) | 0.58 |
| stock_mirror_t21 | 0.81 (12.0) | 0.12 (2.5) | 0.30 (6.7) | -0.28 (-5.5) | -0.09 (-1.2) | 0.54 (14.5) | 0.75 |

## Calendar years (net)

| trial | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calls_d50_t21 | 0.5% | 5.2% | -12.0% | -10.2% | -0.1% | 22.9% | 2.6% | 24.4% |
| calls_d30_t21 | -4.5% | -13.8% | -22.9% | -14.6% | -7.8% | 22.5% | -4.6% | 22.9% |
| calls_d50_t42 | 1.9% | 9.1% | -10.8% | -18.6% | -6.6% | 23.7% | 0.7% | 25.3% |
| stock_mirror_t21 | 6.9% | 24.1% | 10.9% | -2.6% | 8.3% | 21.3% | 10.5% | 36.3% |

## Gate detail

* **calls_d50_t21**: dsr 0.76✗, t_nw 0.99✗, alpha_t -0.89✗, lag_stress 1.04✓, cost_stress 0.77✗, pbo 0.44✓, drawdown 0.21✓, stability 0.62✓
* **calls_d30_t21**: dsr 0.61✗, t_nw 0.51✗, alpha_t -1.14✗, lag_stress 1.06✓, cost_stress 0.35✗, pbo 0.44✓, drawdown 0.24✓, stability 0.25✗
* **calls_d50_t42**: dsr 0.70✗, t_nw 0.83✗, alpha_t -1.14✗, lag_stress 0.81✓, cost_stress 0.68✗, pbo 0.44✓, drawdown 0.20✓, stability 0.62✓

## Provenance

* calls_d50_t21: prereg `739b3af52ff6`, result `44cd9678d5e4`, data `ea498d776e31`, code `f21d98e18ed2`, 2026-10-24T05:01:44.071789+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* calls_d30_t21: prereg `ce5f6d772196`, result `2d69cd3e4334`, data `ea498d776e31`, code `f21d98e18ed2`, 2026-10-24T05:01:44.071789+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* calls_d50_t42: prereg `00c39c66afdd`, result `970f3bda0344`, data `ea498d776e31`, code `f21d98e18ed2`, 2026-10-24T05:01:44.071789+00:00, rerun: DSR methodology: the dispersion of trial Sharpe ratios is now taken over the whole programme, as the trial count already was (it was the family's own trials; v3's three similar trials got a DSR of 0.86 from their small spread). Stored statistics unchanged; no backtest re-run.
* stock_mirror_t21: prereg `570d651c1a8e`, result `f303c28591af`, data `ea498d776e31`, code `5364c1a2f185`, 2026-10-23T21:47:25.693431+00:00, rerun: metric bug: Sharpe/t/PSR/DSR/PBO were computed on raw instead of excess returns (fixed in study.metrics); config unchanged
