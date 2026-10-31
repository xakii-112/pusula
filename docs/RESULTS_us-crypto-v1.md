# Results: `us-crypto-v1`

> Generated from `ledger/experiments.jsonl` by `pusula study report`. Do not edit.
> Registered trials in this family: **3**; deflated-Sharpe trial count (whole programme `us-`): **46**. Sharpe, t and DSR use excess returns.

Development window 2020-01-02 .. 2022-12-30; holdout 2023-01-03 .. 2026-10-22. 

## Holdout summary

| trial | Sharpe | ann. ret | vol | max DD | NW t | FF5+MOM alpha (t) | DSR | lag+1 Sharpe | 2x cost t | turnover/yr | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| crypto_trend_long | 0.29 | 6.9% | 7.9% | 10.9% | 0.61 | 2.9% (1.01) | 0.01 | 0.26 | 0.50 | 1.6 | no: dsr, t_nw, alpha_t, cost_stress |
| crypto_trend_ls | -0.41 | -4.2% | 17.9% | 22.3% | -0.76 | -4.2% (-0.48) | 0.00 | -0.47 | -1.05 | 6.2 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, stability |
| crypto_momentum_ls | -1.31 | -25.9% | 24.1% | 68.2% | -2.73 | -40.8% (-3.38) | 0.00 | -1.33 | -3.84 | 21.8 | no: dsr, t_nw, alpha_t, lag_stress, cost_stress, drawdown, stability |
| equal_weight | -0.18 | -29.6% | 72.6% | 92.5% | -0.37 | -5.6% (-2.47) | n/a | n/a | n/a | 5.2 | benchmark |
| btc_hold | 1.02 | 52.1% | 47.4% | 53.0% | 2.00 | -0.0% (-3.56) | n/a | n/a | n/a | 0.2 | benchmark |

## Development window

| trial | Sharpe | ann. ret | vol | max DD | NW t |
|---|---:|---:|---:|---:|---:|
| crypto_trend_long | 1.39 | 23.9% | 15.9% | 13.9% | 2.21 |
| crypto_trend_ls | 0.78 | 17.5% | 23.2% | 28.7% | 1.31 |
| crypto_momentum_ls | -0.55 | -13.3% | 22.6% | 50.2% | -0.98 |
| equal_weight | 0.83 | 37.3% | 94.3% | 89.4% | 1.44 |
| btc_hold | 0.75 | 32.0% | 73.2% | 76.6% | 1.33 |

## Factor exposures (holdout, HAC)

| trial | btc | cmkt | mkt | R2 |
|---|---:|---:|---:|---:|
| crypto_trend_long | 0.01 (0.9) | 0.07 (8.1) | -0.03 (-2.2) | 0.49 |
| crypto_trend_ls | -0.03 (-1.0) | -0.10 (-4.7) | -0.10 (-2.3) | 0.27 |
| crypto_momentum_ls | 0.20 (4.2) | -0.17 (-5.2) | -0.05 (-0.9) | 0.11 |
| equal_weight | 0.01 (1.3) | 0.99 (166.7) | -0.00 (-0.2) | 1.00 |
| btc_hold | 1.00 (4819447734803888.0) | -0.00 (-2.4) | 0.00 (5.6) | 1.00 |

## Calendar years (net)

| trial | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_trend_long | 17.2% | 74.1% | -6.9% | 13.8% | 6.7% | 3.5% | 2.4% |
| crypto_trend_ls | -8.4% | 57.6% | 12.3% | -15.4% | -2.1% | 15.1% | -10.9% |
| crypto_momentum_ls | 3.4% | 9.5% | -42.4% | -9.8% | -19.0% | -40.5% | -25.9% |
| equal_weight | 134.3% | 639.6% | -85.1% | 72.1% | -4.8% | -76.2% | -32.2% |
| btc_hold | 300.9% | 59.8% | -64.1% | 153.3% | 122.4% | -6.3% | -7.1% |

## Gate detail

* **crypto_trend_long**: dsr 0.01✗, t_nw 0.61✗, alpha_t 1.01✗, lag_stress 0.90✓, cost_stress 0.50✗, pbo 0.14✓, drawdown 0.11✓, stability 0.86✓
* **crypto_trend_ls**: dsr 0.00✗, t_nw -0.76✗, alpha_t -0.48✗, lag_stress n/a✗, cost_stress -1.05✗, pbo 0.14✓, drawdown 0.22✓, stability 0.43✗
* **crypto_momentum_ls**: dsr 0.00✗, t_nw -2.73✗, alpha_t -3.38✗, lag_stress n/a✗, cost_stress -3.84✗, pbo 0.14✓, drawdown 0.68✗, stability 0.29✗

## Provenance

* crypto_trend_long: prereg `e9fd805af6ab`, result `3ca90cab9549`, data `c06de6d8643c`, code `62dad3d4cac5`, 2026-10-30T14:59:51.373788+00:00, rerun: Data defect found in the first run, before any use of its result: Binance reuses symbols. LUNAUSDT relisted at 177,400x and token swaps (COCOS, DREP, QUICK, SUN, BNX) appeared as returns of up to +17,575% a year. A gap of 3+ days now ends a security. Configuration unchanged.
* crypto_trend_ls: prereg `a60a44231fee`, result `b74ba8c8a2fb`, data `c06de6d8643c`, code `62dad3d4cac5`, 2026-10-30T14:59:51.373788+00:00, rerun: Data defect found in the first run, before any use of its result: Binance reuses symbols. LUNAUSDT relisted at 177,400x and token swaps (COCOS, DREP, QUICK, SUN, BNX) appeared as returns of up to +17,575% a year. A gap of 3+ days now ends a security. Configuration unchanged.
* crypto_momentum_ls: prereg `562409882492`, result `978feb224877`, data `c06de6d8643c`, code `62dad3d4cac5`, 2026-10-30T14:59:51.373788+00:00, rerun: Data defect found in the first run, before any use of its result: Binance reuses symbols. LUNAUSDT relisted at 177,400x and token swaps (COCOS, DREP, QUICK, SUN, BNX) appeared as returns of up to +17,575% a year. A gap of 3+ days now ends a security. Configuration unchanged.
* equal_weight: prereg `2ca64b0d07d7`, result `53d0a6945848`, data `c06de6d8643c`, code `62dad3d4cac5`, 2026-10-30T14:59:51.373788+00:00, rerun: Data defect found in the first run, before any use of its result: Binance reuses symbols. LUNAUSDT relisted at 177,400x and token swaps (COCOS, DREP, QUICK, SUN, BNX) appeared as returns of up to +17,575% a year. A gap of 3+ days now ends a security. Configuration unchanged.
* btc_hold: prereg `d64cb30f2fca`, result `09c8447d4f97`, data `c06de6d8643c`, code `62dad3d4cac5`, 2026-10-30T14:59:51.373788+00:00, rerun: Data defect found in the first run, before any use of its result: Binance reuses symbols. LUNAUSDT relisted at 177,400x and token swaps (COCOS, DREP, QUICK, SUN, BNX) appeared as returns of up to +17,575% a year. A gap of 3+ days now ends a security. Configuration unchanged.
