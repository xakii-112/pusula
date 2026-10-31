# us-crypto-v1: post-hoc comparison (not a gate input)

Net of costs, excess returns. Interval: 95%, paired stationary bootstrap (2,000 resamples, mean block 10 sessions).

## development: 2020-01-02 to 2022-12-30

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr | cost/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_trend_long | 1.39 | 23.9% | 15.9% | 13.9% | 0.12 | 1.8 | 0.6% |
| crypto_trend_ls | 0.78 | 17.5% | 23.2% | 28.7% | 0.26 | 5.5 | 2.3% |
| crypto_momentum_ls | -0.55 | -13.3% | 22.6% | 50.2% | 1.11 | 17.4 | 10.1% |
| equal_weight | 0.83 | 37.3% | 94.3% | 89.4% | 0.98 | 4.6 | 7.9% |
| btc_hold | 0.75 | 32.0% | 73.2% | 76.6% | 1.00 | 0.4 | 0.1% |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| crypto_trend_long - btc_hold | +0.64 | [-0.52, +1.76] | 85% |
| crypto_trend_long - equal_weight | +0.56 | [-0.28, +1.40] | 90% |
| crypto_trend_ls - btc_hold | +0.03 | [-1.76, +1.63] | 52% |
| crypto_momentum_ls - equal_weight | -1.38 | [-3.06, +0.21] | 4% |

## holdout: 2023-01-03 to 2026-10-22

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr | cost/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_trend_long | 0.29 | 6.9% | 7.9% | 10.9% | 0.11 | 1.4 | 0.4% |
| crypto_trend_ls | -0.41 | -4.2% | 17.9% | 22.3% | 0.33 | 6.8 | 3.2% |
| crypto_momentum_ls | -1.31 | -25.9% | 24.1% | 68.2% | 1.69 | 25.3 | 13.3% |
| equal_weight | -0.18 | -29.6% | 72.6% | 92.5% | 1.00 | 5.6 | 6.2% |
| btc_hold | 1.02 | 52.1% | 47.4% | 53.0% | 1.00 | 0.0 | 0.0% |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| crypto_trend_long - btc_hold | -0.73 | [-1.63, +0.19] | 7% |
| crypto_trend_long - equal_weight | +0.47 | [-0.23, +1.25] | 91% |
| crypto_trend_ls - btc_hold | -1.43 | [-3.30, +0.44] | 7% |
| crypto_momentum_ls - equal_weight | -1.13 | [-2.55, +0.25] | 5% |

## whole: 2020-01-02 to 2026-10-22

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr | cost/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_trend_long | 0.91 | 14.1% | 12.1% | 13.9% | 0.12 | 1.6 | 0.5% |
| crypto_trend_ls | 0.19 | 4.8% | 20.4% | 28.7% | 0.30 | 6.2 | 2.8% |
| crypto_momentum_ls | -0.99 | -20.6% | 23.4% | 83.3% | 1.43 | 21.8 | 11.3% |
| equal_weight | 0.33 | -5.4% | 82.9% | 97.8% | 0.99 | 5.2 | 7.4% |
| btc_hold | 0.85 | 42.9% | 60.2% | 76.6% | 1.00 | 0.2 | 0.0% |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| crypto_trend_long - btc_hold | +0.06 | [-0.70, +0.82] | 54% |
| crypto_trend_long - equal_weight | +0.59 | [-0.02, +1.17] | 97% |
| crypto_trend_ls - btc_hold | -0.66 | [-1.94, +0.51] | 13% |
| crypto_momentum_ls - equal_weight | -1.31 | [-2.36, -0.33] | 0% |

## 2022 alone: 2022-01-03 to 2022-12-30

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr | cost/yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| crypto_trend_long | -2.42 | -6.9% | 3.5% | 8.0% | 0.04 | 0.7 | 0.2% |
| crypto_trend_ls | 0.50 | 12.4% | 29.1% | 18.7% | 0.37 | 7.8 | 2.9% |
| crypto_momentum_ls | -2.74 | -42.6% | 20.0% | 45.3% | 1.53 | 24.0 | 14.5% |
| equal_weight | -1.63 | -85.2% | 90.9% | 85.7% | 1.01 | 4.7 | 6.9% |
| btc_hold | -1.30 | -64.2% | 63.7% | 66.7% | 1.00 | 0.0 | 0.0% |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| crypto_trend_long - btc_hold | -1.12 | [-2.51, +0.28] | 6% |
| crypto_trend_long - equal_weight | -0.79 | [-1.90, +0.28] | 8% |
| crypto_trend_ls - btc_hold | +1.80 | [-1.64, +4.74] | 85% |
| crypto_momentum_ls - equal_weight | -1.11 | [-3.87, +1.38] | 17% |
