# us-crossasset-v1: post-hoc comparison (not a gate input)

Net of costs, excess returns. Interval: 95%, paired stationary bootstrap (2,000 resamples, mean block 10 sessions).

## development: 2019-01-02 to 2022-12-30

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | 0.74 | 7.7% | 9.0% | 10.6% | 1.40 | 17.5 |
| ca_trend_long | 0.95 | 6.1% | 5.3% | 7.8% | 0.81 | 7.3 |
| ca_trend_equal_risk | 0.75 | 7.7% | 9.0% | 9.4% | 1.33 | 15.5 |
| equal_weight | 0.57 | 6.8% | 10.7% | 19.6% | 1.00 | 0.6 |
| risk_parity_long | 0.32 | 2.8% | 5.8% | 11.5% | 0.96 | 3.3 |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| ca_trend_long - risk_parity_long | +0.63 | [-0.24, +1.38] | 92% |
| ca_trend_long - equal_weight | +0.38 | [-0.51, +1.25] | 77% |
| ca_trend_ls - risk_parity_long | +0.42 | [-1.15, +1.87] | 67% |
| ca_trend_ls - equal_weight | +0.18 | [-1.37, +1.66] | 56% |
| ca_trend_ls - ca_trend_equal_risk | -0.00 | [-0.26, +0.26] | 51% |

## holdout: 2023-01-03 to 2026-10-22

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | -0.05 | 4.2% | 6.5% | 8.2% | 1.26 | 18.4 |
| ca_trend_long | 0.25 | 6.0% | 5.3% | 4.7% | 0.87 | 9.0 |
| ca_trend_equal_risk | -0.32 | 2.7% | 5.8% | 6.4% | 1.16 | 16.0 |
| equal_weight | 0.84 | 11.3% | 7.6% | 8.3% | 1.00 | 0.3 |
| risk_parity_long | 0.56 | 7.5% | 4.7% | 4.7% | 0.98 | 2.8 |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| ca_trend_long - risk_parity_long | -0.31 | [-0.86, +0.19] | 11% |
| ca_trend_long - equal_weight | -0.59 | [-1.24, +0.02] | 3% |
| ca_trend_ls - risk_parity_long | -0.61 | [-1.52, +0.21] | 7% |
| ca_trend_ls - equal_weight | -0.89 | [-1.88, +0.03] | 3% |
| ca_trend_ls - ca_trend_equal_risk | +0.27 | [-0.18, +0.72] | 89% |

## whole: 2019-01-02 to 2026-10-22

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | 0.42 | 6.0% | 7.9% | 16.3% | 1.33 | 18.0 |
| ca_trend_long | 0.61 | 6.1% | 5.3% | 7.8% | 0.84 | 8.1 |
| ca_trend_equal_risk | 0.33 | 5.2% | 7.6% | 13.3% | 1.25 | 15.8 |
| equal_weight | 0.67 | 9.0% | 9.3% | 19.6% | 1.00 | 0.5 |
| risk_parity_long | 0.43 | 5.1% | 5.3% | 11.5% | 0.97 | 3.1 |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| ca_trend_long - risk_parity_long | +0.18 | [-0.37, +0.70] | 72% |
| ca_trend_long - equal_weight | -0.06 | [-0.69, +0.53] | 38% |
| ca_trend_ls - risk_parity_long | -0.01 | [-1.03, +0.97] | 46% |
| ca_trend_ls - equal_weight | -0.25 | [-1.33, +0.78] | 29% |
| ca_trend_ls - ca_trend_equal_risk | +0.08 | [-0.14, +0.32] | 74% |

## 2022 alone: 2022-01-03 to 2022-12-30

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| ca_trend_ls | 0.77 | 10.7% | 12.7% | 10.6% | 1.29 | 14.5 |
| ca_trend_long | 0.37 | 3.0% | 4.9% | 3.7% | 0.50 | 3.7 |
| ca_trend_equal_risk | 0.74 | 10.2% | 12.4% | 9.4% | 1.25 | 13.8 |
| equal_weight | -0.77 | -8.3% | 12.1% | 15.7% | 1.00 | 0.4 |
| risk_parity_long | -1.13 | -6.1% | 6.5% | 10.3% | 0.96 | 3.0 |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| ca_trend_long - risk_parity_long | +1.50 | [-1.43, +3.86] | 87% |
| ca_trend_long - equal_weight | +1.14 | [-1.75, +3.53] | 79% |
| ca_trend_ls - risk_parity_long | +1.90 | [-2.10, +5.79] | 82% |
| ca_trend_ls - equal_weight | +1.53 | [-2.40, +5.39] | 78% |
| ca_trend_ls - ca_trend_equal_risk | +0.02 | [-0.38, +0.60] | 59% |
