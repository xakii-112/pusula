# us-trend-v1: post-hoc comparison (not a gate input)

Holdout 2023-01-03 to 2026-10-22, net of costs, excess returns. Interval: 95%, paired stationary bootstrap (2,000 resamples, mean block 10 sessions).

| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |
|---|---:|---:|---:|---:|---:|---:|
| forecast_hrp | 0.31 | 7.1% | 8.0% | 10.1% | 0.75 | 8.13 |
| signonly_hrp | 0.30 | 7.0% | 8.0% | 10.5% | 0.75 | 25.59 |
| forecast_equal_risk | 0.32 | 7.2% | 8.1% | 10.1% | 0.73 | 6.70 |
| equal_weight | 0.71 | 14.6% | 14.0% | 17.5% | 1.00 | 0.80 |
| trend_heritage | 0.51 | 10.2% | 11.0% | 14.0% | 0.96 | 22.35 |

| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |
|---|---:|---|---:|
| forecast_hrp - equal_weight | -0.40 | [-0.97, +0.15] | 8% |
| forecast_hrp - trend_heritage | -0.20 | [-0.50, +0.12] | 11% |
| signonly_hrp - trend_heritage | -0.20 | [-0.47, +0.07] | 8% |
| signonly_hrp - equal_weight | -0.41 | [-0.90, +0.10] | 4% |
| forecast_hrp - forecast_equal_risk | -0.01 | [-0.12, +0.10] | 41% |
| forecast_hrp - signonly_hrp | +0.00 | [-0.20, +0.21] | 52% |
