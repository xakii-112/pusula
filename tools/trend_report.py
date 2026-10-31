"""Post-hoc comparison for us-trend-v1 (NOT part of any gate): does the portfolio engine beat
trend_heritage and the equal-weight benchmark, with a confidence interval the ledger's point
estimates do not have? Paired stationary bootstrap of the holdout Sharpe difference, net of costs,
plus what each book actually ran at (volatility, exposure, turnover).

    uv run python tools/trend_report.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from pusula.data.pipeline import research_panel
from pusula.engine.backtest import Backtester, BacktestResult
from pusula.live.forward import rf_series
from pusula.research.families import us_equity_v2, us_trend_v1
from pusula.research.study import Trial, make_strategy, window_excess, window_returns
from pusula.stats import sharpe as S
from pusula.stats.bootstrap import sharpe_difference

ROOT = Path(__file__).resolve().parents[1]
LO, HI = us_trend_v1().holdout


def _run(t: Trial) -> tuple[str, BacktestResult]:
    panel = research_panel().panel
    cfg = replace(us_trend_v1().backtest, kill_switch=False)
    res = Backtester(panel, cfg, rf_series(panel)).run(make_strategy(t, cfg.cost_model))
    res.weights = res.weights[:, :0]
    return t.name, res


def main() -> None:
    heritage = next(t for t in us_equity_v2().trials if t.name == "trend_heritage")
    trials = (*us_trend_v1().trials, heritage)
    with ProcessPoolExecutor(5) as ex:
        runs = dict(ex.map(_run, trials))
    x = {n: window_excess(r, LO, HI) for n, r in runs.items()}
    lines = [
        "# us-trend-v1: post-hoc comparison (not a gate input)",
        "",
        f"Holdout {LO} to {HI}, net of costs, excess returns. Interval: 95%, paired stationary "
        "bootstrap (2,000 resamples, mean block 10 sessions).",
        "",
        "| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in runs.items():
        mask = (r.dates >= np.datetime64(LO)) & (r.dates <= np.datetime64(HI))
        raw = window_returns(r, LO, HI)
        lines.append(
            f"| {n} | {S.annualised_sharpe(x[n]):.2f} | {S.annualised_return(raw):.1%} "
            f"| {S.annualised_vol(raw):.1%} | {S.max_drawdown(raw):.1%} "
            f"| {r.gross_exposure[mask].mean():.2f} | {r.turnover[mask].mean() * 252:.2f} |"
        )
    pairs = [
        ("forecast_hrp", "equal_weight"),
        ("forecast_hrp", "trend_heritage"),
        ("signonly_hrp", "trend_heritage"),
        ("signonly_hrp", "equal_weight"),
        ("forecast_hrp", "forecast_equal_risk"),
        ("forecast_hrp", "signonly_hrp"),
    ]
    lines += [
        "",
        "| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |",
        "|---|---:|---|---:|",
    ]
    for a, b in pairs:
        d = sharpe_difference(x[a], x[b])
        lines.append(
            f"| {a} - {b} | {d.diff:+.2f} | [{d.lo:+.2f}, {d.hi:+.2f}] | {d.p_better:.0%} |"
        )
    out = ROOT / "docs" / "TREND_COMPARISON.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
