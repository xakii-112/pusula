"""Post-hoc comparison for us-crossasset-v1 (NOT part of any gate): trend against the passive books,
with an interval, in each window and in the one year it was meant for. Writes
docs/CROSSASSET_COMPARISON.md.

    uv run python tools/crossasset_report.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from pusula.data.pipeline import CROSS_ASSET, etf_data
from pusula.engine.backtest import Backtester, BacktestResult
from pusula.live.forward import rf_series
from pusula.research.families import us_crossasset_v1
from pusula.research.study import Trial, make_strategy, window_excess, window_returns
from pusula.stats import sharpe as S
from pusula.stats.bootstrap import sharpe_difference

ROOT = Path(__file__).resolve().parents[1]
SPEC = us_crossasset_v1()
WINDOWS = {
    "development": SPEC.dev,
    "holdout": SPEC.holdout,
    "whole": (SPEC.dev[0], SPEC.holdout[1]),
    "2022 alone": ("2022-01-03", "2022-12-30"),
}
PAIRS = (
    ("ca_trend_long", "risk_parity_long"),
    ("ca_trend_long", "equal_weight"),
    ("ca_trend_ls", "risk_parity_long"),
    ("ca_trend_ls", "equal_weight"),
    ("ca_trend_ls", "ca_trend_equal_risk"),
)


def _run(t: Trial) -> tuple[str, BacktestResult]:
    panel = etf_data(CROSS_ASSET).panel
    cfg = replace(SPEC.backtest, kill_switch=False)
    res = Backtester(panel, cfg, rf_series(panel)).run(make_strategy(t, cfg.cost_model))
    res.weights = res.weights[:, :0]
    return t.name, res


def main() -> None:
    with ProcessPoolExecutor(5) as ex:
        runs = dict(ex.map(_run, SPEC.trials))
    lines = [
        "# us-crossasset-v1: post-hoc comparison (not a gate input)",
        "",
        "Net of costs, excess returns. Interval: 95%, paired stationary bootstrap (2,000 "
        "resamples, mean block 10 sessions).",
    ]
    for title, (lo, hi) in WINDOWS.items():
        lines += [
            "",
            f"## {title}: {lo} to {hi}",
            "",
            "| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        x = {n: window_excess(r, lo, hi) for n, r in runs.items()}
        for n, r in runs.items():
            mask = (r.dates >= np.datetime64(lo)) & (r.dates <= np.datetime64(hi))
            raw = window_returns(r, lo, hi)
            lines.append(
                f"| {n} | {S.annualised_sharpe(x[n]):.2f} | {S.annualised_return(raw):.1%} "
                f"| {S.annualised_vol(raw):.1%} | {S.max_drawdown(raw):.1%} "
                f"| {r.gross_exposure[mask].mean():.2f} | {r.turnover[mask].mean() * 252:.1f} |"
            )
        lines += [
            "",
            "| a minus b | Sharpe difference | 95% interval | share of resamples a beats b |",
            "|---|---:|---|---:|",
        ]
        for a, b in PAIRS:
            d = sharpe_difference(x[a], x[b])
            lines.append(
                f"| {a} - {b} | {d.diff:+.2f} | [{d.lo:+.2f}, {d.hi:+.2f}] | {d.p_better:.0%} |"
            )
    out = ROOT / "docs" / "CROSSASSET_COMPARISON.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
