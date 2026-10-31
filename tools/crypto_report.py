"""Post-hoc comparison for us-crypto-v1 (NOT part of any gate): the books against holding Bitcoin and
the equal-weight coins, with intervals, by window. Writes docs/CRYPTO_COMPARISON.md.

    uv run python tools/crypto_report.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from pusula.data.crypto import crypto_data
from pusula.engine.backtest import Backtester, BacktestResult
from pusula.live.forward import rf_series
from pusula.research.families import us_crypto_v1
from pusula.research.study import Trial, make_strategy, window_excess, window_returns
from pusula.stats import sharpe as S
from pusula.stats.bootstrap import sharpe_difference

ROOT = Path(__file__).resolve().parents[1]
SPEC = us_crypto_v1()
WINDOWS = {
    "development": SPEC.dev,
    "holdout": SPEC.holdout,
    "whole": (SPEC.dev[0], SPEC.holdout[1]),
    "2022 alone": ("2022-01-03", "2022-12-30"),
}
PAIRS = (
    ("crypto_trend_long", "btc_hold"),
    ("crypto_trend_long", "equal_weight"),
    ("crypto_trend_ls", "btc_hold"),
    ("crypto_momentum_ls", "equal_weight"),
)


def _run(t: Trial) -> tuple[str, BacktestResult]:
    panel = crypto_data().panel
    cfg = replace(SPEC.backtest, kill_switch=False)
    res = Backtester(panel, cfg, rf_series(panel)).run(make_strategy(t, cfg.cost_model))
    res.weights = res.weights[:, :0]
    return t.name, res


def main() -> None:
    with ProcessPoolExecutor(5) as ex:
        runs = dict(ex.map(_run, SPEC.trials))
    lines = [
        "# us-crypto-v1: post-hoc comparison (not a gate input)",
        "",
        "Net of costs, excess returns. Interval: 95%, paired stationary bootstrap (2,000 resamples, "
        "mean block 10 sessions).",
    ]
    for title, (lo, hi) in WINDOWS.items():
        lines += [
            "",
            f"## {title}: {lo} to {hi}",
            "",
            "| book | Sharpe | ann. return | ann. vol | max DD | mean gross | turnover/yr | cost/yr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        x = {n: window_excess(r, lo, hi) for n, r in runs.items()}
        for n, r in runs.items():
            mask = (r.dates >= np.datetime64(lo)) & (r.dates <= np.datetime64(hi))
            raw = window_returns(r, lo, hi)
            cost = float(np.sum(r.costs[mask]) / np.mean(r.equity[mask]) / (mask.sum() / 252))
            lines.append(
                f"| {n} | {S.annualised_sharpe(x[n]):.2f} | {S.annualised_return(raw):.1%} "
                f"| {S.annualised_vol(raw):.1%} | {S.max_drawdown(raw):.1%} "
                f"| {r.gross_exposure[mask].mean():.2f} | {r.turnover[mask].mean() * 252:.1f} | {cost:.1%} |"
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
    out = ROOT / "docs" / "CRYPTO_COMPARISON.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
