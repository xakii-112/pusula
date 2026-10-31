"""Does the spread model, calibrated on index members, hold for the cross-asset ETFs? Compares it with
consolidated quotes on eight mid-year sessions (CALIBRATION.md §1). Reads no returns, so it can run
before a family is registered.

    uv run python tools/etf_spreads.py
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from pusula.data.pipeline import CROSS_ASSET, _default_client, etf_data
from pusula.execution.calibration import measured_half_spread
from pusula.execution.costs import CostModel, cost_inputs

DAYS = (
    "2019-06-14",
    "2020-06-15",
    "2021-06-15",
    "2022-06-15",
    "2023-06-15",
    "2024-06-14",
    "2025-06-16",
    "2026-06-15",
)


def main() -> None:
    panel = etf_data(CROSS_ASSET).panel
    ci = cost_inputs(panel, CostModel())
    rows: list[tuple[str, float, float]] = []
    with _default_client() as client:
        for d in DAYS:
            t = panel.date_index(d)
            for j, ticker in enumerate(panel.symbols):
                quoted = measured_half_spread(client, ticker, dt.date.fromisoformat(d))
                if math.isfinite(quoted) and quoted > 0:
                    rows.append((ticker, quoted * 1e4, float(ci.half_spread[t - 1, j]) * 1e4))
    quoted, model = (np.array([r[i] for r in rows]) for i in (1, 2))
    print(
        f"{len(rows)} samples, median quoted {np.median(quoted):.2f} bp, model {np.median(model):.2f} bp"
    )
    print(
        f"median model/quoted {np.median(model / quoted):.2f}, model >= quoted {np.mean(model >= quoted):.0%}"
    )
    print("| ETF | quoted bp | model bp |\n|---|---:|---:|")
    for ticker in panel.symbols:
        q = [r[1] for r in rows if r[0] == ticker]
        m = [r[2] for r in rows if r[0] == ticker]
        print(f"| {ticker} | {np.median(q):.2f} | {np.median(m):.2f} |")


if __name__ == "__main__":
    main()
