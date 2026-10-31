"""The spread model for crypto, fitted on live Binance books: one log-linear regression of the quoted
half-spread on volatility and dollar volume, across pairs, as `CALIBRATION.md` §1 does for stocks.
Price is left out: a coin's unit is arbitrary. One snapshot of today's market: earlier years were
probably wider, which is one more reason the gate doubles costs. Reads no returns beyond the volatility
input, so it can run before a family is registered.

    uv run python tools/crypto_spreads.py
"""

from __future__ import annotations

import math

import numpy as np

from pusula.data.binance import BinanceClient
from pusula.data.crypto import crypto_data
from pusula.execution.costs import CostModel, cost_inputs


def main() -> None:
    panel = crypto_data().panel
    ci = cost_inputs(panel, CostModel())
    last = panel.n_dates - 1
    members = np.nonzero(panel["in_universe"][last] > 0)[0]
    rows = []
    with BinanceClient() as client:
        for j in members:
            book = client.depth(panel.symbols[j], 5)
            if not book["bids"] or not book["asks"]:
                continue
            bid, ask = float(book["bids"][0][0]), float(book["asks"][0][0])
            mid = (bid + ask) / 2.0
            sigma, adv = float(ci.sigma[last, j]), float(ci.adv_dollars[last, j])
            if mid > 0 and math.isfinite(sigma) and math.isfinite(adv) and sigma > 0 and adv > 0:
                rows.append((panel.symbols[j], (ask - bid) / mid / 2.0, sigma, adv))
    y = np.log([r[1] for r in rows])
    X = np.column_stack(
        [np.ones(len(rows)), np.log([r[2] for r in rows]), np.log([r[3] for r in rows])]
    )
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = np.exp(X @ coef)
    q = np.array([r[1] for r in rows])
    print(f"{len(rows)} pairs at {panel.dates[last]}")
    print(f"intercept {coef[0]:.4f}, sigma {coef[1]:.4f}, ADV {coef[2]:.4f}")
    print(
        f"median quoted half-spread {np.median(q) * 1e4:.2f} bp, p90 {np.quantile(q, 0.9) * 1e4:.2f} bp"
    )
    print(
        f"R2 of the logs {1 - np.var(y - X @ coef) / np.var(y):.2f}, median model/quote {np.median(fitted / q):.2f}"
    )
    for s, h, _, adv in sorted(rows, key=lambda r: -r[3])[:6]:
        print(f"  {s:<12s} {h * 1e4:6.2f} bp  ADV {adv:,.0f}")


if __name__ == "__main__":
    main()
