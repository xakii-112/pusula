"""Calibrate the impact coefficient on tick data, and test the cost model against real books. Writes
docs/IMPACT_CALIBRATION.md. Registered cost models are not changed by this: a different Y is a new
family's configuration. Reads no strategy returns.

    uv run python tools/impact_report.py
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from pusula.data.binance import read_recording
from pusula.data.crypto import crypto_data
from pusula.data.pipeline import _default_client, research_panel
from pusula.data.settings import data_dir
from pusula.execution.costs import cost_inputs
from pusula.micro.impact import (
    bucketise,
    fit_square_root,
    model_cost_bps,
    sweep_cost_bps,
    tick_rule,
)
from pusula.research.families import COSTS_CRYPTO, COSTS_V1

ROOT = Path(__file__).resolve().parents[1]
BUCKET_NS = 300_000_000_000
NY = ZoneInfo("America/New_York")
STOCK_DAYS = ("2025-03-12", "2025-06-11", "2025-09-10", "2025-12-10", "2026-03-11", "2026-06-10")
WINDOWS = ((10, 0), (14, 0))
WINDOW_MINUTES = 60
MAX_PAGES = 4
N_STOCKS = 40
SWEEP_SIZES = (10_000.0, 100_000.0, 1_000_000.0)
SEED = 0


def pooled(fits: list[tuple[np.ndarray, np.ndarray, float, float]]) -> tuple[float, int]:
    """One slope through all buckets of all names, each in its own volatility and volume."""
    xs, ys = [], []
    for q, r, sigma, adv in fits:
        xs.append(np.sign(q) * np.sqrt(np.abs(q) / adv))
        ys.append(r / sigma)
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(x @ y) / float(x @ x), int(x.size)


def crypto_section() -> list[str]:
    rd = crypto_data()
    panel = rd.panel
    ci = cost_inputs(panel, COSTS_CRYPTO)
    last = panel.n_dates - 1
    files = sorted((data_dir() / "book").glob("*USDT-*.jsonl.gz"))
    rows, sweep_rows, fits = [], [], []
    hours = 0.0
    for path in files:
        symbol = path.name.split("-")[0]
        t, p, q, s = [], [], [], []
        latest_snapshot = None
        for rec_t, kind, data in read_recording(path):
            if kind == "trade":
                t.append(int(data["T"]) * 1_000_000)
                p.append(float(data["p"]))
                q.append(float(data["q"]))
                s.append(-1.0 if data["m"] else 1.0)
            elif kind == "snapshot":
                latest_snapshot = data
            del rec_t
        if len(t) < 100 or symbol not in panel.symbols:
            continue
        covered = len(np.unique((np.array(t) - t[0]) // BUCKET_NS)) * BUCKET_NS / 3.6e12
        hours = max(hours, covered)
        j = panel.symbol_index(symbol)
        sigma, adv, half = (
            float(ci.sigma[last, j]),
            float(ci.adv_dollars[last, j]),
            float(ci.half_spread[last, j]),
        )
        b = bucketise(t, p, q, s, BUCKET_NS)
        fit = fit_square_root(b.signed_dollars, b.returns, sigma, adv)
        fits.append((b.signed_dollars, b.returns, sigma, adv))
        rows.append(
            f"| {symbol} | {fit.n} | {fit.y:.2f} | {fit.t:.1f} | {fit.r2:.2f} | {sigma:.1%} | {adv / 1e6:,.0f} |"
        )
        if latest_snapshot is not None:
            asks = [(float(a), float(b_)) for a, b_ in latest_snapshot["asks"]]
            for size in SWEEP_SIZES:
                real = sweep_cost_bps(asks, size)
                best = asks[0][0]
                bid = float(latest_snapshot["bids"][0][0])
                quoted_half = (best - bid) / ((best + bid) / 2.0) / 2.0 * 1e4
                model = model_cost_bps(size, half, sigma, adv, COSTS_CRYPTO)
                sweep_rows.append(
                    f"| {symbol} | {size:,.0f} | {quoted_half + real:.2f} | {model:.2f} |"
                )
    y, n = pooled(fits) if fits else (math.nan, 0)
    return [
        "## Crypto (Binance aggregate trades, aggressor from the exchange)",
        "",
        f"Five-minute buckets over about {hours:.1f} hours of recording: too short for the per-symbol "
        "numbers to mean much, and it grows with the recording. "
        f"Pooled Y = **{y:.2f}** over {n} buckets; `us-crypto-v1` registered {COSTS_CRYPTO.impact_y}.",
        "",
        "| pair | buckets | Y | NW t | R2 | daily vol | ADV, $M |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "### An immediate order against the visible book, last snapshot",
        "",
        "What it would pay against the mid, in bp, before fees, and what the cost model charges for "
        "the same size. The model describes a day's patient execution, so a sweep above it is expected "
        "at sizes that matter; a sweep far below it means the model is conservative.",
        "",
        "| pair | size, $ | sweep | model |",
        "|---|---:|---:|---:|",
        *sweep_rows,
        "",
    ]


def stock_section() -> list[str]:
    rd = research_panel()
    panel = rd.panel
    ci = cost_inputs(panel, COSTS_V1)
    rng = np.random.default_rng(SEED)
    last_members = np.nonzero(panel["in_universe"][-1] > 0)[0]
    picks = rng.choice(last_members, size=N_STOCKS, replace=False)
    fits = []
    per_stock = []
    n_trades = 0
    with _default_client() as client:
        for j in picks:
            sid = panel.symbols[int(j)]
            ticker = sid.split("@")[0]
            all_t: list[np.ndarray] = []
            for day in STOCK_DAYS:
                d = dt.date.fromisoformat(day)
                row = panel.date_index(day) - 1
                sigma, adv = float(ci.sigma[row, j]), float(ci.adv_dollars[row, j])
                if not (math.isfinite(sigma) and math.isfinite(adv) and sigma > 0 and adv > 0):
                    continue
                for hh, mm in WINDOWS:
                    start = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY)
                    tr = client.trades(
                        ticker,
                        start,
                        start + dt.timedelta(minutes=WINDOW_MINUTES),
                        max_pages=MAX_PAGES,
                    )
                    if len(tr) < 50:
                        continue
                    t = tr.index.asi8 if hasattr(tr.index, "asi8") else tr.index.to_numpy()
                    p, q = tr["price"].to_numpy(), tr["size"].to_numpy()
                    b = bucketise(t, p, q, tick_rule(p), BUCKET_NS)
                    n_trades += len(tr)
                    if b.returns.size >= 3:
                        fits.append((b.signed_dollars, b.returns, sigma, adv))
                        all_t.append(b.signed_dollars)
                        per_stock.append(fit_square_root(b.signed_dollars, b.returns, sigma, adv).y)
    y, n = pooled(fits)
    finite = [v for v in per_stock if math.isfinite(v)]
    return [
        "## Stocks (Alpaca consolidated trades, tick rule)",
        "",
        f"{N_STOCKS} current index members drawn at random, {len(STOCK_DAYS)} sessions, two "
        f"{WINDOW_MINUTES}-minute windows each ({n_trades:,} trades, at most {MAX_PAGES} pages per "
        "window, so the most liquid names are cut short). Pooled "
        f"Y = **{y:.2f}** over {n} five-minute buckets; median across windows "
        f"{np.median(finite):.2f}; the registered cost model uses {COSTS_V1.impact_y} and the gate "
        f"doubles it to {2 * COSTS_V1.impact_y}.",
        "",
    ]


def main() -> None:
    lines = [
        "# Impact calibration on tick data",
        "",
        "Generated by `tools/impact_report.py`. Y is the coefficient of "
        "`sigma_daily x sign(Q) x sqrt(|Q| / ADV)` for the return of a five-minute bucket on its "
        "signed flow. It measures aggregate order flow, not one desk's order, and tick-rule signs "
        "are noisy, which biases Y towards zero: read it as an order of magnitude. Registered cost "
        "models do not change because of this; a different Y is a new family's configuration.",
        "",
        *stock_section(),
        *crypto_section(),
    ]
    out = ROOT / "docs" / "IMPACT_CALIBRATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
