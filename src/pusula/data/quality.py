"""Data-quality checks.

Two kinds of rule, kept apart on purpose:

* **Repairs** are pointwise and causal (they look at one bar only): impossible OHLC
  relations and non-positive prices are set to NaN. Nothing else is changed.
* **Flags** may look at neighbouring bars (e.g. a spike that reverses the next day). Using
  them to *edit* history would leak future information into the past, so they only report.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from pusula.core import calendar as cal
from pusula.data.panel import Panel


@dataclass(frozen=True, slots=True)
class QualityReport:
    n_dates: int
    n_symbols: int
    repaired_bars: int
    invalid_vwap: int
    invalid_ohlc: int
    nonpositive: int
    negative_volume: int
    extreme_moves: int
    spike_reversals: int
    stale_runs: int
    interior_gaps: int
    sessions_not_in_calendar: tuple[str, ...]
    calendar_sessions_missing: tuple[str, ...]
    examples: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.n_dates} sessions x {self.n_symbols} securities; "
            f"repaired {self.repaired_bars} bars "
            f"(invalid OHLC {self.invalid_ohlc}, non-positive {self.nonpositive}), "
            f"{self.invalid_vwap} invalid VWAPs; flags: "
            f"extreme moves {self.extreme_moves}, spike reversals {self.spike_reversals}, "
            f"stale runs {self.stale_runs}, interior gaps {self.interior_gaps}; "
            f"calendar: {len(self.sessions_not_in_calendar)} extra, "
            f"{len(self.calendar_sessions_missing)} missing sessions"
        )


def _examples(mask: np.ndarray, panel: Panel, k: int = 5) -> list[str]:
    ii, jj = np.nonzero(mask)
    return [f"{panel.symbols[j]}@{panel.dates[i]}" for i, j in zip(ii[:k], jj[:k], strict=True)]


def check_and_repair(
    panel: Panel, *, extreme: float = 0.6, stale_days: int = 5
) -> tuple[Panel, QualityReport]:
    o, h, lo, c = (panel[f].copy() for f in ("open", "high", "low", "close"))
    v = panel["volume"].copy() if "volume" in panel else np.zeros_like(c)
    with np.errstate(invalid="ignore"):
        nonpos = (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)
        bad_ohlc = (lo > np.fmin(o, c) + 1e-9) | (h < np.fmax(o, c) - 1e-9) | (h < lo)
        neg_vol = v < 0
    repair = (nonpos | bad_ohlc | neg_vol) & np.isfinite(c)
    bar_fields = {"open", "high", "low", "close", "volume", "vwap", "trades"}
    fields = {k: a.copy() for k, a in panel.fields.items()}
    for k, arr in fields.items():
        if k in bar_fields:
            arr[repair] = np.nan
    bad_vwap = np.zeros_like(repair)
    if "vwap" in fields:
        vw = fields["vwap"]
        with np.errstate(invalid="ignore"):
            # VWAP must lie inside the bar's range; zero VWAPs occur in vendor data
            bad_vwap = np.isfinite(vw) & (
                (vw <= 0) | (vw < lo * (1 - 1e-6)) | (vw > h * (1 + 1e-6))
            )
        vw[bad_vwap] = np.nan
    cleaned = Panel(panel.dates, panel.symbols, fields)

    cc = cleaned["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.full_like(cc, np.nan)
        r[1:] = cc[1:] / cc[:-1] - 1.0
        extreme_m = np.abs(r) > extreme
        spike = np.zeros_like(extreme_m)
        spike[1:-1] = (
            (np.abs(r[1:-1]) > 0.25)
            & (np.sign(r[1:-1]) == -np.sign(r[2:]))
            & (np.abs((1 + r[1:-1]) * (1 + r[2:]) - 1) < 0.05)
        )
    vv = cleaned["volume"] if "volume" in cleaned else np.ones_like(cc)
    flat = np.zeros_like(cc, dtype=bool)
    flat[1:] = (cc[1:] == cc[:-1]) & (vv[1:] <= 0)
    stale_runs = 0
    for j in range(cc.shape[1]):
        run = 0
        for i in range(cc.shape[0]):
            run = run + 1 if flat[i, j] else 0
            if run == stale_days:
                stale_runs += 1
    live = np.isfinite(cc)
    gaps = 0
    for j in range(cc.shape[1]):
        idx = np.nonzero(live[:, j])[0]
        if idx.size:
            gaps += int((~live[idx[0] : idx[-1] + 1, j]).sum())

    dset = {d.item() for d in panel.dates}
    if panel.n_dates:
        first, last = panel.dates[0].item(), panel.dates[-1].item()
        expected = {d.item() for d in cal.sessions(first, last)}
    else:
        expected = set()
    extra = tuple(sorted(d.isoformat() for d in dset - expected if isinstance(d, dt.date)))
    missing = tuple(sorted(d.isoformat() for d in expected - dset))
    report = QualityReport(
        n_dates=panel.n_dates,
        n_symbols=panel.n_symbols,
        repaired_bars=int(repair.sum()),
        invalid_vwap=int(bad_vwap.sum()),
        invalid_ohlc=int((bad_ohlc & np.isfinite(c)).sum()),
        nonpositive=int((nonpos & np.isfinite(c)).sum()),
        negative_volume=int(neg_vol.sum()),
        extreme_moves=int(extreme_m.sum()),
        spike_reversals=int(spike.sum()),
        stale_runs=stale_runs,
        interior_gaps=gaps,
        sessions_not_in_calendar=extra,
        calendar_sessions_missing=missing,
        examples={
            "invalid_ohlc": _examples(bad_ohlc & np.isfinite(c), panel),
            "extreme_moves": _examples(extreme_m, panel),
            "spike_reversals": _examples(spike, panel),
        },
    )
    return cleaned, report
