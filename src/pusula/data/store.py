"""On-disk bar store: one Parquet file per security, plus a manifest.

The manifest records, per security, the request that produced the file (ticker, asof,
adjustment, feed, date range), when it was fetched and the SHA-256 of the file bytes. A study
records the manifest digest, so a result can be tied to the exact data it saw.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pusula.core.errors import DataError
from pusula.core.hashing import content_hash
from pusula.data.alpaca import BAR_COLUMNS
from pusula.data.panel import Panel
from pusula.data.settings import atomic_write_text


@dataclass(frozen=True, slots=True)
class StoredSeries:
    security_id: str
    ticker: str
    asof: str | None
    adjustment: str
    feed: str
    start: str
    end: str
    rows: int
    fetched_at: str
    sha256: str


class BarStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bars_dir = root / "bars_1d"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "manifest.json"
        self._manifest: dict[str, StoredSeries] = {}
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifest = {k: StoredSeries(**v) for k, v in raw.items()}

    # ------------------------------------------------------------------ write
    def _file(self, security_id: str) -> Path:
        safe = security_id.replace("@", "__").replace(".", "_")
        return self.bars_dir / f"{safe}.parquet"

    def put(
        self,
        security_id: str,
        ticker: str,
        asof: dt.date | None,
        frame: pd.DataFrame,
        *,
        adjustment: str,
        feed: str,
        fetched_at: dt.datetime,
    ) -> StoredSeries:
        missing = [c for c in BAR_COLUMNS if c not in frame.columns]
        if missing:
            raise DataError(f"{security_id}: missing columns {missing}")
        f = frame.loc[:, list(BAR_COLUMNS)].astype("float64").sort_index()
        f.index = pd.DatetimeIndex(f.index, name="date")
        p = self._file(security_id)
        f.to_parquet(p, engine="pyarrow", index=True)
        rec = StoredSeries(
            security_id=security_id,
            ticker=ticker,
            asof=asof.isoformat() if asof else None,
            adjustment=adjustment,
            feed=feed,
            start=f.index.min().date().isoformat() if len(f) else "",
            end=f.index.max().date().isoformat() if len(f) else "",
            rows=len(f),
            fetched_at=fetched_at.isoformat(),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        self._manifest[security_id] = rec
        return rec

    def flush(self) -> None:
        payload = {k: asdict(v) for k, v in sorted(self._manifest.items())}
        atomic_write_text(self.manifest_path, json.dumps(payload, indent=1))

    # ------------------------------------------------------------------ read
    def has(self, security_id: str) -> bool:
        return security_id in self._manifest and self._file(security_id).exists()

    def get(self, security_id: str) -> pd.DataFrame:
        if not self.has(security_id):
            raise KeyError(security_id)
        return pd.read_parquet(self._file(security_id), engine="pyarrow")

    def ids(self) -> list[str]:
        return sorted(self._manifest)

    def record(self, security_id: str) -> StoredSeries:
        return self._manifest[security_id]

    def manifest_digest(self, ids: Iterable[str] | None = None) -> str:
        keys = sorted(ids) if ids is not None else sorted(self._manifest)
        return content_hash({k: self._manifest[k].sha256 for k in keys if k in self._manifest})

    def panel(self, ids: Sequence[str], start: str | None = None, end: str | None = None) -> Panel:
        """Wide panel of bar fields for ``ids`` on the union of their sessions."""
        frames: dict[str, dict[str, pd.Series]] = {c: {} for c in BAR_COLUMNS}
        for sid in ids:
            df = self.get(sid)
            if start:
                df = df.loc[df.index >= pd.Timestamp(start)]
            if end:
                df = df.loc[df.index <= pd.Timestamp(end)]
            for c in BAR_COLUMNS:
                frames[c][sid] = df[c]
        wide = {c: pd.DataFrame(v).reindex(columns=list(ids)) for c, v in frames.items()}
        p = Panel.from_frames(wide)
        if p.n_dates == 0:
            raise DataError("empty panel")
        return p


def session_calendar_from(panel: Panel, min_coverage: float = 0.5) -> np.ndarray:
    """Sessions on which at least ``min_coverage`` of securities with data that year traded.

    Guards against stray single-security prints on non-sessions (e.g. a holiday OTC print).
    """
    live = np.isfinite(panel["close"])
    frac = live.sum(axis=1) / np.maximum(1, live.any(axis=0).sum())
    return panel.dates[frac >= min_coverage * float(np.median(frac))]
