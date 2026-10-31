"""Fama-French 5 factors + momentum (daily), from the Kenneth R. French Data Library.

Used for two things: the risk-free rate (cash accrual in backtests) and factor attribution
(is a strategy's return explained by known, cheaply harvestable premia?).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from pusula.core.errors import DataError

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
FACTOR_NAMES = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a French daily CSV (percent units) into decimal returns indexed by date."""
    lines = text.splitlines()
    header_i = next((i for i, ln in enumerate(lines) if ln.startswith(",")), None)
    if header_i is None:
        raise DataError("no header row in French CSV")
    cols = [c.strip().lower().replace("-", "_") for c in lines[header_i].split(",")[1:]]
    rows: list[list[object]] = []
    for ln in lines[header_i + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(cols) + 1 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break  # end of the daily block
            continue
        rows.append([parts[0], *[float(p) for p in parts[1:]]])
    df = pd.DataFrame(rows, columns=["date", *cols])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), format="%Y%m%d"), name="date")
    out = df / 100.0
    if out.isna().any().any():
        raise DataError("missing values in French factor file")
    return out


def fetch(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Daily factors (decimal): mkt_rf, smb, hml, rmw, cma, mom, rf. Cached as CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_p = cache_dir / "french_daily.csv"
    if out_p.exists() and not refresh:
        return pd.read_csv(out_p, index_col=0, parse_dates=True)
    frames = []
    for key, fname in FILES.items():
        r = httpx.get(BASE + fname, headers={"User-Agent": "Mozilla/5.0 pusula"}, timeout=120.0)
        if r.status_code != 200:
            raise DataError(f"French library {key}: HTTP {r.status_code}")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = parse_french_csv(z.read(z.namelist()[0]).decode("latin-1"))
        frames.append(df)
    joined = frames[0].join(frames[1], how="inner")
    missing = [c for c in (*FACTOR_NAMES, "rf") if c not in joined.columns]
    if missing:
        raise DataError(f"French factor files lack {missing}")
    joined.to_csv(out_p)
    return joined


def align(factors: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    """Factors on the given sessions; sessions without a factor print are NaN (not filled)."""
    idx = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    return factors.reindex(idx)
