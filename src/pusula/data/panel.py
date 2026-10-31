"""Point-in-time panel: sessions x securities x fields.

Contract
--------
* ``dates`` are strictly increasing trading sessions (``datetime64[D]``).
* Row ``t`` of every field holds only information known at the **close of session t**.
  Open, high, low, close, volume and VWAP of session t satisfy this trivially; anything derived
  (signals, risk forecasts) must be computed causally, and ``alpha.lookahead`` checks it.
* Missing observations are NaN. A security that is not yet listed or already delisted is NaN,
  never forward-filled silently.
* Arrays are read-only. Transformations return new panels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from pusula.core.errors import DataError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
DateArray = npt.NDArray[np.datetime64]


def _frozen(a: npt.ArrayLike, dtype: npt.DTypeLike = np.float64) -> npt.NDArray[np.generic]:
    arr = np.array(a, dtype=dtype, copy=True)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class Panel:
    dates: DateArray
    symbols: tuple[str, ...]
    fields: Mapping[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates).astype("datetime64[D]")
        if dates.ndim != 1:
            raise DataError("dates must be 1-D")
        if dates.size > 1 and not np.all(dates[1:] > dates[:-1]):
            raise DataError("dates must be strictly increasing")
        if len(set(self.symbols)) != len(self.symbols):
            raise DataError("symbols must be unique")
        shape = (dates.size, len(self.symbols))
        frozen: dict[str, FloatArray] = {}
        for name, arr in self.fields.items():
            a = np.asarray(arr)
            if a.shape != shape:
                raise DataError(f"field {name!r} has shape {a.shape}, expected {shape}")
            frozen[name] = _frozen(a)  # type: ignore[assignment]
        object.__setattr__(self, "dates", _frozen(dates, "datetime64[D]"))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "fields", frozen)

    # ------------------------------------------------------------------ shape & access
    @property
    def n_dates(self) -> int:
        return int(self.dates.size)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def __getitem__(self, name: str) -> FloatArray:
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"panel has no field {name!r}; fields: {sorted(self.fields)}") from None

    def get(self, name: str, default: FloatArray | None = None) -> FloatArray | None:
        return self.fields.get(name, default)

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    def date_index(self, date: np.datetime64 | str) -> int:
        d = np.datetime64(date, "D")
        i = int(np.searchsorted(self.dates, d))
        if i >= self.n_dates or self.dates[i] != d:
            raise KeyError(f"{d} is not a session in this panel")
        return i

    def asof_index(self, date: np.datetime64 | str) -> int:
        """Index of the last session ``<= date`` (-1 if none)."""
        d = np.datetime64(date, "D")
        return int(np.searchsorted(self.dates, d, side="right")) - 1

    def symbol_index(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    # ------------------------------------------------------------------ transformations
    def upto(self, t: int) -> Panel:
        """The panel as it was known at the close of session ``t`` (rows 0..t)."""
        if not -1 <= t < self.n_dates:
            raise IndexError(t)
        return Panel(
            self.dates[: t + 1], self.symbols, {k: v[: t + 1] for k, v in self.fields.items()}
        )

    def rows(self, start: int, stop: int) -> Panel:
        return Panel(
            self.dates[start:stop], self.symbols, {k: v[start:stop] for k, v in self.fields.items()}
        )

    def between(self, start: str | np.datetime64, end: str | np.datetime64) -> Panel:
        lo = int(np.searchsorted(self.dates, np.datetime64(start, "D")))
        hi = int(np.searchsorted(self.dates, np.datetime64(end, "D"), side="right"))
        return self.rows(lo, hi)

    def take(self, symbols: Iterable[str]) -> Panel:
        syms = tuple(symbols)
        idx = [self.symbol_index(s) for s in syms]
        return Panel(self.dates, syms, {k: v[:, idx] for k, v in self.fields.items()})

    def with_fields(self, **new: npt.ArrayLike) -> Panel:
        merged: dict[str, FloatArray] = dict(self.fields)
        for k, v in new.items():
            merged[k] = np.asarray(v, dtype=np.float64)
        return Panel(self.dates, self.symbols, merged)

    def without(self, *names: str) -> Panel:
        return Panel(
            self.dates, self.symbols, {k: v for k, v in self.fields.items() if k not in names}
        )

    # ------------------------------------------------------------------ interop
    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(
            self[name], index=pd.DatetimeIndex(self.dates), columns=list(self.symbols)
        )

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> Panel:
        """Build from wide frames (index = dates, columns = symbols), outer-aligned."""
        if not frames:
            raise DataError("no frames")
        index = None
        cols: list[str] = []
        seen: set[str] = set()
        for df in frames.values():
            idx = pd.DatetimeIndex(df.index).normalize()
            index = idx if index is None else index.union(idx)
            for c in df.columns:
                if c not in seen:
                    seen.add(str(c))
                    cols.append(str(c))
        assert index is not None
        index = index.sort_values()
        out: dict[str, FloatArray] = {}
        for name, df in frames.items():
            d = df.copy()
            d.index = pd.DatetimeIndex(d.index).normalize()
            out[name] = d.reindex(index=index, columns=cols).to_numpy(dtype=np.float64)
        return cls(index.values.astype("datetime64[D]"), tuple(cols), out)

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        date: str = "date",
        symbol: str = "symbol",
        values: Iterable[str] | None = None,
    ) -> Panel:
        vals = (
            list(values)
            if values is not None
            else [c for c in df.columns if c not in (date, symbol)]
        )
        if df.duplicated([date, symbol]).any():
            raise DataError("duplicate (date, symbol) rows")
        frames = {v: df.pivot(index=date, columns=symbol, values=v) for v in vals}
        return cls.from_frames(frames)


def nan_equal(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], *, atol: float = 0.0) -> bool:
    """Elementwise equality where NaN == NaN (shape must match)."""
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= atol
    return bool(np.all(both_nan | close))
