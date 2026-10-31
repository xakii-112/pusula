"""NYSE trading calendar (full-day closures and early closes), rule-based.

Research uses the sessions that actually appear in the data; this calendar is the independent
check on that data (a missing or extra session is a data-quality finding) and the scheduler for
live/paper runs, where future sessions are needed.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np

# Unscheduled full-day closures since 2000 (national days of mourning, weather, 9/11).
SPECIAL_CLOSURES: frozenset[dt.date] = frozenset(
    {
        dt.date(2001, 9, 11),
        dt.date(2001, 9, 12),
        dt.date(2001, 9, 13),
        dt.date(2001, 9, 14),
        dt.date(2004, 6, 11),  # President Reagan
        dt.date(2007, 1, 2),  # President Ford
        dt.date(2012, 10, 29),  # Hurricane Sandy
        dt.date(2012, 10, 30),
        dt.date(2018, 12, 5),  # President G.H.W. Bush
        dt.date(2025, 1, 9),  # President Carter
    }
)


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


@lru_cache(maxsize=256)
def holidays(year: int) -> frozenset[dt.date]:
    """Scheduled full-day NYSE holidays in ``year`` plus special closures."""
    out: set[dt.date] = set()
    ny = dt.date(year, 1, 1)
    # NYSE does not close on Friday Dec 31 when Jan 1 is a Saturday (Rule 7.2).
    if ny.weekday() != 5:
        out.add(_observed(ny))
    out.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    out.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    out.add(easter(year) - dt.timedelta(days=2))  # Good Friday
    out.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        out.add(_observed(dt.date(year, 6, 19)))  # Juneteenth
    out.add(_observed(dt.date(year, 7, 4)))
    out.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    out.add(_observed(dt.date(year, 12, 25)))
    out.update(d for d in SPECIAL_CLOSURES if d.year == year)
    return frozenset(out)


def is_session(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def early_closes(year: int) -> frozenset[dt.date]:
    """13:00 ET closes: day after Thanksgiving, Christmas Eve and July 3 when they are sessions."""
    out = {_nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1)}
    for d in (dt.date(year, 12, 24), dt.date(year, 7, 3)):
        if is_session(d):
            out.add(d)
    return frozenset(d for d in out if is_session(d))


def sessions(start: dt.date, end: dt.date) -> np.ndarray:
    """All NYSE sessions in ``[start, end]`` as ``datetime64[D]``."""
    if end < start:
        return np.array([], dtype="datetime64[D]")
    days = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(end, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    keep = [is_session(d.item()) for d in days]
    return days[np.asarray(keep, dtype=bool)]


def next_session(d: dt.date) -> dt.date:
    nxt = d + dt.timedelta(days=1)
    while not is_session(nxt):
        nxt += dt.timedelta(days=1)
    return nxt


def previous_session(d: dt.date) -> dt.date:
    prv = d - dt.timedelta(days=1)
    while not is_session(prv):
        prv -= dt.timedelta(days=1)
    return prv
