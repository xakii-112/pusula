"""Injected time. Nothing below the ``live`` layer reads the wall clock directly."""

from __future__ import annotations

import datetime as dt
from typing import Protocol


class Clock(Protocol):
    def now(self) -> dt.datetime: ...


class WallClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)


class FixedClock:
    """Deterministic clock for tests and replays; ``advance`` moves it forward."""

    def __init__(self, at: dt.datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("FixedClock needs an aware datetime")
        self._at = at

    def now(self) -> dt.datetime:
        return self._at

    def advance(self, delta: dt.timedelta) -> None:
        if delta < dt.timedelta(0):
            raise ValueError("time does not go backwards")
        self._at += delta
