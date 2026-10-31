"""A local limit order book rebuilt from Binance's diff stream, in whole ticks and lots.

The documented procedure: keep the diffs, take a REST snapshot, drop every diff the snapshot already
contains, and require the rest to chain (``U`` of each is the last ``u`` plus one). A broken chain
is reported, never papered over: the replay stops counting until the next snapshot re-anchors it.
Later snapshots also check the book that was maintained in between, which is the measure of whether
this reconstruction can be trusted at all.

Replaying a recording yields events in the exchange's own order, so a backtest sees what a resting
order would have seen. Integer arithmetic throughout, as in meridian: PnL identities hold exactly.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

BUY, SELL = 1, -1


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    tick: float
    lot: float

    def price_ticks(self, price: str | float) -> int:
        return round(float(price) / self.tick)

    def qty_lots(self, qty: str | float) -> int:
        return round(float(qty) / self.lot)


class Side:
    """One side of the book: quantity per price, with the prices kept sorted."""

    def __init__(self, sign: int) -> None:
        self.sign = sign
        self.levels: dict[int, int] = {}
        self._keys: list[int] = []  # ascending in sign * price: best first

    def set(self, price: int, qty: int) -> None:
        key = -self.sign * price
        if qty <= 0:
            if price in self.levels:
                del self.levels[price]
                self._keys.pop(bisect.bisect_left(self._keys, key))
            return
        if price not in self.levels:
            bisect.insort(self._keys, key)
        self.levels[price] = qty

    def best(self) -> int | None:
        return -self.sign * self._keys[0] if self._keys else None

    def top(self, n: int) -> list[tuple[int, int]]:
        out = []
        for key in self._keys[:n]:
            price = -self.sign * key
            out.append((price, self.levels[price]))
        return out

    def clear(self) -> None:
        self.levels.clear()
        self._keys.clear()


class LocalBook:
    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument
        self.bids = Side(BUY)
        self.asks = Side(SELL)
        self.last_update = -1
        self.synced = False

    def snapshot(self, data: dict[str, Any]) -> None:
        inst = self.instrument
        self.bids.clear()
        self.asks.clear()
        for side, rows in ((self.bids, data["bids"]), (self.asks, data["asks"])):
            for price, qty in rows:
                side.set(inst.price_ticks(price), inst.qty_lots(qty))
        self.last_update = int(data["lastUpdateId"])
        self.synced = False  # until a diff chains onto it

    def apply(self, event: dict[str, Any]) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """Apply one diff; returns (in sequence, changes) where a change is (side, price, old, new).
        A diff the snapshot already holds is not an error: it is skipped with no changes."""
        first, last = int(event["U"]), int(event["u"])
        if last <= self.last_update:
            return True, []
        expected = self.last_update + 1
        if self.synced:
            if first != expected:
                self.synced = False
                return False, []
        elif not first <= expected <= last:
            return False, []
        self.synced = True
        self.last_update = last
        changes: list[tuple[int, int, int, int]] = []
        inst = self.instrument
        for side_key, side in (("b", self.bids), ("a", self.asks)):
            for price_s, qty_s in event[side_key]:
                price, qty = inst.price_ticks(price_s), inst.qty_lots(qty_s)
                old = side.levels.get(price, 0)
                if old != qty:
                    changes.append((side.sign, price, old, qty))
                    side.set(price, qty)
        return True, changes

    def mid2(self) -> int | None:
        """Twice the mid, so that it stays a whole number of ticks."""
        b, a = self.bids.best(), self.asks.best()
        return None if b is None or a is None else b + a

    def crossed(self) -> bool:
        b, a = self.bids.best(), self.asks.best()
        return b is not None and a is not None and b >= a


@dataclass(frozen=True, slots=True)
class Check:
    """A later snapshot against the book maintained up to it."""

    time_ns: int
    levels: int
    matches: int


@dataclass(frozen=True, slots=True)
class BookEvent:
    time_ns: int
    changes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class SegmentBreak:
    """The record stops being continuous here: a lost connection or a broken chain. A replay must
    not treat what the market did in between as something a maker lived through."""


@dataclass(frozen=True, slots=True)
class TradeEvent:
    time_ns: int
    price: int
    qty: int
    aggressor: int  # BUY: a buyer lifted the ask


def compare_top(book: LocalBook, data: dict[str, Any], depth: int = 20) -> Check:
    """How many of the snapshot's best ``depth`` levels the maintained book agrees with."""
    inst = book.instrument
    levels = matches = 0
    for side, rows in ((book.bids, data["bids"]), (book.asks, data["asks"])):
        for price_s, qty_s in rows[:depth]:
            levels += 1
            matches += side.levels.get(inst.price_ticks(price_s), 0) == inst.qty_lots(qty_s)
    return Check(0, levels, matches)


@dataclass
class ReplayStats:
    diffs: int = 0
    trades: int = 0
    gaps: int = 0
    snapshots: int = 0
    level_checks: int = 0
    level_matches: int = 0
    crossed: int = 0


def replay(
    records: Iterable[tuple[int, str, dict[str, Any]]], book: LocalBook, stats: ReplayStats
) -> Iterator[BookEvent | TradeEvent | SegmentBreak]:
    """Exchange-ordered events of a recording, while the book is in sequence. Events carry the
    exchange's own event time; trades arrive on the same connection as diffs, in order."""
    inst = book.instrument
    check_against: dict[str, Any] | None = None
    for _, kind, data in records:
        if kind == "snapshot":
            stats.snapshots += 1
            if book.synced:
                # taken ahead of the diffs on file: compare when the book gets there
                check_against = data
                continue  # still in sequence: nothing to re-anchor
            book.snapshot(data)
        elif kind == "depth":
            was_synced = book.synced
            ok, changes = book.apply(data)
            if not ok:
                if was_synced:
                    stats.gaps += 1
                    yield SegmentBreak()
                check_against = None
                continue
            if check_against is not None and book.last_update >= int(check_against["lastUpdateId"]):
                check = compare_top(book, check_against)
                stats.level_checks += check.levels
                stats.level_matches += check.matches
                check_against = None
            if changes:
                stats.diffs += 1
                stats.crossed += book.crossed()
                yield BookEvent(int(data["E"]) * 1_000_000, tuple(changes))
        elif kind == "trade":
            if not book.synced:
                continue
            stats.trades += 1
            yield TradeEvent(
                int(data["T"]) * 1_000_000,
                inst.price_ticks(data["p"]),
                inst.qty_lots(data["q"]),
                SELL if data["m"] else BUY,
            )
        elif kind == "segment":
            book.synced = False
            check_against = None
            if stats.snapshots:
                stats.gaps += 1
            yield SegmentBreak()
