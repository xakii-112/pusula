"""Book-vs-broker reconciliation.

The journal is pusula's truth about what it did; the broker's position report is the truth
about what happened. They are compared every run and any difference is journalled as a
``reconcile`` event with the exact share breaks; a book with an unexplained break is halted
(new decisions are refused) until an operator resolves it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pusula.engine.ledger import Ledger


@dataclass(frozen=True, slots=True)
class Break:
    security: str
    book: int
    broker: int

    @property
    def diff(self) -> int:
        return self.broker - self.book


@dataclass(frozen=True, slots=True)
class Reconciliation:
    breaks: tuple[Break, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not self.breaks

    def as_event(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "breaks": [
                {"security": b.security, "book": b.book, "broker": b.broker} for b in self.breaks
            ],
        }


def reconcile(
    ledger: Ledger, broker_positions: Mapping[str, int], symbol_of: Mapping[str, str] | None = None
) -> Reconciliation:
    """Compare share positions. ``symbol_of`` maps pusula security ids to broker symbols
    (default: the ticker part of ``TICKER@start``)."""
    book: dict[str, int] = {}
    for sid, pos in ledger.positions.items():
        sym = symbol_of[sid] if symbol_of is not None else sid.split("@")[0]
        book[sym] = book.get(sym, 0) + pos.qty
    keys = sorted(set(book) | set(broker_positions))
    breaks = tuple(
        Break(k, book.get(k, 0), int(broker_positions.get(k, 0)))
        for k in keys
        if book.get(k, 0) != int(broker_positions.get(k, 0))
    )
    return Reconciliation(breaks)
