"""Exact trading ledger (integer micro-dollars, integer shares).

Capital flows (deposits, withdrawals, book resets) are their own event type and never mix
with trading P&L, so the P&L identity holds exactly at all times::

    equity - net_contributions == realised + unrealised - fees + financing

NihaiAlgoritma's live equity curve mixed resets of the paper account with strategy losses and
could not be audited afterwards; here the identity is asserted after every event in tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pusula.core.errors import PusulaError


@dataclass(slots=True)
class Position:
    qty: int = 0
    cost_micros: int = 0  # total cost basis of the open quantity (signed with qty)


@dataclass(slots=True)
class Ledger:
    cash: int = 0
    contributions: int = 0
    realised: int = 0
    fees: int = 0
    financing: int = 0  # interest received minus borrow / margin paid
    positions: dict[str, Position] = field(default_factory=dict)

    def contribute(self, micros: int) -> None:
        self.cash += micros
        self.contributions += micros

    def fill(self, security: str, qty: int, price_micros: int, fee_micros: int) -> None:
        if qty == 0:
            return
        if price_micros <= 0 or fee_micros < 0:
            raise PusulaError(f"invalid fill price {price_micros} / fee {fee_micros}")
        pos = self.positions.setdefault(security, Position())
        self.cash -= qty * price_micros + fee_micros
        self.fees += fee_micros
        if pos.qty == 0 or (pos.qty > 0) == (qty > 0):
            pos.cost_micros += qty * price_micros
            pos.qty += qty
        else:
            closing = min(abs(qty), abs(pos.qty)) * (1 if qty > 0 else -1)
            # average cost of the part being closed, exact: cost * closing / qty, rounded once
            avg_part = (pos.cost_micros * -closing) // pos.qty if pos.qty else 0
            # realised = proceeds - cost (for a long close: -closing > 0 shares sold)
            realised = (-closing) * price_micros - avg_part
            self.realised += realised
            pos.cost_micros -= avg_part
            pos.qty += closing
            rest = qty - closing
            if rest:
                pos.cost_micros += rest * price_micros
                pos.qty += rest
        if pos.qty == 0:
            # residual rounding of the integer division lands in realised P&L, exactly once
            self.realised -= pos.cost_micros
            pos.cost_micros = 0
            del self.positions[security]

    def financing_flow(self, micros: int) -> None:
        self.cash += micros
        self.financing += micros

    def market_value(self, marks: Mapping[str, int]) -> int:
        return sum(p.qty * marks[s] for s, p in self.positions.items())

    def unrealised(self, marks: Mapping[str, int]) -> int:
        return sum(p.qty * marks[s] - p.cost_micros for s, p in self.positions.items())

    def equity(self, marks: Mapping[str, int]) -> int:
        return self.cash + self.market_value(marks)

    def check_identity(self, marks: Mapping[str, int]) -> None:
        lhs = self.equity(marks) - self.contributions
        rhs = self.realised + self.unrealised(marks) - self.fees + self.financing
        if lhs != rhs:
            raise PusulaError(f"P&L identity broken: {lhs} != {rhs}")
