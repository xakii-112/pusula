"""Order lifecycle.

States and legal transitions (anything else raises ``OrderStateError``)::

    PENDING_NEW -> NEW | REJECTED
    NEW -> PARTIALLY_FILLED | FILLED | PENDING_CANCEL | CANCELED | EXPIRED
    PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | PENDING_CANCEL | CANCELED | EXPIRED
    PENDING_CANCEL -> CANCELED | FILLED | PARTIALLY_FILLED   (a fill can race the cancel)

Client order ids are deterministic (hash of strategy, session, security and sequence), so a
retried submission after a crash is recognised as the same order by the broker (idempotency),
never sent twice.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from pusula.core.errors import OrderStateError
from pusula.core.hashing import sha256_hex


class OrderState(enum.StrEnum):
    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


TERMINAL = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED}
)

_ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PENDING_NEW: frozenset({OrderState.NEW, OrderState.REJECTED}),
    OrderState.NEW: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {OrderState.CANCELED, OrderState.FILLED, OrderState.PARTIALLY_FILLED}
    ),
}


def client_order_id(strategy: str, session: str, security: str, seq: int = 0) -> str:
    return "pz-" + sha256_hex(f"{strategy}|{session}|{security}|{seq}")[:28]


@dataclass(slots=True)
class Order:
    client_id: str
    security: str
    qty: int  # signed: + buy, - sell
    session: str
    tif: str = "day"
    order_type: str = "market"  # executed by the venue/algo as a VWAP-style day order
    limit_micros: int | None = None
    state: OrderState = OrderState.PENDING_NEW
    filled: int = 0  # signed, same sign as qty
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.qty == 0:
            raise ValueError("zero-quantity order")

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    def transition(self, new: OrderState, note: str = "") -> None:
        if self.state in TERMINAL:
            raise OrderStateError(f"{self.client_id}: {self.state} is terminal, cannot go to {new}")
        if new not in _ALLOWED[self.state]:
            raise OrderStateError(f"{self.client_id}: illegal transition {self.state} -> {new}")
        self.history.append(f"{self.state}->{new}{(' ' + note) if note else ''}")
        self.state = new

    def fill(self, qty: int) -> None:
        if qty == 0:
            return
        if (qty > 0) != (self.qty > 0):
            raise OrderStateError(f"{self.client_id}: fill side does not match order side")
        if abs(self.filled + qty) > abs(self.qty):
            raise OrderStateError(f"{self.client_id}: overfill ({self.filled}+{qty} of {self.qty})")
        # validate the transition before mutating: a rejected fill must leave the order untouched
        self.transition(
            OrderState.FILLED if self.filled + qty == self.qty else OrderState.PARTIALLY_FILLED
        )
        self.filled += qty
