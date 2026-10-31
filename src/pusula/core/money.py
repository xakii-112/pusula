"""Exact money arithmetic.

Cash, prices, fees and PnL in the trading ledger are integers in micro-dollars (1e-6 USD);
quantities are integer shares or contracts. Research code works in float64 returns and
weights, which is standard and fast; the ledger, where every cent must reconcile, never does.
Conversions from float happen exactly once, at the boundary, with explicit rounding.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

MICROS = 1_000_000
CENT_MICROS = 10_000


def to_micros(amount: float | int | str | Decimal) -> int:
    """Dollars -> integer micro-dollars, round-half-even.

    Floats go through ``repr`` so that ``0.1`` becomes exactly 100_000 rather than the binary
    approximation.
    """
    if isinstance(amount, float):
        d = Decimal(repr(amount))
    elif isinstance(amount, Decimal):
        d = amount
    else:
        d = Decimal(amount)
    if not d.is_finite():
        raise ValueError(f"non-finite amount {amount!r}")
    return int((d * MICROS).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def to_dollars(micros: int) -> float:
    return micros / MICROS


def notional(price_micros: int, qty: int) -> int:
    """Signed notional in micros; exact (integer product)."""
    return price_micros * qty


def apply_rate(micros: int, rate: float | Decimal) -> int:
    """``micros * rate`` rounded half-even to a whole micro (fees, borrow, interest)."""
    r = Decimal(repr(rate)) if isinstance(rate, float) else rate
    return int((Decimal(micros) * r).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def round_to_cents(micros: int) -> int:
    """Round a micro amount to whole cents (broker fee convention), half-even."""
    q, r = divmod(micros, CENT_MICROS)
    half = CENT_MICROS // 2
    if r > half or (r == half and q % 2 == 1):
        q += 1
    return q * CENT_MICROS


def format_usd(micros: int) -> str:
    sign = "-" if micros < 0 else ""
    m = abs(micros)
    dollars, rem = divmod(m, MICROS)
    return f"{sign}${dollars:,}.{rem // CENT_MICROS:02d}"
