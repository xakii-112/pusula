"""Exception types. Errors are raised at boundaries with a reason; nothing swallows them."""

from __future__ import annotations


class PusulaError(Exception):
    """Base class."""


class DataError(PusulaError):
    """Input data violates its contract (schema, ordering, point-in-time)."""


class LookaheadError(PusulaError):
    """A computation used information that was not available at decision time."""


class RegistryError(PusulaError):
    """Experiment registry contract violated (missing pre-registration, tampering, rerun)."""


class SolverError(PusulaError):
    """The portfolio optimiser could not produce a valid solution."""


class OrderStateError(PusulaError):
    """Illegal order-state transition."""
