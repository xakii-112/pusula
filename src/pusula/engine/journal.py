"""Append-only, hash-chained event journal; the ledger is a fold over it.

State is never "the last JSON file written": it is whatever replaying the journal produces.
After a crash the book is rebuilt by replay, and a reconciliation against the broker compares
that rebuilt state with the broker's positions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pusula.core.errors import PusulaError
from pusula.core.hashing import canonical, content_hash
from pusula.engine.ledger import Ledger

GENESIS = "0" * 64
KINDS = frozenset(
    {"capital", "order", "fill", "financing", "mark", "decision", "risk", "reconcile"}
)


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    kind: str
    session: str
    data: dict[str, Any]
    prev: str
    hash: str


class Journal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.events.append(
                        Event(r["seq"], r["kind"], r["session"], r["data"], r["prev"], r["hash"])
                    )
            self.verify()

    @staticmethod
    def _digest(seq: int, kind: str, session: str, data: Mapping[str, Any], prev: str) -> str:
        return content_hash(
            {"seq": seq, "kind": kind, "session": session, "data": data, "prev": prev}
        )

    def verify(self) -> None:
        prev = GENESIS
        for i, e in enumerate(self.events):
            if (
                e.seq != i
                or e.prev != prev
                or self._digest(e.seq, e.kind, e.session, e.data, e.prev) != e.hash
            ):
                raise PusulaError(f"journal integrity failure at seq {i}")
            prev = e.hash

    def append(self, kind: str, session: str, data: Mapping[str, Any]) -> Event:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        body = canonical(dict(data))
        seq = len(self.events)
        prev = self.events[-1].hash if self.events else GENESIS
        e = Event(seq, kind, session, body, prev, self._digest(seq, kind, session, body, prev))
        self.events.append(e)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(
                    json.dumps(
                        {
                            "seq": e.seq,
                            "kind": e.kind,
                            "session": e.session,
                            "data": e.data,
                            "prev": e.prev,
                            "hash": e.hash,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return e

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def replay(self) -> tuple[Ledger, dict[str, int]]:
        """Rebuild the ledger and the last marks from the events."""
        led, marks, _ = self._fold()
        return led, marks

    def equity_by_session(self) -> list[tuple[str, int]]:
        """Equity (micros) at every mark event, i.e. at each processed close. A resumed book
        rebuilds its equity curve from this, not from the current process's memory."""
        return self._fold()[2]

    def _fold(self) -> tuple[Ledger, dict[str, int], list[tuple[str, int]]]:
        led = Ledger()
        marks: dict[str, int] = {}
        curve: list[tuple[str, int]] = []
        for e in self.events:
            d = e.data
            if e.kind == "capital":
                led.contribute(int(d["micros"]))
            elif e.kind == "fill":
                led.fill(
                    str(d["security"]), int(d["qty"]), int(d["price_micros"]), int(d["fee_micros"])
                )
            elif e.kind == "financing":
                led.financing_flow(int(d["micros"]))
            elif e.kind == "mark":
                today = {str(k): int(v) for k, v in d["marks"].items()}
                marks.update(today)
                curve.append((e.session, led.equity(today)))
        return led, marks, curve
