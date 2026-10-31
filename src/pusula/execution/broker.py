"""Brokers: a deterministic simulator and an Alpaca paper-trading adapter.

Both implement :class:`Broker`. The simulator fills day orders at the session's execution
price (VWAP by default) plus the cost model, at most ``max_participation`` of ADV; the rest of
the order expires at the close. The Alpaca adapter is **dry-run by default**: it records the
exact request it would send and sends nothing until constructed with ``dry_run=False``, which
the live runner only does when the operator passes an explicit flag.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from pusula.core.money import to_micros
from pusula.execution.costs import CostModel, trade_cost
from pusula.execution.orders import Order, OrderState

ORDER_RETRIES = 6


@dataclass(frozen=True, slots=True)
class FillReport:
    client_id: str
    security: str
    qty: int
    price_micros: int
    fee_micros: int


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """What the simulator needs to fill one session's orders (per security)."""

    exec_price: Mapping[str, float]
    half_spread: Mapping[str, float]
    adv_dollars: Mapping[str, float]
    sigma: Mapping[str, float]


class Broker(Protocol):
    def submit(self, orders: Sequence[Order]) -> None: ...

    def execute_session(self, snapshot: MarketSnapshot) -> list[FillReport]: ...


@dataclass
class SimBroker:
    cost_model: CostModel = field(default_factory=CostModel)
    open_orders: dict[str, Order] = field(default_factory=dict)
    seen_ids: set[str] = field(default_factory=set)

    def submit(self, orders: Sequence[Order]) -> None:
        for o in orders:
            if o.client_id in self.seen_ids:
                continue  # idempotent: a resubmitted id is the same order
            self.seen_ids.add(o.client_id)
            o.transition(OrderState.NEW, "accepted by simulator")
            self.open_orders[o.client_id] = o

    def execute_session(self, snapshot: MarketSnapshot) -> list[FillReport]:
        fills: list[FillReport] = []
        cm = self.cost_model
        for cid, o in list(self.open_orders.items()):
            px = snapshot.exec_price.get(o.security, float("nan"))
            if not (np.isfinite(px) and px > 0):
                o.transition(OrderState.EXPIRED, "no price")
                del self.open_orders[cid]
                continue
            adv = snapshot.adv_dollars.get(o.security, float("nan"))
            cap_sh = int(cm.max_participation * adv / px) if np.isfinite(adv) else 0
            qty = int(np.sign(o.remaining)) * min(abs(o.remaining), max(cap_sh, 0))
            if qty != 0:
                cost = float(
                    trade_cost(
                        np.array([qty * px]),
                        np.array([snapshot.half_spread.get(o.security, np.nan)]),
                        np.array([adv]),
                        np.array([snapshot.sigma.get(o.security, np.nan)]),
                        cm,
                    )[0]
                )
                rep = FillReport(cid, o.security, qty, to_micros(px), to_micros(round(cost, 6)))
                o.fill(qty)
                fills.append(rep)
            if o.state not in (OrderState.FILLED,):
                o.transition(OrderState.EXPIRED, f"day order: {o.remaining} unfilled")
            del self.open_orders[cid]
        return fills


@dataclass
class AlpacaPaperBroker:
    """Alpaca paper trading (https://paper-api.alpaca.markets). Dry-run unless told otherwise."""

    key_id: str
    secret: str
    dry_run: bool = True
    base_url: str = "https://paper-api.alpaca.markets"
    transport: httpx.BaseTransport | None = None
    sent: list[dict[str, Any]] = field(default_factory=list)
    retry_wait: float = 2.0

    def _post_order(self, c: httpx.Client, payload: dict[str, Any]) -> httpx.Response | str:
        """A throttled or failing broker is waited for: a plan of hundreds of orders runs into the
        rate limit, and that is not a refusal of the order. The ids make a repeat harmless."""
        problem = ""
        for attempt in range(ORDER_RETRIES):
            try:
                r = c.post("/v2/orders", json=payload)
            except httpx.TransportError as exc:
                problem = type(exc).__name__
            else:
                if r.status_code != 429 and r.status_code < 500:
                    return r
                problem = f"HTTP {r.status_code}"
            time.sleep(self.retry_wait * 2**attempt)
        return problem

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=30.0,
            transport=self.transport,
            headers={"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret},
        )

    @staticmethod
    def order_payload(o: Order) -> dict[str, Any]:
        return {
            "symbol": o.security.split("@")[0],
            "qty": str(abs(o.qty)),
            "side": "buy" if o.qty > 0 else "sell",
            "type": o.order_type,
            "time_in_force": o.tif,
            "client_order_id": o.client_id,
        }

    def submit(self, orders: Sequence[Order]) -> None:
        if not orders:
            return
        if self.dry_run:
            for o in orders:
                self.sent.append({"dry_run": True, **self.order_payload(o)})
                o.transition(OrderState.NEW, "dry-run (not sent)")
            return
        with self._client() as c:
            for o in orders:
                payload = self.order_payload(o)
                r = self._post_order(c, payload)
                if isinstance(r, str):
                    o.transition(OrderState.REJECTED, f"gave up: {r}")
                elif r.status_code == 422 and "client_order_id" in r.text:
                    o.transition(OrderState.NEW, "already accepted (idempotent resubmit)")
                elif r.status_code in (200, 201):
                    o.transition(OrderState.NEW, f"alpaca id {r.json().get('id')}")
                else:
                    o.transition(OrderState.REJECTED, f"HTTP {r.status_code}: {r.text[:120]}")
                self.sent.append(payload)

    def submit_exact(self, orders: Sequence[tuple[str, str, float, str]]) -> dict[str, str]:
        """(client id, symbol, signed quantity, time in force) as market orders, in fractions where
        the quantity has them. Returns what was refused and why. Nothing is sent on a dry run."""
        problems: dict[str, str] = {}
        if self.dry_run:
            return problems
        with self._client() as c:
            for client_id, symbol, qty, tif in orders:
                payload = {
                    "symbol": symbol,
                    "qty": format(abs(qty), ".9f").rstrip("0").rstrip("."),
                    "side": "buy" if qty > 0 else "sell",
                    "type": "market",
                    "time_in_force": tif,
                    "client_order_id": client_id,
                }
                r = self._post_order(c, payload)
                self.sent.append(payload)
                if isinstance(r, str):
                    problems[client_id] = f"gave up: {r}"
                elif r.status_code not in (200, 201) and not (
                    r.status_code == 422 and "client_order_id" in r.text
                ):
                    problems[client_id] = f"HTTP {r.status_code}: {r.text[:120]}"
        return problems

    def positions(self) -> dict[str, int]:
        """Whole shares of the US stocks held."""
        with self._client() as c:
            r = c.get("/v2/positions")
            r.raise_for_status()
            return {
                p["symbol"]: int(float(p["qty"]))
                for p in r.json()
                if p.get("asset_class", "us_equity") == "us_equity"
            }

    def exact_positions(self) -> dict[str, float]:
        """Everything held, in fractions, a coin as ``BTC/USD`` (the feed spells it ``BTCUSD``)."""
        with self._client() as c:
            r = c.get("/v2/positions")
            r.raise_for_status()
            out: dict[str, float] = {}
            for p in r.json():
                name = str(p["symbol"])
                if p.get("asset_class") == "crypto" and "/" not in name:
                    name = name.removesuffix("USD") + "/USD"
                out[name] = float(p["qty"])
            return out

    def assets(self, asset_class: str) -> list[dict[str, Any]]:
        """The broker's active assets of a class: what can be traded, and how."""
        with self._client() as c:
            r = c.get("/v2/assets", params={"status": "active", "asset_class": asset_class})
            r.raise_for_status()
            found: list[dict[str, Any]] = r.json()
            return found

    def equity(self) -> float:
        with self._client() as c:
            r = c.get("/v2/account")
            r.raise_for_status()
            return float(r.json()["equity"])

    def fills(self, day: str) -> list[dict[str, Any]]:
        """The account's fills on a day, from the broker's activity feed."""
        with self._client() as c:
            r = c.get("/v2/account/activities/FILL", params={"date": day})
            r.raise_for_status()
            out: list[dict[str, Any]] = r.json()
            return out

    def execute_session(self, snapshot: MarketSnapshot) -> list[FillReport]:
        raise NotImplementedError("fills come from the broker's activity feed, not a simulation")
