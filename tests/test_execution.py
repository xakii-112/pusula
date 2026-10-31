from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pusula.core.errors import OrderStateError, PusulaError
from pusula.engine.journal import Journal
from pusula.engine.ledger import Ledger
from pusula.execution.broker import AlpacaPaperBroker, MarketSnapshot, SimBroker
from pusula.execution.costs import CostModel, trade_cost
from pusula.execution.orders import Order, OrderState, client_order_id


def test_order_state_machine() -> None:
    o = Order(client_order_id("s", "2024-01-02", "AAPL"), "AAPL", 100, "2024-01-02")
    with pytest.raises(OrderStateError):
        o.fill(10)  # not acknowledged yet
    o.transition(OrderState.NEW)
    o.fill(40)
    assert o.state is OrderState.PARTIALLY_FILLED and o.remaining == 60
    with pytest.raises(OrderStateError, match="overfill"):
        o.fill(61)
    with pytest.raises(OrderStateError, match="side"):
        o.fill(-1)
    o.fill(60)
    assert o.state is OrderState.FILLED
    with pytest.raises(OrderStateError, match="terminal"):
        o.transition(OrderState.CANCELED)


def test_client_ids_are_deterministic() -> None:
    a = client_order_id("strat", "2024-01-02", "MSFT", 0)
    assert a == client_order_id("strat", "2024-01-02", "MSFT", 0)
    assert a != client_order_id("strat", "2024-01-02", "MSFT", 1)


fills = st.lists(
    st.tuples(
        st.sampled_from(["A", "B", "C"]),
        st.integers(-500, 500).filter(lambda q: q != 0),
        st.integers(1_000_000, 500_000_000),
        st.integers(0, 5_000_000),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=300, deadline=None)
@given(fills, st.integers(1_000_000, 900_000_000))
def test_ledger_pnl_identity_exact(seq: list[tuple[str, int, int, int]], mark: int) -> None:
    led = Ledger()
    led.contribute(10**12)
    for sec, q, px, fee in seq:
        led.fill(sec, q, px, fee)
        led.financing_flow(-17)
        marks = dict.fromkeys(led.positions, mark)
        led.check_identity(marks)  # integer equality after every event


def test_ledger_round_trip_realises_exact_pnl() -> None:
    led = Ledger()
    led.contribute(1_000_000_000)
    led.fill("A", 3, 10_000_000, 0)
    led.fill("A", -1, 13_000_000, 0)
    led.fill("A", -2, 7_000_000, 0)
    assert led.realised == 3_000_000 - 6_000_000
    assert "A" not in led.positions
    led.fill("B", -10, 5_000_000, 1)  # short
    led.fill("B", 15, 4_000_000, 1)  # cover and flip long 5
    assert led.realised == -3_000_000 + 10_000_000
    assert led.positions["B"].qty == 5
    led.check_identity({"B": 4_500_000})


def test_ledger_rejects_bad_fill() -> None:
    with pytest.raises(PusulaError):
        Ledger().fill("A", 1, 0, 0)


def test_journal_replay_and_tamper(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "j.jsonl"
    j = Journal(p)
    j.append("capital", "2024-01-02", {"micros": 1_000_000_000})
    j.append(
        "fill",
        "2024-01-03",
        {"security": "A", "qty": 10, "price_micros": 50_000_000, "fee_micros": 5},
    )
    j.append("financing", "2024-01-03", {"micros": 12})
    j.append("mark", "2024-01-03", {"marks": {"A": 51_000_000}})
    led, marks = Journal(p).replay()
    assert led.cash == 1_000_000_000 - 500_000_005 + 12
    led.check_identity(marks)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["data"]["qty"] = 11
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(PusulaError, match="integrity"):
        Journal(p)


def _snap(px: float = 100.0, adv: float = 1e8) -> MarketSnapshot:
    return MarketSnapshot({"A": px}, {"A": 2e-4}, {"A": adv}, {"A": 0.02})


def test_sim_broker_fills_caps_and_is_idempotent() -> None:
    b = SimBroker(CostModel(max_participation=0.01))
    o = Order("id1", "A", 20_000, "2024-01-02")
    b.submit([o])
    b.submit([Order("id1", "A", 20_000, "2024-01-02")])  # duplicate id ignored
    rep = b.execute_session(_snap(adv=1e8))  # cap = 1% of $100M / $100 = 10,000 shares
    assert len(rep) == 1 and rep[0].qty == 10_000
    assert o.state is OrderState.EXPIRED and o.remaining == 10_000
    exp_fee = trade_cost(
        np.array([1e6]),
        np.array([2e-4]),
        np.array([1e8]),
        np.array([0.02]),
        CostModel(max_participation=0.01),
    )[0]
    assert rep[0].fee_micros == pytest.approx(exp_fee * 1e6, abs=1)


def test_alpaca_dry_run_sends_nothing() -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={"id": "x"})

    b = AlpacaPaperBroker("k", "s", transport=httpx.MockTransport(handler))
    o = Order("pz-1", "AAPL@2020-01-01", -5, "2024-01-02")
    b.submit([o])
    assert not calls
    assert b.sent[0]["symbol"] == "AAPL" and b.sent[0]["side"] == "sell" and b.sent[0]["dry_run"]


def test_alpaca_live_submit_and_idempotent_resubmit() -> None:
    seen: set[str] = set()

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.headers["APCA-API-KEY-ID"] == "k"
        if body["client_order_id"] in seen:
            return httpx.Response(422, json={"message": "client_order_id must be unique"})
        seen.add(body["client_order_id"])
        return httpx.Response(200, json={"id": "abc"})

    b = AlpacaPaperBroker("k", "s", dry_run=False, transport=httpx.MockTransport(handler))
    o1 = Order("pz-2", "MSFT", 3, "2024-01-02")
    b.submit([o1])
    o2 = Order("pz-2", "MSFT", 3, "2024-01-02")
    b.submit([o2])
    assert o1.state is OrderState.NEW and o2.state is OrderState.NEW
    assert "idempotent" in o2.history[-1]
