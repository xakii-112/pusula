"""The order book replay, the market maker's queue and accounting, and the impact estimators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pusula.execution.costs import CostModel
from pusula.micro import impact as I
from pusula.micro import maker as M
from pusula.micro.book import (
    BUY,
    SELL,
    BookEvent,
    Instrument,
    LocalBook,
    ReplayStats,
    SegmentBreak,
    TradeEvent,
    compare_top,
    replay,
)
from pusula.research import mm_study as R

INST = Instrument("X", 0.01, 0.001)


def _snap(uid: int, bids: list[list[str]], asks: list[list[str]]) -> dict[str, Any]:
    return {"lastUpdateId": uid, "bids": bids, "asks": asks}


def _diff(
    first: int, last: int, bids: list[list[str]], asks: list[list[str]], e: int = 1
) -> dict[str, Any]:
    return {"U": first, "u": last, "b": bids, "a": asks, "E": e}


def _book() -> LocalBook:
    b = LocalBook(INST)
    b.snapshot(_snap(10, [["100.00", "1"], ["99.99", "2"]], [["100.02", "1"], ["100.03", "2"]]))
    return b


# ---------------------------------------------------------------------------- book
def test_side_keeps_best_first() -> None:
    b = _book()
    assert b.bids.best() == 10000 and b.asks.best() == 10002
    assert b.mid2() == 20002
    b.bids.set(10001, 5)
    assert b.bids.best() == 10001
    b.bids.set(10001, 0)
    assert b.bids.best() == 10000
    assert b.asks.top(2) == [(10002, 1000), (10003, 2000)]


def test_diff_anchoring_follows_the_documented_procedure() -> None:
    b = _book()
    assert b.apply(_diff(5, 10, [], []))[0] is True  # already in the snapshot: skipped
    assert not b.synced
    assert b.apply(_diff(12, 13, [], []))[0] is False  # leaves a hole after the snapshot
    ok, changes = b.apply(_diff(9, 12, [["100.00", "3"]], []))  # straddles it
    assert ok and b.synced and changes == [(BUY, 10000, 1000, 3000)]
    assert b.apply(_diff(13, 14, [], [["100.02", "0"]]))[1] == [(SELL, 10002, 1000, 0)]
    assert b.asks.best() == 10003


def test_a_broken_chain_is_reported_and_stops_the_book() -> None:
    b = _book()
    b.apply(_diff(11, 12, [], []))
    assert b.apply(_diff(14, 15, [], []))[0] is False
    assert not b.synced


def test_a_later_snapshot_checks_the_maintained_book() -> None:
    b = _book()
    b.apply(_diff(11, 12, [["100.00", "3"]], []))
    same = _snap(12, [["100.00", "3"], ["99.99", "2"]], [["100.02", "1"], ["100.03", "2"]])
    other = _snap(12, [["100.00", "9"], ["99.99", "2"]], [["100.02", "1"], ["100.03", "2"]])
    assert compare_top(b, same).matches == 4
    assert compare_top(b, other).matches == 3


def test_replay_yields_events_and_counts_gaps() -> None:
    records = [
        (0, "segment", {"started": True}),
        (1, "snapshot", _snap(10, [["100.00", "1"]], [["100.02", "1"]])),
        (2, "trade", {"T": 5, "p": "100.02", "q": "0.5", "m": False}),
        (3, "depth", _diff(11, 12, [], [["100.02", "0.5"]], e=6)),
        (4, "depth", _diff(15, 16, [], [], e=7)),
    ]
    book = LocalBook(INST)
    stats = ReplayStats()
    events = list(replay(records, book, stats))
    # the trade came before the first chained diff: the book was not in sequence yet
    assert [type(e) for e in events] == [SegmentBreak, BookEvent, SegmentBreak]
    assert stats.gaps == 1 and stats.snapshots == 1
    assert events[1].time_ns == 6_000_000


def test_a_later_snapshot_is_compared_when_the_book_reaches_it() -> None:
    records = [
        (1, "snapshot", _snap(10, [["100.00", "1"]], [["100.02", "1"]])),
        (2, "depth", _diff(11, 12, [["100.00", "2"]], [], e=6)),
        (3, "snapshot", _snap(13, [["100.00", "2"]], [["100.02", "9"]])),  # ahead of the book
        (4, "depth", _diff(13, 13, [], [["100.02", "9"]], e=7)),
    ]
    stats = ReplayStats()
    list(replay(records, LocalBook(INST), stats))
    assert stats.snapshots == 2 and stats.level_checks == 2 and stats.level_matches == 2


def test_aggressor_side_comes_from_the_maker_flag() -> None:
    records = [
        (1, "snapshot", _snap(10, [["100.00", "1"]], [["100.02", "1"]])),
        (2, "depth", _diff(11, 12, [["100.00", "2"]], [], e=6)),
        (3, "trade", {"T": 7, "p": "100.00", "q": "1", "m": True}),
        (4, "trade", {"T": 8, "p": "100.02", "q": "1", "m": False}),
    ]
    book = LocalBook(INST)
    events = list(replay(records, book, ReplayStats()))
    trades = [e for e in events if isinstance(e, TradeEvent)]
    assert [t.aggressor for t in trades] == [SELL, BUY]


# ---------------------------------------------------------------------------- market maker
def _market() -> LocalBook:
    b = LocalBook(INST)
    b.snapshot(_snap(10, [["100.00", "1"]], [["100.02", "1"]]))
    b.synced = True
    return b


SEC = 1_000_000_000


def _spec(**kw: Any) -> M.MakerSpec:
    args: dict[str, Any] = {"notional": 100.0, "latency_ns": 0, "decide_every_ns": 1000 * SEC}
    args.update(kw)
    return M.MakerSpec(**args)


def _sim(**kw: Any) -> tuple[M.MakerSim, LocalBook]:
    book = _market()
    sim = M.MakerSim(_spec(**kw), book)
    sim.on_book(BookEvent(0, ()), {})
    sim.decide(0)
    return sim, book


def test_quotes_join_the_touch_behind_what_is_displayed() -> None:
    sim, _ = _sim()
    q = sim.quotes[BUY][0]
    assert (q.price, q.ahead, q.remaining) == (10000, 1000, sim.res.size_lots)


def test_a_trade_consumes_the_queue_before_the_order() -> None:
    sim, _ = _sim()
    q = sim.quotes[BUY][0]
    sim.on_trade(TradeEvent(1, 10000, 400, SELL))
    assert q.ahead == 600 and not sim.res.fills
    sim.on_trade(TradeEvent(2, 10000, 600 + 100, SELL))
    assert sim.res.fills[0].qty == 100 and sim.res.position == 100


def test_a_trade_through_fills_the_whole_order() -> None:
    sim, _ = _sim()
    sim.on_trade(TradeEvent(1, 9999, 1, SELL))
    assert sim.res.position == sim.res.size_lots
    assert not sim.quotes[BUY]


def test_cancels_move_the_order_up_in_proportion() -> None:
    sim, _ = _sim()
    q = sim.quotes[BUY][0]
    sim.on_book(BookEvent(1, ((BUY, 10000, 1000, 500),)), {})
    assert q.ahead == 500  # all of the level is ahead of it


def test_a_decrease_explained_by_trades_is_not_a_cancel() -> None:
    sim, _ = _sim()
    q = sim.quotes[BUY][0]
    sim.on_book(BookEvent(1, ((BUY, 10000, 1000, 600),)), {(BUY, 10000): 400})
    assert q.ahead == 1000


def test_the_pessimistic_queue_never_advances_on_cancels() -> None:
    sim, _ = _sim(cancel_ahead=0.0)
    q = sim.quotes[BUY][0]
    sim.on_book(BookEvent(1, ((BUY, 10000, 1000, 500),)), {})
    assert q.ahead == 1000


def test_latency_keeps_an_order_out_until_it_arrives() -> None:
    sim, _ = _sim(latency_ns=100)
    sim.on_trade(TradeEvent(50, 9999, 1, SELL))
    assert sim.res.position == 0
    sim.on_trade(TradeEvent(150, 9999, 1, SELL))
    assert sim.res.position == sim.res.size_lots


def test_a_cancel_races_the_fill_for_its_latency() -> None:
    sim, book = _sim(latency_ns=100)
    sim.on_trade(TradeEvent(100, 10000, 0, SELL))
    book.bids.set(10000, 0)
    book.bids.set(9998, 1000)
    sim.decide(200)
    assert sim.quotes[BUY][0].dead_at == 300
    sim.on_trade(TradeEvent(250, 9990, 1, SELL))  # still live: filled at the old price
    assert sim.res.position == sim.res.size_lots


def test_the_same_price_keeps_its_place() -> None:
    sim, _ = _sim()
    q = sim.quotes[BUY][0]
    q.ahead = 7
    sim.decide(10)
    assert sim.quotes[BUY] == [q] and q.ahead == 7


def test_position_limit_stops_a_side() -> None:
    sim, _ = _sim(max_position_mult=1)
    sim.on_trade(TradeEvent(1, 9999, 1, SELL))
    sim.decide(2)
    assert not [q for q in sim.quotes[BUY] if q.dead_at is None]
    assert [q for q in sim.quotes[SELL] if q.dead_at is None]


def test_identity_edge_plus_inventory_is_exact() -> None:
    sim, book = _sim()
    sim.on_trade(TradeEvent(1, 9999, 1, SELL))  # bought at 100.00, mid 100.01
    book.bids.set(10010, 1000)
    book.asks.set(10012, 1000)
    book.bids.set(10000, 0)
    book.asks.set(10002, 0)
    sim.on_book(BookEvent(2, ()), {})
    res = sim.res
    res.check_identity()
    size = res.size_lots
    assert res.edge == size * 2  # half a spread in half-ticks x lots
    assert res.inventory == size * (20022 - 20002)
    assert res.pnl == res.edge + res.inventory


def test_fees_and_breakeven() -> None:
    res = M.MakerResult(_spec())
    res.cash2 = 2_000
    res.traded_notional = 10_000_000
    assert res.breakeven_fee_bps() == pytest.approx(1000 / 10_000_000 * 1e4)
    gross = 2_000 * 0.01 * 0.001 / 2
    assert res.net(0.0, 0.01, 0.001) == pytest.approx(gross)
    assert res.net(-0.5, 0.01, 0.001) > gross
    assert res.net(10.0, 0.01, 0.001) == pytest.approx(gross - 1000 * 1e-3 * 0.01 * 10_000 / 1000)


def test_markouts_measure_the_move_against_the_maker() -> None:
    sim, book = _sim()
    sim.on_trade(TradeEvent(0, 9999, 1, SELL))  # bought at mid 100.01
    book.bids.set(9990, 1000)
    book.asks.set(9992, 1000)
    book.bids.set(10000, 0)
    book.asks.set(10002, 0)
    sim.on_book(BookEvent(2 * SEC, ()), {})
    rows = sim.res.markouts[1 * SEC]
    assert rows and rows[0][0] == pytest.approx((19982 - 20002) / 20002 * 1e4)
    assert not sim.res.markouts[10 * SEC]


def test_skews_lean_the_right_way() -> None:
    book = _market()
    assert M.target_quotes(M.MakerSpec("symmetric"), book, 0.0, 0.0) == (10000, 10002)
    long_book = M.target_quotes(M.MakerSpec("stoikov", gamma=2.0), book, 5.0, 1.0)
    assert long_book is not None and long_book[0] < 10000  # long: less keen to buy
    book.bids.set(10000, 9000)
    leaning = M.target_quotes(M.MakerSpec("imbalance", imbalance_bps=5.0), book, 0.0, 0.0)
    assert leaning is not None and leaning[1] > 10002  # bids heavy: expects up, asks higher


def test_a_quote_never_crosses_the_other_touch() -> None:
    book = _market()
    q = M.target_quotes(M.MakerSpec("imbalance", imbalance_bps=50.0), book, 0.0, 0.0)
    book2 = _market()
    book2.bids.set(10000, 9000)
    q = M.target_quotes(M.MakerSpec("imbalance", imbalance_bps=50.0), book2, 0.0, 0.0)
    assert q is not None and q[0] < 10002 and q[1] > 10000


def test_run_makers_gives_each_the_same_market_and_checks_the_identity() -> None:
    book = _market()
    events: list[BookEvent | TradeEvent] = [
        BookEvent(0, ()),
        TradeEvent(10, 9999, 5, SELL),
        BookEvent(SEC, ()),
    ]
    a, b = M.run_makers([_spec(), _spec(cancel_ahead=0.0)], events, book)
    assert a.traded_lots == b.traded_lots == a.size_lots


def test_a_break_closes_the_stretch_and_the_next_starts_flat() -> None:
    """The market moves a lot while nothing is recorded: that move is nobody's PnL."""
    book = _market()

    def events() -> Any:
        yield BookEvent(0, ())
        yield TradeEvent(10, 9999, 5, SELL)  # the maker buys
        yield SegmentBreak()
        book.bids.set(20000, 1000)  # back at twice the price
        book.asks.set(20002, 1000)
        book.bids.set(10000, 0)
        book.asks.set(10002, 0)
        yield BookEvent(SEC, ())

    res = M.run_makers([_spec()], events(), book)[0]
    res.check_identity()
    assert res.inventory == 0  # the jump is not inventory PnL
    assert res.position == 0  # flat across the break
    assert res.edge == res.traded_lots * 2


def test_merge_adds_stretches_and_keeps_hours() -> None:
    spec = _spec()
    a, b = M.MakerResult(spec), M.MakerResult(spec)
    for r in (a, b):
        for h in M.MARKOUTS_NS:
            r.markouts[h] = []
    a.cash2, a.edge, a.traded_notional, a.hours, a.blocks = 10, 10, 100, 1, [(4, 40)]
    b.cash2, b.edge, b.traded_notional, b.hours, b.blocks = 6, 6, 50, 1, [(2, 20)]
    m = M.merge_results(spec, [a, b])
    assert (m.pnl, m.traded_notional, m.hours) == (16, 150, 2)
    assert m.blocks == [(4, 40), (12, 120)]


def test_hour_blocks_do_not_pile_up_after_a_quiet_spell() -> None:
    sim, _ = _sim()
    sim._mark_mid(1)
    sim._mark_mid(1 + 4 * M.BLOCK_NS)
    sim._mark_mid(2 + 4 * M.BLOCK_NS)
    assert len(sim.res.blocks) == 1


# ---------------------------------------------------------------------------- the study
def test_trials_are_four_and_registered_values_are_fixed() -> None:
    assert [t.name for t in R.TRIALS] == ["touch", "wide_2bp", "stoikov", "imbalance"]
    v = R.variants(R.TRIALS[0].spec)
    assert v["latency"].latency_ns == R.LATENCY_STRESS_NS and v["queue"].cancel_ahead == 0.0


def test_a_short_recording_cannot_promote() -> None:
    good = list(np.full(100, 1.0) + np.random.default_rng(0).normal(0, 0.1, 100))
    out = R.evaluate_trial({"base": good, "latency": good, "queue": good}, 4, 0.0)
    assert not out["promoted"]
    assert any("sample" in f for f in out["failed"])


def test_gate_passes_only_a_large_positive_sample_and_needs_the_stresses() -> None:
    rng = np.random.default_rng(0)
    good = list(1.0 + rng.normal(0, 0.5, R.MIN_HOURS * 2))
    bad = list(rng.normal(-0.5, 0.5, R.MIN_HOURS * 2))
    assert R.evaluate_trial({"base": good, "latency": good, "queue": good}, 4, 0.01)["promoted"]
    out = R.evaluate_trial({"base": good, "latency": bad, "queue": good}, 4, 0.01)
    assert out["failed"] == ["latency_stress"]


# ---------------------------------------------------------------------------- impact
def test_tick_rule() -> None:
    assert list(I.tick_rule([10, 11, 11, 10, 10, 12])) == [0, 1, 1, -1, -1, 1]
    assert list(I.tick_rule([10, 10, 10])) == [0, 0, 0]


def test_bucketise_signs_dollars_and_needs_two_trades() -> None:
    b = I.bucketise([0, 1, 10, 20, 21], [10, 11, 50, 10, 10], [1, 1, 1, 2, 2], [1, -1, 1, 1, 1], 10)
    assert b.signed_dollars[0] == pytest.approx(10 - 11)
    assert b.returns[0] == pytest.approx(math.log(1.1))
    assert b.signed_dollars.size == 2  # the lone trade at t=10 has no return


def test_square_root_fit_recovers_a_planted_coefficient() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(0, 1e6, 5000)
    sigma, adv, y = 0.02, 1e9, 0.7
    r = y * sigma * np.sign(q) * np.sqrt(np.abs(q) / adv) + rng.normal(0, 1e-4, q.size)
    fit = I.fit_square_root(q, r, sigma, adv)
    assert fit.y == pytest.approx(0.7, abs=0.02)
    assert fit.t > 10


def test_square_root_fit_refuses_too_little() -> None:
    assert math.isnan(I.fit_square_root([1.0], [0.1], 0.02, 1e9).y)


def test_sweep_walks_the_book() -> None:
    levels = [(100.0, 1.0), (101.0, 1.0)]
    assert I.sweep_cost_bps(levels, 50.0) == pytest.approx(0.0)
    # 100 dollars at 100, 101 dollars at 101: 201 dollars for 2 units
    assert I.sweep_cost_bps(levels, 201.0) == pytest.approx(50.0)
    assert math.isnan(I.sweep_cost_bps(levels, 1e6))


def test_model_cost_is_what_the_cost_model_charges() -> None:
    m = CostModel()
    got = I.model_cost_bps(1e6, 1e-4, 0.02, 1e9, m)
    assert got == pytest.approx(1.0 + m.impact_y * 0.02 * math.sqrt(1e-3) * 1e4)
