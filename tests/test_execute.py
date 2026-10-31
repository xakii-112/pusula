"""Mirroring a book in a broker account: the plan, its limits and the locks on a live account."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from pusula import cli
from pusula.execution.broker import AlpacaPaperBroker
from pusula.execution.orders import OrderState
from pusula.live import execute as X

NOW = dt.datetime(2026, 10, 30, 12, tzinfo=dt.UTC)
ARMED = {
    "PUSULA_LIVE_ARM": "f/b",
    "PUSULA_LIVE_MAX_CAPITAL": "1000",
    "PUSULA_ALPACA_LIVE_KEY": "k",
    "PUSULA_ALPACA_LIVE_SECRET": "s",
}


def _plan(**kw: object) -> X.Plan:
    args: dict[str, object] = {
        "family": "f",
        "book": "b",
        "session": "2026-10-29",
        "target": {"AAA@2016-01-01": 1000, "BBB@2016-01-01": -500},
        "prices": {"AAA@2016-01-01": 10.0, "BBB@2016-01-01": 20.0},
        "book_equity": 100_000.0,
        "account_equity": 10_000.0,
        "held": {},
    }
    args.update(kw)
    return X.make_plan(**args)  # type: ignore[arg-type]


def test_plan_scales_the_book_to_the_account_towards_zero() -> None:
    p = _plan(target={"AAA@2016-01-01": 1009}, prices={"AAA@2016-01-01": 10.0})
    assert p.scale == pytest.approx(0.1)
    assert [(o.symbol, o.qty) for o in p.orders] == [("AAA", 100)]  # 100.9 rounds towards zero


def test_plan_trades_only_the_difference_and_closes_what_the_book_left() -> None:
    p = _plan(held={"AAA": 60, "ZZZ": 5})
    q = {o.symbol: o.qty for o in p.orders}
    assert q == {"AAA": 40, "BBB": -50, "ZZZ": -5}


def test_client_ids_are_stable_across_runs() -> None:
    assert [o.client_id for o in _plan().orders] == [o.client_id for o in _plan().orders]


def test_plan_refuses_a_target_it_cannot_price() -> None:
    with pytest.raises(X.ExecutionRefusedError, match="no price"):
        _plan(prices={"AAA@2016-01-01": 10.0})


def test_plan_limits_stop_a_wrong_scale() -> None:
    p = _plan(target={"AAA@2016-01-01": 5000}, book_equity=10_000.0)
    with pytest.raises(X.ExecutionRefusedError, match="gross"):
        X.check_plan(p)
    big = _plan(target={"AAA@2016-01-01": 500}, book_equity=10_000.0)
    with pytest.raises(X.ExecutionRefusedError, match="of the account"):
        X.check_plan(big)


def test_a_position_the_book_cannot_price_is_refused() -> None:
    with pytest.raises(X.ExecutionRefusedError, match="not priced"):
        X.check_plan(_plan(held={"ZZZ": 5}))


@pytest.mark.parametrize(
    ("snap", "why"),
    [
        ({"as_of": "2026-10-29", "books": {}}, "no open"),
        ({"as_of": "2026-10-29", "books": {"b": {"halted": True}}}, "halted"),
        ({"as_of": "2026-10-28", "books": {"b": {}}}, "pusula forward"),
        ({"as_of": "2026-10-29", "rehearsal": True, "books": {"b": {}}}, "rehearsal"),
    ],
)
def test_a_book_with_nothing_to_mirror_is_refused(snap: dict[str, object], why: str) -> None:
    with pytest.raises(X.ExecutionRefusedError, match=why):
        X.check_book(snap, "b", "2026-10-29")


def test_live_needs_every_condition() -> None:
    passed = {"status": "passes"}
    assert X.live_refusals(True, passed, True, "f", "b", ARMED) == []
    assert any("promotion" in r for r in X.live_refusals(False, passed, True, "f", "b", ARMED))
    assert any(
        "forward" in r for r in X.live_refusals(True, {"status": "fails"}, True, "f", "b", ARMED)
    )
    assert any("forward" in r for r in X.live_refusals(True, None, True, "f", "b", ARMED))
    assert any("stresses" in r for r in X.live_refusals(True, passed, False, "f", "b", ARMED))
    other = {**ARMED, "PUSULA_LIVE_ARM": "f/other"}
    assert any("ARM" in r for r in X.live_refusals(True, passed, True, "f", "b", other))
    assert any(
        "MAX_CAPITAL" in r
        for r in X.live_refusals(
            True, passed, True, "f", "b", {**ARMED, "PUSULA_LIVE_MAX_CAPITAL": "x"}
        )
    )
    assert any(
        "KEY" in r
        for r in X.live_refusals(
            True, passed, True, "f", "b", {"PUSULA_LIVE_ARM": "f/b", "PUSULA_LIVE_MAX_CAPITAL": "9"}
        )
    )


def test_nothing_in_the_ledger_can_go_live_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--live` on a real family stops at the locks, before any broker is built."""
    called: list[str] = []
    monkeypatch.setattr(X, "live_broker", lambda *a: called.append("broker"))
    code = cli.main(["execute", "--family", "us-equity-v2", "--book", "xs_momentum", "--live"])
    assert code == 2 and not called


def test_an_account_mirrors_one_book(tmp_path: Path) -> None:
    (tmp_path / "account.json").write_text(json.dumps({"owner": "f/other"}), encoding="utf-8")
    with pytest.raises(X.ExecutionRefusedError, match="already mirrors"):
        X.check_account(tmp_path, "f", "b", "2026-10-29", {}, False)


def test_an_account_holding_something_else_is_refused(tmp_path: Path) -> None:
    with pytest.raises(X.ExecutionRefusedError, match="did not open"):
        X.check_account(tmp_path, "f", "b", "2026-10-29", {"AAA": 1}, False)
    X.check_account(tmp_path, "f", "b", "2026-10-29", {"AAA": 1}, True)


def test_a_session_is_sent_once(tmp_path: Path) -> None:
    (tmp_path / "account.json").write_text(
        json.dumps({"owner": "f/b", "last_session": "2026-10-29"}), encoding="utf-8"
    )
    with pytest.raises(X.ExecutionRefusedError, match="already sent"):
        X.check_account(tmp_path, "f", "b", "2026-10-29", {}, False)


def _broker(handler: object) -> AlpacaPaperBroker:
    return AlpacaPaperBroker("k", "s", dry_run=False, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_send_plan_sends_and_records(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        assert str(r.url).startswith("https://paper-api.alpaca.markets/v2/orders")
        sent.append(json.loads(r.content))
        return httpx.Response(200, json={"id": "x"})

    orders = X.send_plan(_plan(), _broker(handler), tmp_path, NOW)
    assert all(o.state == OrderState.NEW for o in orders)
    assert {s["symbol"] for s in sent} == {"AAA", "BBB"}
    assert {s["side"] for s in sent} == {"buy", "sell"}
    state = json.loads((tmp_path / "account.json").read_text(encoding="utf-8"))
    assert state["owner"] == "f/b" and state["last_session"] == "2026-10-29"
    assert (tmp_path / "plan-2026-10-29.json").exists()


def test_a_rejected_order_does_not_mark_the_session_done(tmp_path: Path) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "insufficient"})

    orders = X.send_plan(_plan(), _broker(handler), tmp_path, NOW)
    assert all(o.state == OrderState.REJECTED for o in orders)
    assert not (tmp_path / "account.json").exists()


def test_a_resubmitted_order_is_recognised(tmp_path: Path) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="client_order_id must be unique")

    orders = X.send_plan(_plan(), _broker(handler), tmp_path, NOW)
    assert all(o.state == OrderState.NEW for o in orders)


def test_fill_quality_signs_a_cost_positive_on_both_sides() -> None:
    plan = {
        "session": "s",
        "orders": [
            {"symbol": "AAA", "qty": 100, "reference_price": 10.0},
            {"symbol": "BBB", "qty": -100, "reference_price": 10.0},
            {"symbol": "CCC", "qty": 5, "reference_price": 10.0},
        ],
    }
    fills = [
        {"symbol": "AAA", "side": "buy", "qty": "60", "price": "10.10"},
        {"symbol": "AAA", "side": "buy", "qty": "40", "price": "10.10"},
        {"symbol": "BBB", "side": "sell", "qty": "100", "price": "9.90"},
    ]
    q = X.fill_quality(plan, fills)
    assert q["mean_bps"] == pytest.approx(100.0)
    assert q["unfilled"] == 1
    assert q["orders"][0]["filled"] == 100


# ---------------------------------------------------------------------------- all the books
AAA, BBB = "AAA@2016-01-01", "BBB@2016-01-01"
PRICES = {AAA: 10.0, BBB: 20.0, "AAA": 10.0, "BBB": 20.0}


def _combined(*books: X.BookTarget, **kw: object) -> X.Plan:
    args: dict[str, object] = {
        "slots": len(books),
        "session": "2026-10-29",
        "prices": PRICES,
        "account_equity": 100_000.0,
        "held": {},
    }
    args.update(kw)
    return X.make_combined_plan(books, **args)  # type: ignore[arg-type]


def _target(name: str, target: dict[str, int], equity: float = 1_000_000.0) -> X.BookTarget:
    return X.BookTarget(name, target, equity, "2026-10-29")


WHOLE = X.Tradable(fractionable=False, shortable=True)
FRACTIONS = X.Tradable(fractionable=True, shortable=True)
TERMS = {"AAA": FRACTIONS, "BBB": FRACTIONS}


def test_books_share_the_capital_and_net_by_symbol() -> None:
    a = _target("f/a", {AAA: 10_000, BBB: -1_000})
    b = _target("f/b", {AAA: -4_000})
    p = _combined(a, b, stocks=TERMS)
    assert p.scale == pytest.approx(0.5)
    # each has 50,000 of capital against a 1M book: 5% of its shares
    assert {o.symbol: o.qty for o in p.orders} == {"AAA": 300, "BBB": -50}
    assert p.books == 2
    assert p.gross_after == pytest.approx((300 * 10 + 50 * 20) / 100_000)


def test_a_small_account_keeps_its_longs_in_fractions() -> None:
    """Whole shares kept 57.7% of the gross of 28 books in a $100,000 account."""
    a, b = _target("f/a", {AAA: 2_000}), _target("f/b", {AAA: 500})
    p = _combined(a, b, stocks=TERMS)
    assert [o.qty for o in p.orders] == pytest.approx([125.0])
    three = _combined(_target("f/a", {AAA: 7}), stocks=TERMS, prices={AAA: 1000.0, "AAA": 1000.0})
    assert three.orders[0].qty == pytest.approx(0.7)


def test_a_stock_without_fractions_rounds_to_the_nearest_share() -> None:
    p = _combined(
        _target("f/a", {AAA: 7}), stocks={"AAA": WHOLE}, prices={AAA: 1000.0, "AAA": 1000.0}
    )
    assert [o.qty for o in p.orders] == [1.0]


def test_shorts_are_whole_and_only_what_the_broker_lends() -> None:
    assert X._stock_target(-1.4, 0.0, FRACTIONS) == -1.0
    assert X._stock_target(-1.6, 0.0, FRACTIONS) == -2.0
    no_borrow = X.Tradable(fractionable=True, shortable=False)
    assert X._stock_target(-5.0, 0.0, no_borrow) == 0.0
    p = _combined(_target("f/a", {AAA: -10_000}), stocks={"AAA": no_borrow})
    assert not p.orders
    assert any("will not lend" in n for n in p.notes)


def test_a_position_changing_sign_goes_through_zero_first() -> None:
    assert X._stock_target(5.0, -3.0, FRACTIONS) == 0.0
    assert X._stock_target(-5.0, 3.0, FRACTIONS) == 0.0
    assert X._stock_target(-5.0, -3.0, FRACTIONS) == -5.0


def test_an_order_too_small_to_matter_is_not_sent() -> None:
    p = _combined(_target("f/a", {AAA: 2}), stocks=TERMS)  # 0.2 shares, $2 ... the edge
    assert [o.symbol for o in p.orders] == ["AAA"]
    q = _combined(_target("f/a", {AAA: 1}), stocks=TERMS)  # $1
    assert not q.orders


def test_each_book_can_be_given_a_fixed_amount() -> None:
    a = _target("f/a", {AAA: 10_000})  # 1% of a 1M book's shares for 10,000 dollars
    p = _combined(a, _target("f/b", {BBB: 1_000}), capital_per_book=10_000.0, stocks=TERMS)
    assert {o.symbol: o.qty for o in p.orders} == {"AAA": 100, "BBB": 10}
    assert p.scale == pytest.approx(0.1)


def test_an_account_too_small_for_the_amount_is_refused_and_says_what_to_do() -> None:
    a = _target("f/a", {AAA: 10_000})
    with pytest.raises(X.ExecutionRefusedError, match="raise the paper balance"):
        _combined(a, slots=28, capital_per_book=100_000.0, stocks=TERMS)


def test_a_new_record_starts_after_a_reset_and_keeps_the_old(tmp_path: Path) -> None:
    (tmp_path / "account.json").write_text('{"owner": "all/books"}', encoding="utf-8")
    (tmp_path / "plan-2026-10-30.json").write_text("{}", encoding="utf-8")
    moved = X.archive_record(tmp_path, NOW)
    assert moved is not None and (moved / "account.json").exists()
    assert not (tmp_path / "account.json").exists()
    assert X.archive_record(tmp_path, NOW) is None


def test_a_halted_book_keeps_its_share_in_cash() -> None:
    a = _target("f/a", {AAA: 10_000})
    assert _combined(a, slots=2, stocks=TERMS).orders[0].qty == 500  # half the capital


def test_a_net_position_the_books_left_is_closed() -> None:
    p = _combined(_target("f/a", {AAA: 10_000}), held={"AAA": 100, "BBB": 7}, stocks=TERMS)
    assert {o.symbol: o.qty for o in p.orders} == {"AAA": 900, "BBB": -7}


# ---------------------------------------------------------------------------- coins in the account
COIN = X.Tradable(True, False, 0.0001, 0.0001)


def _crypto_book(target: dict[str, int], equity: float = 1_000_000.0) -> X.BookTarget:
    return X.BookTarget("c/x", target, equity, "2026-10-29", True)


def test_coin_names() -> None:
    assert X.coin_of("BTCUSDT") == "BTC/USD"
    assert X.coin_of("LUNAUSDT#2") == "LUNA/USD"


def test_a_crypto_book_is_added_in_dollars_and_bought_at_the_coins_own_price() -> None:
    """Its units are rescaled: 1,000 units at 500 is $500,000, whatever a real coin costs."""
    p = _combined(
        _crypto_book({"BTCUSDT": 1_000}),
        prices={"BTCUSDT": 500.0},
        coins={"BTC/USD": COIN},
        coin_prices={"BTC/USD": 50_000.0},
    )
    o = p.orders[0]
    assert (o.symbol, o.tif) == ("BTC/USD", "gtc")
    assert o.qty == pytest.approx(50_000.0 / 50_000.0)  # 5% of a 1M book's $500,000 at 100k capital


def test_a_coin_the_broker_does_not_list_stays_in_cash_and_is_said() -> None:
    p = _combined(_crypto_book({"XYZUSDT": 1_000}), prices={"XYZUSDT": 500.0})
    assert not p.orders
    assert any("does not list" in n for n in p.notes)
    assert p.captured == 1.0  # nothing could be held of what is listed


def test_a_net_crypto_short_is_left_flat() -> None:
    p = _combined(
        _crypto_book({"BTCUSDT": -1_000}),
        prices={"BTCUSDT": 500.0},
        coins={"BTC/USD": COIN},
        coin_prices={"BTC/USD": 50_000.0},
    )
    assert not p.orders
    assert any("cannot short" in n for n in p.notes)


def test_crypto_books_net_and_stocks_do_not_mix_with_them() -> None:
    p = _combined(
        _crypto_book({"BTCUSDT": 1_000}),
        _target("f/a", {AAA: 10_000}),
        prices={**PRICES, "BTCUSDT": 500.0},
        stocks=TERMS,
        coins={"BTC/USD": COIN},
        coin_prices={"BTC/USD": 50_000.0},
    )
    assert {o.symbol for o in p.orders} == {"AAA", "BTC/USD"}


def test_a_coin_held_and_no_longer_tradable_stops_the_plan() -> None:
    with pytest.raises(X.ExecutionRefusedError, match="no longer tradable"):
        _combined(_target("f/a", {AAA: 10_000}), held={"XYZ/USD": 1.0}, stocks=TERMS)


def test_coin_increment_and_minimum() -> None:
    coarse = X.Tradable(True, False, 0.5, 0.1)
    p = _combined(
        _crypto_book({"BTCUSDT": 1_000}),
        prices={"BTCUSDT": 500.0},
        coins={"BTC/USD": coarse},
        coin_prices={"BTC/USD": 300_000.0},
    )  # $50,000 is 0.1667 coins: 0.1 after the increment, under the minimum order
    assert not p.orders


def test_exact_orders_go_out_with_their_fractions(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        sent.append(json.loads(r.content))
        return httpx.Response(200, json={"id": "x"})

    p = _combined(
        _crypto_book({"BTCUSDT": 1_000}),
        _target("f/a", {AAA: 7}),
        prices={**PRICES, "BTCUSDT": 500.0},
        stocks=TERMS,
        coins={"BTC/USD": COIN},
        coin_prices={"BTC/USD": 50_000.0},
    )
    X.send_plan(p, _broker(handler), tmp_path, NOW)
    by = {s["symbol"]: s for s in sent}
    assert by["AAA"]["qty"] == "0.35" and by["AAA"]["time_in_force"] == "day"
    assert by["BTC/USD"]["time_in_force"] == "gtc"
    state = json.loads((tmp_path / "account.json").read_text(encoding="utf-8"))
    assert state["owner"] == "all/books" and state["start_equity"] == 100_000.0


def test_a_refused_exact_order_is_recorded(tmp_path: Path) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "no"})

    p = _combined(_target("f/a", {AAA: 10_000}), _target("f/b", {BBB: 1_000}), stocks=TERMS)
    orders = X.send_plan(p, _broker(handler), tmp_path, NOW)
    assert all(o.state == OrderState.REJECTED for o in orders)
    plan = json.loads((tmp_path / "plan-2026-10-29.json").read_text(encoding="utf-8"))
    assert set(plan["outcome"].values()) == {"rejected"}


def test_exact_positions_name_coins_as_the_orders_do() -> None:
    rows = [
        {"symbol": "AAA", "qty": "1.5", "asset_class": "us_equity"},
        {"symbol": "BTCUSD", "qty": "0.25", "asset_class": "crypto"},
    ]
    b = _broker(lambda r: httpx.Response(200, json=rows))
    assert b.exact_positions() == {"AAA": 1.5, "BTC/USD": 0.25}
    assert b.positions() == {"AAA": 1}


def test_a_dry_run_sends_no_exact_order() -> None:
    def boom(r: httpx.Request) -> httpx.Response:
        raise AssertionError("sent")

    b = AlpacaPaperBroker("k", "s", dry_run=True, transport=httpx.MockTransport(boom))
    assert b.submit_exact([("i", "AAA", 0.5, "day")]) == {}


def test_fill_quality_reads_fractions_and_the_feeds_spelling_of_a_coin() -> None:
    plan = {"session": "s", "orders": [{"symbol": "BTC/USD", "qty": 0.5, "reference_price": 100.0}]}
    q = X.fill_quality(plan, [{"symbol": "BTCUSD", "side": "buy", "qty": "0.5", "price": "101"}])
    assert q["mean_bps"] == pytest.approx(100.0)


def test_tracking_compares_the_account_with_the_mix(tmp_path: Path) -> None:
    from pusula.engine.journal import Journal

    root = tmp_path / "f" / "a"
    root.mkdir(parents=True)
    j = Journal(root / "journal.jsonl")
    j.append("capital", "2026-10-28", {"micros": 1_000_000_000})
    j.append("mark", "2026-10-28", {"marks": {}})
    j.append("capital", "2026-10-29", {"micros": 100_000_000})
    j.append("mark", "2026-10-29", {"marks": {}})
    book = X.BookTarget("f/a", {}, 1100.0, "2026-10-29")
    state = {"start_equity": 100_000.0, "start_session": "2026-10-28"}
    t = X.mirror_tracking(tmp_path, [book], 2, state, 104_000.0)
    assert t is not None
    assert t["account"] == pytest.approx(0.04)
    assert t["mix"] == pytest.approx(0.05)  # 10% on half the capital
    assert t["gap"] == pytest.approx(-0.01)
    assert X.mirror_tracking(tmp_path, [book], 2, {}, 1.0) is None


def test_combined_plan_refuses_a_security_without_a_price() -> None:
    with pytest.raises(X.ExecutionRefusedError, match="no price"):
        _combined(_target("f/a", {"CCC@2016-01-01": 5}))


def test_combined_client_ids_are_stable() -> None:
    a = _target("f/a", {AAA: 10_000})
    assert [o.client_id for o in _combined(a).orders] == [o.client_id for o in _combined(a).orders]


def _forward(root: Path, family: str, as_of: str, books: dict[str, dict[str, object]]) -> None:
    from pusula.engine.journal import Journal

    fam = root / family
    fam.mkdir(parents=True)
    (fam / "dashboard.json").write_text(
        json.dumps({"as_of": as_of, "books": books}), encoding="utf-8"
    )
    for name in books:
        (fam / name).mkdir()
        j = Journal(fam / name / "journal.jsonl")
        j.append("capital", as_of, {"micros": 1_000_000_000_000})
        j.append(
            "fill",
            as_of,
            {
                "security": AAA,
                "qty": 100,
                "price_micros": 10_000_000,
                "fee_micros": 0,
                "client_id": "c",
            },
        )
        j.append("mark", as_of, {"marks": {AAA: 10_000_000}})
        state = {"pending": [], "journal_head": j.events[-1].hash}
        (fam / name / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_gather_counts_halted_books_as_shares_and_notes_stale_ones(tmp_path: Path) -> None:
    _forward(tmp_path, "f", "2026-10-29", {"a": {}, "h": {"halted": True}})
    _forward(tmp_path, "g", "2026-10-28", {"b": {}})
    books, slots, notes = X.gather_books(tmp_path, ["f", "g", "none"], "2026-10-29")
    assert [b.name for b in books] == ["f/a", "g/b"]
    assert slots == 3
    assert any("f/h: halted" in n for n in notes)
    assert any("g/b: at 2026-10-28" in n for n in notes)
    assert any("none: no forward" in n for n in notes)


def test_gather_refuses_when_nothing_has_reached_the_session(tmp_path: Path) -> None:
    _forward(tmp_path, "f", "2026-10-28", {"a": {}})
    with pytest.raises(X.ExecutionRefusedError, match="pusula forward"):
        X.gather_books(tmp_path, ["f"], "2026-10-29")


def test_all_books_can_never_go_live() -> None:
    assert cli.main(["execute", "--all", "--live", "--send"]) == 2


def test_a_book_must_be_named_without_all() -> None:
    assert cli.main(["execute"]) == 2


def test_a_partly_refused_plan_keeps_the_owner_and_can_be_retried(tmp_path: Path) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if json.loads(r.content)["symbol"] == "BBB":
            return httpx.Response(403, json={"message": "not shortable"})
        return httpx.Response(200, json={"id": "x"})

    X.send_plan(_plan(), _broker(handler), tmp_path, NOW)
    state = json.loads((tmp_path / "account.json").read_text(encoding="utf-8"))
    assert state["owner"] == "f/b" and state["last_session"] is None
    X.check_account(tmp_path, "f", "b", "2026-10-29", {"AAA": 100}, False)  # may run again


def test_a_throttled_order_is_waited_for_not_refused() -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"id": "x"})

    b = _broker(handler)
    b.retry_wait = 0.0
    from pusula.execution.orders import Order

    o = Order("id", "AAA", 1, "s")
    b.submit([o])
    assert o.state == OrderState.NEW and calls["n"] == 3


def test_a_broker_that_stays_down_ends_in_a_refusal_without_a_secret() -> None:
    from pusula.execution.orders import Order

    b = _broker(lambda r: httpx.Response(503))
    b.retry_wait = 0.0
    o = Order("id", "AAA", 1, "s")
    b.submit([o])
    assert o.state == OrderState.REJECTED and "HTTP 503" in o.history[-1]


def test_last_fills_are_reported_once(tmp_path: Path) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/v2/account/activities/FILL":
            if r.url.params["date"] == "2026-10-30":
                return httpx.Response(
                    200, json=[{"symbol": "AAA", "side": "buy", "qty": "50", "price": "10.10"}]
                )
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"id": "x"})

    b = _broker(handler)
    X.send_plan(_plan(), b, tmp_path, NOW)
    q = X.report_last_fills(tmp_path, b, dt.date(2026, 10, 31))
    assert q is not None and q["mean_bps"] == pytest.approx(100.0)
    assert (tmp_path / "fills-2026-10-29.json").exists()
    assert X.report_last_fills(tmp_path, b, dt.date(2026, 10, 31)) is None


def test_no_fills_yet_is_not_a_report(tmp_path: Path) -> None:
    b = _broker(
        lambda r: (
            httpx.Response(200, json=[])
            if "activities" in r.url.path
            else httpx.Response(200, json={"id": "x"})
        )
    )
    X.send_plan(_plan(), b, tmp_path, NOW)
    assert X.report_last_fills(tmp_path, b, dt.date(2026, 10, 30)) is None
    assert not (tmp_path / "fills-2026-10-29.json").exists()


def test_health_warns_of_a_mirror_that_could_not_plan(tmp_path: Path) -> None:
    from pusula.live.health import check_mirror

    (tmp_path / "refused.json").write_text('{"why": "account too small"}', encoding="utf-8")
    assert "account too small" in check_mirror(tmp_path)[0].message


def test_health_warns_of_refused_orders(tmp_path: Path) -> None:
    from pusula.live.health import check_mirror

    assert check_mirror(tmp_path) == []
    (tmp_path / "plan-2026-10-29.json").write_text(
        json.dumps({"session": "2026-10-29", "outcome": {"a": "new", "b": "rejected"}}),
        encoding="utf-8",
    )
    alerts = check_mirror(tmp_path)
    assert len(alerts) == 1 and "refused 1 of 2" in alerts[0].message


def test_broker_reads_account_and_fills() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/v2/account":
            return httpx.Response(200, json={"equity": "123.5"})
        if r.url.path == "/v2/account/activities/FILL":
            assert r.url.params["date"] == "2026-10-30"
            return httpx.Response(200, json=[{"symbol": "AAA"}])
        return httpx.Response(404)

    b = _broker(handler)
    assert b.equity() == 123.5
    assert b.fills("2026-10-30") == [{"symbol": "AAA"}]
