"""Mirror one forward book in a broker account: real orders, with the record kept honest.

The book stays the record. Its journal is a simulation at model prices, and this never writes to it.
What it does: take the book's latest target, scale it to the account's equity, compare with what the
account holds, and send the difference as day market orders with deterministic client ids. A re-run
is not a second order. Each run first reports how the last orders filled against the prices the book
assumed, which on a real account is the only measurement of the cost model that exists; on a paper
account fills are simulated and say little about impact.

Paper needs ``--send``. A live account is refused by :func:`live_refusals` unless everything holds:

* the trial passed the promotion gate in the ledger;
* its forward record passed the same gate on its own, stresses included (RESEARCH_PROTOCOL §4);
* the operator armed exactly this book: ``PUSULA_LIVE_ARM`` holds ``family/book``;
* a capital limit is set (``PUSULA_LIVE_MAX_CAPITAL``) and separate live keys exist.

No flag bypasses these. An account mirrors one book, or all of them together: ``--all`` gives every
open book an equal share of the capital, adds their targets symbol by symbol and sends the net.
Longs and shorts of different books cancel, so the account is then a test of the plumbing and of
the mix as a whole, and no longer the record of any one strategy. It is for paper only.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pusula.core.errors import PusulaError
from pusula.core.money import to_dollars
from pusula.data.settings import atomic_write_text, environment, home
from pusula.engine.journal import Journal
from pusula.execution.broker import AlpacaPaperBroker
from pusula.execution.orders import Order, OrderState, client_order_id

LIVE_URL = "https://api.alpaca.markets"
MAX_GROSS = 2.05  # Reg T overnight, and a little room for rounding
MAX_ORDER_SHARE = 0.35  # of equity, one order: a wrong scale must not reach the market
MIN_ORDER_DOLLARS = 2.0  # the broker's fractional minimum is $1; below this it is only churn


class ExecutionRefusedError(PusulaError):
    """The plan was not sent. The message says why."""


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    symbol: str
    security: str
    qty: float  # whole shares for a single book; fractions, and coins, in the combined plan
    reference_price: float
    client_id: str
    tif: str = "day"


@dataclass(frozen=True, slots=True)
class Plan:
    family: str
    book: str
    session: str
    scale: float
    account_equity: float
    book_equity: float
    gross_after: float
    orders: tuple[PlannedOrder, ...]
    live: bool = False
    books: int = 1
    captured: float = 1.0  # of the target's gross, after rounding to whole shares
    notes: tuple[str, ...] = ()


def symbol_of(security: str) -> str:
    return security.split("@", maxsplit=1)[0]


def book_target(journal: Journal, state: Mapping[str, Any]) -> dict[str, int]:
    """What the book wants to hold next: its latest undone decision, else what it holds."""
    pending = state.get("pending") or []
    if pending:
        return {str(k): int(v) for k, v in pending[-1][1].items() if int(v) != 0}
    ledger, _ = journal.replay()
    return {s: int(p.qty) for s, p in ledger.positions.items() if p.qty != 0}


def make_plan(
    family: str,
    book: str,
    session: str,
    target: Mapping[str, int],
    prices: Mapping[str, float],
    book_equity: float,
    account_equity: float,
    held: Mapping[str, int],
    capital_fraction: float = 1.0,
    live: bool = False,
) -> Plan:
    """Orders that take ``held`` to the target scaled to the account. Pure, so it can be tested."""
    if not 0.0 < capital_fraction <= 1.0:
        raise ExecutionRefusedError(f"capital fraction {capital_fraction} outside (0, 1]")
    if book_equity <= 0 or account_equity <= 0:
        raise ExecutionRefusedError("no equity to scale to")
    scale = capital_fraction * account_equity / book_equity
    wanted: dict[str, tuple[str, int]] = {}
    for security, qty in target.items():
        price = prices.get(security, math.nan)
        if not (math.isfinite(price) and price > 0):
            raise ExecutionRefusedError(f"{security}: no price to size it")
        shares = int(qty * scale)  # towards zero: never more than the scaled book
        if shares != 0:
            wanted[symbol_of(security)] = (security, shares)
    by_symbol = {symbol_of(s): s for s in target}
    orders: list[PlannedOrder] = []
    for symbol in sorted(set(wanted) | set(held)):
        security, shares = wanted.get(symbol, (by_symbol.get(symbol, symbol), 0))
        dq = shares - int(held.get(symbol, 0))
        if dq == 0:
            continue
        price = prices.get(security, prices.get(symbol, math.nan))
        orders.append(
            PlannedOrder(
                symbol,
                security,
                dq,
                float(price),
                client_order_id(f"{family}/{book}", session, security),
            )
        )
    gross = sum(abs(q) * prices[s] for s, q in wanted.values()) / account_equity
    return Plan(
        family,
        book,
        session,
        scale,
        account_equity,
        book_equity,
        gross,
        tuple(orders),
        live,
    )


@dataclass(frozen=True, slots=True)
class BookTarget:
    name: str  # family/book
    target: Mapping[str, int]
    equity: float
    as_of: str
    crypto: bool = False


COMBINED = ("all", "books")  # what the account's owner is called when it mirrors them together


def gather_books(
    forward: Path, families: Iterable[str], session: str, crypto: Iterable[str] = ()
) -> tuple[list[BookTarget], int, list[str]]:
    """Every open book of ``families``, the number of capital shares, and what to know about them.
    A halted book keeps its share in cash, so that it does not lever the others up. A book behind
    the last session (its family sat a night out) keeps its last target until it catches up."""
    books: list[BookTarget] = []
    slots = 0
    notes: list[str] = []
    crypto = set(crypto)
    for family in families:
        path = forward / family / "dashboard.json"
        if not path.exists():
            notes.append(f"{family}: no forward snapshot, left out")
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("rehearsal"):
            raise ExecutionRefusedError(f"{family}: this snapshot is a rehearsal")
        as_of = str(snapshot.get("as_of"))
        for name, entry in sorted(snapshot.get("books", {}).items()):
            slots += 1
            label = f"{family}/{name}"
            if entry.get("halted"):
                notes.append(f"{label}: halted, its share stays in cash")
                continue
            root = forward / family / name
            journal = Journal(root / "journal.jsonl")
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            books.append(
                BookTarget(
                    label,
                    book_target(journal, state),
                    book_equity_of(journal),
                    as_of,
                    family in crypto,
                )
            )
            if as_of != session:
                notes.append(f"{label}: at {as_of}, holding its last target")
    if not any(b.as_of == session for b in books):
        raise ExecutionRefusedError(f"no book has reached {session}: run `pusula forward` first")
    return books, slots, notes


@dataclass(frozen=True, slots=True)
class Tradable:
    """What the broker says about one asset. A stock absent from the table is whole shares only."""

    fractionable: bool = False
    shortable: bool = False
    min_size: float = 0.0
    increment: float = 0.0


def coin_of(security: str) -> str:
    """``BTCUSDT`` and a relisted ``LUNAUSDT#2`` as the broker's ``BTC/USD`` and ``LUNA/USD``."""
    return security.split("#", maxsplit=1)[0].removesuffix("USDT") + "/USD"


def _stock_target(exact: float, held: float, asset: Tradable) -> float:
    """Where a stock can go tonight. Longs in fractions if the broker allows it. Shorts are whole
    shares, the nearest, and only in what it will lend. A position changing sign goes through zero
    first: the broker refuses one order that does both, so the far side waits for the next night."""
    if exact > 0:
        target = math.floor(exact * 1e6) / 1e6 if asset.fractionable else float(round(exact))
    elif asset.shortable:
        target = float(round(exact))
    else:
        target = 0.0
    return 0.0 if target * held < 0 else target


def make_combined_plan(
    books: Sequence[BookTarget],
    slots: int,
    session: str,
    prices: Mapping[str, float],
    account_equity: float,
    held: Mapping[str, float],
    capital_fraction: float = 1.0,
    notes: Sequence[str] = (),
    stocks: Mapping[str, Tradable] | None = None,
    coins: Mapping[str, Tradable] | None = None,
    coin_prices: Mapping[str, float] | None = None,
    capital_per_book: float | None = None,
) -> Plan:
    """The net of every book's target, each scaled to its share of the account. A crypto book's
    units are rescaled (``data.crypto``), so it is added up in dollars and bought at the coin's own
    price. What the broker cannot hold stays in cash, and the plan says how much."""
    if not 0.0 < capital_fraction <= 1.0:
        raise ExecutionRefusedError(f"capital fraction {capital_fraction} outside (0, 1]")
    if slots <= 0 or account_equity <= 0:
        raise ExecutionRefusedError("no books or no equity to scale to")
    share = capital_fraction / slots
    if capital_per_book is not None:
        needed = capital_per_book * slots
        if needed > account_equity:
            raise ExecutionRefusedError(
                f"{slots} books at {capital_per_book:,.0f} each need {needed:,.0f}, the account "
                f"has {account_equity:,.0f}: raise the paper balance in Alpaca's dashboard"
            )
        share = capital_per_book / account_equity
    exact: dict[str, float] = {}
    price_of: dict[str, float] = {}
    coin_dollars: dict[str, float] = {}
    for b in books:
        if b.equity <= 0:
            raise ExecutionRefusedError(f"{b.name}: no equity")
        scale = share * account_equity / b.equity
        for security, qty in b.target.items():
            price = prices.get(security, math.nan)
            if not (math.isfinite(price) and price > 0):
                raise ExecutionRefusedError(f"{security} of {b.name}: no price to size it")
            if b.crypto:
                coin = coin_of(security)
                coin_dollars[coin] = coin_dollars.get(coin, 0.0) + qty * scale * price
            else:
                symbol = symbol_of(security)
                exact[symbol] = exact.get(symbol, 0.0) + qty * scale
                price_of[symbol] = price
    stocks = stocks or {}
    coins = coins or {}
    coin_prices = coin_prices or {}
    all_notes = list(notes)
    orders: list[PlannedOrder] = []
    gross = gross_exact = 0.0
    refused_shorts = 0
    for symbol in sorted(set(exact) | {h for h in held if "/" not in h}):
        asset = stocks.get(symbol, Tradable())
        now = float(held.get(symbol, 0.0))
        want = exact.get(symbol, 0.0)
        target = _stock_target(want, now, asset)
        refused_shorts += want < 0 and not asset.shortable
        price = price_of.get(symbol, prices.get(symbol, math.nan))
        gross_exact += abs(want) * (price if math.isfinite(price) else 0.0)
        gross += abs(target) * (price if math.isfinite(price) else 0.0)
        dq = target - now
        if abs(dq) * (price if math.isfinite(price) else 1e9) < MIN_ORDER_DOLLARS:
            continue
        orders.append(
            PlannedOrder(
                symbol,
                symbol,
                dq,
                float(price),
                client_order_id("/".join(COMBINED), session, symbol),
            )
        )
    if refused_shorts:
        all_notes.append(f"{refused_shorts} shorts the broker will not lend: left flat")
    missing_dollars = clipped_dollars = 0.0
    for coin in sorted(set(coin_dollars) | {h for h in held if "/" in h}):
        dollars = coin_dollars.get(coin, 0.0)
        terms = coins.get(coin)
        price = coin_prices.get(coin, math.nan)
        now = float(held.get(coin, 0.0))
        if terms is None or not (math.isfinite(price) and price > 0):
            missing_dollars += abs(dollars)
            if not now:
                continue
            raise ExecutionRefusedError(f"{coin}: held by the account and no longer tradable")
        if dollars < 0:  # spot crypto cannot be short here
            clipped_dollars -= dollars
        target = max(dollars, 0.0) / price
        if terms.increment > 0:
            target = math.floor(target / terms.increment) * terms.increment
        gross_exact += abs(dollars)
        gross += target * price
        dq = target - now
        if abs(dq) < terms.min_size or abs(dq) * price < MIN_ORDER_DOLLARS:
            continue
        orders.append(
            PlannedOrder(
                coin,
                coin,
                dq,
                price,
                client_order_id("/".join(COMBINED), session, coin),
                "gtc",
            )
        )
    if missing_dollars:
        all_notes.append(f"{missing_dollars:,.0f} of crypto the broker does not list: left in cash")
    if clipped_dollars:
        all_notes.append(
            f"{clipped_dollars:,.0f} of net crypto shorts: left flat, spot cannot short"
        )
    return Plan(
        COMBINED[0],
        COMBINED[1],
        session,
        share,
        account_equity,
        sum(b.equity for b in books) / max(len(books), 1),
        gross / account_equity,
        tuple(orders),
        False,
        len(books),
        gross / gross_exact if gross_exact > 0 else 1.0,
        tuple(all_notes),
    )


def check_plan(plan: Plan) -> None:
    """Limits of the plan itself, independent of how it was computed."""
    if plan.gross_after > MAX_GROSS:
        raise ExecutionRefusedError(
            f"gross after the plan {plan.gross_after:.2f} above {MAX_GROSS}"
        )
    for o in plan.orders:
        if not math.isfinite(o.reference_price):
            raise ExecutionRefusedError(
                f"{o.symbol}: held by the account and not priced by the book"
            )
        share = abs(o.qty) * o.reference_price / plan.account_equity
        if share > MAX_ORDER_SHARE:
            raise ExecutionRefusedError(
                f"{o.symbol}: one order is {share:.0%} of the account's equity"
            )


def check_book(snapshot: Mapping[str, Any], book: str, expected_session: str) -> None:
    """A halted or stale book has nothing to mirror."""
    entry = snapshot.get("books", {}).get(book)
    if entry is None:
        raise ExecutionRefusedError(f"{book}: no open forward book in the snapshot")
    if entry.get("halted"):
        raise ExecutionRefusedError(f"{book}: halted ({entry.get('kill_reason')})")
    if snapshot.get("rehearsal"):
        raise ExecutionRefusedError("this snapshot is a rehearsal")
    if str(snapshot.get("as_of")) != expected_session:
        raise ExecutionRefusedError(
            f"snapshot at {snapshot.get('as_of')}, last completed session {expected_session}: "
            "run `pusula forward` first"
        )


def live_refusals(
    promoted: bool,
    review_book: Mapping[str, Any] | None,
    stress_run: bool,
    family: str,
    book: str,
    env: Mapping[str, str],
) -> list[str]:
    """Why a live account may not mirror this book; empty only if every condition holds."""
    out: list[str] = []
    if not promoted:
        out.append("the trial did not pass the promotion gate in the ledger")
    status = (review_book or {}).get("status")
    if status != "passes":
        out.append(f"its forward record has not passed the gate (status: {status})")
    if not stress_run:
        out.append("the forward review ran without its stresses (`pusula review --stress`)")
    if env.get("PUSULA_LIVE_ARM") != f"{family}/{book}":
        out.append(f"PUSULA_LIVE_ARM is not {family}/{book}")
    limit = env.get("PUSULA_LIVE_MAX_CAPITAL", "")
    if not limit.replace(".", "", 1).isdigit() or float(limit) <= 0:
        out.append("PUSULA_LIVE_MAX_CAPITAL is not set")
    if not (env.get("PUSULA_ALPACA_LIVE_KEY") and env.get("PUSULA_ALPACA_LIVE_SECRET")):
        out.append("PUSULA_ALPACA_LIVE_KEY / PUSULA_ALPACA_LIVE_SECRET are not set")
    return out


def state_dir(root: Path | None = None) -> Path:
    return (root or home()) / "execution"


def _load_state(root: Path) -> dict[str, Any]:
    path = root / "account.json"
    if not path.exists():
        return {}
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return state


def check_account(
    root: Path, family: str, book: str, session: str, held: Mapping[str, float], adopt: bool
) -> None:
    """An account mirrors one book, and starts flat or with what the last run left there."""
    state = _load_state(root)
    owner = state.get("owner")
    if owner and owner != f"{family}/{book}":
        raise ExecutionRefusedError(f"this account already mirrors {owner}")
    if not owner and any(held.values()) and not adopt:
        raise ExecutionRefusedError(
            f"the account holds {len(held)} positions this program did not open; "
            "flatten it or pass --adopt"
        )
    if state.get("last_session") == session:
        raise ExecutionRefusedError(f"orders for {session} were already sent")


def send_plan(plan: Plan, broker: AlpacaPaperBroker, root: Path, now: dt.datetime) -> list[Order]:
    """Submit, then record what happened to each order. Written before and after, so that a crash
    in between still leaves the plan on file and its deterministic ids make a retry harmless."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"plan-{plan.session}.json"
    atomic_write_text(path, json.dumps(asdict(plan), indent=1))
    orders = [
        Order(o.client_id, o.symbol, int(o.qty), plan.session)
        for o in plan.orders
        if o.qty != 0 and float(o.qty).is_integer() and o.tif == "day" and plan.books == 1
    ]
    broker.submit(orders)
    outcome = {o.client_id: str(o.state) for o in orders}
    if plan.books != 1:
        problems = broker.submit_exact([(o.client_id, o.symbol, o.qty, o.tif) for o in plan.orders])
        outcome = {
            o.client_id: "rejected" if o.client_id in problems else "new" for o in plan.orders
        }
        orders = [Order(o.client_id, o.symbol, 1, plan.session) for o in plan.orders]
        for o in orders:
            if o.client_id in problems:
                o.transition(OrderState.REJECTED, problems[o.client_id])
            else:
                o.transition(OrderState.NEW, "accepted")
    atomic_write_text(path, json.dumps({**asdict(plan), "outcome": outcome}, indent=1))
    rejected = sum(o.state == OrderState.REJECTED for o in orders)
    if rejected < len(orders) or not orders:
        # whatever was accepted is this program's from now on; the session counts as done only if
        # nothing was refused, so a re-run retries the rest and its ids make the others harmless
        prior = _load_state(root)
        atomic_write_text(
            root / "account.json",
            json.dumps(
                {
                    "owner": f"{plan.family}/{plan.book}",
                    "last_session": plan.session if not rejected else prior.get("last_session"),
                    "sent_at": now.isoformat(),
                    "start_equity": prior.get("start_equity", plan.account_equity),
                    "start_session": prior.get("start_session", plan.session),
                    "book_share": plan.scale,
                }
            ),
        )
    return orders


def report_last_fills(
    root: Path, broker: AlpacaPaperBroker, today: dt.date
) -> dict[str, Any] | None:
    """How the last plan filled, once it has: written next to it, and returned the first time.
    Fills come from the broker's feed, day by day since the plan went out."""
    plans = sorted(root.glob("plan-*.json"))
    state = _load_state(root)
    if not plans or not state.get("sent_at"):
        return None
    plan = json.loads(plans[-1].read_text(encoding="utf-8"))
    out = root / f"fills-{plan['session']}.json"
    if out.exists():
        return None
    start = dt.datetime.fromisoformat(state["sent_at"]).date()
    fills: list[dict[str, Any]] = []
    for offset in range((today - start).days + 1):
        fills += broker.fills((start + dt.timedelta(days=offset)).isoformat())
    if not fills:
        return None
    quality = fill_quality(plan, fills)
    atomic_write_text(out, json.dumps(quality, indent=1))
    return quality


def fill_quality(plan: Mapping[str, Any], fills: list[dict[str, Any]]) -> dict[str, Any]:
    """Volume-weighted fill price of each planned order against the price the book's decision saw,
    signed so that a positive number is a cost, in basis points of what was traded."""
    by_symbol: dict[str, list[tuple[float, float]]] = {}
    for f in fills:
        qty = float(f["qty"]) * (1 if f["side"] == "buy" else -1)
        name = str(f["symbol"]).replace("/", "")  # the feed spells a coin BTCUSD
        by_symbol.setdefault(name, []).append((qty, float(f["price"])))
    rows = []
    cost = traded = 0.0
    for o in plan["orders"]:
        got = by_symbol.get(str(o["symbol"]).replace("/", ""), [])
        filled = sum(q for q, _ in got)
        if not filled:
            rows.append({"symbol": o["symbol"], "ordered": o["qty"], "filled": 0})
            continue
        avg = sum(q * p for q, p in got) / filled
        ref = float(o["reference_price"])
        side = 1 if filled > 0 else -1
        bps = side * (avg - ref) / ref * 1e4
        rows.append(
            {"symbol": o["symbol"], "ordered": o["qty"], "filled": filled, "avg": avg, "bps": bps}
        )
        cost += bps * abs(filled) * ref
        traded += abs(filled) * ref
    return {
        "session": plan["session"],
        "orders": rows,
        "traded": traded,
        "mean_bps": cost / traded if traded else math.nan,
        "unfilled": sum(1 for r in rows if not r["filled"]),
    }


def archive_record(root: Path, now: dt.datetime) -> Path | None:
    """Move the account's record aside, for an account that has been reset: its plans, fills and
    tracking describe money that no longer exists there. Nothing is deleted."""
    files = [p for p in root.glob("*.json") if p.is_file()]
    if not files:
        return None
    target = root / f"archive-{now.strftime('%Y%m%dT%H%M%S')}"
    target.mkdir(parents=True)
    for p in files:
        p.rename(target / p.name)
    return target


def book_return_since(journal: Journal, start_session: str) -> float:
    """A book's return from the close of ``start_session``; zero if it opened after it did not."""
    curve = journal.equity_by_session()
    base = [v for s, v in curve if s <= start_session]
    first = base[-1] if base else (curve[0][1] if curve else 0)
    return curve[-1][1] / first - 1.0 if first else 0.0


def mirror_tracking(
    forward: Path, books: Sequence[BookTarget], slots: int, state: Mapping[str, Any], equity: float
) -> dict[str, Any] | None:
    """The account against the mix it mirrors, since the first plan. The account is the only record
    of what really filled; the books are the record of each strategy. The gap between them is what
    netting, rounding, what the broker would not hold and its fills cost together."""
    start, session = state.get("start_equity"), state.get("start_session")
    if not start or not session:
        return None
    mix = 0.0
    for b in books:
        journal = Journal(forward / b.name / "journal.jsonl")
        mix += book_return_since(journal, str(session)) * float(state.get("book_share", 1 / slots))
    actual = equity / float(start) - 1.0
    return {
        "since": session,
        "account": actual,
        "mix": mix,
        "gap": actual - mix,
    }


def render_plan(plan: Plan, largest: int = 15) -> str:
    kind = "LIVE" if plan.live else "paper"
    lines = [
        f"{plan.family}/{plan.book} on the {kind} account, decision close {plan.session}",
        f"account equity {plan.account_equity:,.0f}, book {plan.book_equity:,.0f}, "
        f"scale {plan.scale:.5f}, gross after {plan.gross_after:.2f}, {len(plan.orders)} orders",
    ]
    if plan.books > 1:
        lines.append(
            f"{plan.books} books net together; what can be held is {plan.captured:.1%} "
            "of their gross"
        )
    lines += [f"  note: {n}" for n in plan.notes]
    ordered = sorted(plan.orders, key=lambda o: -abs(o.qty) * (o.reference_price or 0.0))
    lines += [
        f"  {o.qty:+12.4f} {o.symbol:<9s} ~{abs(o.qty) * o.reference_price:>12,.0f}"
        for o in ordered[:largest]
    ]
    if len(ordered) > largest:
        lines.append(f"  and {len(ordered) - largest} smaller")
    return "\n".join(lines)


def book_equity_of(journal: Journal) -> float:
    curve = journal.equity_by_session()
    if not curve:
        raise ExecutionRefusedError("the book has no equity history yet")
    return to_dollars(curve[-1][1])


def live_broker(env: Mapping[str, str] | None = None) -> AlpacaPaperBroker:
    """The same adapter on the live endpoint; callers have passed :func:`live_refusals`."""
    e = environment() if env is None else env
    return AlpacaPaperBroker(
        e["PUSULA_ALPACA_LIVE_KEY"],
        e["PUSULA_ALPACA_LIVE_SECRET"],
        dry_run=False,
        base_url=LIVE_URL,
    )
