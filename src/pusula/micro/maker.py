"""A market maker replayed against a real order book, with its place in the queue tracked.

meridian (Desktop/jane) measured market making in a synthetic market, and said its promotion was a
statement about that market. This keeps its accounting and its question (is spread capture worth
the adverse selection?) and answers on the book that actually traded:

* **Queue.** An order joins behind what is displayed at its price, ``latency`` after it was decided.
  A trade at its price consumes what is ahead of it first. A decrease that no trade explains is a
  cancel, and takes from those ahead in proportion to their share of the level (``cancel_ahead``: 0
  says nobody ahead ever cancels, the pessimistic bound). A trade through its price fills it whole.
* **No footprint.** The order does not change what others do; true only for sizes small against the
  level, and the sizes here are.
* **Accounting.** In half-ticks and lots: ``PnL = edge + inventory`` holds as an integer equality on
  every run, before fees, and fees are one number per fill: one replay gives every fee tier.
* **Adverse selection.** Each fill's mid move after 1, 10 and 60 seconds, against the maker.
* **Breaks.** Where the record is not continuous, the makers stop, marked at the last mid, and new
  ones start flat. What the market did while nothing was recorded is nobody's PnL.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pusula.micro.book import BUY, SELL, BookEvent, LocalBook, SegmentBreak, TradeEvent

MARKOUTS_NS = (1_000_000_000, 10_000_000_000, 60_000_000_000)
EWMA_HALF_LIFE_NS = 60_000_000_000
BLOCK_NS = 3_600_000_000_000  # results are also kept per hour: the unit of the statistics


@dataclass(frozen=True, slots=True)
class MakerSpec:
    kind: str = "symmetric"  # symmetric | stoikov | imbalance
    half_spread_bps: float = 0.0  # 0 joins the touch
    notional: float = 1000.0  # quote currency per side
    max_position_mult: int = 5  # of the size
    decide_every_ns: int = 1_000_000_000
    latency_ns: int = 50_000_000
    gamma: float = 0.0  # stoikov: skew in volatilities (per decision) per size of inventory
    imbalance_bps: float = 0.0  # imbalance: skew in bps of the mid at a fully one-sided touch
    cancel_ahead: float = 1.0  # share of the proportional cancel rule; 0 is the pessimistic bound


@dataclass
class _Quote:
    side: int
    price: int
    remaining: int
    ahead: int
    live_at: int  # when it reaches the exchange
    dead_at: int | None = None


@dataclass
class Fill:
    time_ns: int
    side: int
    price: int
    qty: int
    mid2: int


@dataclass
class MakerResult:
    spec: MakerSpec
    fills: list[Fill] = field(default_factory=list)
    edge: int = 0  # half-ticks x lots: what the fills earned against the mid at the time
    inventory: int = 0  # half-ticks x lots: what the position earned as the mid moved
    cash2: int = 0  # half-ticks x lots
    position: int = 0
    final_mid2: int = 0
    max_abs_position: int = 0
    traded_lots: int = 0
    traded_notional: int = 0  # ticks x lots
    markouts: dict[int, list[tuple[float, int]]] = field(default_factory=dict)
    blocks: list[tuple[int, int]] = field(default_factory=list)  # pnl2, notional at each hour's end
    decisions: int = 0
    size_lots: int = 0
    hours: int = 0

    @property
    def pnl(self) -> int:
        return self.cash2 + self.position * self.final_mid2

    def check_identity(self) -> None:
        if self.pnl != self.edge + self.inventory:
            raise AssertionError(f"PnL {self.pnl} != {self.edge} + {self.inventory}")

    def net(self, fee_bps: float, tick: float, lot: float) -> float:
        """PnL after fees, in quote currency; a negative fee is a rebate."""
        gross = self.pnl * tick * lot / 2.0
        return gross - self.traded_notional * tick * lot * fee_bps * 1e-4

    def breakeven_fee_bps(self) -> float:
        """The maker fee at which this run's PnL is zero; negative means it needs a rebate."""
        if not self.traded_notional:
            return math.nan
        return self.pnl / 2.0 / self.traded_notional * 1e4

    def hourly_net(self, fee_bps: float, tick: float, lot: float) -> list[float]:
        """Net PnL of each completed hour, in quote currency."""
        out = []
        prev_pnl = prev_notional = 0
        for pnl2, notional in self.blocks:
            gross = (pnl2 - prev_pnl) * tick * lot / 2.0
            out.append(gross - (notional - prev_notional) * tick * lot * fee_bps * 1e-4)
            prev_pnl, prev_notional = pnl2, notional
        return out

    def markout_bps(self, horizon_ns: int, tick: float) -> float:
        """Lot-weighted mid move after a fill, in bps of the fill's mid, favourable positive."""
        rows = self.markouts.get(horizon_ns, [])
        lots = sum(q for _, q in rows)
        return sum(v * q for v, q in rows) / lots if lots else math.nan


def _touch_imbalance(book: LocalBook) -> float:
    b, a = book.bids.top(1), book.asks.top(1)
    if not b or not a or b[0][1] + a[0][1] == 0:
        return 0.0
    return (b[0][1] - a[0][1]) / (b[0][1] + a[0][1])


def target_quotes(
    spec: MakerSpec, book: LocalBook, inventory_sizes: float, sigma_ticks: float
) -> tuple[int, int] | None:
    """Bid and ask for this decision. The half-spread is at least half the market's, so a quote of
    0 bps is at the touch; a skew may improve a side, and neither side crosses the other's touch."""
    best_bid, best_ask = book.bids.best(), book.asks.best()
    if best_bid is None or best_ask is None or best_bid >= best_ask:
        return None
    mid = (best_bid + best_ask) / 2.0
    skew = 0.0
    if spec.kind == "stoikov":
        skew = -spec.gamma * inventory_sizes * sigma_ticks
    elif spec.kind == "imbalance":
        skew = spec.imbalance_bps * 1e-4 * mid * _touch_imbalance(book)
    elif spec.kind != "symmetric":
        raise ValueError(f"unknown maker {spec.kind!r}")
    half = max((best_ask - best_bid) / 2.0, spec.half_spread_bps * 1e-4 * mid)
    bid = min(math.floor(mid + skew - half), best_ask - 1)
    ask = max(math.ceil(mid + skew + half), best_bid + 1)
    if bid >= ask:
        return best_bid, best_ask
    return bid, ask


class MakerSim:
    def __init__(self, spec: MakerSpec, book: LocalBook) -> None:
        self.spec = spec
        self.book = book
        self.res = MakerResult(spec)
        self.quotes: dict[int, list[_Quote]] = {BUY: [], SELL: []}
        self.next_decide = 0
        self.last_mid2: int | None = None
        self.var_ticks = 0.0
        self.last_var_t: int | None = None
        self.block_end: int | None = None
        self.pending: dict[int, deque[tuple[int, int, int, int]]] = {
            h: deque() for h in MARKOUTS_NS
        }
        for horizon in MARKOUTS_NS:
            self.res.markouts[horizon] = []

    # ------------------------------------------------------------- the world moves
    def on_book(self, ev: BookEvent, traded: Mapping[tuple[int, int], int]) -> None:
        """``traded``: lots that traded at (resting side, price) since the last book event: the
        part of a decrease that was not a cancel."""
        for sign, price, old, new in ev.changes:
            drop = old - new
            if drop <= 0 or not old:
                continue
            cancels = max(drop - traded.get((sign, price), 0), 0)
            if not cancels:
                continue
            for q in self.quotes[sign]:
                if q.price == price and q.live_at <= ev.time_ns and q.ahead:
                    q.ahead -= min(q.ahead, round(self.spec.cancel_ahead * cancels * q.ahead / old))
        self._mark_mid(ev.time_ns)

    def on_trade(self, ev: TradeEvent) -> None:
        resting = -ev.aggressor  # the side the trade hit
        for q in list(self.quotes[resting]):
            if q.live_at > ev.time_ns or (q.dead_at is not None and q.dead_at <= ev.time_ns):
                continue
            through = ev.price < q.price if resting == BUY else ev.price > q.price
            if through:  # it traded beyond this quote: everything at this price went first
                self._fill(q, q.remaining, ev.time_ns)
            elif ev.price == q.price:
                used = min(q.ahead, ev.qty)
                q.ahead -= used
                self._fill(q, min(ev.qty - used, q.remaining), ev.time_ns)
            if q.remaining == 0:
                self.quotes[resting].remove(q)

    def _fill(self, q: _Quote, qty: int, t: int) -> None:
        if qty <= 0:
            return
        q.remaining -= qty
        mid2 = self.last_mid2
        assert mid2 is not None, "quotes exist only after the first mark"
        res = self.res
        signed = q.side * qty
        res.cash2 -= signed * q.price * 2
        res.edge += signed * (mid2 - q.price * 2)
        res.position += signed
        res.max_abs_position = max(res.max_abs_position, abs(res.position))
        res.traded_lots += qty
        res.traded_notional += qty * q.price
        res.fills.append(Fill(t, q.side, q.price, qty, mid2))
        for h in MARKOUTS_NS:
            self.pending[h].append((t + h, q.side, qty, mid2))

    def _mark_mid(self, t: int) -> None:
        mid2 = self.book.mid2()
        if mid2 is None:
            return
        res = self.res
        if self.last_mid2 is not None and mid2 != self.last_mid2:
            res.inventory += res.position * (mid2 - self.last_mid2)
            move = (mid2 - self.last_mid2) / 2.0
            if self.last_var_t is not None:
                dt = max(t - self.last_var_t, 1)
                decay = 0.5 ** (dt / EWMA_HALF_LIFE_NS)
                # variance per decision interval, in ticks squared
                scaled = move * move * self.spec.decide_every_ns / dt
                self.var_ticks = decay * self.var_ticks + (1.0 - decay) * scaled
            self.last_var_t = t
        self.last_mid2 = mid2
        res.final_mid2 = mid2
        for h, queue in self.pending.items():
            while queue and queue[0][0] <= t:
                _, side, qty, at = queue.popleft()
                res.markouts[h].append((side * (mid2 - at) / at * 1e4, qty))
        if self.block_end is None:
            self.block_end = t + BLOCK_NS
        elif t >= self.block_end:
            res.blocks.append((res.pnl, res.traded_notional))
            res.hours += 1
            self.block_end = t + BLOCK_NS  # a quiet spell makes one long block, not several empty

    # ------------------------------------------------------------- the maker decides
    def _size(self) -> int:
        if not self.res.size_lots:
            inst = self.book.instrument
            mid = (self.last_mid2 or 0) / 2.0 * inst.tick
            self.res.size_lots = max(round(self.spec.notional / max(mid, 1e-12) / inst.lot), 1)
        return self.res.size_lots

    def decide(self, t: int) -> None:
        self.res.decisions += 1
        spec = self.spec
        size = self._size()
        limit = spec.max_position_mult * size
        inventory = self.res.position / size
        target = target_quotes(spec, self.book, inventory, math.sqrt(max(self.var_ticks, 0.0)))
        for side in (BUY, SELL):
            price = None if target is None else (target[0] if side == BUY else target[1])
            if price is not None and abs(self.res.position + side * size) > limit:
                price = None
            keep = False
            for q in self.quotes[side]:
                if q.dead_at is not None:
                    continue
                if price is not None and q.price == price:
                    keep = True  # same price: it keeps its place in the queue
                else:
                    q.dead_at = t + spec.latency_ns
            if price is not None and not keep:
                levels = (self.book.bids if side == BUY else self.book.asks).levels
                self.quotes[side].append(
                    _Quote(side, price, size, levels.get(price, 0), t + spec.latency_ns)
                )

    def sweep_dead(self, t: int) -> None:
        for side in (BUY, SELL):
            self.quotes[side] = [q for q in self.quotes[side] if q.dead_at is None or q.dead_at > t]


def merge_results(spec: MakerSpec, parts: Sequence[MakerResult]) -> MakerResult:
    """Consecutive stretches of one maker as one result: flat between them, so PnL simply adds."""
    out = MakerResult(spec)
    for h in MARKOUTS_NS:
        out.markouts[h] = []
    pnl = 0
    notional = 0
    for r in parts:
        r.check_identity()
        out.fills += r.fills
        out.edge += r.edge
        out.inventory += r.inventory
        out.traded_lots += r.traded_lots
        out.traded_notional += r.traded_notional
        out.max_abs_position = max(out.max_abs_position, r.max_abs_position)
        out.decisions += r.decisions
        out.hours += r.hours
        out.size_lots = out.size_lots or r.size_lots
        out.blocks += [(pnl + a, notional + b) for a, b in r.blocks]
        for h in MARKOUTS_NS:
            out.markouts[h] += r.markouts[h]
        pnl += r.pnl
        notional += r.traded_notional
    out.cash2 = pnl  # flat at the end of every stretch: all of it is cash
    out.check_identity()
    return out


def run_makers(
    specs: Sequence[MakerSpec],
    events: Iterable[BookEvent | TradeEvent | SegmentBreak],
    book: LocalBook,
) -> list[MakerResult]:
    """One pass over ``events`` (they keep ``book`` current as they are produced) through several
    makers. They see the same market and do not see each other."""
    sims = [MakerSim(s, book) for s in specs]
    parts: list[list[MakerResult]] = [[] for _ in specs]
    traded: dict[tuple[int, int], int] = {}
    started = False
    for ev in events:
        if isinstance(ev, SegmentBreak):
            for part, sim in zip(parts, sims, strict=True):
                part.append(sim.res)
            sims = [MakerSim(s, book) for s in specs]
            traded = {}
            started = False
            continue
        if isinstance(ev, TradeEvent):
            key = (-ev.aggressor, ev.price)
            traded[key] = traded.get(key, 0) + ev.qty
            for sim in sims:
                sim.on_trade(ev)
            continue
        for sim in sims:
            sim.on_book(ev, traded)
            sim.sweep_dead(ev.time_ns)
        traded = {}
        if book.mid2() is None:
            continue
        if not started:
            started = True
            for sim in sims:
                sim.next_decide = ev.time_ns
        for sim in sims:
            if ev.time_ns >= sim.next_decide:
                sim.decide(ev.time_ns)
                sim.next_decide = ev.time_ns + sim.spec.decide_every_ns
    for part, sim in zip(parts, sims, strict=True):
        part.append(sim.res)
    return [merge_results(s, part) for s, part in zip(specs, parts, strict=True)]
