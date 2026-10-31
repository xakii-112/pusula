"""Paper-trading engine: the production path, in integer shares and micro-dollars.

It drives the *same* strategy object through the *same* :class:`DecisionContext` builder as the
research backtester, but every effect goes through orders, a broker, fills, an exact ledger and
the hash-chained journal. ``tests/test_parity.py`` runs both on the same market and requires
their daily returns to agree up to share rounding: research and production cannot drift apart
silently (they did in NihaiAlgoritma, Tur-101 found three divergences).

``step(e)`` processes one session; the forward (live) runner calls it once per new session
after rebuilding the panel, so history and live are one loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pusula.core.hashing import array_digest
from pusula.core.money import to_dollars, to_micros
from pusula.engine.backtest import Backtester
from pusula.engine.journal import Journal
from pusula.engine.ledger import Ledger
from pusula.engine.reconcile import reconcile
from pusula.execution.broker import Broker, MarketSnapshot, SimBroker
from pusula.execution.orders import Order, client_order_id
from pusula.portfolio.context import Strategy
from pusula.risk.limits import KillSwitch, gate_targets


@dataclass
class PaperEngine:
    bt: Backtester
    strategy: Strategy
    name: str
    journal: Journal = field(default_factory=Journal)
    broker: Broker | None = None
    initial_capital: float = 10_000_000.0
    start: int = 0
    pending: list[tuple[int, dict[str, int]]] = field(default_factory=list)
    executed_upto: int = -1
    halted: bool = False
    prev_equity: int = 0
    equity_curve: dict[int, int] = field(default_factory=dict)
    kill: KillSwitch = field(default_factory=KillSwitch)
    ledger: Ledger = field(default_factory=Ledger)
    opened: bool = False
    last_step: int = -1
    _prepared: bool = False

    def __post_init__(self) -> None:
        if self.broker is None:
            self.broker = SimBroker(self.bt.cfg.cost_model)
        cfg = self.bt.cfg
        self.kill = KillSwitch(cfg.max_drawdown, cfg.max_daily_loss, derisk_drawdown=1.0)

    # ------------------------------------------------------------------ helpers
    def _session(self, e: int) -> str:
        return str(self.bt.panel.dates[e])

    def _emit(self, kind: str, session: str, data: dict[str, object]) -> None:
        """Append to the journal and apply to the live ledger (replay must give the same)."""
        ev = self.journal.append(kind, session, data)
        d = ev.data
        if kind == "capital":
            self.ledger.contribute(int(d["micros"]))
        elif kind == "fill":
            self.ledger.fill(
                str(d["security"]), int(d["qty"]), int(d["price_micros"]), int(d["fee_micros"])
            )
        elif kind == "financing":
            self.ledger.financing_flow(int(d["micros"]))

    def _marks(self, e: int) -> dict[str, int]:
        out = {}
        for s in self.ledger.positions:
            j = self.bt.panel.symbol_index(s)
            px = self.bt.close_ff[e, j]
            out[s] = to_micros(float(px)) if math.isfinite(px) else 0
        return out

    # ------------------------------------------------------------------ persistence
    def control_state(self) -> dict[str, object]:
        """Everything that is not in the journal, keyed by session dates (row numbers move when
        the panel is rebuilt; dates do not)."""
        d = self.bt.panel.dates
        return {
            "name": self.name,
            "start": str(d[self.start]),
            "last_step": str(d[self.last_step]) if self.last_step >= 0 else None,
            "pending": [[str(d[r]), tgt] for r, tgt in self.pending],
            "executed_upto": str(d[self.executed_upto]) if self.executed_upto >= 0 else None,
            "halted": self.halted,
            "prev_equity": self.prev_equity,
            "kill": {
                "peak": self.kill.peak,
                "tripped": self.kill.tripped,
                "reason": self.kill.reason,
            },
            "journal_head": self.journal.events[-1].hash if self.journal.events else None,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Resume from ``control_state`` + the journal (the ledger is rebuilt by replay)."""
        p = self.bt.panel
        head = self.journal.events[-1].hash if self.journal.events else None
        if head != state["journal_head"]:
            raise ValueError("journal head does not match the saved state: refusing to resume")
        led, _ = self.journal.replay()
        self.ledger = led
        # the equity history lives in the journal, not in a previous process's memory
        self.equity_curve = {
            p.date_index(session): eq for session, eq in self.journal.equity_by_session()
        }
        self.opened = True
        self.start = p.date_index(str(state["start"]))
        self.last_step = p.date_index(str(state["last_step"])) if state["last_step"] else -1
        pend = state["pending"]
        assert isinstance(pend, list)
        self.pending = [
            (p.date_index(str(dd)), {str(k): int(v) for k, v in dict(tgt).items()})
            for dd, tgt in pend
        ]
        ex = state["executed_upto"]
        self.executed_upto = p.date_index(str(ex)) if ex else -1
        self.halted = bool(state["halted"])
        self.prev_equity = int(state["prev_equity"])  # type: ignore[call-overload]
        k = state["kill"]
        assert isinstance(k, dict)
        self.kill.peak, self.kill.tripped, self.kill.reason = (
            float(k["peak"]),
            bool(k["tripped"]),
            str(k["reason"]),
        )

    def reconcile_with(self, broker_positions: dict[str, int], session: str) -> bool:
        """Journal a book-vs-broker comparison; any break halts the book (no new decisions)."""
        rec = reconcile(self.ledger, broker_positions)
        self._emit("reconcile", session, rec.as_event())
        if not rec.clean:
            self.halted = True
            self._emit("risk", session, {"halt": f"{len(rec.breaks)} reconciliation breaks"})
        return rec.clean

    def run(self, t0: int, t1: int) -> None:
        for e in range(t0, t1 + 1):
            self.step(e, final=(e == t1))

    # ------------------------------------------------------------------ one session
    def step(self, e: int, final: bool = False) -> None:
        bt, p = self.bt, self.bt.panel
        session = self._session(e)
        if e <= self.last_step:
            raise ValueError(f"session {session} already processed (last {self.last_step})")
        if not self._prepared:
            self.strategy.prepare(p)
            self._prepared = True
        if not self.opened:
            self.opened = True
            self.start = e
            self._emit(
                "capital",
                session,
                {"micros": to_micros(self.initial_capital), "note": f"open book {self.name}"},
            )
            self.prev_equity = to_micros(self.initial_capital)
        self.last_step = e
        led = self.ledger
        cm = bt.cfg.cost_model
        if e > self.start:
            # 1. delistings
            for sid, pos in list(led.positions.items()):
                j = p.symbol_index(sid)
                if bt.last_valid[j] < e and pos.qty != 0:
                    px = float(bt.close_ff[e - 1, j])
                    hs = float(bt.ci.half_spread[max(int(bt.last_valid[j]), 0), j])
                    fee = to_micros(
                        abs(pos.qty) * px * (hs if math.isfinite(hs) else cm.spread_cap_bps * 1e-4)
                    )
                    self._emit(
                        "fill",
                        session,
                        {
                            "security": sid,
                            "qty": -pos.qty,
                            "price_micros": to_micros(px),
                            "fee_micros": fee,
                            "client_id": f"delist-{sid}",
                            "reason": "delisted",
                        },
                    )
            # 2. execute the latest due target through the broker
            due = [x for x in self.pending if x[0] <= e - bt.cfg.lag and x[0] > self.executed_upto]
            if due:
                d_row, tgt = due[-1]
                self.executed_upto = d_row
                self.pending = [x for x in self.pending if x[0] > d_row]
                held = {s: pos.qty for s, pos in led.positions.items()}
                orders = []
                for sid in sorted(set(tgt) | set(held)):
                    dq = tgt.get(sid, 0) - held.get(sid, 0)
                    if dq != 0:
                        orders.append(
                            Order(client_order_id(self.name, session, sid), sid, dq, session)
                        )
                assert self.broker is not None
                self.broker.submit(orders)
                for o in orders:
                    self._emit(
                        "order",
                        session,
                        {
                            "client_id": o.client_id,
                            "security": o.security,
                            "qty": o.qty,
                            "state": str(o.state),
                        },
                    )
                snap = MarketSnapshot(
                    exec_price={
                        o.security: float(bt.px_exec[e, p.symbol_index(o.security)]) for o in orders
                    },
                    half_spread={
                        o.security: float(bt.ci.half_spread[d_row, p.symbol_index(o.security)])
                        for o in orders
                    },
                    adv_dollars={
                        o.security: float(bt.ci.adv_dollars[d_row, p.symbol_index(o.security)])
                        for o in orders
                    },
                    sigma={
                        o.security: float(bt.ci.sigma[d_row, p.symbol_index(o.security)])
                        for o in orders
                    },
                )
                for f in self.broker.execute_session(snap):
                    self._emit(
                        "fill",
                        session,
                        {
                            "security": f.security,
                            "qty": f.qty,
                            "price_micros": f.price_micros,
                            "fee_micros": f.fee_micros,
                            "client_id": f.client_id,
                        },
                    )
                # 3. financing over the night
            prev_marks = {
                s: to_micros(float(bt.close_ff[e - 1, p.symbol_index(s)])) for s in led.positions
            }
            short_val = sum(max(-pos.qty * prev_marks[s], 0) for s, pos in led.positions.items())
            interest = round(led.cash * float(bt.rf[e]))
            if led.cash < 0:
                interest -= round(-led.cash * cm.financing_spread_annual / 252.0)
            borrow = round(short_val * cm.borrow_annual / 252.0)
            if interest - borrow != 0:
                self._emit(
                    "financing",
                    session,
                    {"micros": interest - borrow, "interest": interest, "borrow": borrow},
                )
        # 4. marks
        marks = self._marks(e)
        self._emit("mark", session, {"marks": marks})
        eq = led.equity(marks)
        led.check_identity(marks)
        self.equity_curve[e] = eq
        # 5. kill switch
        if (
            self.bt.cfg.kill_switch
            and e > self.start
            and not self.halted
            and self.kill.update(to_dollars(eq), to_dollars(self.prev_equity)) == 0.0
        ):
            self.halted = True
            self._emit("risk", session, {"kill_switch": self.kill.reason})
            self.pending.append((e, {}))
        self.prev_equity = eq
        # 6. decision
        if self.halted or final or (e - self.start) % self.strategy.rebalance_every != 0:
            return
        N = p.n_symbols
        mv = np.zeros(N)
        for s, pos in led.positions.items():
            mv[p.symbol_index(s)] = pos.qty * to_dollars(marks[s])
        eq_d = to_dollars(eq)
        w_now = mv / eq_d if eq_d > 0 else np.zeros(N)
        ctx = bt.context(e, eq_d, w_now)
        target = self.strategy.decide(ctx)
        gr = gate_targets(
            target,
            w_now,
            p.symbols,
            bt.cfg.limits,
            price=bt.close[e],
            adv_dollars=bt.ci.adv_dollars[e],
            equity=eq_d,
            beta=ctx.beta,
        )
        target_sh: dict[str, int] = {}
        for jj in np.nonzero(np.abs(gr.weights) > 1e-9)[0]:
            j = int(jj)
            c = float(bt.close[e, j])
            if math.isfinite(c) and c > 0:
                sh = int(gr.weights[j] * eq_d / c)  # toward zero: never exceed the target
                if sh != 0:
                    target_sh[p.symbols[j]] = sh
            else:
                pos_held = led.positions.get(p.symbols[j])
                if pos_held is not None:
                    target_sh[p.symbols[j]] = pos_held.qty
        self._emit(
            "decision",
            session,
            {
                "strategy": self.name,
                "weights_digest": array_digest(gr.weights),
                "breaches": gr.breaches,
                "n_targets": len(target_sh),
            },
        )
        self.pending.append((e, target_sh))
