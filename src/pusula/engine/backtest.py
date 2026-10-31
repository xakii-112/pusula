"""Event-loop backtester (research engine).

Timeline of one session ``e``::

    open/VWAP of e : execute the latest target decided at or before close (e - lag)
    close of e     : mark to market, accrue interest / borrow, kill-switch check,
                     and (if a rebalance day) build the DecisionContext from rows <= e and ask
                     the strategy for new targets, which pass the risk gate.

Positions are held in shares (float here; the paper ledger holds integer shares and integer
micro-dollars and is reconciled against this engine in ``tests/test_parity.py``). Targets are
converted to shares at the decision close; the trade at execution is ``target - current``
shares, capped by participation of ADV, priced at the execution price plus the cost model.

Delisting: when a security's prices end, the position is closed at its last close (minus
spread) on the next session. Interior gaps (halts) hold the position at the last price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pusula.data.panel import FloatArray, Panel
from pusula.execution.costs import CostInputs, CostModel, cost_inputs, trade_cost
from pusula.portfolio.context import DecisionContext, Strategy
from pusula.risk.limits import GateResult, KillSwitch, RiskLimits, gate_targets
from pusula.risk.model import FactorRisk, RiskModelSpec, estimate, simple_returns


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_equity: float = 10_000_000.0
    lag: int = 1
    exec_price: str = "vwap"  # vwap | open | close
    start: str | None = None  # first decision date
    end: str | None = None
    cost_model: CostModel = field(default_factory=CostModel)
    risk_spec: RiskModelSpec = field(default_factory=RiskModelSpec)
    limits: RiskLimits = field(default_factory=RiskLimits)
    kill_switch: bool = True
    max_drawdown: float = 0.25
    max_daily_loss: float = 0.08
    min_price: float = 5.0


@dataclass
class BacktestResult:
    strategy: str
    dates: npt.NDArray[np.datetime64]
    equity: FloatArray
    returns: FloatArray  # net daily returns (first entry 0)
    gross_returns: FloatArray  # before trading costs and borrow
    costs: FloatArray  # $ per day
    borrow: FloatArray
    interest: FloatArray
    turnover: FloatArray  # traded $ / equity
    gross_exposure: FloatArray
    net_exposure: FloatArray
    n_long: npt.NDArray[np.int64]
    n_short: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float32]  # T x N at each close
    beta_exposure: FloatArray
    events: list[str]
    unfilled_frac: FloatArray  # share of desired trade $ not filled (capacity)
    rf: FloatArray  # daily risk-free rate applied to cash (for excess returns)
    halted_on: str | None = None  # session the kill switch flattened the book, if it did

    @property
    def excess_returns(self) -> FloatArray:
        """Net returns minus the risk-free rate: the basis of every Sharpe-type statistic.
        (A book sitting in cash after a kill switch earns T-bills with almost no volatility; on
        raw returns that looked like a Sharpe of 80, us-equity-v1, 2026-10-24.)"""
        out = self.returns - self.rf
        out[0] = 0.0
        return out

    def summary(self) -> dict[str, float]:
        from pusula.stats import sharpe as S

        r = self.returns[1:]
        x = self.excess_returns[1:]
        gx = (self.gross_returns - self.rf)[1:]
        avg_eq = float(np.mean(self.equity))
        return {
            "ann_return": S.annualised_return(r),
            "ann_vol": S.annualised_vol(r),
            "sharpe": S.annualised_sharpe(x),
            "sharpe_gross": S.annualised_sharpe(gx),
            "max_drawdown": S.max_drawdown(r),
            "turnover_annual": float(np.mean(self.turnover[1:]) * 252),
            "cost_drag_annual": float(np.sum(self.costs) / avg_eq / (r.size / 252)),
            "avg_gross": float(np.mean(self.gross_exposure[1:])),
            "avg_net": float(np.mean(self.net_exposure[1:])),
            "avg_names": float(np.mean(self.n_long[1:] + self.n_short[1:])),
            "avg_beta": float(np.nanmean(self.beta_exposure[1:]))
            if np.isfinite(self.beta_exposure).any()
            else math.nan,
            "unfilled": float(np.nanmean(self.unfilled_frac[1:]))
            if np.isfinite(self.unfilled_frac).any()
            else 0.0,
        }


def _exec_prices(panel: Panel, how: str) -> FloatArray:
    if how == "close":
        return panel["close"]
    base = panel["open"] if how == "open" or "vwap" not in panel else panel["vwap"]
    fb = np.where(np.isfinite(base), base, panel["open"])
    return np.where(np.isfinite(fb), fb, panel["close"])


def _ffill(x: FloatArray) -> FloatArray:
    out = x.copy()
    for i in range(1, out.shape[0]):
        m = ~np.isfinite(out[i])
        out[i, m] = out[i - 1, m]
    return out


class Backtester:
    def __init__(
        self, panel: Panel, config: BacktestConfig, rf_daily: FloatArray | None = None
    ) -> None:
        self.panel = panel
        self.cfg = config
        T = panel.n_dates
        self.rf = (
            np.zeros(T) if rf_daily is None else np.where(np.isfinite(rf_daily), rf_daily, 0.0)
        )
        self.close = panel["close"]
        self.close_ff = _ffill(self.close)
        self.mark = np.where(np.isfinite(self.close_ff), self.close_ff, 0.0)  # valuation price
        self.px_exec = _exec_prices(panel, config.exec_price)
        self.returns = simple_returns(self.close)
        self.ci: CostInputs = cost_inputs(panel, config.cost_model)
        self.member = (
            (panel["in_universe"] > 0) if "in_universe" in panel else np.isfinite(self.close)
        )
        finite = np.isfinite(self.close)
        last = np.full(panel.n_symbols, -1)
        for j in range(panel.n_symbols):
            idx = np.nonzero(finite[:, j])[0]
            if idx.size:
                last[j] = idx[-1]
        self.last_valid = last
        self._risk_cache: dict[int, FactorRisk | None] = {}

    # ------------------------------------------------------------------ helpers
    def risk_at(self, t: int) -> FactorRisk | None:
        if t not in self._risk_cache:
            m = self.member[t] & np.isfinite(self.close[t])
            self._risk_cache[t] = estimate(self.returns, t, m, self.cfg.risk_spec)
        return self._risk_cache[t]

    def eligible_at(self, t: int) -> npt.NDArray[np.bool_]:
        c = self.close[t]
        return (
            self.member[t]
            & np.isfinite(c)
            & (c >= self.cfg.min_price)
            & np.isfinite(self.ci.adv_dollars[t])
        )

    def context(self, t: int, equity: float, w_current: FloatArray) -> DecisionContext:
        risk = self.risk_at(t)
        beta = np.full(self.panel.n_symbols, np.nan)
        if risk is not None:
            mw = np.full(risk.n, 1.0 / risk.n)
            beta[risk.cols] = risk.beta(mw)
        return DecisionContext(
            t=t,
            date=self.panel.dates[t],
            equity=equity,
            w_current=w_current,
            eligible=self.eligible_at(t),
            close=self.close[t],
            half_spread=self.ci.half_spread[t],
            adv_dollars=self.ci.adv_dollars[t],
            sigma=self.ci.sigma[t],
            risk=risk,
            beta=beta,
        )

    # ------------------------------------------------------------------ main loop
    def run(self, strategy: Strategy) -> BacktestResult:
        cfg, p = self.cfg, self.panel
        T, N = p.n_dates, p.n_symbols
        strategy.prepare(p)
        t0 = (
            0 if cfg.start is None else int(np.searchsorted(p.dates, np.datetime64(cfg.start, "D")))
        )
        t1 = (
            T - 1
            if cfg.end is None
            else int(np.searchsorted(p.dates, np.datetime64(cfg.end, "D"), "right")) - 1
        )
        q = np.zeros(N)  # shares
        cash = cfg.initial_equity
        equity = np.full(T, np.nan)
        rets = np.zeros(T)
        gross_r = np.zeros(T)
        costs = np.zeros(T)
        borrow = np.zeros(T)
        interest = np.zeros(T)
        turnover = np.zeros(T)
        gexp = np.zeros(T)
        nexp = np.zeros(T)
        nl = np.zeros(T, dtype=np.int64)
        ns = np.zeros(T, dtype=np.int64)
        bexp = np.full(T, np.nan)
        unfilled = np.full(T, np.nan)
        W = np.zeros((T, N), dtype=np.float32)
        events: list[str] = []
        pending: list[tuple[int, FloatArray]] = []  # (decision row, target shares)
        executed_upto = -1
        ks = KillSwitch(cfg.max_drawdown, cfg.max_daily_loss, derisk_drawdown=1.0)
        equity[t0] = cash
        prev_eq = cash
        halted = False
        halted_on: str | None = None
        cm = cfg.cost_model
        for e in range(t0, t1 + 1):
            day_cost = 0.0
            traded = 0.0
            desired = 0.0
            if e > t0:
                # 1. delistings: close positions whose prices have ended
                gone = (self.last_valid < e) & (q != 0)
                for j in np.nonzero(gone)[0]:
                    px = self.close_ff[e - 1, j]
                    val = q[j] * px
                    hs = float(self.ci.half_spread[max(int(self.last_valid[j]), 0), j])
                    c = abs(val) * (hs if math.isfinite(hs) else cm.spread_cap_bps * 1e-4)
                    cash += val - c
                    day_cost += c
                    events.append(
                        f"{p.dates[e]} delisted {p.symbols[j]}: closed {val:,.0f} at last close"
                    )
                    q[j] = 0.0
                # 2. execution of the latest due target
                due = [x for x in pending if x[0] <= e - cfg.lag and x[0] > executed_upto]
                if due:
                    d_row, q_tgt = due[-1]
                    executed_upto = d_row
                    pending = [x for x in pending if x[0] > d_row]
                    px = self.px_exec[e]
                    tradable = np.isfinite(px) & (px > 0)
                    dq = np.where(tradable, q_tgt - q, 0.0)
                    want = dq * np.where(tradable, px, 0.0)
                    cap = cm.max_participation * self.ci.adv_dollars[d_row]
                    cap = np.where(np.isfinite(cap), cap, 0.0)
                    scale = np.where(np.abs(want) > cap, cap / np.maximum(np.abs(want), 1e-12), 1.0)
                    dq = dq * scale
                    trade_val = dq * np.where(tradable, px, 0.0)
                    c = trade_cost(
                        trade_val,
                        self.ci.half_spread[d_row],
                        self.ci.adv_dollars[d_row],
                        self.ci.sigma[d_row],
                        cm,
                    )
                    cash -= float(trade_val.sum()) + float(c.sum())
                    day_cost += float(c.sum())
                    q += dq
                    traded = float(np.abs(trade_val).sum())
                    desired = float(np.abs(want).sum())
                # 3. financing over the night e-1 -> e
                short_val = float(np.maximum(-q * self.mark[e - 1], 0.0).sum())
                b = short_val * cm.borrow_annual / 252.0
                i_ = cash * self.rf[e]
                if cash < 0:  # debit balance: broker margin rate = rf + spread
                    i_ -= -cash * cm.financing_spread_annual / 252.0
                cash += i_ - b
                borrow[e], interest[e] = b, i_
            # 4. mark to market
            mv = q * self.mark[e]
            eq = cash + float(mv.sum())
            equity[e] = eq
            if e > t0:
                rets[e] = eq / prev_eq - 1.0
                gross_r[e] = (eq + day_cost + borrow[e]) / prev_eq - 1.0
                costs[e] = day_cost
                turnover[e] = traded / prev_eq
                unfilled[e] = 1.0 - traded / desired if desired > 0 else np.nan
            w_now = mv / eq if eq > 0 else np.zeros(N)
            W[e] = w_now.astype(np.float32)
            gexp[e], nexp[e] = float(np.abs(w_now).sum()), float(w_now.sum())
            nl[e], ns[e] = int((w_now > 1e-6).sum()), int((w_now < -1e-6).sum())
            # 5. kill switch
            if cfg.kill_switch and e > t0:
                scale_ks = ks.update(eq, prev_eq)
                if scale_ks == 0.0 and not halted:
                    halted = True
                    events.append(f"{p.dates[e]} KILL SWITCH: {ks.reason}; flattening")
                    halted_on = str(p.dates[e])
                    pending.append((e, np.zeros(N)))
            prev_eq = eq
            # 6. decision
            if halted or e >= t1 or (e - t0) % strategy.rebalance_every != 0:
                continue
            ctx = self.context(e, eq, w_now)
            if ctx.risk is not None:
                bexp[e] = float(np.nansum(np.where(np.isfinite(ctx.beta), ctx.beta * w_now, 0.0)))
            target = strategy.decide(ctx)
            gr: GateResult = gate_targets(
                target,
                w_now,
                p.symbols,
                cfg.limits,
                price=self.close[e],
                adv_dollars=self.ci.adv_dollars[e],
                equity=eq,
                beta=ctx.beta,
            )
            if gr.breaches:
                events.append(f"{p.dates[e]} gate: " + "; ".join(gr.breaches))
            with np.errstate(invalid="ignore", divide="ignore"):
                q_tgt = np.where(
                    np.isfinite(self.close[e]) & (self.close[e] > 0),
                    gr.weights * eq / self.close[e],
                    q,
                )
            pending.append((e, q_tgt))
        sl = slice(t0, t1 + 1)
        return BacktestResult(
            strategy=strategy.name,
            dates=p.dates[sl],
            equity=equity[sl],
            returns=rets[sl],
            gross_returns=gross_r[sl],
            costs=costs[sl],
            borrow=borrow[sl],
            interest=interest[sl],
            turnover=turnover[sl],
            gross_exposure=gexp[sl],
            net_exposure=nexp[sl],
            n_long=nl[sl],
            n_short=ns[sl],
            weights=W[sl],
            beta_exposure=bexp[sl],
            events=events,
            unfilled_frac=unfilled[sl],
            rf=self.rf[sl].copy(),
            halted_on=halted_on,
        )
