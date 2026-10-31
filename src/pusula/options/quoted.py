"""Option strategies priced at the quotes the market showed, not at a model's value.

``overlay`` bought any strike at one ATM volatility plus a fixed spread. This buys at the ask and
sells at the bid of the contract that was listed, on the date it was quoted, pays a commission, and
settles at expiry on the unadjusted close. Between quotes (the chain has three dates a week) a
position is marked at its last volatility, which keeps the smile: a 20-delta put stays at its own
vol. A split rescales the strike and the quantity, as the exchange does.

What it leaves out, in the results as well: early assignment of short positions, margin beyond the
cash-securing of written puts, and capacity. The chain's quotes are end-of-day snapshots, so a fill
at them assumes the close was tradable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from pusula.data.panel import FloatArray, Panel
from pusula.options import black

FEE_PER_SHARE = 0.0065  # $0.65 a contract
STOCK_COST = 0.0005  # one-way, as in options_study
SPLIT_JUMP = 0.15  # a day's change in raw / adjusted beyond any dividend
DAYS = 365.0


@dataclass(frozen=True, slots=True)
class QuotedSpec:
    kind: str  # long_call | short_put | covered_call | straddle_ls
    target_delta: float = 0.5
    dte: int = 30
    dte_range: tuple[int, int] = (20, 45)
    max_names: int = 50
    max_rel_spread: float = 0.5  # of mid, at entry
    premium_cap: float = 0.25  # long calls: of equity
    straddle_premium: float = 0.10  # straddle_ls: premium on each side, of equity
    spread_mult: float = 1.0  # 2.0 in the cost stress: half-spreads doubled, around mid
    entry_delay: int = 0  # chain dates after the first possible one (the lag stress)


@dataclass
class _Leg:
    symbol: str
    j: int
    strike: float
    cp: int
    qty: float  # signed, in shares of underlying
    expiry: pd.Timestamp
    vol: float


@dataclass
class QuotedResult:
    dates: npt.NDArray[np.datetime64]
    equity: FloatArray
    returns: FloatArray
    n_positions: npt.NDArray[np.int64]
    premium: FloatArray  # signed premium flow on entry: + received, - paid
    fees: FloatArray
    rolls: int = 0
    payoffs: list[float] = field(default_factory=list)


def _entry_price(bid: float, ask: float, buying: bool, mult: float) -> float:
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0 * mult
    return mid + half if buying else max(mid - half, 0.0)


def pick_contracts(
    chain: pd.DataFrame, cp: str, target_delta: float, dte: int, dte_range: tuple[int, int]
) -> pd.DataFrame:
    """For each symbol the contract nearest ``target_delta`` (absolute) in the expiry nearest
    ``dte`` days: one row per symbol. Ties go to the first row, so the choice is deterministic."""
    c = chain[chain["call_put"] == cp]
    c = c[(c["bid"] > 0) & (c["ask"] > c["bid"]) & c["dte"].between(*dte_range)]
    if c.empty:
        return c
    c = c.assign(_e=(c["dte"] - dte).abs())
    best = c.groupby("symbol", observed=True)["_e"].transform("min")
    c = c[c["_e"] == best]
    c = c.assign(_d=(c["delta"].abs() - target_delta).abs())
    return c.sort_values(["symbol", "_d", "strike"], kind="mergesort").drop_duplicates("symbol")


def _rel_spread(c: pd.DataFrame) -> pd.Series:
    return (c["ask"] - c["bid"]) / ((c["ask"] + c["bid"]) / 2.0)


def _atm_straddles(chain: pd.DataFrame, spec: QuotedSpec) -> pd.DataFrame:
    """Per symbol, the call nearest delta 0.5 and the put at the same strike and expiry."""
    calls = pick_contracts(chain, "Call", 0.5, spec.dte, spec.dte_range)
    puts = chain[(chain["call_put"] == "Put") & (chain["bid"] > 0) & (chain["ask"] > chain["bid"])]
    return calls.merge(
        puts[["symbol", "expiration", "strike", "bid", "ask", "vol"]],
        on=["symbol", "expiration", "strike"],
        suffixes=("", "_p"),
    )


@dataclass
class _Book:
    cash: float
    legs: list[_Leg] = field(default_factory=list)
    stock: dict[int, float] = field(default_factory=dict)  # j -> units of adjusted price

    def mark(
        self,
        close_raw: FloatArray,
        close_adj: FloatArray,
        today: pd.Timestamp,
        quotes: dict[tuple[str, pd.Timestamp, float, int], tuple[float, float]],
        r: float,
    ) -> float:
        total = self.cash
        for j, units in self.stock.items():
            px = close_adj[j]
            if math.isfinite(px):
                total += units * float(px)
        for lg in self.legs:
            key = (lg.symbol, lg.expiry, lg.strike, lg.cp)
            quoted = quotes.get(key)
            if quoted is not None:
                mid, vol = quoted
                if math.isfinite(vol) and vol > 0:
                    lg.vol = vol
                total += lg.qty * mid
                continue
            S = float(close_raw[lg.j])
            if not math.isfinite(S):
                total += 0.0
                continue
            T = max((lg.expiry - today).days, 0) / DAYS
            total += lg.qty * float(black.price(S, lg.strike, T, r, 0.0, lg.vol, lg.cp))
        return total


def _quote_table(
    day: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp, float, int], tuple[float, float]]:
    mids = (day["bid"].to_numpy() + day["ask"].to_numpy()) / 2.0
    cps = np.where(day["call_put"].to_numpy() == "Call", 1, -1)
    return {
        (s, e, float(k), int(c)): (float(m), float(v))
        for s, e, k, c, m, v in zip(
            day["symbol"].to_numpy(),
            pd.DatetimeIndex(day["expiration"]),
            day["strike"].to_numpy(),
            cps,
            mids,
            day["vol"].to_numpy(),
            strict=True,
        )
    }


def _open_positions(
    book: _Book,
    spec: QuotedSpec,
    day: pd.DataFrame,
    panel: Panel,
    row: int,
    score: FloatArray,
    close_raw: FloatArray,
    close_adj: FloatArray,
    rv: FloatArray,
) -> tuple[float, float]:
    """Open the book's positions at today's quotes; returns (premium flow, fees)."""
    members = panel["in_universe"][row - 1] > 0  # known at the previous close
    allowed = {
        panel.symbols[j].split("@")[0]: j
        for j in np.nonzero(members & np.isfinite(close_raw[row]))[0]
    }
    day = day[day["symbol"].isin(list(allowed))]
    if day.empty:
        return 0.0, 0.0
    equity = book.cash
    premium = fees = 0.0

    def add(sym: str, row_: pd.Series, cp: int, qty: float, bid: float, ask: float) -> None:
        nonlocal premium, fees
        px = _entry_price(bid, ask, qty > 0, spec.spread_mult)
        premium -= qty * px
        fees += abs(qty) * FEE_PER_SHARE
        book.legs.append(
            _Leg(
                sym,
                allowed[sym],
                float(row_["strike"]),
                cp,
                qty,
                pd.Timestamp(row_["expiration"]),
                float(row_["vol"]),
            )
        )

    if spec.kind == "straddle_ls":
        s = _atm_straddles(day, spec)
        if s.empty:
            return 0.0, 0.0
        spread = (s["ask"] + s["ask_p"] - s["bid"] - s["bid_p"]) / (
            (s["ask"] + s["ask_p"] + s["bid"] + s["bid_p"]) / 2.0
        )
        s = s[spread <= spec.max_rel_spread].copy()
        j = s["symbol"].map(allowed).to_numpy()
        s["gap"] = s["vol"].to_numpy(dtype=np.float64) - rv[row - 1, j]
        s = s[np.isfinite(s["gap"])].sort_values(["gap", "symbol"], kind="mergesort")
        n = min(spec.max_names // 2, len(s) // 2)
        if n == 0:
            return 0.0, 0.0
        per_side = spec.straddle_premium * equity / n
        for sign, group in ((1.0, s.iloc[:n]), (-1.0, s.iloc[-n:])):
            for _, r in group.iterrows():
                mid = (r["bid"] + r["ask"] + r["bid_p"] + r["ask_p"]) / 2.0
                qty = sign * per_side / mid
                sym = str(r["symbol"])
                add(sym, r, 1, qty, float(r["bid"]), float(r["ask"]))
                put_row = r.copy()
                put_row["vol"] = r["vol_p"]
                add(sym, put_row, -1, qty, float(r["bid_p"]), float(r["ask_p"]))
    else:
        cp, side = ("Put", -1) if spec.kind == "short_put" else ("Call", 1)
        c = pick_contracts(day, cp, spec.target_delta, spec.dte, spec.dte_range)
        if c.empty:
            return 0.0, 0.0
        c = c[_rel_spread(c) <= spec.max_rel_spread].copy()
        if spec.kind == "long_call":
            j = c["symbol"].map(allowed).to_numpy()
            c["rank"] = score[row - 1, j]
            c = c[np.isfinite(c["rank"]) & (c["rank"] > 0)]
            c = c.sort_values(["rank", "symbol"], ascending=[False, True], kind="mergesort")
        else:
            # the tightest markets: investable, and known at the time
            c["rank"] = _rel_spread(c)
            c = c.sort_values(["rank", "symbol"], kind="mergesort")
        c = c.head(spec.max_names)
        if c.empty:
            return 0.0, 0.0
        per = equity / len(c)
        scale = 1.0
        if spec.kind == "long_call":
            # the delta-dollars of a stock book, within a premium budget
            cost = 0.0
            for _, r in c.iterrows():
                S = float(close_raw[row, allowed[str(r["symbol"])]])
                cost += per / (abs(float(r["delta"])) * S) * float(r["ask"])
            scale = min(1.0, spec.premium_cap * equity / cost) if cost > 0 else 0.0
        for _, r in c.iterrows():
            sym = str(r["symbol"])
            j_ = allowed[sym]
            S = float(close_raw[row, j_])
            if spec.kind == "long_call":
                d = max(abs(float(r["delta"])), 0.05)
                qty = scale * per / (d * S)
            elif spec.kind == "short_put":
                qty = -per / float(r["strike"])  # cash-secured
            else:  # covered call
                qty = -per / S
                adj = float(close_adj[row, j_])
                units = per / adj
                book.stock[j_] = units
                book.cash -= units * adj * (1.0 + STOCK_COST)
            add(sym, r, side, qty, float(r["bid"]), float(r["ask"]))
    book.cash += premium - fees
    return premium, fees


def _apply_splits(
    legs: list[_Leg],
    raw_before: FloatArray,
    adj_before: FloatArray,
    raw_now: FloatArray,
    adj_now: FloatArray,
) -> None:
    """A split moves the raw price and not the adjusted one; the contract follows it."""
    for lg in legs:
        a = float(raw_before[lg.j]) / float(adj_before[lg.j])
        b = float(raw_now[lg.j]) / float(adj_now[lg.j])
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0:
            continue
        ratio = b / a
        if abs(ratio - 1.0) > SPLIT_JUMP:
            lg.strike *= ratio
            lg.qty /= ratio


def run_quoted(
    spec: QuotedSpec,
    panel: Panel,
    close_raw: FloatArray,
    chain_by_date: dict[pd.Timestamp, pd.DataFrame],
    score: FloatArray,
    rv: FloatArray,
    rf_daily: FloatArray,
    start: int,
    end: int,
    initial_equity: float = 10_000_000.0,
) -> QuotedResult:
    """One cycle at a time: open at the first quoted date with nothing open, hold to expiry."""
    dates = panel.dates
    close_adj = panel["close"]
    book = _Book(initial_equity)
    n = end - start + 1
    eq = np.full(n, np.nan)
    npos = np.zeros(n, dtype=np.int64)
    prem = np.zeros(n)
    fees = np.zeros(n)
    payoffs: list[float] = []
    rolls = 0
    delay = 0
    last_raw = close_raw[start].copy()
    for e in range(start, end + 1):
        today = pd.Timestamp(dates[e])
        i = e - start
        r_ann = 252.0 * float(rf_daily[e]) if math.isfinite(rf_daily[e]) else 0.0
        if e > start and math.isfinite(rf_daily[e]):
            book.cash *= 1.0 + float(rf_daily[e])
        finite = np.isfinite(close_raw[e])
        last_raw = np.where(finite, close_raw[e], last_raw)
        if e > start:
            _apply_splits(book.legs, close_raw[e - 1], close_adj[e - 1], close_raw[e], close_adj[e])
        # expiries settle on the unadjusted close (the last one, if the stock has stopped trading)
        still: list[_Leg] = []
        for lg in book.legs:
            if lg.expiry <= today:
                pay = lg.qty * max(lg.cp * (float(last_raw[lg.j]) - lg.strike), 0.0)
                book.cash += pay
                payoffs.append(pay)
            else:
                still.append(lg)
        settled = len(still) < len(book.legs)
        book.legs = still
        if settled and not book.legs and book.stock:
            for j, units in book.stock.items():
                px = close_adj[e, j]
                if not math.isfinite(px):
                    px = float(close_adj[: e + 1, j][np.isfinite(close_adj[: e + 1, j])][-1])
                book.cash += units * float(px) * (1.0 - STOCK_COST)
            book.stock = {}
        day = chain_by_date.get(today)
        if not book.legs and not book.stock and day is not None and e > start:
            if delay >= spec.entry_delay:
                p, f = _open_positions(book, spec, day, panel, e, score, close_raw, close_adj, rv)
                prem[i], fees[i] = p, f
                rolls += 1
                delay = 0
            else:
                delay += 1
        quotes: dict[tuple[str, pd.Timestamp, float, int], tuple[float, float]] = {}
        if day is not None and book.legs:
            held = {lg.symbol for lg in book.legs}
            quotes = _quote_table(day[day["symbol"].isin(held)])
        eq[i] = book.mark(last_raw, close_adj[e], today, quotes, r_ann)
        npos[i] = len(book.legs)
        if eq[i] <= 0:  # the book is gone: a loss of everything is the honest end of it
            eq[i:] = 0.0
            break
    rets = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rets[1:] = np.where(eq[:-1] > 0, eq[1:] / eq[:-1] - 1.0, 0.0)
    return QuotedResult(dates[start : end + 1], eq, rets, npos, prem, fees, rolls, payoffs)


def stressed(spec: QuotedSpec) -> dict[str, QuotedSpec]:
    return {
        "base": spec,
        "lag2": replace(spec, entry_delay=spec.entry_delay + 1),
        "cost2": replace(spec, spread_mult=2.0 * spec.spread_mult),
    }
