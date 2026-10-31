"""The full option engine: SVI smiles, American trees, surfaces and quote-priced strategies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pusula.data.panel import Panel
from pusula.options import american as A
from pusula.options import black
from pusula.options import quoted as Q
from pusula.options import surface as V
from pusula.options import svi as I

TRUE = I.SVIParams(a=0.01, b=0.08, rho=-0.5, m=0.02, sigma=0.15)


# ---------------------------------------------------------------------------- SVI
def test_svi_recovers_a_smile() -> None:
    k = np.linspace(-0.4, 0.4, 25)
    fit = I.fit_svi(k, TRUE.total_variance(k))
    assert fit.rmse < 1e-5
    grid = np.linspace(-0.4, 0.4, 101)
    assert np.allclose(fit.params.total_variance(grid), TRUE.total_variance(grid), atol=2e-4)


def test_svi_fit_keeps_the_wing_constraints() -> None:
    rng = np.random.default_rng(0)
    k = np.linspace(-0.5, 0.5, 15)
    w = TRUE.total_variance(k) * np.exp(rng.normal(0.0, 0.05, k.size))
    p = I.fit_svi(k, w).params
    assert p.b >= 0 and -1 < p.rho < 1 and p.sigma > 0
    assert p.b * (1.0 + abs(p.rho)) <= 4.0 + 1e-9  # Lee's moment bound


def test_svi_refuses_too_little() -> None:
    with pytest.raises(ValueError, match="at least"):
        I.fit_svi([0.0, 0.1], [0.1, 0.1])
    with pytest.raises(ValueError, match="positive"):
        I.fit_svi(np.linspace(-1, 1, 6), [-0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def test_butterfly_check_catches_an_arbitrage() -> None:
    assert I.butterfly_arbitrage_free(TRUE)
    # Gatheral & Jacquier's example of a slice with butterfly arbitrage
    bad = I.SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    assert not I.butterfly_arbitrage_free(bad)


def test_a_fit_is_pushed_away_from_butterfly_arbitrage() -> None:
    """Noisy quotes with a hump: the plain fit would have negative density within the range."""
    k = np.array([-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3])
    w = np.array([0.05, 0.02, 0.012, 0.011, 0.010, 0.011, 0.012, 0.02, 0.05]) * 1.0
    fit = I.fit_svi(k, w)
    assert I.butterfly_arbitrage_free(fit.params, np.linspace(fit.k_lo, fit.k_hi, 201), tol=1e-3)


def test_a_later_slice_is_lifted_above_an_earlier_one() -> None:
    k = np.linspace(-0.3, 0.3, 13)
    low = I.SVIParams(a=0.004, b=0.05, rho=-0.5, m=0.0, sigma=0.15)
    fit = I.fit_svi(k, low.total_variance(k), floor=TRUE)
    assert fit.lifted > 0
    grid = np.linspace(fit.k_lo, fit.k_hi, 41)
    assert np.all(fit.params.total_variance(grid) >= TRUE.total_variance(grid) - 1e-12)


def test_calendar_check() -> None:
    later = I.SVIParams(a=0.02, b=0.08, rho=-0.5, m=0.02, sigma=0.15)
    assert I.calendar_violations([(0.1, TRUE), (0.2, later)]) == []
    v = I.calendar_violations([(0.1, later), (0.2, TRUE)])
    assert len(v) == 1 and v[0][2] > 0


# ---------------------------------------------------------------------------- American trees
def test_tree_converges_to_the_closed_form() -> None:
    for cp in (1, -1):
        assert abs(A.european_check(100.0, 105.0, 0.5, 0.03, 0.01, 0.25, cp)) < 0.01


def test_hull_american_put() -> None:
    """Hull's example: S=50, K=50, r=10%, vol=40%, five months; the limit of the tree is 4.28."""
    assert A.price(50.0, 50.0, 5.0 / 12.0, 0.10, 0.0, 0.40, -1, steps=500) == pytest.approx(
        4.283, abs=0.01
    )


def test_call_without_dividends_is_never_exercised_early() -> None:
    assert A.early_exercise_premium(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 1) == pytest.approx(
        0.0, abs=1e-6
    )


def test_put_has_an_early_exercise_premium_and_respects_intrinsic() -> None:
    assert A.early_exercise_premium(80.0, 100.0, 1.0, 0.05, 0.0, 0.3, -1) > 0.1
    assert A.price(50.0, 100.0, 1.0, 0.05, 0.0, 0.3, -1) >= 50.0 - 1e-9


def test_a_dividend_gives_a_call_an_early_exercise_premium() -> None:
    div = [(0.25, 5.0)]
    assert A.early_exercise_premium(100.0, 90.0, 0.5, 0.02, 0.0, 0.2, 1, div) > 0.05
    # and a dividend lowers a call, raises a put
    assert A.price(100.0, 100.0, 0.5, 0.02, 0.0, 0.2, 1, True, div) < A.price(
        100.0, 100.0, 0.5, 0.02, 0.0, 0.2, 1
    )
    assert A.price(100.0, 100.0, 0.5, 0.02, 0.0, 0.2, -1, True, div) > A.price(
        100.0, 100.0, 0.5, 0.02, 0.0, 0.2, -1
    )


def test_greeks_match_the_closed_form_for_an_american_call_without_dividends() -> None:
    g = A.greeks(100.0, 100.0, 0.5, 0.03, 0.0, 0.25, 1)
    assert g["delta"] == pytest.approx(
        float(black.delta(100, 100, 0.5, 0.03, 0, 0.25, 1)), abs=0.01
    )
    assert g["gamma"] == pytest.approx(float(black.gamma(100, 100, 0.5, 0.03, 0, 0.25)), rel=0.1)
    assert g["vega"] == pytest.approx(float(black.vega(100, 100, 0.5, 0.03, 0, 0.25)), rel=0.02)
    assert g["theta"] < 0


def test_implied_vol_round_trip_and_refusal() -> None:
    p = A.price(100.0, 95.0, 0.5, 0.03, 0.0, 0.32, -1, steps=100)
    assert A.implied_vol(p, 100.0, 95.0, 0.5, 0.03, 0.0, -1) == pytest.approx(0.32, abs=0.002)
    assert math.isnan(A.implied_vol(1.0, 50.0, 100.0, 0.5, 0.03, 0.0, -1))  # below intrinsic


# ---------------------------------------------------------------------------- surfaces
def _slice_rows(
    expiry: str, asof: str, params: I.SVIParams, F: float, r: float, spread: float = 0.02
) -> list[dict[str, object]]:
    T = (pd.Timestamp(expiry) - pd.Timestamp(asof)).days / V.DAYS
    rows: list[dict[str, object]] = []
    for K in np.arange(F * 0.7, F * 1.31, F * 0.05):
        vol = float(params.implied_vol(math.log(K / F), T))
        for name, cp in (("Call", 1), ("Put", -1)):
            mid = float(black.price(F * math.exp(-r * T), K, T, r, 0.0, vol, cp))
            rows.append(
                {
                    "expiration": pd.Timestamp(expiry),
                    "strike": float(K),
                    "call_put": name,
                    "bid": mid * (1 - spread),
                    "ask": mid * (1 + spread),
                    "vol": vol,
                }
            )
    return rows


def test_parity_forward_recovers_the_forward() -> None:
    rows = pd.DataFrame(_slice_rows("2026-03-01", "2026-01-30", TRUE, 102.0, 0.04))
    T = 30 / V.DAYS
    assert V.parity_forward(rows, T, 0.04) == pytest.approx(102.0, abs=0.05)


def test_slice_fit_recovers_the_vols_and_says_so() -> None:
    rows = pd.DataFrame(_slice_rows("2026-03-01", "2026-01-30", TRUE, 100.0, 0.0))
    f = V.fit_slice(rows, pd.Timestamp("2026-01-30"), 0.0)
    assert f.rmse_vol < 0.005
    assert f.butterfly_free
    T = f.T
    assert f.vol(100.0) == pytest.approx(float(TRUE.implied_vol(0.0, T)), abs=0.01)


def test_a_thin_slice_is_refused() -> None:
    rows = pd.DataFrame(_slice_rows("2026-03-01", "2026-01-30", TRUE, 100.0, 0.0)).iloc[10:14]
    with pytest.raises(V.SurfaceError):
        V.fit_slice(rows, pd.Timestamp("2026-01-30"), 0.0)


def _surface() -> V.VolSurface:
    later = I.SVIParams(a=0.03, b=0.10, rho=-0.5, m=0.02, sigma=0.15)
    rows = _slice_rows("2026-03-01", "2026-01-30", TRUE, 100.0, 0.0) + _slice_rows(
        "2026-05-01", "2026-01-30", later, 100.0, 0.0
    )
    return V.build_surface(pd.DataFrame(rows), "X", pd.Timestamp("2026-01-30"), 0.0)


def test_surface_interpolates_in_total_variance_and_does_not_extrapolate() -> None:
    s = _surface()
    a, b = s.slices
    T = (a.T + b.T) / 2.0
    w_mid = (float(a.params.total_variance(0.0)) + float(b.params.total_variance(0.0))) / 2.0
    assert s.vol(100.0, T) == pytest.approx(math.sqrt(w_mid / T), abs=1e-6)
    with pytest.raises(V.SurfaceError):
        s.vol(100.0, b.T + 0.1)
    assert s.calendar_free


def test_surface_prices_the_smile_where_one_vol_cannot() -> None:
    """The point of the surface: a 25% OTM put is worth more than ATM vol says."""
    s = _surface()
    T = s.slices[0].T
    atm = s.vol(100.0, T)
    flat = float(black.price(100.0, 75.0, T, 0.0, 0.0, atm, -1))
    assert s.price(75.0, T, 0.0, -1) > 2.0 * flat


# ---------------------------------------------------------------------------- quoted strategies
def _market(n: int = 40, price: float = 100.0, drift: float = 0.0) -> tuple[Panel, np.ndarray]:
    dates = np.array(pd.bdate_range("2026-01-05", periods=n), dtype="datetime64[D]")
    close = price * (1.0 + drift) ** np.arange(n)
    c = close[:, None]
    p = Panel(
        dates,
        ("AAA@2016-01-01",),
        {
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": np.ones_like(c),
            "in_universe": np.ones_like(c),
        },
    )
    return p, c


def _quote(
    day: str, cp: str, K: float, bid: float, ask: float, delta: float, exp: str
) -> dict[str, object]:
    d, e = pd.Timestamp(day), pd.Timestamp(exp)
    return {
        "date": d,
        "symbol": "AAA",
        "expiration": e,
        "strike": K,
        "call_put": cp,
        "bid": bid,
        "ask": ask,
        "vol": 0.3,
        "delta": delta,
        "dte": (e - d).days,
    }


def _run(
    spec: Q.QuotedSpec, rows: list[dict[str, object]], p: Panel, c: np.ndarray
) -> Q.QuotedResult:
    frame = pd.DataFrame(rows)
    chain = {pd.Timestamp(k): v for k, v in frame.groupby("date")}
    score = np.ones_like(c)
    rv = np.full_like(c, 0.2)
    return Q.run_quoted(spec, p, c, chain, score, rv, np.zeros(p.n_dates), 0, p.n_dates - 1)


def test_pick_contracts_takes_the_nearest_expiry_then_delta() -> None:
    rows = [
        _quote("2026-01-06", "Call", 100.0, 2.0, 2.2, 0.50, "2026-02-05"),
        _quote("2026-01-06", "Call", 110.0, 1.0, 1.2, 0.30, "2026-02-05"),
        _quote("2026-01-06", "Call", 100.0, 3.0, 3.2, 0.52, "2026-02-20"),
        _quote("2026-01-06", "Call", 95.0, 4.0, 4.2, 0.70, "2026-01-16"),  # outside the range
    ]
    got = Q.pick_contracts(pd.DataFrame(rows), "Call", 0.3, 30, (20, 45))
    assert len(got) == 1
    assert (got["strike"].iloc[0], got["expiration"].iloc[0]) == (110.0, pd.Timestamp("2026-02-05"))


def test_a_long_call_pays_the_ask_and_settles_on_the_raw_close() -> None:
    p, c = _market(drift=0.005)
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Call", 100.0, 2.0, 2.4, 0.5, exp)]
    res = _run(Q.QuotedSpec("long_call", dte_range=(0, 60), max_rel_spread=1.0), rows, p, c)
    # delta-dollars of the whole book: 10M / (0.5 x S), capped by a 25% premium budget
    S = float(c[1, 0])
    qty_uncapped = 10_000_000.0 / (0.5 * S)
    scale = min(1.0, 0.25 * 10_000_000.0 / (qty_uncapped * 2.4))
    qty = qty_uncapped * scale
    pay = qty * max(float(c[25, 0]) - 100.0, 0.0)
    expected = 10_000_000.0 - qty * 2.4 - qty * Q.FEE_PER_SHARE + pay
    assert res.equity[25] == pytest.approx(expected, rel=1e-9)
    assert res.rolls >= 1


def test_a_written_put_keeps_the_bid_when_it_expires_worthless() -> None:
    p, c = _market(drift=0.001)
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 90.0, 1.0, 1.2, -0.2, exp)]
    res = _run(Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0), rows, p, c)
    qty = 10_000_000.0 / 90.0
    assert res.equity[25] == pytest.approx(
        10_000_000.0 + qty * 1.0 - qty * Q.FEE_PER_SHARE, rel=1e-9
    )


def test_a_written_put_loses_what_the_stock_falls_through_it() -> None:
    p, c = _market(drift=-0.01)
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 90.0, 1.0, 1.2, -0.2, exp)]
    res = _run(Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0), rows, p, c)
    qty = 10_000_000.0 / 90.0
    loss = qty * (90.0 - float(c[25, 0]))
    assert res.equity[25] == pytest.approx(
        10_000_000.0 + qty * 1.0 - qty * Q.FEE_PER_SHARE - loss, rel=1e-9
    )


def test_doubling_the_cost_doubles_the_half_spread_around_mid() -> None:
    assert Q._entry_price(1.0, 1.2, True, 1.0) == pytest.approx(1.2)
    assert Q._entry_price(1.0, 1.2, False, 1.0) == pytest.approx(1.0)
    assert Q._entry_price(1.0, 1.2, True, 2.0) == pytest.approx(1.3)
    assert Q._entry_price(1.0, 1.2, False, 2.0) == pytest.approx(0.9)


def test_a_wide_market_is_not_traded() -> None:
    p, c = _market()
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 90.0, 1.0, 3.0, -0.2, exp)]
    res = _run(Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=0.5), rows, p, c)
    assert np.allclose(res.equity[np.isfinite(res.equity)], 10_000_000.0)


def test_a_position_opens_only_after_the_signal_is_known() -> None:
    """Row 0 is the first session: nothing can be decided before a previous close exists."""
    p, c = _market()
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[0]), "Put", 90.0, 1.0, 1.2, -0.2, exp)]
    res = _run(Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0), rows, p, c)
    assert res.n_positions.sum() == 0


def test_the_lag_stress_enters_one_quoted_date_later() -> None:
    p, c = _market()
    exp = str(p.dates[25])
    rows = [
        _quote(str(p.dates[1]), "Put", 90.0, 1.0, 1.2, -0.2, exp),
        _quote(str(p.dates[3]), "Put", 90.0, 1.0, 1.2, -0.2, exp),
    ]
    spec = Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0)
    base, lagged = _run(spec, rows, p, c), _run(Q.stressed(spec)["lag2"], rows, p, c)
    assert base.n_positions[1] == 1 and lagged.n_positions[1] == 0 and lagged.n_positions[3] == 1


def test_a_position_marks_at_its_own_vol_between_quotes() -> None:
    p, c = _market()
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 90.0, 1.0, 1.2, -0.2, exp)]
    res = _run(Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0), rows, p, c)
    assert np.isfinite(res.equity).all()
    assert res.equity[2] != res.equity[1]  # valued at the model, not frozen


def test_a_book_wiped_out_stays_at_zero() -> None:
    p, c = _market(drift=-0.2)
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 90.0, 1.0, 1.2, -0.2, exp)]
    spec = Q.QuotedSpec("short_put", 0.2, dte_range=(0, 60), max_rel_spread=1.0)
    frame = pd.DataFrame(rows)
    chain = {pd.Timestamp(k): v for k, v in frame.groupby("date")}
    # written with ten times the capital at risk: the cash-securing is what normally prevents this
    res = Q.run_quoted(
        spec, p, c, chain, np.ones_like(c), np.ones_like(c), np.zeros(p.n_dates), 0, 39, 1e6
    )
    assert res.equity[-1] >= 0.0
    assert np.isfinite(res.returns).all()


def test_a_split_rescales_the_contract_and_keeps_its_value() -> None:
    p, c = _market(n=40, drift=0.0)
    raw = c.copy()
    raw[10:] = raw[10:] / 2.0  # 2-for-1 on row 10, the adjusted series does not move
    exp = str(p.dates[25])
    rows = [_quote(str(p.dates[1]), "Put", 110.0, 10.0, 10.2, -0.5, exp)]
    frame = pd.DataFrame(rows)
    chain = {pd.Timestamp(k): v for k, v in frame.groupby("date")}
    spec = Q.QuotedSpec("short_put", 0.5, dte_range=(0, 60), max_rel_spread=1.0)
    res = Q.run_quoted(
        spec, p, raw, chain, np.ones_like(c), np.ones_like(c), np.zeros(p.n_dates), 0, 39
    )
    qty = 10_000_000.0 / 110.0
    # a put 10 in the money before the split is 5 in the money on twice the shares after it
    expected = 10_000_000.0 + qty * 10.0 - qty * Q.FEE_PER_SHARE - qty * 10.0
    assert res.equity[25] == pytest.approx(expected, rel=1e-9)


def test_straddles_buy_the_cheap_vol_and_sell_the_dear() -> None:
    dates = np.array(pd.bdate_range("2026-01-05", periods=30), dtype="datetime64[D]")
    c = np.full((30, 2), 100.0)
    p = Panel(
        dates,
        ("AAA@2016-01-01", "BBB@2016-01-01"),
        {
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": np.ones_like(c),
            "in_universe": np.ones_like(c),
        },
    )
    day, exp = str(dates[1]), str(dates[25])
    rows = []
    for sym, vol in (("AAA", 0.2), ("BBB", 0.5)):
        for cp, d in (("Call", 0.5), ("Put", -0.5)):
            r = _quote(day, cp, 100.0, 2.0, 2.2, d, exp)
            r["symbol"], r["vol"] = sym, vol
            rows.append(r)
    frame = pd.DataFrame(rows)
    spec = Q.QuotedSpec("straddle_ls", dte_range=(0, 60), max_names=2, max_rel_spread=1.0)
    book = Q._Book(1_000_000.0)
    Q._open_positions(book, spec, frame, p, 1, np.ones_like(c), c, c, np.full_like(c, 0.3))
    by_symbol = {lg.symbol: lg.qty for lg in book.legs}
    assert by_symbol["AAA"] > 0 and by_symbol["BBB"] < 0
    assert len(book.legs) == 4
