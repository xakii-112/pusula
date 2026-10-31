"""Binance data, the crypto universe, sessions, relistings, and the book recording."""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from pusula.core.errors import DataError
from pusula.data import binance as B
from pusula.data import crypto as C
from pusula.data.panel import Panel
from pusula.research.families import us_crypto_v1
from pusula.research.study import registered_config

NOW = dt.datetime(2026, 10, 30, tzinfo=dt.UTC)


def _bars(days: pd.DatetimeIndex, price: float = 10.0, step: float = 0.0) -> pd.DataFrame:
    c = price * (1.0 + step) ** np.arange(len(days))
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6 / c,
            "trades": 100.0,
            "vwap": c,
        },
        index=days,
    )


def _put(store: Any, symbol: str, frame: pd.DataFrame) -> None:
    store.put(symbol, symbol, None, frame, adjustment="raw", feed="binance", fetched_at=NOW)
    store.flush()


# ---------------------------------------------------------------------------- pairs
@pytest.mark.parametrize(
    ("base", "quote", "member"),
    [
        ("BTC", "USDT", True),
        ("LUNA", "USDT", True),
        ("BTC", "BUSD", False),
        ("USDC", "USDT", False),
        ("EUR", "USDT", False),
        ("BTCUP", "USDT", False),
        ("ETHBEAR", "USDT", False),
        ("UP", "USDT", True),  # a coin that is only called UP
    ],
)
def test_member_pairs(base: str, quote: str, member: bool) -> None:
    assert C.is_member_pair({"baseAsset": base, "quoteAsset": quote}) is member


def test_klines_frame_reads_dollars_and_keeps_empty_typed() -> None:
    row = [1793318400000, "1", "2", "0.5", "1.5", "10", 0, "15", 7, "0", "0", "0"]
    df = B.klines_frame([row])
    assert df["quote_volume"].iloc[0] == 15.0 and df["trades"].iloc[0] == 7.0
    assert df.index[0] == pd.Timestamp("2026-10-30")
    assert B.klines_frame([]).empty


def test_client_retries_then_gives_a_clean_error() -> None:
    calls = {"n": 0}

    def flaky(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": 1})

    with B.BinanceClient(httpx.MockTransport(flaky), min_gap=0.0) as c:
        assert c.get("/x") == {"ok": 1}
    with (
        B.BinanceClient(
            httpx.MockTransport(lambda r: httpx.Response(400, text="bad")), min_gap=0.0
        ) as c,
        pytest.raises(DataError, match="HTTP 400"),
    ):
        c.get("/x")


def test_klines_pages_through_the_limit() -> None:
    first = pd.Timestamp("2020-01-01")

    def handler(r: httpx.Request) -> httpx.Response:
        start = pd.Timestamp(int(r.url.params["startTime"]), unit="ms")
        n = min(B.KLINE_LIMIT, 1500 - (start - first).days)
        rows = [
            [
                int((start + pd.Timedelta(days=i)).timestamp() * 1000),
                "1",
                "1",
                "1",
                "1",
                "1",
                0,
                "1",
                1,
            ]
            for i in range(n)
        ]
        return httpx.Response(200, json=rows)

    with B.BinanceClient(httpx.MockTransport(handler), min_gap=0.0) as c:
        df = c.klines("X", first, first + pd.Timedelta(days=1499))
    assert len(df) == 1500 and df.index.is_unique


# ---------------------------------------------------------------------------- sessions
def test_weekend_folds_into_monday() -> None:
    days = pd.date_range("2026-10-23", "2026-10-26", freq="D")  # Friday .. Monday
    frame = _bars(days)
    frame["close"] = [1.0, 2.0, 3.0, 4.0]
    frame["high"] = [1.5, 9.0, 3.5, 4.5]
    frame["low"] = [0.5, 0.5, 0.1, 3.5]
    frame["open"] = [1.0, 1.0, 2.0, 3.0]
    frame["volume"] = [10.0, 20.0, 30.0, 40.0]
    frame["vwap"] = [1.0, 1.0, 2.0, 3.0]
    out = C.fold_to_sessions(frame)
    assert list(out.index) == [pd.Timestamp("2026-10-23"), pd.Timestamp("2026-10-26")]
    mon = out.loc["2026-10-26"]
    assert (mon["open"], mon["close"], mon["high"], mon["low"]) == (1.0, 4.0, 9.0, 0.1)
    assert mon["volume"] == 90.0
    assert mon["vwap"] == pytest.approx((20 + 60 + 120) / 90)


def test_folding_keeps_the_compounded_return() -> None:
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    frame = _bars(days, step=0.01)
    out = C.fold_to_sessions(frame)
    assert out["close"].iloc[-1] / out["close"].iloc[0] == pytest.approx(
        frame["close"].iloc[-1] / frame.loc[out.index[0], "close"]
    )


def test_a_session_waits_for_its_own_day(tmp_path: Path) -> None:
    store = C.store_of(tmp_path)
    _put(store, "AAAUSDT", _bars(pd.date_range("2026-01-01", "2026-03-08", freq="D")))  # a Sunday
    p = C.crypto_data(tmp_path).panel
    assert p.dates[-1] == np.datetime64("2026-03-06")  # Monday's bar needs Monday


# ---------------------------------------------------------------------------- relistings
def test_a_relisted_symbol_is_a_new_security(tmp_path: Path) -> None:
    """LUNAUSDT, 2022-05-13 -> 05-31: the same name, another coin, 177,400 times the price."""
    store = C.store_of(tmp_path)
    before = _bars(pd.date_range("2022-01-03", "2022-05-13", freq="D"), price=0.00005)
    after = _bars(pd.date_range("2022-05-31", "2022-09-30", freq="D"), price=8.87)
    _put(store, "LUNAUSDT", pd.concat([before, after]))
    p = C.crypto_data(tmp_path).panel
    assert set(p.symbols) == {"LUNAUSDT", "LUNAUSDT#2"}
    close = p["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        moves = close[1:] / close[:-1]
    assert np.nanmax(moves) < 1.5
    old, new = p.symbol_index("LUNAUSDT"), p.symbol_index("LUNAUSDT#2")
    assert not (np.isfinite(close[:, old]) & np.isfinite(close[:, new])).any()


def test_a_short_gap_does_not_split() -> None:
    days = pd.date_range("2022-01-03", periods=40, freq="D").delete([10])
    assert len(C.split_relistings("X", _bars(days))) == 1
    wide = pd.date_range("2022-01-03", periods=40, freq="D").delete([10, 11, 12])
    assert list(C.split_relistings("X", _bars(wide))) == ["X", "X#2"]


# ---------------------------------------------------------------------------- universe
def test_members_are_chosen_without_the_future() -> None:
    dates = np.array(pd.date_range("2026-01-01", periods=120, freq="B"), dtype="datetime64[D]")
    rng = np.random.default_rng(1)
    dv = rng.uniform(2e6, 3e6, (120, 5))
    closes = np.ones((120, 5))
    a = C.monthly_members(dv, dates, closes)
    changed = dv.copy()
    changed[70:, 0] = 1e12  # huge volume later
    b = C.monthly_members(changed, dates, closes)
    assert np.array_equal(a[:70], b[:70])


def test_top_n_and_minimum_volume() -> None:
    dates = np.array(pd.date_range("2026-01-01", periods=120, freq="B"), dtype="datetime64[D]")
    N = C.TOP_N + 10
    dv = np.tile(np.arange(1, N + 1) * 1e6, (120, 1))
    dv[:, 0] = 1e3
    closes = np.ones((120, N))
    m = C.monthly_members(dv, dates, closes)
    assert m[-1].sum() == C.TOP_N
    assert m[-1, 0] == 0 and m[-1, -1] == 1


def test_a_member_leaves_when_its_price_ends() -> None:
    dates = np.array(pd.date_range("2026-01-01", periods=120, freq="B"), dtype="datetime64[D]")
    dv = np.full((120, 3), 5e6)
    closes = np.ones((120, 3))
    closes[100:, 1] = np.nan
    m = C.monthly_members(dv, dates, closes)
    assert m[99, 1] == 1 and m[100, 1] == 0


def test_rescaling_keeps_returns_and_dollars(tmp_path: Path) -> None:
    store = C.store_of(tmp_path)
    frame = _bars(pd.date_range("2026-01-01", "2026-03-31", freq="D"), price=0.00001, step=0.01)
    _put(store, "SHIBUSDT", frame)
    p = C.crypto_data(tmp_path).panel
    close, vol, vwap = p["close"][:, 0], p["volume"][:, 0], p["vwap"][:, 0]
    assert close[0] == pytest.approx(100.0)
    start = pd.Timestamp(p.dates[0])  # 1 January is a holiday: the first bar closes on the 2nd
    assert close[-1] / close[0] == pytest.approx(
        frame["close"].iloc[-1] / frame.loc[start, "close"]
    )
    raw = C.fold_to_sessions(frame)
    assert (vol * vwap)[5] == pytest.approx(raw["volume"].iloc[5] * raw["vwap"].iloc[5])


def test_update_fetches_only_what_is_missing(tmp_path: Path) -> None:
    asked: list[str] = []

    class Client:
        def symbols(self) -> list[dict[str, str]]:
            return [
                {"symbol": "AAAUSDT", "baseAsset": "AAA", "quoteAsset": "USDT"},
                {"symbol": "USDCUSDT", "baseAsset": "USDC", "quoteAsset": "USDT"},
            ]

        def klines(self, symbol: str, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
            asked.append(f"{symbol} {start.date()}")
            days = pd.date_range(start, stop, freq="D")
            f = _bars(days)
            return pd.DataFrame(
                {
                    "open": f["open"],
                    "high": f["high"],
                    "low": f["low"],
                    "close": f["close"],
                    "volume": f["volume"],
                    "quote_volume": f["volume"] * f["close"],
                    "trades": f["trades"],
                }
            )

    client = Client()
    C.update_crypto(tmp_path, dt.date(2018, 1, 10), client, log=False)  # type: ignore[arg-type]
    C.update_crypto(tmp_path, dt.date(2018, 1, 12), client, log=False)  # type: ignore[arg-type]
    assert asked == ["AAAUSDT 2018-01-01", "AAAUSDT 2018-01-11"]
    assert len(C.store_of(tmp_path).get("AAAUSDT")) == 12


def test_factors_read_bitcoin_and_members_known_before_the_day() -> None:
    dates = np.array(pd.date_range("2026-01-01", periods=4, freq="B"), dtype="datetime64[D]")
    close = np.array([[100.0, 10.0], [110.0, 10.0], [110.0, 20.0], [110.0, 20.0]])
    p = Panel(
        dates,
        (C.BTC, "XUSDT"),
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones_like(close),
            "in_universe": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
        },
    )
    french = pd.DataFrame(
        {"mkt_rf": 0.0, "rf": 0.0}, index=pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    )
    f = C.crypto_factors(p, french)
    assert f["btc"].iloc[0] == pytest.approx(0.10)
    # X doubled on the third session, while it was not yet a known member: not in that day's market
    assert f["cmkt"].iloc[1] == pytest.approx(0.0)


def test_family_registers_its_universe_and_factors() -> None:
    spec = us_crypto_v1()
    cfg = registered_config(spec, spec.trials[0])
    assert cfg["universe"] == [C.UNIVERSE]
    assert cfg["factors"] == list(C.CRYPTO_FACTORS)
    assert {t.name for t in spec.trials if t.benchmark} == {"equal_weight", "btc_hold"}


# ---------------------------------------------------------------------------- recording
def _write(path: Path, *events: tuple[str, object]) -> None:
    sink = B._Sink(path)
    for kind, data in events:
        sink.write(kind, data)
    sink.close()


def test_recording_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "x" / "r.jsonl.gz"
    _write(path, ("snapshot", {"lastUpdateId": 1}), ("depth", {"u": 2}), ("trade", {"p": "1"}))
    got = [(k, d) for _, k, d in B.read_recording(path)]
    assert got == [("snapshot", {"lastUpdateId": 1}), ("depth", {"u": 2}), ("trade", {"p": "1"})]


def test_a_recording_cut_by_a_crash_reads_up_to_the_cut(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl.gz"
    sink = B._Sink(path)
    sink.write("depth", {"u": 1})
    sink.flush()
    sink.write("depth", {"u": 2})  # never flushed
    size = path.stat().st_size
    data = path.read_bytes()
    assert size > 0
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(data)
    assert [d["u"] for _, _, d in B.read_recording(cut)] == [1]
    sink.close()
    assert [d["u"] for _, _, d in B.read_recording(path)] == [1, 2]


def test_a_half_written_line_is_not_read(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"t": 1, "k": "depth", "d": {}}) + "\n" + '{"t": 2, "k"')
    assert len(list(B.read_recording(path))) == 1
