"""Data pipeline end to end against a fake Alpaca: universe -> store -> panel -> updates.

The fake server paginates small pages, drops one symbol from multi-symbol requests (the real
API did this to DOW next to DD), honours ``asof`` and can apply a split to a history.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pandas as pd
import pytest

from pusula.core import calendar as cal
from pusula.data import universe as U
from pusula.data.alpaca import AlpacaClient
from pusula.data.pipeline import build_store, research_panel, update_store
from pusula.data.settings import AlpacaCredentials


class FakeAlpaca:
    def __init__(self, tickers: list[str], end: dt.date, drop_in_batch: str = "BBB") -> None:
        rng = np.random.default_rng(0)
        self.sessions = [d.item() for d in cal.sessions(dt.date(2016, 1, 4), end)]
        self.px: dict[str, np.ndarray] = {
            t: 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(self.sessions)))) for t in tickers
        }
        self.drop = drop_in_batch
        self.requests: list[dict[str, str]] = []
        self.page = 300

    def extend(self, end: dt.date) -> None:
        rng = np.random.default_rng(1)
        new = [d.item() for d in cal.sessions(self.sessions[-1] + dt.timedelta(days=1), end)]
        for t, p in self.px.items():
            self.px[t] = np.concatenate(
                [p, p[-1] * np.exp(np.cumsum(rng.normal(0, 0.01, len(new))))]
            )
        self.sessions += new

    def split(self, ticker: str, ratio: float) -> None:
        """A split today: the vendor rescales the whole adjusted history."""
        self.px[ticker] = self.px[ticker] / ratio

    def handler(self, req: httpx.Request) -> httpx.Response:
        q = {k: v[0] for k, v in parse_qs(urlparse(str(req.url)).query).items()}
        self.requests.append(q)
        syms = q["symbols"].split(",")
        start, end = dt.date.fromisoformat(q["start"]), dt.date.fromisoformat(q["end"])
        rows: list[tuple[str, dict[str, object]]] = []
        for s in syms:
            if s not in self.px or (s == self.drop and len(syms) > 1):
                continue
            for i, d in enumerate(self.sessions):
                if start <= d <= end:
                    c = float(self.px[s][i])
                    rows.append(
                        (
                            s,
                            {
                                "t": f"{d.isoformat()}T05:00:00Z",
                                "o": c,
                                "h": c * 1.01,
                                "l": c * 0.99,
                                "c": c,
                                "v": 1e6,
                                "vw": c,
                                "n": 1000,
                            },
                        )
                    )
        off = int(q.get("page_token", "0"))
        chunk = rows[off : off + self.page]
        bars: dict[str, list[dict[str, object]]] = {}
        for s, b in chunk:
            bars.setdefault(s, []).append(b)
        nxt = str(off + self.page) if off + self.page < len(rows) else None
        return httpx.Response(200, json={"bars": bars, "next_page_token": nxt})

    def client(self) -> AlpacaClient:
        return AlpacaClient(
            AlpacaCredentials("k", "s"), transport=httpx.MockTransport(self.handler)
        )


def _history() -> U.IndexHistory:
    current = ("AAA", "BBB", *[f"T{i:03d}" for i in range(3)])
    changes = (U.Change(dt.date(2017, 3, 1), "BBB", "CCC", "test"),)
    return U.IndexHistory(
        current, {"AAA": "Tech", "BBB": "Energy"}, changes, dt.date(2018, 6, 29), "x"
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeAlpaca:
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    return FakeAlpaca(["AAA", "BBB", "CCC", "T000", "T001", "T002", "SPY"], dt.date(2018, 6, 29))


def test_build_store_and_research_panel(tmp_path: Path, fake: FakeAlpaca) -> None:
    store = build_store(
        tmp_path, end=dt.date(2018, 6, 29), client=fake.client(), history=_history(), log=False
    )
    ids = store.ids()
    assert "CCC@2016-01-01" in ids and "BBB@2017-03-01" in ids and "SPY" in ids
    assert store.record("BBB@2017-03-01").rows == len(
        fake.sessions
    )  # re-asked alone after the drop
    assert any(r["symbols"] == "BBB" for r in fake.requests)
    assert any(r.get("asof") == "2017-02-28" and "CCC" in r["symbols"] for r in fake.requests)
    assert any("page_token" in r for r in fake.requests)  # pagination exercised
    rd = research_panel(tmp_path)
    p = rd.panel
    m = p["in_universe"]
    i_before, i_after = p.date_index("2017-02-28"), p.date_index("2017-03-01")
    assert (
        m[i_before, p.symbol_index("CCC@2016-01-01")] == 1
        and m[i_after, p.symbol_index("CCC@2016-01-01")] == 0
    )
    assert (
        m[i_before, p.symbol_index("BBB@2017-03-01")] == 0
        and m[i_after, p.symbol_index("BBB@2017-03-01")] == 1
    )
    assert rd.benchmarks.symbols == ("SPY",)
    assert rd.quality.calendar_sessions_missing == ()
    assert rd.sectors["AAA@2016-01-01"] == "Tech"


def test_update_appends_and_redownloads_after_split(tmp_path: Path, fake: FakeAlpaca) -> None:
    build_store(
        tmp_path, end=dt.date(2018, 6, 29), client=fake.client(), history=_history(), log=False
    )
    fake.extend(dt.date(2018, 7, 13))
    rebuilt = update_store(tmp_path, end=dt.date(2018, 7, 13), client=fake.client(), log=False)
    assert rebuilt == []
    from pusula.data.store import BarStore

    st = BarStore(tmp_path)
    aaa = st.get("AAA@2016-01-01")
    assert aaa.index.max() == pd.Timestamp("2018-07-13") and not aaa.index.has_duplicates
    np.testing.assert_allclose(aaa["close"].to_numpy(), fake.px["AAA"])
    # a split rescales the vendor's adjusted history: splicing would create a fake -75% day
    fake.extend(dt.date(2018, 7, 20))
    fake.split("AAA", 4.0)
    rebuilt = update_store(tmp_path, end=dt.date(2018, 7, 20), client=fake.client(), log=False)
    assert rebuilt == ["AAA@2016-01-01"]
    aaa = BarStore(tmp_path).get("AAA@2016-01-01")
    np.testing.assert_allclose(aaa["close"].to_numpy(), fake.px["AAA"])
    r = aaa["close"].pct_change().dropna()
    assert r.abs().max() < 0.2
    meta = json.loads((tmp_path / "manifest.json").read_text())
    assert meta["AAA@2016-01-01"]["rows"] == len(fake.sessions)


def test_client_retries_rate_limit_then_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from pusula.core.errors import DataError
    from pusula.data.alpaca import BarRequest

    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] < 3 else 403, text="forbidden")

    cli = AlpacaClient(AlpacaCredentials("k", "s"), transport=httpx.MockTransport(handler))
    with pytest.raises(DataError, match="HTTP 403"):
        cli.daily_bars(BarRequest(("AAA",), dt.date(2020, 1, 1), dt.date(2020, 1, 31)))
    assert calls["n"] == 3
