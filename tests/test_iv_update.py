from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
import pytest

from pusula.core.errors import DataError
from pusula.data.iv import DoltClient, update_iv


def _store(tmp_path: Path) -> Path:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-25", "2026-08-26"]),
            "symbol": ["A", "A"],
            "iv": [0.2, 0.21],
        }
    ).to_parquet(tmp_path / "iv_history.parquet", index=False)
    (tmp_path / "iv_history.json").write_text(json.dumps({"rows": 2}), encoding="utf-8")
    return tmp_path


def _dolt(pages: dict[str, list[dict[str, object]]], calls: list[str], fail_first: int = 0):  # type: ignore[no-untyped-def]
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        q = parse_qs(urlparse(str(req.url)).query)["q"][0]
        calls.append(q)
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(429)
        if "DISTINCT date" in q:
            rows = [{"date": d} for d in sorted(pages)]
        elif "dolt_branches" in q:
            rows = [{"hash": "abc123"}]
        else:
            d = q.split("date = '")[1][:10]
            off = int(q.split("OFFSET ")[1])
            rows = pages[d][off : off + 2]
        return httpx.Response(200, json={"query_execution_status": "Success", "rows": rows})

    return httpx.MockTransport(handler)


def test_update_appends_new_dates_with_pagination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    store = _store(tmp_path)
    pages = {
        "2026-08-27": [
            {"date": "2026-08-27", "act_symbol": s, "iv_current": "0.25"} for s in "ABC"
        ],
        "2026-08-28": [
            {"date": "2026-08-28", "act_symbol": "A", "iv_current": "9.99"}
        ],  # junk value
    }
    calls: list[str] = []
    cli = DoltClient(transport=_dolt(pages, calls, fail_first=1), polite_delay=0)
    real_paged = cli.paged
    cli.paged = lambda sql, page=1000: real_paged(sql, page=2)  # type: ignore[method-assign]
    added = update_iv(store, cli, log=False)
    assert added == 3  # the 9.99 IV is rejected as implausible
    df = pd.read_parquet(store / "iv_history.parquet")
    assert len(df) == 5 and df["date"].max() == pd.Timestamp("2026-08-27")
    meta = json.loads((store / "iv_history.json").read_text())
    assert meta["updates"][-1]["dolt_commit"] == "abc123"
    assert any("OFFSET 2" in c for c in calls)  # paginated
    assert all("ORDER BY" in c for c in calls if "iv_current" in c)
    assert not any("dolt_log" in c for c in calls)  # sorting the commit log timed out


def test_a_night_without_new_dates_asks_for_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls: list[str] = []
    cli = DoltClient(transport=_dolt({}, calls), polite_delay=0)
    assert update_iv(_store(tmp_path), cli, log=False) == 0
    assert len(calls) == 1 and "DISTINCT date" in calls[0]


def test_server_error_is_never_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query_execution_status": "Error",
                "query_execution_message": "syntax error",
                "rows": [{"x": 1}],
            },
        )

    with pytest.raises(DataError, match="syntax error"):
        DoltClient(transport=httpx.MockTransport(handler), polite_delay=0).query("SELECT bad")


def test_bootstrap_from_whole_table_csv(tmp_path: Path) -> None:
    from pusula.data.iv import bootstrap_iv

    csv = (
        b"date,act_symbol,hv_current,iv_current\n2019-02-11,AAPL,0.3,0.25\n"
        b"2019-02-11,MSFT,0.2,0.22\n2019-02-12,AAPL,0.3,7.5\n"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=csv)

    out = bootstrap_iv(tmp_path / "iv", transport=httpx.MockTransport(handler))
    df = pd.read_parquet(out)
    assert len(df) == 2  # the 7.5 IV is rejected
    meta = json.loads((tmp_path / "iv" / "iv_history.json").read_text())
    assert meta["end"] == "2019-02-11" and len(meta["source_sha256"]) == 64
    assert not (tmp_path / "iv" / "volatility_history.csv.partial").exists()


def test_bootstrap_failure_leaves_no_store(tmp_path: Path) -> None:
    from pusula.data.iv import bootstrap_iv

    with pytest.raises(DataError, match="HTTP 503"):
        bootstrap_iv(tmp_path / "iv", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert not (tmp_path / "iv" / "iv_history.parquet").exists()
