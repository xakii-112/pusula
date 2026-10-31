"""The desktop window (with a stand-in for the webview) and the overview it opens on."""

from __future__ import annotations

import datetime as dt
import json
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from pusula import cli
from pusula.research.registry import Registry
from pusula.web import desktop as D
from pusula.web.app import create_app


class _Event:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, fn: Any) -> _Event:
        self.handlers.append(fn)
        return self


class _FakeWebview:
    """Records the window and, while it is "open", asks the server what the page would."""

    class errors:  # noqa: N801
        class WebViewException(Exception):  # noqa: N818
            pass

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.kwargs: dict[str, Any] = {}
        self.url = ""
        self.title = ""
        self.health_while_open = 0
        self.window = types.SimpleNamespace(events=types.SimpleNamespace(shown=_Event()))

    def create_window(self, title: str, url: str, **kw: Any) -> Any:
        self.title, self.url, self.kwargs = title, url, kw
        return self.window

    def start(self, **kw: Any) -> None:
        if self.fail:
            raise self.errors.WebViewException("no runtime")
        self.health_while_open = httpx.get(self.url + "healthz").status_code


def test_server_lives_exactly_as_long_as_the_window(tmp_path: Path) -> None:
    fake = _FakeWebview()
    D.open_window(tmp_path / "ledger.jsonl", webview=fake)
    assert fake.title == "Pusula"
    assert fake.url.startswith("http://127.0.0.1:")
    assert fake.health_while_open == 200
    assert fake.kwargs["maximized"] is True
    with pytest.raises(httpx.TransportError):
        httpx.get(fake.url + "healthz", timeout=1.0)


def test_a_failed_window_stops_the_server_and_says_what_to_do(tmp_path: Path) -> None:
    fake = _FakeWebview(fail=True)
    with pytest.raises(D.DesktopError, match="WebView2"):
        D.open_window(tmp_path / "ledger.jsonl", webview=fake)
    with pytest.raises(httpx.TransportError):
        httpx.get(fake.url + "healthz", timeout=1.0)


def test_server_starts_without_a_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under pythonw there is no stdout, and uvicorn's default logging config dies on it."""
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)
    server = D.AppServer(tmp_path / "ledger.jsonl", None)
    server.start()
    assert httpx.get(server.url + "healthz").status_code == 200
    server.stop()
    assert not server.running


def test_report_failure_keeps_the_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "home", lambda: tmp_path)
    D.report_failure("broken")
    assert "broken" in (tmp_path / "logs" / "app.err.log").read_text(encoding="utf-8")


def test_cli_reports_a_window_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    said: list[str] = []

    def boom(*a: object, **k: object) -> None:
        raise D.DesktopError("no window")

    monkeypatch.setattr(D, "open_window", boom)
    monkeypatch.setattr(D, "report_failure", said.append)
    assert cli.main(["app"]) == 1
    assert said == ["no window"]


def test_icon_is_a_valid_ico() -> None:
    data = D.ICON.read_bytes()
    assert data[:4] == b"\x00\x00\x01\x00"
    assert int.from_bytes(data[4:6], "little") == 5


# ---------------------------------------------------------------------------- overview
NOW = dt.datetime(2026, 10, 30, tzinfo=dt.UTC)


def _result(reg: Registry, family: str, name: str, sharpe: float, benchmark: bool) -> None:
    h = reg.preregister(
        family=family,
        name=name,
        hypothesis="h",
        config={"benchmark": benchmark},
        gates={},
        windows={},
        now=NOW,
    )
    verdict = {"promoted": False, "benchmark": True} if benchmark else {"promoted": False}
    reg.record_result(
        h,
        metrics={"holdout": {"sharpe": sharpe}},
        verdict=verdict,
        data_digest="d",
        now=NOW,
    )


def _client(tmp_path: Path) -> TestClient:
    reg = Registry(tmp_path / "ledger.jsonl")
    _result(reg, "us-a-v1", "x", 0.2, False)
    _result(reg, "us-a-v1", "y", 0.7, False)
    _result(reg, "us-a-v1", "equal_weight", 0.5, True)
    _result(reg, "us-b-v1", "z", -0.1, False)
    fwd = tmp_path / "forward"
    root = fwd / "us-a-v1"
    root.mkdir(parents=True)
    snap = {
        "as_of": "2026-10-29",
        "books": {
            "x": {
                "equity_curve": [["2026-10-28", 100.0], ["2026-10-29", 102.0]],
                "halted": False,
                "last_trades": {"n_trades": 4},
            },
            "y": {
                "equity_curve": [["2026-10-28", 100.0], ["2026-10-29", 99.0]],
                "halted": True,
                "last_trades": {"n_trades": 0},
            },
        },
    }
    (root / "dashboard.json").write_text(json.dumps(snap), encoding="utf-8")
    return TestClient(create_app(tmp_path / "ledger.jsonl", fwd))


def test_overview_summarises_every_family(tmp_path: Path) -> None:
    ov = _client(tmp_path).get("/api/overview").json()
    assert ov["trials"] == 3 and ov["promoted"] == 0
    a, b = ov["families"]
    assert (a["family"], a["candidates"], a["promoted"]) == ("us-a-v1", 2, 0)
    assert a["best"] == {"name": "y", "sharpe": 0.7}
    assert a["benchmark"] == {"name": "equal_weight", "sharpe": 0.5}
    assert a["forward"]["as_of"] == "2026-10-29"
    assert (a["forward"]["books"], a["forward"]["traded"], a["forward"]["halted"]) == (2, 1, 1)
    assert a["forward"]["best"] == {"book": "x", "move": pytest.approx(0.02)}
    assert a["forward"]["worst"] == {"book": "y", "move": pytest.approx(-0.01)}
    assert b["benchmark"] is None and b["forward"] is None


def test_execution_reports_the_account_and_its_last_plan(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/execution").json() == {"accounts": {}}
    root = tmp_path / "execution" / "paper"
    root.mkdir(parents=True)
    (root / "account.json").write_text(
        json.dumps({"owner": "f/b", "last_session": "2026-10-29"}), encoding="utf-8"
    )
    (root / "plan-2026-10-29.json").write_text(
        json.dumps(
            {"orders": [{}, {}], "gross_after": 1.5, "outcome": {"a": "new", "b": "rejected"}}
        ),
        encoding="utf-8",
    )
    got = client.get("/api/execution").json()["accounts"]["paper"]
    assert got["owner"] == "f/b"
    assert (got["orders"], got["rejected"], got["gross_after"]) == (2, 1, 1.5)


def test_overview_stamp_moves_when_a_snapshot_is_rewritten(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.get("/api/overview").json()["stamp"]
    path = tmp_path / "forward" / "us-a-v1" / "dashboard.json"
    import os

    os.utime(path, (first + 100, first + 100))
    assert client.get("/api/overview").json()["stamp"] == pytest.approx(first + 100)
