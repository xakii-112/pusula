"""Read-only operations dashboard.

Binds to 127.0.0.1 and exposes no mutating endpoint: operating actions (flatten, reset, order
submission) are CLI commands with their own confirmation, never a button on a web page
(NihaiAlgoritma's dashboard writes raced the main loop; BULGU-106-30).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from pusula.core.errors import RegistryError
from pusula.data.settings import home
from pusula.research.registry import Registry

INDEX = Path(__file__).with_name("index.html")


def _forward_summary(snap: dict[str, Any]) -> dict[str, Any]:
    """A family's night: its books, how many traded or stopped, the session's extremes."""
    moves: list[tuple[float, str]] = []
    traded = halted = 0
    for name, book in snap.get("books", {}).items():
        traded += bool(book.get("last_trades", {}).get("n_trades"))
        halted += bool(book.get("halted"))
        curve = book.get("equity_curve", [])
        if len(curve) >= 2:
            moves.append((float(curve[-1][1]) / float(curve[-2][1]) - 1.0, str(name)))
    moves.sort()
    return {
        "as_of": snap.get("as_of"),
        "books": len(snap.get("books", {})),
        "traded": traded,
        "halted": halted,
        "best": {"book": moves[-1][1], "move": moves[-1][0]} if moves else None,
        "worst": {"book": moves[0][1], "move": moves[0][0]} if moves else None,
    }


def _family_summary(reg: Registry, family: str, forward: Path) -> dict[str, Any]:
    latest = reg.latest_results(family)
    cands: list[tuple[float, str]] = []
    bench: list[tuple[float, str]] = []
    promoted = 0
    for name, e in latest.items():
        sharpe = e.body["metrics"].get("holdout", {}).get("sharpe")
        verdict = e.body["verdict"]
        promoted += bool(verdict.get("promoted"))
        if isinstance(sharpe, (int, float)):
            (bench if verdict.get("benchmark") else cands).append((float(sharpe), str(name)))
    best = max(cands, default=None)
    # the equal-weight book when there is one: it is what every family is compared with
    plain = [b for b in bench if b[1] == "equal_weight"] or bench
    path = forward / family / "dashboard.json"
    return {
        "family": family,
        "candidates": reg.family_trials(family),
        "promoted": promoted,
        "best": {"name": best[1], "sharpe": best[0]} if best else None,
        "benchmark": {"name": plain[0][1], "sharpe": plain[0][0]} if plain else None,
        "forward": _forward_summary(json.loads(path.read_text(encoding="utf-8")))
        if path.exists()
        else None,
    }


def _stamp(ledger: Path, forward: Path) -> float:
    """Latest change of anything the window shows, so that an open window follows the nightly run
    by asking one cheap question."""
    paths = [ledger, forward.parent / "health.json"]
    if forward.is_dir():
        for family in forward.iterdir():
            paths += [family / "dashboard.json", family / "review.json"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def create_app(ledger_path: Path, forward_root: Path | None = None) -> FastAPI:
    app = FastAPI(title="pusula", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX.read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        try:
            reg = Registry(ledger_path)
            ok, head = True, reg.head
        except RegistryError as exc:
            ok, head = False, str(exc)
        return {"ledger_ok": ok, "ledger_head": head}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        forward = forward_root or home() / "forward"
        try:
            reg = Registry(ledger_path)
        except RegistryError as exc:
            return {
                "ledger_ok": False,
                "ledger_head": str(exc),
                "trials": 0,
                "promoted": 0,
                "families": [],
                "stamp": _stamp(ledger_path, forward),
            }
        names = sorted({str(e.body["family"]) for e in reg.entries if e.kind == "prereg"})
        families = [_family_summary(reg, n, forward) for n in names]
        return {
            "ledger_ok": True,
            "ledger_head": reg.head,
            "trials": sum(f["candidates"] for f in families),
            "promoted": sum(f["promoted"] for f in families),
            "families": families,
            "stamp": _stamp(ledger_path, forward),
        }

    @app.get("/api/execution")
    def execution() -> dict[str, Any]:
        """Which book a broker account mirrors, and its last plan. It cannot send anything."""
        root = (forward_root.parent if forward_root else home()) / "execution"
        out: dict[str, Any] = {"accounts": {}}
        for mode in ("paper", "live"):
            path = root / mode / "account.json"
            if not path.exists():
                continue
            account: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            plans = sorted((root / mode).glob("plan-*.json"))
            if plans:
                plan = json.loads(plans[-1].read_text(encoding="utf-8"))
                account["orders"] = len(plan.get("orders", []))
                account["gross_after"] = plan.get("gross_after")
                outcome = plan.get("outcome", {})
                account["rejected"] = sum(v == "rejected" for v in outcome.values())
                account["captured"] = plan.get("captured")
                account["books"] = plan.get("books")
            tracking = root / mode / "tracking.json"
            if tracking.exists():
                account["tracking"] = json.loads(tracking.read_text(encoding="utf-8"))
            out["accounts"][mode] = account
        return out

    @app.get("/api/families")
    def families() -> dict[str, Any]:
        reg = Registry(ledger_path)
        names = sorted({str(e.body["family"]) for e in reg.entries if e.kind == "prereg"})
        return {"families": names}

    @app.get("/api/forward/{family}")
    def forward(family: str) -> dict[str, Any]:
        p = (forward_root or home() / "forward") / family / "dashboard.json"
        if not p.exists():
            raise HTTPException(404, "no forward snapshot yet: run `pusula forward`")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """What the last ``pusula health`` found; its age is part of the answer."""
        p = (forward_root.parent if forward_root else home()) / "health.json"
        if not p.exists():
            raise HTTPException(404, "no health report yet: run `pusula health`")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    @app.get("/api/review/{family}")
    def review(family: str) -> dict[str, Any]:
        p = (forward_root or home() / "forward") / family / "review.json"
        if not p.exists():
            raise HTTPException(404, "no forward review yet: run `pusula review`")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    @app.get("/api/results/{family}")
    def results(family: str) -> dict[str, Any]:
        reg = Registry(ledger_path)
        out = []
        for name, e in reg.latest_results(family).items():
            m = e.body["metrics"]
            out.append(
                {
                    "name": name,
                    "holdout": m["holdout"],
                    "dev": m["dev"],
                    "factor": m["factor"],
                    "dsr": m.get("dsr"),
                    "pbo": m.get("pbo"),
                    "verdict": e.body["verdict"],
                    "summary": m["summary"],
                    "created_at": e.body["created_at"],
                }
            )
        from pusula.research.report import _programme

        prog = _programme(family)
        return {
            "family": family,
            "trials": reg.family_trials(family),
            "programme": prog,
            "programme_trials": reg.program_trials(prog),
            "results": out,
        }

    return app
