"""Structural rules: layered imports (no cycles by construction), no wall clock below `live`,
no broad exception swallowing."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "pusula"
LAYERS = [
    "core",
    "stats",
    "data",
    "alpha",
    "risk",
    "execution",
    "micro",
    "portfolio",
    "ml",
    "engine",
    "options",
    "research",
    "live",
    "web",
    "cli",
]
RANK = {name: i for i, name in enumerate(LAYERS)}


def _modules() -> list[tuple[str, ast.Module, Path]]:
    out = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC)
        layer = rel.parts[0].removesuffix(".py") if rel.parts[0] != "__init__.py" else "core"
        out.append((layer, ast.parse(p.read_text(encoding="utf-8")), p))
    return out


def _imports(tree: ast.Module) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pusula."):
            names.append(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            names += [a.name.split(".")[1] for a in node.names if a.name.startswith("pusula.")]
    return names


def test_layering() -> None:
    bad = []
    for layer, tree, path in _modules():
        rel = path.relative_to(SRC)
        bad.extend(
            f"{rel} ({layer}) imports {dep}"
            for dep in _imports(tree)
            if dep in RANK and RANK[dep] > RANK[layer]
        )
    assert not bad, "\n".join(bad)


def test_no_wall_clock_below_live() -> None:
    allowed = {
        "live",
        "web",
        "cli",
        "data",
        "research",
    }  # data stamps fetch time; research stamps ledger
    bad = []
    for layer, tree, path in _modules():
        if layer in allowed or path.name == "clock.py":  # WallClock is the one sanctioned reader
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"now", "today", "utcnow"}:
                src = ast.unparse(node)
                if "datetime" in src or "date." in src or src.startswith("dt."):
                    bad.append(f"{path.relative_to(SRC)}: {src}")
    assert not bad, "\n".join(bad)


def test_no_silent_exception_swallowing() -> None:
    bad = []
    for _, tree, path in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                broad = node.type is None or (
                    isinstance(node.type, ast.Name)
                    and node.type.id in {"Exception", "BaseException"}
                )
                silent = all(isinstance(s, ast.Pass) for s in node.body)
                if broad or silent:
                    bad.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not bad, "\n".join(bad)
