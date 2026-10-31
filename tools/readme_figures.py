"""Draw the README figures from the research ledger, and nothing else.

Every number comes from the latest result of each trial in ``ledger/experiments.jsonl``, so the
figures can be rebuilt from a fresh clone with no market data:

    uv run --with matplotlib python tools/readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
BENCHMARK = "#52514e"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
DSR_LINE = 0.95

# Which of the trials repeat under a later family with the same parameters: count them once.
SUPERSEDED = {"us-equity-v1"}


def latest_results() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with (ROOT / "ledger" / "experiments.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["kind"] == "result":
                body = entry["body"]
                out[(body["family"], body["name"])] = body
    return out


def style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def trial_map(res: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Every registered strategy against the line it would have had to cross."""
    points = [
        (b["metrics"]["holdout"]["sharpe"], b["metrics"]["dsr"], fam, name)
        for (fam, name), b in res.items()
        if b["metrics"]["dsr"] is not None
        and "sharpe" in b["metrics"]["holdout"]
        and fam not in SUPERSEDED
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
        s=58,
        color=SERIES,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
    )
    ax.axhline(DSR_LINE, color=CRITICAL, linewidth=2, zorder=2)
    ax.text(
        ax.get_xlim()[0],
        DSR_LINE + 0.015,
        "Promotion needs DSR ≥ 0.95, among other checks",
        color=INK,
        fontsize=10,
        va="bottom",
    )
    for fam, name, dx, dy, ha in (
        ("us-equity-v2", "xs_momentum", 8, 8, "left"),
        ("us-options-v1", "calls_d50_t21", 8, 4, "left"),
        ("us-optsurface-v1", "iv_spread_ls", 8, 8, "left"),
    ):
        b = res[(fam, name)]
        ax.annotate(
            name,
            (b["metrics"]["holdout"]["sharpe"], b["metrics"]["dsr"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=10,
            color=INK_2,
        )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Holdout Sharpe (net of costs)", color=INK_2, fontsize=10)
    ax.set_ylabel("Deflated Sharpe ratio", color=INK_2, fontsize=10)
    save(fig, "trials_vs_gate.png")
    print(f"trials_vs_gate.png: {len(points)} trials, max DSR {max(p[1] for p in points):.3f}")


def equity_v2_bars(res: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = sorted(
        (
            (name, b["metrics"]["holdout"]["sharpe"])
            for (fam, name), b in res.items()
            if fam == "us-equity-v2" and name != "equal_weight"
        ),
        key=lambda r: r[1],
    )
    bench = res[("us-equity-v2", "equal_weight")]["metrics"]["holdout"]["sharpe"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), facecolor=SURFACE)
    style(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)
    y = range(len(rows))
    ax.barh(list(y), [r[1] for r in rows], height=0.5, color=SERIES, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=4)
    ax.plot(
        [bench, bench],
        [-0.6, len(rows) - 0.45],
        color=BENCHMARK,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.text(
        bench,
        len(rows) - 0.35,
        f"Equal-weight S&P 500: {bench:.2f}",
        color=INK_2,
        fontsize=10,
        ha="center",
        va="bottom",
    )
    ax.set_yticks(list(y), [r[0] for r in rows], color=INK, fontsize=10)
    for i, (_, v) in enumerate(rows):
        ax.text(
            v + (0.025 if v >= 0 else -0.025),
            i,
            f"{v:.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            color=INK,
        )
    ax.set_xlim(-1.35, 1.2)
    ax.set_ylim(-0.6, len(rows) + 0.15)
    save(fig, "us_equity_v2_sharpe.png")


def gate_grid(res: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = sorted(
        (
            (name, b["verdict"]["checks"])
            for (fam, name), b in res.items()
            if fam == "us-equity-v2" and b["verdict"].get("checks")
        ),
        key=lambda r: -res[("us-equity-v2", r[0])]["metrics"]["holdout"]["sharpe"],
    )
    checks = [c["name"] for c in rows[0][1]]
    fig, ax = plt.subplots(figsize=(9.2, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=INK)
    for i, (_, cs) in enumerate(rows):
        for j, c in enumerate(cs):
            ok = bool(c["passed"])
            ax.add_patch(
                plt.Rectangle(
                    (j + 0.04, i + 0.06),
                    0.92,
                    0.88,
                    facecolor=GOOD if ok else CRITICAL,
                    edgecolor="none",
                    alpha=0.9,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                "pass" if ok else "fail",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlim(0, len(checks))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(checks))], checks)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))], [r[0] for r in rows])
    save(fig, "us_equity_v2_gate.png")


def main() -> None:
    res = latest_results()
    trial_map(res)
    equity_v2_bars(res)
    gate_grid(res)


if __name__ == "__main__":
    main()
