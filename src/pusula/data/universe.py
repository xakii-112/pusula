"""Point-in-time S&P 500 membership.

Survivorship bias is the single largest source of false backtests in equity research: today's
index members are, by construction, the companies that survived. Membership is rebuilt for any
date by rolling the dated change log back from today's list:

* a ticker **added after** date D was not a member on D;
* a ticker **removed after** date D was a member on D.

Each membership spell becomes a *security* with the stable id ``TICKER@YYYY-MM-DD`` (the
spell's first day, or the history start). The id never changes, also not when the company later
leaves the index, so positions and journals keep referring to the same thing. Prices for a
closed spell are requested ``asof`` its last day, so the ticker resolves to the company that
carried it then, not to whoever uses the symbol today.

Known limits (documented, not hidden): the change log is community-maintained; a few
early-2010s entries are incomplete. GICS sectors are known for current members only.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from pusula.core.errors import DataError
from pusula.core.hashing import sha256_hex

CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
USER_AGENT = "pusula-research/0.1 (point-in-time index membership; contact via repository)"

_TICKER = re.compile(r"^[A-Z][A-Z0-9]*(\.[A-Z])?$")


def normalise_ticker(raw: object) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    t = str(raw).strip().upper().replace("-", ".")
    t = re.sub(r"\[.*?\]", "", t).strip()
    return t if _TICKER.match(t) else None


@dataclass(frozen=True, slots=True)
class Change:
    date: dt.date
    added: str | None
    removed: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Spell:
    """One continuous membership spell: member on sessions ``start <= d < end``."""

    ticker: str
    start: dt.date
    end: dt.date | None  # None = still a member

    @property
    def security_id(self) -> str:
        return f"{self.ticker}@{self.start.isoformat()}"

    @property
    def asof(self) -> dt.date | None:
        return None if self.end is None else self.end - dt.timedelta(days=1)


@dataclass(frozen=True, slots=True)
class IndexHistory:
    current: tuple[str, ...]
    sectors: dict[str, str]
    changes: tuple[Change, ...]
    snapshot_date: dt.date
    source_sha256: str

    def members_on(self, d: dt.date) -> frozenset[str]:
        m = set(self.current)
        for ch in sorted(self.changes, key=lambda c: c.date, reverse=True):
            if ch.date <= d:
                break
            if ch.added:
                m.discard(ch.added)
            if ch.removed:
                m.add(ch.removed)
        return frozenset(m)

    def spells(self, since: dt.date) -> list[Spell]:
        """Membership spells overlapping ``[since, snapshot]``."""
        events = sorted({c.date for c in self.changes if c.date > since})
        checkpoints = [since, *events]
        state: dict[str, dt.date] = {}  # ticker -> spell start
        out: list[Spell] = []
        prev: frozenset[str] = frozenset()
        for i, d in enumerate(checkpoints):
            now = self.members_on(d)
            if i == 0:
                state = dict.fromkeys(now, since)
            else:
                for t in now - prev:
                    state[t] = d
                out.extend(Spell(t, state.pop(t), d) for t in prev - now)
            prev = now
        out.extend(Spell(t, s, None) for t, s in state.items())
        return sorted(out, key=lambda s: (s.ticker, s.start))


def _level(label: object, i: int) -> str:
    """Lower-cased level ``i`` of a (possibly MultiIndex tuple) column label."""
    if isinstance(label, tuple):
        return str(label[i]).lower()
    return str(label).lower()


def parse_tables(current_html: str, changes_html: str, snapshot: dt.date) -> IndexHistory:
    cur_tabs = pd.read_html(io.StringIO(current_html))
    cur = cur_tabs[0]
    if "Symbol" not in cur.columns:
        raise DataError("current-members table has no Symbol column")
    current: list[str] = []
    sectors: dict[str, str] = {}
    for _, row in cur.iterrows():
        t = normalise_ticker(row["Symbol"])
        if t is None:
            continue
        current.append(t)
        if "GICS Sector" in cur.columns:
            sectors[t] = str(row["GICS Sector"])
    raw: pd.DataFrame | None = None
    for tab in [*cur_tabs, *pd.read_html(io.StringIO(changes_html))]:
        top = [_level(c, 0) for c in tab.columns]
        if any("effective" in u for u in top) and any("added" in u for u in top):
            raw = tab
            break
    if raw is None:
        raise DataError("no dated change table found: point-in-time membership cannot be built")
    table: pd.DataFrame = raw
    cols: list[object] = list(table.columns)

    def pick(top_key: str, sub_key: str) -> pd.Series:
        for c in cols:
            if top_key in _level(c, 0) and sub_key in _level(c, -1):
                return table[c]
        raise DataError(f"change table lacks {top_key}/{sub_key}")

    dates = pd.to_datetime(pick("effective", "effective"), errors="coerce", format="mixed")
    added = pick("added", "ticker")
    removed = pick("removed", "ticker")
    try:
        reason = pick("reason", "reason").astype(str)
    except DataError:
        reason = pd.Series([""] * len(raw))
    changes = []
    when: list[object] = list(dates)  # NaT for unparseable rows
    for d, a, r, why in zip(when, added, removed, reason, strict=True):
        if not isinstance(d, pd.Timestamp):
            continue
        ch = Change(d.date(), normalise_ticker(a), normalise_ticker(r), str(why))
        if ch.added or ch.removed:
            changes.append(ch)
    if len(current) < 450:
        raise DataError(f"implausible current membership size {len(current)}")
    digest = sha256_hex(current_html + "\x00" + changes_html)
    return IndexHistory(tuple(current), sectors, tuple(changes), snapshot, digest)


def fetch_html(url: str) -> str:
    # stdlib client on purpose: Wikimedia's edge rejects httpx's default header set (HTTP 403
    # for every User-Agent tried, 2026-10-23) while accepting the same UA from urllib.
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body: bytes = r.read()
    except urllib.error.HTTPError as exc:
        raise DataError(f"GET {url} -> HTTP {exc.code}") from exc
    return body.decode("utf-8")


def load_or_fetch(
    cache_dir: Path, *, refresh: bool = False, today: dt.date | None = None
) -> IndexHistory:
    """Load the cached snapshot (HTML is stored verbatim, so parsing is reproducible)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_p = cache_dir / "snapshot.json"
    cur_p, chg_p = cache_dir / "current.html", cache_dir / "changes.html"
    if refresh or not (meta_p.exists() and cur_p.exists() and chg_p.exists()):
        snap = today or dt.date.today()
        cur_p.write_text(fetch_html(CURRENT_URL), encoding="utf-8")
        chg_p.write_text(fetch_html(CHANGES_URL), encoding="utf-8")
        meta_p.write_text(json.dumps({"snapshot_date": snap.isoformat()}), encoding="utf-8")
    snap = dt.date.fromisoformat(json.loads(meta_p.read_text(encoding="utf-8"))["snapshot_date"])
    return parse_tables(cur_p.read_text(encoding="utf-8"), chg_p.read_text(encoding="utf-8"), snap)


def membership_mask(
    spells: Iterable[Spell], dates: npt.NDArray[np.datetime64], security_ids: Sequence[str]
) -> npt.NDArray[np.float64]:
    """T x N 0/1 float mask: 1 where the security is an index member at that session's close."""
    col = {s: i for i, s in enumerate(security_ids)}
    mask = np.zeros((dates.size, len(security_ids)), dtype=np.float64)
    for sp in spells:
        j = col.get(sp.security_id)
        if j is None:
            continue
        lo = np.searchsorted(dates, np.datetime64(sp.start, "D"))
        hi = dates.size if sp.end is None else np.searchsorted(dates, np.datetime64(sp.end, "D"))
        mask[lo:hi, j] = 1.0
    return mask
