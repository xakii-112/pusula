"""Point-in-time company fundamentals from SEC EDGAR (XBRL), aligned to the research panel.

The rules, each one a way fundamentals backtests go wrong:

1. **Known when accepted, not when the period ended.** Every XBRL fact carries the accession it
   was reported in, and the accession's EDGAR acceptance time (UTC, to the second) decides the
   first session whose close can use it: the first session that closes *after* acceptance. A
   10-K for a December year end is typically public in February; reading it at the period end
   reads the future.
2. **As known then.** At any time the value of a period is the latest one reported by then. A
   restatement filed in 2021 never reaches a 2019 session.
3. **One concept, several tags, per filing.** Companies switch XBRL tags (Apple's revenue moved
   from ``Revenues`` to ``RevenueFromContractWithCustomer...`` in 2018 and its ``Revenues``
   facts stop there); a canonical concept is a tag chain resolved within each filing.
4. **Quarter four is derived.** 10-Ks report the fiscal year, not its last quarter: Q4 = FY -
   nine-month year-to-date of the same fiscal year (same start date).
5. **Removed members need their own CIK.** Wikipedia gives the CIK of current members only.
   A removed member is matched by name through EDGAR's entity search and accepted only if the
   candidate's name (or a former name) matches and it filed 10-K/10-Q reports during the
   membership spell. Unresolved securities are listed in the master, never dropped silently.
6. **Financials are excluded** from accruals and gross profitability (SIC 6000-6799), as in the
   papers that define them (Sloan 1996; Novy-Marx 2013): bank cash flows and "cost of revenue"
   mean something else.

Events produced (one row per filing or release), mapped to panel fields:

* ``ann_day0``: 1.0 on the reaction session of an earnings release (8-K Item 2.02 accepted),
  0.0 otherwise.
* ``sue``: standardised unexpected quarterly net income at the 10-Q/10-K that reports it,
  seasonal random walk with drift (Foster 1977): (Q_t - Q_{t-4} - mean d) / sd d over the
  previous eight seasonal differences d (at least six).
* ``accruals``: (net income - operating cash flow) / average total assets, fiscal year, at the
  10-K (Sloan 1996).
* ``gross_profitability``: (revenue - cost of revenue) / total assets, fiscal year, at the
  10-K (Novy-Marx 2013).

Event fields are NaN except on the session an event becomes usable; signals decide how long a
value stays live (``alpha.signals``).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pusula.core.calendar import early_closes, previous_session
from pusula.core.errors import DataError
from pusula.core.hashing import content_hash
from pusula.data import universe as U
from pusula.data.panel import FloatArray, Panel
from pusula.data.sec import SecClient

# ------------------------------------------------------------------------ concepts
CONCEPTS: dict[str, tuple[str, ...]] = {
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "cfo": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "assets": ("Assets",),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "cogs": (
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    "gross_profit": ("GrossProfit",),
}
_TAG = {
    tag: (concept, rank) for concept, chain in CONCEPTS.items() for rank, tag in enumerate(chain)
}
REPORT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"})
ORIGINAL_FORMS = frozenset({"10-K", "10-Q", "10-KT"})
ANNUAL_FORMS = frozenset({"10-K", "10-KT"})
FINANCIAL_SIC = (6000, 6799)

QUARTER_DAYS = (80, 100)
NINE_MONTH_DAYS = (250, 290)
YEAR_DAYS = (340, 390)

# (start, end) of a reported period; start is NaT for an instant (balance-sheet) fact
Period = tuple[Any, pd.Timestamp]


def _no_start(start: object) -> bool:
    """A period without a start date: pandas yields the ``NaT`` singleton for it."""
    return start is None or start is pd.NaT


def _days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).days)


# ------------------------------------------------------------------------ distillation
def distill_facts(payload: Mapping[str, Any]) -> pd.DataFrame:
    """The facts of ``CONCEPTS`` (USD, report forms only) as a long frame, tag chain resolved
    within each filing: per (concept, accession, start, end) the earliest tag in the chain."""
    rows: list[tuple[object, ...]] = []
    gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    for tag, block in gaap.items():
        hit = _TAG.get(tag)
        if hit is None:
            continue
        concept, rank = hit
        for f in (block.get("units") or {}).get("USD", []):
            form = str(f.get("form") or "")
            if form not in REPORT_FORMS or f.get("end") is None or f.get("val") is None:
                continue
            try:
                val = float(f["val"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            rows.append(
                (concept, tag, rank, f.get("start"), f["end"], val, str(f.get("accn")), form)
            )
    cols = ["concept", "tag", "rank", "start", "end", "val", "accn", "form"]
    df = pd.DataFrame(rows, columns=cols)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna(subset=["end"])
    key = ["concept", "accn", "start", "end"]
    df = df.sort_values([*key, "rank"], na_position="first").drop_duplicates(key, keep="first")
    return df.drop(columns="rank").reset_index(drop=True)


def distill_filings(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Report filings and earnings releases (8-K Item 2.02) with their acceptance time (UTC)."""
    rec = (payload.get("filings") or {}).get("recent") or {}
    n = len(rec.get("form", []))
    rows = []
    for i in range(n):
        form = str(rec["form"][i])
        items = str((rec.get("items") or [""] * n)[i] or "")
        is_release = form == "8-K" and "2.02" in items.split(",")
        if form not in REPORT_FORMS and not is_release:
            continue
        rows.append(
            (
                str(rec["accessionNumber"][i]),
                form,
                rec["filingDate"][i],
                rec["acceptanceDateTime"][i],
                items,
                (rec.get("reportDate") or [""] * n)[i] or None,
            )
        )
    df = pd.DataFrame(rows, columns=["accn", "form", "filed", "accepted", "items", "report"])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce", utc=True, format="mixed")
    df["report"] = pd.to_datetime(df["report"], errors="coerce")
    return df.drop_duplicates("accn").sort_values("accepted").reset_index(drop=True)


# ------------------------------------------------------------------------ point-in-time series
def quarters(
    latest: Mapping[Period, float],
) -> dict[pd.Timestamp, float]:
    """Quarter end -> quarterly value: reported three-month figures, Q4 derived as FY - 9M."""
    q: dict[pd.Timestamp, tuple[int, float]] = {}
    nine: dict[pd.Timestamp, list[tuple[pd.Timestamp, float]]] = {}
    years = []
    for (s, e), v in latest.items():
        if _no_start(s):
            continue
        d = _days(s, e)
        if QUARTER_DAYS[0] <= d <= QUARTER_DAYS[1]:
            if e not in q or abs(d - 91) < q[e][0]:
                q[e] = (abs(d - 91), v)
        elif NINE_MONTH_DAYS[0] <= d <= NINE_MONTH_DAYS[1]:
            nine.setdefault(s, []).append((e, v))
        elif YEAR_DAYS[0] <= d <= YEAR_DAYS[1]:
            years.append((s, e, v))
    out = {e: v for e, (_, v) in q.items()}
    for s, e, v in years:
        if any(abs((e - k).days) <= 7 for k in out):
            continue
        for e9, v9 in nine.get(s, []):
            if 80 <= (e - e9).days <= 100:
                out[e] = v - v9
                break
    return out


def _match(target: pd.Timestamp, keys: Iterable[pd.Timestamp], tol: int) -> pd.Timestamp | None:
    best: tuple[int, pd.Timestamp] | None = None
    for k in keys:
        d = abs((k - target).days)
        if d <= tol and (best is None or d < best[0]):
            best = (d, k)
    return None if best is None else best[1]


def surprise(
    q: Mapping[pd.Timestamp, float], end: pd.Timestamp, n_hist: int = 8, min_hist: int = 6
) -> float | None:
    """Foster (1977) standardised unexpected earnings for the quarter ending ``end``."""
    e = _match(end, q.keys(), 7)
    if e is None:
        return None
    ends = sorted(k for k in q if k <= e)

    def diff(k: pd.Timestamp) -> float | None:
        prev = _match(k - pd.Timedelta(days=364), ends, 20)
        if prev is None or prev >= k:
            return None
        return q[k] - q[prev]

    now = diff(e)
    if now is None:
        return None
    hist = [d for k in ends[:-1][-n_hist:] if (d := diff(k)) is not None]
    if len(hist) < min_hist:
        return None
    sd = float(np.std(hist, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return (now - float(np.mean(hist))) / sd


def _year_value(latest: Mapping[Period, float], end: pd.Timestamp) -> float | None:
    for (s, e), v in latest.items():
        if (
            not _no_start(s)
            and abs((e - end).days) <= 7
            and YEAR_DAYS[0] <= _days(s, e) <= YEAR_DAYS[1]
        ):
            return v
    return None


def _instant(latest: Mapping[Period, float], end: pd.Timestamp, tol: int) -> float | None:
    best: tuple[int, float] | None = None
    for (s, e), v in latest.items():
        if _no_start(s):
            d = abs((e - end).days)
            if d <= tol and (best is None or d < best[0]):
                best = (d, v)
    return None if best is None else best[1]


def company_events(facts: pd.DataFrame, filings: pd.DataFrame, sic: int | None) -> pd.DataFrame:
    """One company's events: earnings releases, SUE at each 10-Q/10-K, annual ratios at each
    10-K. Every value uses only facts accepted no later than the event's own filing."""
    out: list[dict[str, object]] = [
        {"accepted": t, "field": "ann_day0", "value": 1.0}
        for t in filings.loc[filings["form"] == "8-K", "accepted"]
    ]
    if facts.empty:
        return pd.DataFrame(out, columns=["accepted", "field", "value"])
    order = filings.dropna(subset=["accepted"]).set_index("accn")["accepted"]
    f = facts[facts["accn"].isin(order.index)].copy()
    f["accepted"] = f["accn"].map(order)
    f = f.sort_values("accepted", kind="stable")
    own_end = f.groupby("accn")["end"].max()
    stream = list(zip(f["accepted"], f["concept"], f["start"], f["end"], f["val"], strict=True))
    known: dict[str, dict[Period, float]] = {c: {} for c in CONCEPTS}
    ptr = 0
    financial = sic is not None and FINANCIAL_SIC[0] <= sic <= FINANCIAL_SIC[1]
    reports = filings[filings["form"].isin(ORIGINAL_FORMS)].dropna(subset=["accepted"])
    for _, rep in reports.sort_values("accepted", kind="stable").iterrows():
        t = rep["accepted"]
        while ptr < len(stream) and stream[ptr][0] <= t:  # facts known by this filing
            _, c, s, e, v = stream[ptr]
            known[c][(s, e)] = float(v)
            ptr += 1
        end = rep["report"] if pd.notna(rep["report"]) else own_end.get(rep["accn"])
        if end is None or pd.isna(end):
            continue
        ni = known["net_income"]
        s = surprise(quarters(ni), end)
        if s is not None:
            out.append({"accepted": t, "field": "sue", "value": s})
        if rep["form"] not in ANNUAL_FORMS or financial:
            continue
        a_end = _instant(known["assets"], end, 7)
        if a_end is None or a_end <= 0:
            continue
        a_prev = _instant(known["assets"], end - pd.Timedelta(days=365), 30)
        ni_y, cfo_y = _year_value(ni, end), _year_value(known["cfo"], end)
        if ni_y is not None and cfo_y is not None and a_prev is not None and a_prev > 0:
            out.append(
                {
                    "accepted": t,
                    "field": "accruals",
                    "value": (ni_y - cfo_y) / (0.5 * (a_end + a_prev)),
                }
            )
        gp = _year_value(known["gross_profit"], end)
        if gp is None:
            rev = _year_value(known["revenue"], end)
            cogs = _year_value(known["cogs"], end)
            gp = None if rev is None or cogs is None else rev - cogs
        if gp is not None:
            out.append({"accepted": t, "field": "gross_profitability", "value": gp / a_end})
    return pd.DataFrame(out, columns=["accepted", "field", "value"])


# ------------------------------------------------------------------------ security master
# corporate suffixes and joiners that do not identify a company
_STOP = frozenset(
    {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "COMPANIES",
        "LTD",
        "LIMITED",
        "PLC",
        "LLC",
        "LP",
        "HOLDINGS",
        "HOLDING",
        "GROUP",
        "THE",
        "NV",
        "SA",
        "AG",
        "SE",
        "NEW",
        "DE",
        "AND",
        "CLASS",
        "A",
        "B",
        "C",
    }
)


def _int_or_none(v: object) -> int | None:
    """An integer code (CIK, SIC) from a table cell or JSON field; None if missing/unreadable."""
    try:
        f = float(str(v))
    except ValueError:
        return None
    return int(f) if np.isfinite(f) else None


def _words(name: str) -> list[str]:
    """Upper-case words without corporate suffixes; apostrophes join ("Kohl's" -> KOHLS)."""
    s = re.sub(r"['’]", "", name.upper()).replace("&", " AND ")
    s = re.sub(r"\(.*?\)", " ", s)  # EDGAR appends current tickers in parentheses
    return [w for w in re.sub(r"[^A-Z0-9]+", " ", s).split() if w not in _STOP]


def name_tokens(name: str) -> frozenset[str]:
    return frozenset(_words(name))


def name_similarity(index_name: str, entity: str) -> float:
    """Share of the index name's words found in the EDGAR name, first words equal. Measured
    against the index name, not the shorter of the two: dividing by the shorter matched
    "Coca-Cola Enterprises" to COCA COLA CO and "Time Warner Cable" to TIME INC (2026-10-24)."""
    wa, wb = _words(index_name), _words(entity)
    if not wa or not wb or wa[0] != wb[0]:
        return 0.0
    ta = set(wa)
    return len(ta & set(wb)) / len(ta)


def search_queries(name: str, ticker: str) -> list[str]:
    """EDGAR's entity search is a word-prefix index on current names: "VF Corporation" and
    "Macy's" find nothing, "V F Corp" and "Macy" do. Several forms are tried."""
    core = _words(name)
    raw_first = re.sub(r"[^A-Za-z0-9 ]+", " ", name.split("'", maxsplit=1)[0]).split()
    out = [
        ticker,
        name,
        " ".join(core),
        " ".join(core[:2]),
        core[0] if core else "",
        raw_first[0] if raw_first else "",
    ]
    seen: list[str] = []
    for raw in out:
        q = raw.strip()
        if q and q not in seen:
            seen.append(q)
    return seen


@dataclass(frozen=True, slots=True)
class MasterRow:
    security_id: str
    ticker: str
    name: str
    cik: int | None
    sic: int | None
    method: str  # current_table | search | override | unresolved
    note: str


def table_names(
    current_html: str, changes_html: str
) -> tuple[dict[str, tuple[str, int | None]], dict[tuple[dt.date, str], str]]:
    """Current members: ticker -> (security name, CIK). Removals: (date, ticker) -> name."""
    cur = pd.read_html(io.StringIO(current_html))[0]
    current: dict[str, tuple[str, int | None]] = {}
    for _, row in cur.iterrows():
        t = U.normalise_ticker(row.get("Symbol"))
        if t is None:
            continue
        current[t] = (str(row.get("Security", "")), _int_or_none(row.get("CIK")))
    removed: dict[tuple[dt.date, str], str] = {}
    for tab in pd.read_html(io.StringIO(changes_html)):
        top = [U._level(c, 0) for c in tab.columns]
        if not (any("effective" in u for u in top) and any("added" in u for u in top)):
            continue
        cols = list(tab.columns)

        def pick(
            top_key: str, sub_key: str, cols: Sequence[object] = cols, tab: pd.DataFrame = tab
        ) -> pd.Series:
            for c in cols:
                if top_key in U._level(c, 0) and sub_key in U._level(c, -1):
                    return tab[c]
            raise DataError(f"change table lacks {top_key}/{sub_key}")

        dates = pd.to_datetime(pick("effective", "effective"), errors="coerce", format="mixed")
        for d, tk, nm in zip(
            dates, pick("removed", "ticker"), pick("removed", "security"), strict=True
        ):
            t = U.normalise_ticker(tk)
            if isinstance(d, pd.Timestamp) and t and isinstance(nm, str):
                removed[(d.date(), t)] = nm
        break
    return current, removed


def _report_dates(sub: Mapping[str, Any]) -> list[dt.date]:
    rec = (sub.get("filings") or {}).get("recent") or {}
    return [
        dt.date.fromisoformat(d)
        for f, d in zip(rec.get("form", []), rec.get("filingDate", []), strict=False)
        if f in REPORT_FORMS
    ]


def covers_spell(sub: Mapping[str, Any], start: dt.date, end: dt.date | None) -> bool:
    """The entity filed 10-K/10-Q reports near both ends of the spell (an entity that only
    started filing after the spell began is a successor, not the member)."""
    ds = _report_dates(sub)
    near_start = any(
        start - dt.timedelta(days=400) <= d <= start + dt.timedelta(days=400) for d in ds
    )
    if end is None:
        return near_start
    return near_start and any(
        end - dt.timedelta(days=400) <= d <= end + dt.timedelta(days=200) for d in ds
    )


def _tickers(sub: Mapping[str, Any]) -> set[str]:
    return {t for t in (U.normalise_ticker(x) for x in sub.get("tickers") or []) if t}


def _sim_with_former(name: str, entity: str, sub: Mapping[str, Any] | None) -> float:
    former = [str(x.get("name", "")) for x in (sub or {}).get("formerNames", [])]
    return max([name_similarity(name, entity), *(name_similarity(name, x) for x in former)])


def _sic(sub: Mapping[str, Any] | None) -> int | None:
    return _int_or_none(sub.get("sic")) if sub else None


def _master_ciks(master: pd.DataFrame) -> list[tuple[str, int | None]]:
    """(security id, CIK or None) per master row."""
    return [
        (str(s), _int_or_none(c)) for s, c in zip(master["security_id"], master["cik"], strict=True)
    ]


def _master_sics(master: pd.DataFrame) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for c, s in zip(master["cik"], master["sic"], strict=True):
        cik = _int_or_none(c)
        if cik is not None:
            out[cik] = _int_or_none(s)
    return out


def resolve_master(
    spells: Sequence[U.Spell],
    current: Mapping[str, tuple[str, int | None]],
    removed: Mapping[tuple[dt.date, str], str],
    client: SecClient,
    *,
    since: dt.date,
    overrides: Mapping[str, tuple[int | None, str]] | None = None,
    name_confirmed: frozenset[str] = frozenset(),
    submissions: dict[int, tuple[dict[str, Any], str] | None] | None = None,
    min_similarity: float = 1.0,
) -> list[MasterRow]:
    """One row per membership spell (security id). ``submissions`` is a CIK -> (payload, sha)
    cache the caller may reuse, so no filing index is fetched twice. For overrides listed in
    ``name_confirmed`` the name test is skipped (the filing-coverage test is not)."""
    subs = submissions if submissions is not None else {}

    def sub_of(cik: int) -> dict[str, Any] | None:
        if cik not in subs:
            subs[cik] = client.submissions(cik)
        got = subs[cik]
        return None if got is None else got[0]

    ov = overrides or {}
    rows: list[MasterRow] = []
    for sp in spells:
        sid = sp.security_id
        start = max(sp.start, since)
        if sp.end is None:
            nm = current.get(sp.ticker, ("", None))[0]
        else:
            nm = removed.get((sp.end, sp.ticker)) or next(
                (
                    v
                    for (d, t), v in sorted(removed.items(), key=lambda kv: kv[0][0])
                    if t == sp.ticker
                ),
                "",
            )
        if sid in ov:
            cik, why = ov[sid]
            sub = sub_of(cik) if cik else None
            ok = cik is None or (
                sub is not None
                and covers_spell(sub, start, sp.end)
                and (
                    sid in name_confirmed
                    or sp.ticker in _tickers(sub)
                    or _sim_with_former(nm, str(sub.get("name", "")), sub) >= min_similarity
                )
            )
            method = "override" if ok else "override_unverified"
            rows.append(MasterRow(sid, sp.ticker, nm, cik, _sic(sub), method, why))
            continue
        if sp.end is None:
            cik = current.get(sp.ticker, ("", None))[1]
            method = "current_table" if cik else "unresolved"
            note = "" if cik else "current member without a CIK in the table"
            rows.append(
                MasterRow(sid, sp.ticker, nm, cik, _sic(sub_of(cik)) if cik else None, method, note)
            )
            continue
        cands: list[tuple[int, str]] = []
        if sp.ticker in current and current[sp.ticker][1]:
            cands.append((int(current[sp.ticker][1] or 0), current[sp.ticker][0]))
        for q in search_queries(nm, sp.ticker) if nm else [sp.ticker]:
            cands.extend(client.search_entities(q)[:6])
        # rank: ticker still carried by the entity, then name similarity (current or former)
        best: tuple[int, float, int, str] | None = None
        seen: set[int] = set()
        for cik, entity in cands:
            if cik in seen:
                continue
            seen.add(cik)
            sim = name_similarity(nm, entity)
            tick = (
                f"({sp.ticker}" in entity.upper()
                or f" {sp.ticker})" in entity.upper()
                or f", {sp.ticker}" in entity.upper()
            )
            if sim < min_similarity and not tick and len(seen) > 12:
                continue  # unpromising and far down the list: not worth a request
            sub = sub_of(cik)
            if sub is None:
                continue
            sim = _sim_with_former(nm, entity, sub)
            has_tick = sp.ticker in _tickers(sub)
            if (sim < min_similarity and not has_tick) or not covers_spell(sub, start, sp.end):
                continue
            key = (int(has_tick), sim, cik, entity)
            if best is None or key[:2] > best[:2]:
                best = key
        if best is None:
            rows.append(
                MasterRow(
                    sid, sp.ticker, nm, None, None, "unresolved", f"no verified match for {nm!r}"
                )
            )
        else:
            by_ticker, sim, cik, entity = best
            how = (
                "ticker+name"
                if by_ticker and sim >= min_similarity
                else ("ticker" if by_ticker else "name")
            )
            rows.append(
                MasterRow(
                    sid,
                    sp.ticker,
                    nm,
                    cik,
                    _sic(sub_of(cik)),
                    "search",
                    f"{entity} [{how}, similarity {sim:.2f}]",
                )
            )
    return rows


# ------------------------------------------------------------------------ store
def store_dir(root: Path) -> Path:
    return root / "fundamentals"


def build_fundamentals(
    root: Path,
    client: SecClient,
    *,
    history: U.IndexHistory | None = None,
    since: dt.date = dt.date(2016, 1, 1),
    overrides: Mapping[str, tuple[int | None, str]] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Resolve the security master, download company facts and filing indexes for every CIK,
    and write the distilled store. Returns the manifest."""
    from pusula.data.cik_overrides import CIK_OVERRIDES, NAME_CONFIRMED

    udir = root / "universe"
    hist = history or U.load_or_fetch(udir)
    current, removed = table_names(
        (udir / "current.html").read_text(encoding="utf-8"),
        (udir / "changes.html").read_text(encoding="utf-8"),
    )
    subs: dict[int, tuple[dict[str, Any], str] | None] = {}
    master = resolve_master(
        hist.spells(since),
        current,
        removed,
        client,
        since=since,
        overrides=CIK_OVERRIDES if overrides is None else overrides,
        name_confirmed=NAME_CONFIRMED,
        submissions=subs,
    )
    ciks = sorted({r.cik for r in master if r.cik})
    tables: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    provenance: dict[str, dict[str, object]] = {}
    for k, cik in enumerate(ciks):
        got_s = subs[cik] if cik in subs else client.submissions(cik)
        tables[cik] = _fetch_company(client, cik, got_s, provenance)
        if log and (k + 1) % 50 == 0:
            print(f"[fundamentals] {k + 1}/{len(ciks)} companies", flush=True)
    mdf = pd.DataFrame([asdict(r) for r in master])
    manifest = {
        "source": "SEC EDGAR data.sec.gov companyfacts + submissions",
        "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "since": since.isoformat(),
        "requests": client.requests,
        "ciks": provenance,
    }
    return _write_store(root, mdf, tables, manifest)


def _fetch_company(
    client: SecClient,
    cik: int,
    got_s: tuple[dict[str, Any], str] | None,
    provenance: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    got_f = client.companyfacts(cik)
    fil = distill_filings(got_s[0] if got_s else {})
    fac = distill_facts(got_f[0]) if got_f else distill_facts({})
    provenance[str(cik)] = {
        "submissions_sha256": got_s[1] if got_s else None,
        "companyfacts_sha256": got_f[1] if got_f else None,
        "facts": len(fac),
        "filings": len(fil),
        "fetched": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    return fac, fil


def _write_store(
    root: Path,
    master: pd.DataFrame,
    tables: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write facts, filings, the per-company events derived from them, the master and the
    manifest. Events are computed here once, not in every backtest worker."""
    from pusula.data.settings import atomic_write_text

    out = store_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    sic = _master_sics(master)
    facts_l, filings_l, events_l = [], [], []
    for cik, (fac, fil) in sorted(tables.items()):
        facts_l.append(fac.assign(cik=cik))
        filings_l.append(fil.assign(cik=cik))
        events_l.append(
            company_events(fac, fil.reset_index(drop=True), sic.get(cik)).assign(cik=cik)
        )
    pd.concat(facts_l, ignore_index=True).to_parquet(out / "facts.parquet", index=False)
    pd.concat(filings_l, ignore_index=True).to_parquet(out / "filings.parquet", index=False)
    events = pd.concat(events_l, ignore_index=True)
    events.to_parquet(out / "events.parquet", index=False)
    master.to_csv(out / "master.csv", index=False)
    manifest.update(
        {
            "securities": len(master),
            "resolved": int(master["cik"].notna().sum()),
            "by_method": master["method"].value_counts().to_dict(),
            "companies": len(tables),
            "events": events["field"].value_counts().to_dict(),
        }
    )
    manifest["digest"] = store_digest(manifest, master)
    atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=1, default=str))
    return manifest


def update_fundamentals(
    root: Path,
    client: SecClient,
    *,
    active_days: int = 400,
    log: bool = True,
) -> dict[str, Any]:
    """Nightly refresh. Membership spells new since the build are resolved; for every company
    with a spell active in the last ``active_days`` the recent filing index is read, and only a
    company with a report or release not yet stored has its company facts downloaded again."""
    from pusula.data.cik_overrides import CIK_OVERRIDES, NAME_CONFIRMED

    sd = store_dir(root)
    manifest = json.loads((sd / "manifest.json").read_text(encoding="utf-8"))
    master = read_master(root)
    facts = pd.read_parquet(sd / "facts.parquet")
    filings = pd.read_parquet(sd / "filings.parquet")
    udir = root / "universe"
    hist = U.load_or_fetch(udir)
    since = dt.date.fromisoformat(manifest.get("since", "2016-01-01"))
    known = set(master["security_id"])
    new_spells = [sp for sp in hist.spells(since) if sp.security_id not in known]
    subs: dict[int, tuple[dict[str, Any], str] | None] = {}
    if new_spells:
        current, removed = table_names(
            (udir / "current.html").read_text(encoding="utf-8"),
            (udir / "changes.html").read_text(encoding="utf-8"),
        )
        rows = resolve_master(
            new_spells,
            current,
            removed,
            client,
            since=since,
            overrides=CIK_OVERRIDES,
            name_confirmed=NAME_CONFIRMED,
            submissions=subs,
        )
        master = pd.concat([master, pd.DataFrame([asdict(r) for r in rows])], ignore_index=True)
        if log:
            print(f"[fundamentals] {len(rows)} new membership spells resolved", flush=True)
    cutoff = dt.date.today() - dt.timedelta(days=active_days)
    spell_end = {sp.security_id: sp.end for sp in hist.spells(since)}
    active_set: set[int] = set()
    for sid, cik_ in _master_ciks(master):
        end = spell_end.get(sid)
        if cik_ is not None and (end is None or end >= cutoff):
            active_set.add(cik_)
    active = sorted(active_set)
    empty_facts = facts.iloc[0:0].drop(columns="cik")
    tables: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    stored = sorted({c for x in filings["cik"].unique() if (c := _int_or_none(x)) is not None})
    have_facts = {_int_or_none(x) for x in facts["cik"].unique()}
    for c in stored:
        tables[c] = (
            facts[facts["cik"] == c].drop(columns="cik") if c in have_facts else empty_facts,
            filings[filings["cik"] == c].drop(columns="cik"),
        )
    provenance: dict[str, dict[str, object]] = dict(manifest.get("ciks", {}))
    refreshed = 0
    for cik in active:
        got = subs.get(cik) or client.submissions(cik, archive=cik not in tables)
        if got is None:
            continue
        fresh = distill_filings(got[0])
        have = set(tables[cik][1]["accn"]) if cik in tables else set()
        if cik in tables and set(fresh["accn"]) <= have:
            continue
        if cik in tables:  # keep the older (archive) filings, add the new ones
            fresh = pd.concat([tables[cik][1], fresh[~fresh["accn"].isin(have)]], ignore_index=True)
            fresh = fresh.sort_values("accepted").reset_index(drop=True)
        fac, _ = _fetch_company(client, cik, got, provenance)
        tables[cik] = (fac, fresh)
        refreshed += 1
    if log:
        print(f"[fundamentals] {refreshed} of {len(active)} active companies refreshed", flush=True)
    manifest.update(
        {"updated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "ciks": provenance}
    )
    manifest["requests"] = client.requests
    return _write_store(root, master, tables, manifest)


def store_digest(manifest: Mapping[str, Any], master: pd.DataFrame) -> str:
    """Content digest: the raw-response hashes of every company plus the security master."""
    shas = {
        k: [v.get("submissions_sha256"), v.get("companyfacts_sha256")]
        for k, v in manifest["ciks"].items()
    }
    # pandas 3 keeps missing values missing under astype(str): build the rows explicitly
    m = sorted((s, "" if c is None else str(c)) for s, c in _master_ciks(master))
    return content_hash({"responses": shas, "master": m})


def read_master(root: Path) -> pd.DataFrame:
    df = pd.read_csv(store_dir(root) / "master.csv")
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce").astype("Int64")
    return df


# ------------------------------------------------------------------------ panel alignment
def session_closes_utc(dates: np.ndarray) -> np.ndarray:
    """Close time of each session (16:00 ET, 13:00 on early closes) as UTC datetime64[ns]."""
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    out = []
    for d in pd.DatetimeIndex(dates):
        day = d.date()
        hour = 13 if day in early_closes(day.year) else 16
        out.append(
            pd.Timestamp(dt.datetime(day.year, day.month, day.day, hour, tzinfo=ny)).tz_convert(
                "UTC"
            )
        )
    return pd.DatetimeIndex(out).tz_localize(None).values.astype("datetime64[ns]")


@dataclass(frozen=True, slots=True)
class SessionClock:
    """Close times of a panel's sessions, plus the close of the session before the first."""

    closes: np.ndarray  # datetime64[ns], UTC
    before_first: np.datetime64

    @classmethod
    def of(cls, dates: np.ndarray) -> SessionClock:
        prev = previous_session(pd.Timestamp(dates[0]).date())
        return cls(
            session_closes_utc(dates), session_closes_utc(np.array([np.datetime64(prev, "D")]))[0]
        )

    def reaction_index(self, accepted: pd.Series) -> np.ndarray:
        """Row of the first session that closes after each acceptance time; -1 for events at or
        before the close preceding the panel, after its last close, or without a timestamp."""
        ts = (
            pd.DatetimeIndex(pd.to_datetime(accepted, utc=True))
            .tz_localize(None)
            .values.astype("datetime64[ns]")
        )
        idx = np.searchsorted(self.closes, ts, side="right")
        missing = pd.isna(pd.Series(accepted)).to_numpy()
        bad = (idx >= self.closes.size) | ((idx == 0) & (ts <= self.before_first)) | missing
        return np.where(bad, -1, idx)


FIELDS = ("ann_day0", "sue", "accruals", "gross_profitability")


def coverage(panel: Panel, holds: Mapping[str, int]) -> pd.DataFrame:
    """Share of index-member sessions on which each event field has an event no older than its
    hold, by calendar year and by whether the security is still a member at the panel's end
    (removed members are where survivorship bias would hide). Reads no returns."""
    member = panel["in_universe"] > 0
    survivor = member[-1]
    years = pd.DatetimeIndex(panel.dates).year.to_numpy()
    rows_idx = np.arange(panel.n_dates)[:, None]
    out: list[dict[str, object]] = []
    live: dict[str, np.ndarray] = {}
    for f, hold in holds.items():
        x = panel[f]
        ev = (x > 0) if f == "ann_day0" else np.isfinite(x)
        last = np.maximum.accumulate(np.where(ev, rows_idx, -1), axis=0)
        live[f] = (last >= 0) & (rows_idx - last <= hold)
    for y in np.unique(years):
        for group, cols in (("current", survivor), ("removed", ~survivor)):
            m = member[years == y][:, cols]
            n = int(m.sum())
            if n == 0:
                continue
            row: dict[str, object] = {"year": int(y), "members": group, "member_days": n}
            for f in holds:
                row[f] = float((live[f][years == y][:, cols] & m).sum() / n)
            out.append(row)
    return pd.DataFrame(out)


def fundamental_fields(panel: Panel, root: Path) -> dict[str, FloatArray]:
    """T x N event fields (see module docstring) for the panel's securities."""
    master = read_master(root)
    events = pd.read_parquet(store_dir(root) / "events.parquet")
    sid_cik = {s: c for s, c in _master_ciks(master) if c is not None}
    cols: dict[int, list[int]] = {}
    for j, sid in enumerate(panel.symbols):
        c = sid_cik.get(sid)
        if c is not None:
            cols.setdefault(c, []).append(j)
    shape = (panel.n_dates, panel.n_symbols)
    out = {f: np.full(shape, np.nan) for f in FIELDS}
    out["ann_day0"] = np.zeros(shape)
    ev_by = {_int_or_none(c): g for c, g in events.groupby("cik")}
    clock = SessionClock.of(panel.dates)
    for cik, js in cols.items():
        ev = ev_by.get(cik)
        if ev is None or ev.empty:
            continue
        ev = ev.sort_values("accepted", kind="stable")
        rows = clock.reaction_index(ev["accepted"])
        for field_name, r, v in zip(ev["field"], rows, ev["value"], strict=True):
            if r < 0:
                continue
            for j in js:
                out[field_name][r, j] = v  # a later event on the same session wins
    return out
