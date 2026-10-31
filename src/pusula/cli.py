"""Command-line interface: ``uv run pusula <command>``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "ledger" / "experiments.jsonl"


def program_of(family: str) -> str:
    """``us-equity-v2`` -> ``us-equity-``: versions of one research programme share a DSR count."""
    from pusula.research.families import PROGRAMME_OVERRIDE

    if family in PROGRAMME_OVERRIDE:
        return PROGRAMME_OVERRIDE[family]
    head, sep, tail = family.rpartition("-")
    return head + sep if sep and tail.startswith("v") else family


def results_path(family: str) -> Path:
    out = REPO / "docs"
    out.mkdir(parents=True, exist_ok=True)  # absent in the container image
    return out / f"RESULTS_{family}.md"


def _data_digest(
    bars_digest: str, fundamentals: bool = False, surface: bool = False, short: bool = False
) -> str:
    """Bars manifest digest, combined with the implied-vol import's source hash if present and,
    for a family that reads them, the fundamentals store digest, the option-surface stream's
    SHA-256 and the short-volume files' hash chain."""
    from pusula.core.hashing import content_hash
    from pusula.data.settings import data_dir

    digest = bars_digest
    meta = data_dir() / "iv" / "iv_history.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        iv = {"source": m.get("source_sha256"), "updates": m.get("updates", [])}
        if not iv["updates"]:  # same form as results recorded before incremental updates existed
            digest = content_hash({"bars": bars_digest, "iv": iv["source"]})
        else:
            digest = content_hash({"bars": bars_digest, "iv": iv})
    if fundamentals:
        from pusula.data.fundamentals import store_dir

        man = json.loads((store_dir(data_dir()) / "manifest.json").read_text(encoding="utf-8"))
        digest = content_hash({"data": digest, "fundamentals": man["digest"]})
    if surface:
        meta = json.loads((data_dir() / "optskew" / "option_skew.json").read_text(encoding="utf-8"))
        digest = content_hash({"data": digest, "option_surface": meta["source_sha256"]})
    if short:
        sv = json.loads((data_dir() / "shortvol" / "short_volume.json").read_text(encoding="utf-8"))
        digest = content_hash({"data": digest, "short_volume": [sv["chain_sha256"], sv["last"]]})
    return digest


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def cmd_data(args: argparse.Namespace) -> int:
    from pusula.data.pipeline import build_store, research_panel, update_store

    if args.action == "fetch":
        build_store(refresh_universe=args.refresh_universe)
    elif args.action == "update":
        update_store()
    elif args.action == "iv-update":
        from pusula.data.iv import LEGACY_DOLT, bootstrap_iv, import_iv, update_iv
        from pusula.data.settings import data_dir

        store = data_dir() / "iv"
        if not (store / "iv_history.parquet").exists():
            if LEGACY_DOLT.exists():
                import_iv(store)
            else:
                print(f"[iv] no store: bootstrapping from DoltHub -> {bootstrap_iv(store)}")
        print(f"[iv] {update_iv(store)} rows added")
    elif args.action in ("fundamentals", "fundamentals-update"):
        from pusula.data import fundamentals as F
        from pusula.data.sec import SecClient
        from pusula.data.settings import data_dir

        with SecClient() as client:
            if (
                args.action == "fundamentals"
                or not (F.store_dir(data_dir()) / "manifest.json").exists()
            ):
                man = F.build_fundamentals(data_dir(), client)
            else:
                man = F.update_fundamentals(data_dir(), client)
        print(
            f"[fundamentals] {man['resolved']}/{man['securities']} securities with a CIK "
            f"({man['by_method']}), {man['companies']} companies, events {man['events']}, "
            f"{man['requests']} requests, digest {man['digest'][:16]}"
        )
        return 0
    elif args.action == "option-surface":
        from pusula.data.optskew import bootstrap_skew
        from pusula.data.settings import data_dir

        print(f"[optskew] wrote {bootstrap_skew(data_dir() / 'optskew')}")
        return 0
    elif args.action == "crypto":
        from pusula.data.crypto import update_crypto

        print(f"[crypto] {update_crypto()} daily bars added")
        return 0
    elif args.action == "option-chain":
        from pusula.data.optchain import bootstrap_chain
        from pusula.data.settings import data_dir

        print(f"[optchain] wrote {bootstrap_chain(data_dir() / 'optchain')}")
        return 0
    elif args.action == "record-book":
        from pusula.data.binance import record_book
        from pusula.data.settings import data_dir

        symbols = args.symbols.split(",")
        print(record_book(symbols, data_dir() / "book", args.hours * 3600.0))
        return 0
    elif args.action == "cross-asset":
        from pusula.data.pipeline import CROSS_ASSET, fetch_etfs

        print(f"[data] fetched {fetch_etfs(CROSS_ASSET)}")
        return 0
    elif args.action == "short-volume":
        from pusula.data.pipeline import last_completed_session
        from pusula.data.settings import data_dir
        from pusula.data.shortvol import update_shortvol

        n = update_shortvol(data_dir() / "shortvol", last_completed_session())
        print(f"[shortvol] {n} sessions added")
        return 0
    rd = research_panel()
    print(rd.quality.summary())
    print(
        f"securities {rd.panel.n_symbols}, sessions {rd.panel.n_dates} "
        f"({rd.panel.dates[0]} .. {rd.panel.dates[-1]}), data digest {rd.data_digest[:16]}"
    )
    return 0


def _loader() -> tuple[object, object]:
    """Research panel (+ ``iv``, the company-filing, option-surface and short-volume fields
    when their stores exist) and the daily T-bill rate."""
    from pusula.data.pipeline import research_panel
    from pusula.live.forward import rf_series

    panel = _with_aux(research_panel()).panel
    return panel, rf_series(panel)


def _universe_loader(universe: tuple[str, ...]) -> tuple[object, object]:
    """As ``_loader``, for a family with a universe of its own."""
    from pusula.data.pipeline import universe_data
    from pusula.live.forward import rf_series

    panel = universe_data(universe).panel
    return panel, rf_series(panel)


def _loader_of(spec: Any) -> Any:
    """The (picklable) panel loader of a family's backtests."""
    return partial(_universe_loader, spec.universe) if spec.universe else _loader


def _research_of(spec: Any) -> Any:
    """The research data a family runs on, with what it can read attached."""
    from pusula.data.pipeline import research_panel, universe_data

    return universe_data(spec.universe) if spec.universe else _with_aux(research_panel())


def _factors_of(spec: Any, panel: Any, french: Any) -> Any:
    """The frame the family's alpha check reads."""
    from pusula.data.pipeline import is_crypto

    if not is_crypto(spec.universe):
        return french
    from pusula.data.crypto import crypto_factors

    return crypto_factors(panel, french)


def _with_aux(rd: Any) -> Any:
    """Attach every auxiliary store that exists (one order everywhere)."""
    from pusula.data.pipeline import (
        with_fundamentals,
        with_implied_vol,
        with_option_surface,
        with_short_volume,
    )

    return with_short_volume(with_option_surface(with_fundamentals(with_implied_vol(rd))))


def _study_from_ledger(args: argparse.Namespace, reg: Any, S: Any, render: Any) -> int:
    """Actions that need only the ledger (any family, also those without a StudySpec)."""
    if args.action == "reevaluate":
        from pusula.research.registry import code_manifest

        if not args.rerun_reason:
            print("reevaluate needs --rerun-reason", file=sys.stderr)
            return 2
        res = S.reevaluate(
            reg, args.family, program_of(args.family), _now(), args.rerun_reason, code_manifest()
        )
        results_path(args.family).write_text(render(reg, args.family), encoding="utf-8")
        for name, r in res.items():
            failed = ",".join(r["verdict"].get("failed", [])) or "-"
            print(f"{name:32s} DSR {r['dsr']:.2f}  failed: {failed}")
        return 0
    if args.action == "report":
        out = results_path(args.family)
        out.write_text(render(reg, args.family), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    return 2


def cmd_study(args: argparse.Namespace) -> int:
    from pusula.research import study as S
    from pusula.research.families import FAMILIES
    from pusula.research.registry import Registry
    from pusula.research.report import render

    reg = Registry(LEDGER)
    if args.action in ("reevaluate", "report"):
        return _study_from_ledger(args, reg, S, render)
    spec = FAMILIES[args.family]()
    if args.action == "register":
        hashes = S.register(spec, reg, _now())
        for k, v in hashes.items():
            print(f"registered {args.family}/{k}: {v[:16]}")
        return 0
    # run: only the registered configurations, byte for byte
    from pusula.core.errors import RegistryError

    try:
        hashes = S.verify_registration(spec, reg)
    except RegistryError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    from pusula.data import factors as french
    from pusula.data.pipeline import research_panel, universe_data
    from pusula.data.settings import data_dir
    from pusula.research.registry import code_manifest

    files = code_manifest()  # the code that runs, captured before it runs
    rd = universe_data(spec.universe) if spec.universe else research_panel()
    runs = S.run_backtests(spec, _loader_of(spec), workers=args.workers)
    outcomes = S.evaluate_study(
        spec,
        runs,
        _factors_of(spec, rd.panel, french.fetch(data_dir() / "factors")),
        reg.program_trials(program_of(args.family)),
        S.programme_srs(reg, program_of(args.family), args.family),
    )
    S.record(
        spec,
        reg,
        outcomes,
        hashes,
        _data_digest(
            rd.data_digest,
            fundamentals=needs_fundamentals(spec),
            surface=needs_option_surface(spec),
            short=needs_short_volume(spec),
        ),
        _now(),
        rerun_reason=args.rerun_reason,
        code_files=files,
    )
    results_path(args.family).write_text(render(reg, args.family), encoding="utf-8")
    for o in outcomes:
        verdict = o.verdict
        if o.benchmark:
            status = "BENCHMARK"
        elif verdict.get("promoted"):
            status = "PROMOTED"
        else:
            status = "rejected: " + ",".join(verdict.get("failed", []))
        print(f"{o.name:24s} holdout Sharpe {o.holdout['sharpe']:6.2f}  DSR {o.dsr:5.2f}  {status}")
    return 0


def _signal_names(spec: Any) -> set[str]:
    names = {s.name for t in spec.trials for s in t.signals}
    return names | {s.name for t in spec.trials if t.ml is not None for s in t.ml.features}


def needs_implied_vol(spec: Any) -> bool:
    """Does any trial of the family read the option-market field ``iv``?"""
    from pusula.alpha.signals import OPTION_SIGNALS

    return bool(_signal_names(spec) & set(OPTION_SIGNALS))


def needs_option_surface(spec: Any) -> bool:
    """Does any trial of the family read option-surface fields (``data.optskew``)?"""
    from pusula.alpha.signals import SURFACE_SIGNALS

    return bool(_signal_names(spec) & set(SURFACE_SIGNALS))


def needs_fundamentals(spec: Any) -> bool:
    """Does any trial of the family read company-filing fields (``data.fundamentals``)?"""
    from pusula.alpha.signals import FUNDAMENTAL_SIGNALS

    return bool(_signal_names(spec) & set(FUNDAMENTAL_SIGNALS))


def needs_short_volume(spec: Any) -> bool:
    """Does any trial of the family read FINRA short-sale volume (``data.shortvol``)?"""
    from pusula.alpha.signals import SHORT_SIGNALS

    return bool(_signal_names(spec) & set(SHORT_SIGNALS))


def cmd_forward(args: argparse.Namespace) -> int:
    import httpx

    from pusula.core.errors import DataError
    from pusula.data import crypto as C
    from pusula.data import fundamentals as F
    from pusula.data.iv import update_iv
    from pusula.data.pipeline import (
        is_crypto,
        last_completed_session,
        research_panel,
        universe_data,
        update_store,
    )
    from pusula.data.sec import SecClient
    from pusula.data.settings import data_dir
    from pusula.data.shortvol import update_shortvol
    from pusula.live.forward import step_books
    from pusula.research.families import FAMILIES, FORWARD_FAMILIES
    from pusula.research.registry import Registry

    families = list(FORWARD_FAMILIES) if args.family == "all" else [args.family]
    if args.rehearse_from and not args.root:
        print(
            "--rehearse-from needs --root (a rehearsal never writes the real books)",
            file=sys.stderr,
        )
        return 2
    reg = Registry(LEDGER)
    has_fund = (F.store_dir(data_dir()) / "manifest.json").exists()
    has_short = (data_dir() / "shortvol" / "short_volume.parquet").exists()
    fund_failed = short_failed = iv_failed = crypto_failed = False
    if not args.no_update:
        if (data_dir() / "crypto" / "manifest.json").exists() and any(
            is_crypto(FAMILIES[f]().universe) for f in families
        ):
            try:
                C.update_crypto(log=False)
            except (DataError, httpx.HTTPError) as exc:
                crypto_failed = True  # as for SEC below: only the crypto family sits out
                print(f"[forward] crypto update failed: {exc}", file=sys.stderr)
        update_store()  # no bars, no book: an Alpaca failure stops the night, as it should
        if (data_dir() / "iv" / "iv_history.parquet").exists():
            try:  # bootstrap is a one-off: `pusula data iv-update`
                update_iv(data_dir() / "iv")
            except (DataError, httpx.HTTPError) as exc:
                # DoltHub's SQL API times out at times (2026-10-27: "context deadline
                # exceeded" stopped every book that night); only the option-implied families
                # sit out, and the next good night replays their sessions
                iv_failed = True
                print(f"[forward] implied-vol update failed: {exc}", file=sys.stderr)
        if has_short and any(needs_short_volume(FAMILIES[f]()) for f in families):
            try:  # the build is a one-off: `pusula data short-volume`
                update_shortvol(data_dir() / "shortvol", last_completed_session(), log=False)
            except (DataError, httpx.HTTPError) as exc:
                short_failed = True  # as for SEC below: only the short-sale families sit out
                print(f"[forward] short-volume update failed: {exc}", file=sys.stderr)
        if has_fund and any(needs_fundamentals(FAMILIES[f]()) for f in families):
            try:  # the build is a one-off: `pusula data fundamentals`
                with SecClient() as client:
                    F.update_fundamentals(data_dir(), client)
            except (DataError, httpx.HTTPError) as exc:
                # An SEC outage must not stop the other books. The filing families sit this
                # night out (not traded on a stale store) and catch up on the next good night:
                # events carry acceptance times, so the missed sessions are replayed with
                # exactly what was known on each of them.
                fund_failed = True
                print(f"[forward] fundamentals update failed: {exc}", file=sys.stderr)
    # one panel for every family of this night's run
    rd = _with_aux(research_panel())
    for family in families:
        spec = FAMILIES[family]()
        if needs_implied_vol(spec) and (iv_failed or "iv" not in rd.panel):
            why = "implied-vol update failed" if iv_failed else "no implied-vol store"
            print(
                f"[forward] {family}: skipped, {why} (run `pusula data iv-update`)",
                file=sys.stderr,
            )
            continue
        if needs_fundamentals(spec) and (fund_failed or "ann_day0" not in rd.panel):
            why = "fundamentals update failed" if fund_failed else "no fundamentals store"
            print(
                f"[forward] {family}: skipped, {why} (run `pusula data fundamentals-update`)",
                file=sys.stderr,
            )
            continue
        if needs_short_volume(spec) and (short_failed or "short_volume" not in rd.panel):
            why = "short-volume update failed" if short_failed else "no short-volume store"
            print(
                f"[forward] {family}: skipped, {why} (run `pusula data short-volume`)",
                file=sys.stderr,
            )
            continue
        if needs_option_surface(spec):
            # The option-surface store is a one-off ~10 GB stream with no incremental update;
            # a forward book would trade on features that stop at the stream's end date.
            print(
                f"[forward] {family}: skipped, the option-surface store has no nightly update",
                file=sys.stderr,
            )
            continue
        when = {
            e.body["name"]: dt.datetime.fromisoformat(e.body["created_at"])
            for e in reg.preregs(family)
        }
        root = None
        if args.rehearse_from:
            # A rehearsal replays the forward loop from an earlier date into a separate
            # directory. It demonstrates the machinery; it is never a trial's forward record.
            start = dt.datetime.fromisoformat(args.rehearse_from).replace(tzinfo=dt.UTC)
            when = dict.fromkeys(when, start)
            root = Path(args.root) / family
        book_data = rd
        if spec.universe:
            if crypto_failed and is_crypto(spec.universe):
                print(f"[forward] {family}: skipped, crypto update failed", file=sys.stderr)
                continue
            try:
                book_data = universe_data(spec.universe)
            except DataError as exc:
                print(f"[forward] {family}: skipped, {exc}", file=sys.stderr)
                continue
        snap = step_books(spec, when, rd=book_data, root=root, rehearsal=root is not None)
        summary = {
            k: {"equity": v["equity"], "gross": v["gross"], "n": v["n_positions"]}
            for k, v in snap["books"].items()
        }
        print(json.dumps({family: summary}, indent=1))
    # a non-zero exit shows in the scheduled task's result
    return 1 if fund_failed or short_failed or iv_failed or crypto_failed else 0


def cmd_health(args: argparse.Namespace) -> int:
    """Exit 0 when nothing is critical. ``--notify`` tells the outside world (opt-in by env)."""
    from pusula.data.pipeline import last_completed_session
    from pusula.data.settings import home
    from pusula.live.health import render_health, run_health, session_digest, write_health
    from pusula.live.notify import NotifyError, deliver, deliver_digest
    from pusula.research.families import FORWARD_FAMILIES

    now = _now()
    report = run_health(
        FORWARD_FAMILIES,
        LEDGER,
        now,
        last_completed_session(now),
        forward_exit=args.forward_exit,
        err_log=Path(args.err_log) if args.err_log else None,
    )
    write_health(report)
    print(render_health(report))
    if args.notify:
        try:
            print(f"[health] {deliver(report.alerts, home() / 'alert_state.json', now)}")
            if args.digest:
                text, session = session_digest(report)
                print(f"[health] {deliver_digest(text, session, home() / 'digest_state.json')}")
        except NotifyError as exc:
            print(f"[health] notification failed: {exc}", file=sys.stderr)
            return 1
    return 0 if report.healthy else 1


def cmd_review(args: argparse.Namespace) -> int:
    """Each forward book against the gate; ``--stress`` re-runs the research engine over the
    forward window (lag and cost stresses, drift between book and engine)."""
    import pandas as pd

    from pusula.data import factors as french
    from pusula.data.settings import data_dir, home
    from pusula.live.decisions import SignalCache
    from pusula.live.forward import first_session_after
    from pusula.live.review import render_review, review_family, stress_spec, write_review
    from pusula.live.signal_record import signal_record
    from pusula.research.families import FAMILIES, FORWARD_FAMILIES
    from pusula.research.registry import Registry
    from pusula.research.study import run_backtests

    reg = Registry(LEDGER)
    fac = french.fetch(data_dir() / "factors")
    rf = pd.Series(fac["rf"]).sort_index()
    families = list(FORWARD_FAMILIES) if args.family == "all" else [args.family]
    caches: dict[tuple[str, ...], Any] = {}  # one panel per universe, its signals shared
    for family in families:
        root = home() / "forward" / family
        if not root.is_dir():
            print(f"[review] {family}: no forward books yet", file=sys.stderr)
            continue
        spec = FAMILIES[family]()
        research = None
        first = min(dt.datetime.fromisoformat(e.body["created_at"]) for e in reg.preregs(family))
        if args.stress:
            research = run_backtests(
                stress_spec(spec, first.date().isoformat()),
                _loader_of(spec),
                workers=args.workers,
            )
        if spec.universe not in caches:
            caches[spec.universe] = SignalCache(_research_of(spec).panel)
        cache = caches[spec.universe]
        factors = _factors_of(spec, cache.panel, fac)
        rev = review_family(spec, reg, root, rf, factors, program_of(family), research)
        rev["signals"] = signal_record(
            spec, cache.panel, first_session_after(cache.panel, first), cache
        )
        write_review(rev, root)
        print(render_review(rev))
    return 0


def _refuse(why: str) -> int:
    print(f"[execute] refused: {why}", file=sys.stderr)
    return 2


def _security_prices(panel: Any) -> dict[str, float]:
    """The last close of each security of a panel, and of each ticker: what an account holds has no
    other name, and a ticker's latest security is the one still trading."""
    import numpy as np

    close = panel["close"]
    out: dict[str, float] = {}
    for j, sid in enumerate(panel.symbols):
        known = close[:, j][np.isfinite(close[:, j])]
        if not known.size:
            continue
        out[sid] = float(known[-1])  # a name halted today keeps its last price
        ticker = sid.split("@", maxsplit=1)[0]
        if ticker not in out or np.isfinite(close[-1, j]):
            out[ticker] = float(known[-1])
    return out


def _tradable(broker: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """What the paper account can hold: stocks and their terms, coins, and each coin's own price
    (a crypto panel's prices are rescaled)."""
    from pusula.data.crypto import store_of
    from pusula.live.execute import Tradable

    stocks = {
        a["symbol"]: Tradable(
            bool(a.get("fractionable")),
            bool(a.get("shortable") and a.get("easy_to_borrow")),
        )
        for a in broker.assets("us_equity")
        if a.get("tradable")
    }
    coins = {
        a["symbol"]: Tradable(
            True,
            False,
            float(a.get("min_order_size") or 0.0),
            float(a.get("min_trade_increment") or 0.0),
        )
        for a in broker.assets("crypto")
        if a.get("tradable") and str(a["symbol"]).endswith("/USD")
    }
    store = store_of()
    prices: dict[str, float] = {}
    for coin in coins:
        pair = coin.removesuffix("/USD") + "USDT"
        if store.has(pair):
            bars = store.get(pair)
            if len(bars):
                prices[coin] = float(bars["close"].iloc[-1])
    return stocks, coins, prices


def _execute_all(args: argparse.Namespace) -> int:
    """Every open book of every family the broker can trade, netted in one paper account."""
    from pusula.core.errors import DataError
    from pusula.data.pipeline import is_crypto, last_completed_session
    from pusula.data.settings import alpaca_credentials, atomic_write_text, home
    from pusula.execution.broker import AlpacaPaperBroker
    from pusula.live import execute as X
    from pusula.research.families import FAMILIES, FORWARD_FAMILIES

    if args.live:
        return _refuse("--all is for paper: a live account mirrors one book that earned it")
    creds = alpaca_credentials()
    if creds is None:
        return _refuse("no Alpaca credentials")
    broker = AlpacaPaperBroker(creds.key_id, creds.secret, dry_run=not args.send)
    specs = {f: FAMILIES[f]() for f in FORWARD_FAMILIES}
    crypto = [f for f, s in specs.items() if is_crypto(s.universe)]
    store = X.state_dir() / "paper"
    session = last_completed_session().isoformat()
    if args.reset_record:
        print(f"[execute] the old record is now in {X.archive_record(store, _now())}")
    refused = store / "refused.json"
    try:
        books, slots, notes = X.gather_books(home() / "forward", specs, session, crypto)
        prices: dict[str, float] = {}
        for universe in {s.universe for s in specs.values()}:
            spec = next(s for s in specs.values() if s.universe == universe)
            prices.update(_security_prices(_research_of(spec).panel))
        quality = X.report_last_fills(store, broker, _now().date())
        if quality is not None:
            print(
                f"[execute] last plan, {quality['session']}: {quality['mean_bps']:.1f} bp against "
                f"the decision close on {quality['traded']:,.0f} traded, "
                f"{quality['unfilled']} orders unfilled"
            )
        held = broker.exact_positions()
        X.check_account(store, *X.COMBINED, session, held, args.adopt)
        equity = broker.equity()
        state = (
            json.loads((store / "account.json").read_text(encoding="utf-8"))
            if (store / "account.json").exists()
            else {}
        )
        tracking = X.mirror_tracking(home() / "forward", books, slots, state, equity)
        if tracking is not None:
            atomic_write_text(store / "tracking.json", json.dumps(tracking, indent=1))
            print(
                f"[execute] since {tracking['since']}: account {tracking['account']:+.2%}, "
                f"the books' mix {tracking['mix']:+.2%}, gap {tracking['gap']:+.2%}"
            )
        stocks, coins, coin_prices = _tradable(broker)
        plan = X.make_combined_plan(
            books,
            slots,
            session,
            prices,
            equity,
            held,
            args.capital_fraction,
            notes,
            stocks,
            coins,
            coin_prices,
            args.capital_per_book,
        )
        X.check_plan(plan)
    except (X.ExecutionRefusedError, DataError) as exc:
        if args.send:  # a night that sent nothing has to reach the operator, as a failed run does
            store.mkdir(parents=True, exist_ok=True)
            atomic_write_text(refused, json.dumps({"why": str(exc)}))
        return _refuse(str(exc))
    refused.unlink(missing_ok=True)
    print(X.render_plan(plan))
    if not args.send:
        print("[execute] dry run: nothing sent (--send sends it)")
        return 0
    orders = X.send_plan(plan, broker, store, _now())
    bad = [o for o in orders if o.state.value == "rejected"]
    print(f"[execute] sent {len(orders) - len(bad)} orders, {len(bad)} rejected")
    for o in bad[:10]:
        print(f"[execute] {o.security}: {o.history[-1]}", file=sys.stderr)
    return 1 if bad else 0


def cmd_execute(args: argparse.Namespace) -> int:
    """Mirror a forward book in a broker account. Without ``--send``: only the plan."""
    if args.all:
        return _execute_all(args)
    if not (args.family and args.book):
        return _refuse("name a book with --family and --book, or pass --all")
    from pusula.core.errors import DataError
    from pusula.data.pipeline import is_crypto, last_completed_session
    from pusula.data.settings import alpaca_credentials, environment, home
    from pusula.engine.journal import Journal
    from pusula.execution.broker import AlpacaPaperBroker
    from pusula.live import execute as X
    from pusula.research.families import FAMILIES
    from pusula.research.registry import Registry

    spec = FAMILIES[args.family]()
    if is_crypto(spec.universe):
        return _refuse("a crypto book is a research record: the Alpaca adapter trades US shares")
    root = home() / "forward" / args.family
    paths = root / args.book
    if args.live:
        latest = Registry(LEDGER).latest_results(args.family).get(args.book)
        promoted = bool(latest and latest.body["verdict"].get("promoted"))
        review = root / "review.json"
        rev = json.loads(review.read_text(encoding="utf-8")) if review.exists() else {}
        refusals = X.live_refusals(
            promoted,
            rev.get("books", {}).get(args.book),
            bool(rev.get("stress_run")),
            args.family,
            args.book,
            environment(),
        )
        if refusals:
            return _refuse("no live account: " + "; ".join(refusals))
        if not args.send:
            return _refuse("--live needs --send")
    if not (root / "dashboard.json").exists() or not (paths / "state.json").exists():
        return _refuse(f"no forward book {args.family}/{args.book}")
    if args.live:
        broker = X.live_broker()
    else:
        creds = alpaca_credentials()
        if creds is None:
            return _refuse("no Alpaca credentials")
        broker = AlpacaPaperBroker(creds.key_id, creds.secret, dry_run=not args.send)
    snapshot = json.loads((root / "dashboard.json").read_text(encoding="utf-8"))
    state = json.loads((paths / "state.json").read_text(encoding="utf-8"))
    store = X.state_dir() / ("live" if args.live else "paper")
    session = last_completed_session().isoformat()
    try:
        X.check_book(snapshot, args.book, session)
        journal = Journal(paths / "journal.jsonl")
        target = X.book_target(journal, state)
        panel = _research_of(spec).panel
        last = panel.n_dates - 1
        prices = {
            s: float(panel["close"][last, panel.symbol_index(s)])
            for s in {*target, *journal.replay()[0].positions}
            if s in panel.symbols
        }
        quality = X.report_last_fills(store, broker, _now().date())
        if quality is not None:
            print(
                f"[execute] last plan, {quality['session']}: {quality['mean_bps']:.1f} bp against "
                f"the decision close on {quality['traded']:,.0f} traded, "
                f"{quality['unfilled']} orders unfilled"
            )
        held = broker.positions()
        X.check_account(store, args.family, args.book, session, held, args.adopt)
        account = broker.equity()
        if args.live:
            account = min(account, float(environment()["PUSULA_LIVE_MAX_CAPITAL"]))
        plan = X.make_plan(
            args.family,
            args.book,
            session,
            target,
            prices,
            X.book_equity_of(journal),
            account,
            held,
            args.capital_fraction,
            args.live,
        )
        X.check_plan(plan)
    except (X.ExecutionRefusedError, DataError) as exc:
        return _refuse(str(exc))
    print(X.render_plan(plan))
    if not args.send:
        print("[execute] dry run: nothing sent (--send sends it)")
        return 0
    orders = X.send_plan(plan, broker, store, _now())
    bad = [o for o in orders if o.state.value == "rejected"]
    print(f"[execute] sent {len(orders) - len(bad)} orders, {len(bad)} rejected")
    for o in bad[:10]:
        print(f"[execute] {o.security}: {o.history[-1]}", file=sys.stderr)
    return 1 if bad else 0


def cmd_explain(args: argparse.Namespace) -> int:
    from pusula.data.settings import home
    from pusula.engine.journal import Journal
    from pusula.live.decisions import explain_security
    from pusula.research.families import FAMILIES

    spec = FAMILIES[args.family]()
    trial = next((t for t in spec.trials if t.name == args.book), None)
    path = home() / "forward" / args.family / args.book / "journal.jsonl"
    if trial is None or not path.exists():
        print(f"no forward book {args.family}/{args.book}", file=sys.stderr)
        return 2
    panel = _research_of(spec).panel
    print(explain_security(spec, trial, Journal(path), panel, args.security))
    return 0


def cmd_options(args: argparse.Namespace) -> int:
    from pusula.research import options_study as O
    from pusula.research.registry import Registry, code_manifest
    from pusula.research.report import render

    reg = Registry(LEDGER)
    if args.action == "register":
        for k, v in O.register(reg, _now()).items():
            print(f"registered {O.FAMILY}/{k}: {v[:16]}")
        return 0
    if args.action == "report":
        out = results_path(O.FAMILY)
        out.write_text(render(reg, O.FAMILY), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    from pusula.data import factors as french
    from pusula.data.iv import import_iv, iv_panel
    from pusula.data.pipeline import research_panel
    from pusula.data.settings import data_dir
    from pusula.live.forward import rf_series

    files = code_manifest()
    rd = research_panel()
    store = data_dir() / "iv"
    if not (store / "iv_history.parquet").exists():
        import_iv(store)
    iv = iv_panel(rd.panel, store)
    out_ = O.run(
        reg,
        rd.panel,
        iv,
        rf_series(rd.panel),
        french.fetch(data_dir() / "factors"),
        rd.data_digest,
        _now(),
        rerun_reason=args.rerun_reason,
        code_files=files,
    )
    results_path(O.FAMILY).write_text(render(reg, O.FAMILY), encoding="utf-8")
    for name, r in out_.items():
        print(f"{name:20s} holdout Sharpe {r['holdout']['sharpe']:6.2f}  {r['verdict']}")
    return 0


def cmd_optquote(args: argparse.Namespace) -> int:
    """The us-optquote-v1 family: option strategies at the chain's own quotes."""
    from pusula.research import optquote_study as Q
    from pusula.research.registry import Registry, code_manifest
    from pusula.research.report import render

    reg = Registry(LEDGER)
    if args.action == "register":
        for k, v in Q.register(reg, _now()).items():
            print(f"registered {Q.FAMILY}/{k}: {v[:16]}")
        return 0
    if args.action == "report":
        out = results_path(Q.FAMILY)
        out.write_text(render(reg, Q.FAMILY), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    from pusula.core.hashing import content_hash
    from pusula.data import factors as french
    from pusula.data.pipeline import fetch_raw_bars, raw_closes, research_panel
    from pusula.data.settings import data_dir
    from pusula.live.forward import rf_series

    files = code_manifest()
    print(f"[optquote] fetched raw bars of {fetch_raw_bars()} securities", flush=True)
    rd = research_panel()
    store = data_dir() / "optchain"
    chain = Q.load_chain(store, rd.panel)
    print(f"[optquote] {len(chain)} quoted dates", flush=True)
    meta = json.loads((store / "chain.json").read_text(encoding="utf-8"))
    res = Q.run(
        reg,
        rd.panel,
        raw_closes(rd.panel),
        chain,
        rf_series(rd.panel),
        french.fetch(data_dir() / "factors"),
        content_hash({"bars": rd.data_digest, "chain": meta["source_sha256"]}),
        _now(),
        rerun_reason=args.rerun_reason,
        code_files=files,
    )
    results_path(Q.FAMILY).write_text(render(reg, Q.FAMILY), encoding="utf-8")
    for name, r in res.items():
        print(
            f"{name:18s} dev {r['dev']['sharpe']:6.2f}  holdout {r['holdout']['sharpe']:6.2f}  "
            f"failed: {','.join(r['verdict'].get('failed', [])) or '-'}"
        )
    return 0


def _replay_symbol(job: tuple[str, str, float, float, tuple[Any, ...]]) -> tuple[str, Any, Any]:
    from pusula.micro.book import Instrument
    from pusula.research.mm_study import run_symbol

    path, symbol, tick, lot, specs = job
    results, stats = run_symbol(Path(path), Instrument(symbol, tick, lot), specs)
    return symbol, results, stats


def cmd_micro(args: argparse.Namespace) -> int:
    """The market-making family on recorded order books."""
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict

    from pusula.core.hashing import sha256_hex
    from pusula.data.binance import BinanceClient, fetch_instruments
    from pusula.data.settings import data_dir
    from pusula.research import mm_study as M
    from pusula.research.registry import Registry, code_manifest

    reg = Registry(LEDGER)
    if args.action == "register":
        for k, v in M.register(reg, _now()).items():
            print(f"registered {M.FAMILY}/{k}: {v[:16]}")
        return 0
    files = code_manifest()
    store = data_dir() / "book"
    recordings = sorted(store.glob("*USDT-*.jsonl.gz"))
    if not recordings:
        print("no recordings: run `pusula data record-book`", file=sys.stderr)
        return 2
    symbols = sorted({p.name.split("-")[0] for p in recordings})
    table = store / "instruments.json"
    with BinanceClient() as client:
        fetch_instruments(client, symbols, table)
    instruments = M.load_instruments(table)
    names = [(t.name, v) for t in M.TRIALS for v in M.variants(t.spec)]
    specs = tuple(M.variants(t.spec)[v] for t in M.TRIALS for v in M.variants(t.spec))
    jobs = [
        (str(p), s, instruments[s].tick, instruments[s].lot, specs)
        for p in recordings
        for s in [p.name.split("-")[0]]
    ]
    hours: dict[str, dict[str, list[float]]] = {t.name: {} for t in M.TRIALS}
    by_symbol: dict[str, dict[str, Any]] = {t.name: {} for t in M.TRIALS}
    stats_out: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for symbol, results, stats in ex.map(_replay_symbol, jobs):
            tick, lot = instruments[symbol].tick, instruments[symbol].lot
            for (name, variant), res in zip(names, results, strict=True):
                hours[name].setdefault(variant, []).extend(
                    res.hourly_net(M.RETAIL_FEE_BPS, tick, lot)
                )
                if variant == "base":
                    by_symbol[name][symbol] = M.summarise(res, tick, lot)
            prior = stats_out.setdefault(symbol, dict.fromkeys(asdict(stats), 0.0))
            for k, v in asdict(stats).items():
                prior[k] += float(v)
            print(f"[micro] {symbol}: {stats.diffs:,} diffs, {stats.gaps} breaks", flush=True)
    import numpy as np

    srs = []
    for name in hours:
        x = np.asarray(hours[name]["base"], dtype=np.float64)
        if x.size > 3 and x.std() > 0:
            srs.append(float(x.mean() / x.std(ddof=1)))
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    outcomes = {
        name: {
            "verdict": M.evaluate_trial(hours[name], len(M.TRIALS), var_sr),
            "by_symbol": by_symbol[name],
        }
        for name in hours
    }
    digest = sha256_hex("|".join(f"{p.name}:{p.stat().st_size}" for p in recordings))
    M.record(reg, outcomes, digest, _now(), args.rerun_reason, files)
    out = results_path(M.FAMILY)
    out.write_text(M.render_results(outcomes, stats_out), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    from pusula.web.app import create_app

    fwd = Path(args.forward_root) if args.forward_root else None
    uvicorn.run(create_app(LEDGER, fwd), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """The dashboard in its own window. Under ``pythonw`` nobody sees stderr, hence the report."""
    from pusula.web import desktop

    fwd = Path(args.forward_root) if args.forward_root else None
    try:
        desktop.open_window(LEDGER, fwd)
    except desktop.DesktopError as exc:
        desktop.report_failure(str(exc))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pusula", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="build / update the point-in-time bar store")
    d.add_argument(
        "action",
        choices=[
            "fetch",
            "update",
            "iv-update",
            "fundamentals",
            "fundamentals-update",
            "option-surface",
            "short-volume",
            "cross-asset",
            "crypto",
            "option-chain",
            "record-book",
            "quality",
        ],
    )
    d.add_argument("--refresh-universe", action="store_true")
    d.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,LINKUSDT,ARBUSDT")
    d.add_argument("--hours", type=float, default=1.0, help="record-book: how long")
    d.set_defaults(fn=cmd_data)
    s = sub.add_parser("study", help="pre-register, run and report a study family")
    s.add_argument("action", choices=["register", "run", "report", "reevaluate"])
    s.add_argument("--family", default="us-equity-v1")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--rerun-reason", default=None)
    s.set_defaults(fn=cmd_study)
    f = sub.add_parser("forward", help="advance the forward paper books by the new sessions")
    f.add_argument("--family", default="all", help="a family name, or 'all' (FORWARD_FAMILIES)")
    f.add_argument("--no-update", action="store_true")
    f.add_argument("--rehearse-from", default=None, help="ISO date; replay into --root (demo only)")
    f.add_argument("--root", default=None)
    f.set_defaults(fn=cmd_forward)
    h = sub.add_parser("health", help="dead-man check of the nightly system and its records")
    h.add_argument("--notify", action="store_true", help="send alerts, ping the heartbeat")
    h.add_argument("--digest", action="store_true", help="with --notify: the night's summary")
    h.add_argument("--forward-exit", type=int, default=None, help="the nightly run's exit code")
    h.add_argument("--err-log", default=None, help="the nightly run's stderr, for the alert")
    h.set_defaults(fn=cmd_health)
    r = sub.add_parser("review", help="forward record of each book against the gate")
    r.add_argument("--family", default="all", help="a family name, or 'all' (FORWARD_FAMILIES)")
    r.add_argument("--stress", action="store_true", help="also re-run the research engine")
    r.add_argument("--workers", type=int, default=6)
    r.set_defaults(fn=cmd_review)
    e = sub.add_parser("execute", help="mirror forward books in the broker account")
    e.add_argument("--family", default=None)
    e.add_argument("--book", default=None)
    e.add_argument("--all", action="store_true", help="every open book, netted in one account")
    e.add_argument("--send", action="store_true", help="send the orders (default: show the plan)")
    e.add_argument("--adopt", action="store_true", help="the account's positions are this book's")
    e.add_argument("--capital-fraction", type=float, default=1.0)
    e.add_argument(
        "--capital-per-book", type=float, default=None, help="--all: dollars each book gets"
    )
    e.add_argument(
        "--reset-record",
        action="store_true",
        help="--all: the account was reset, start a new record",
    )
    e.add_argument("--live", action="store_true", help="a live account: refused unless earned")
    e.set_defaults(fn=cmd_execute)
    x = sub.add_parser("explain", help="why a forward book holds or traded a security")
    x.add_argument("--family", required=True)
    x.add_argument("--book", required=True)
    x.add_argument("--security", required=True, help="security id, or a ticker if unambiguous")
    x.set_defaults(fn=cmd_explain)
    o = sub.add_parser(
        "options", help="the us-options-v1 family (trend views expressed with calls)"
    )
    o.add_argument("action", choices=["register", "run", "report"])
    o.add_argument("--rerun-reason", default=None)
    o.set_defaults(fn=cmd_options)
    q = sub.add_parser("optquote", help="the us-optquote-v1 family (strategies at real quotes)")
    q.add_argument("action", choices=["register", "run", "report"])
    q.add_argument("--rerun-reason", default=None)
    q.set_defaults(fn=cmd_optquote)
    m = sub.add_parser("micro", help="the mm-crypto-v1 family (market making on real books)")
    m.add_argument("action", choices=["register", "run"])
    m.add_argument("--workers", type=int, default=6)
    m.add_argument("--rerun-reason", default=None)
    m.set_defaults(fn=cmd_micro)
    w = sub.add_parser("web", help="read-only dashboard on 127.0.0.1")
    w.add_argument("--port", type=int, default=8780)
    w.add_argument(
        "--forward-root", default=None, help="read forward snapshots from here (rehearsals)"
    )
    w.set_defaults(fn=cmd_web)
    a = sub.add_parser("app", help="the dashboard as a desktop program, in its own window")
    a.add_argument(
        "--forward-root", default=None, help="read forward snapshots from here (rehearsals)"
    )
    a.set_defaults(fn=cmd_app)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
