# Operations

## Setup

```bash
uv sync                       # Python 3.12, locked dependencies
uv run pytest -n auto         # full suite (419 tests)
uv run pytest -n auto --cov=pusula   # with line coverage (89%)
```

Credentials: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in the environment, or
`PUSULA_ENV_FILE=<path to a .env>`. Values are never written to disk by pusula.
Data lives in `PUSULA_HOME` (default `~/.pusula`; not `AppData`, see ARCHITECTURE "Data
locations"). SEC EDGAR needs no key; requests identify as `pusula-research/0.1` (no personal
contact details; `data.sec.gov` and `efts.sec.gov` accept that, `www.sec.gov` does not and is
not used). `PUSULA_SEC_USER_AGENT` overrides the string if SEC ever requires a contact.

## Commands

| command | what it does |
|---|---|
| `pusula data fetch` | build the point-in-time store (universe snapshot + daily bars of every member since 2016 + ETFs) |
| `pusula data update` | append newly completed sessions; re-download any security whose history was rescaled by a corporate action |
| `pusula data iv-update` | append newly published 30-day ATM implied vols from DoltHub (`post-no-preference/options`, append-only, Dolt commit recorded) |
| `pusula data fundamentals` | build the SEC EDGAR fundamentals store (security master incl. removed members, XBRL facts, filing index, events; ~5,000 requests, ~10 min; no key) |
| `pusula data fundamentals-update` | refresh it: resolve new membership spells, read the recent filing index of every active company, re-download company facts only where a new report or earnings release appeared (the nightly run does this) |
| `pusula data option-surface` | one-off: stream DoltHub's option-chain table (~11 GB of CSV, read through the database's compressed ZIP export in ~50 min; the table's own CSV endpoint is cut after about an hour) and keep only the daily per-stock features (smirk, volatility spread, term slope; a few MB). No nightly update yet, so the option-surface family has no forward books |
| `pusula data short-volume` | build or extend the FINRA Reg SHO daily short-sale volume store (one public file per session since 2018-08-01: ~2,070 files and ~20 min on the first build, every symbol kept, ~220 MB; then one file a night; no key; each file's SHA-256 chained). The nightly run extends it |
| `uv run python tools/shortvol_report.py` | write `docs/SHORTVOL_DATA.md` (identity check of every member year against its own bars, earlier tickers, coverage; no returns read). Run it after a membership change: a newly renamed member shows up as a year below 0.35 and needs an entry in `src/pusula/data/ticker_history.py` |
| `uv run python tools/robustness.py --workers 12` | write `docs/ROBUSTNESS.md`: parameter neighbours and a noise floor of registered trials (exploratory: nothing there may be selected) |
| `uv run python tools/crossasset_report.py` | write `docs/CROSSASSET_COMPARISON.md`: post-hoc intervals for `us-crossasset-v1` |
| `uv run python tools/fundamentals_report.py` | write `docs/FUNDAMENTALS_DATA.md` (CIK provenance, overrides, coverage; no returns read) |
| `pusula data cross-asset` | one-off: the 18 ETFs of `us-crossasset-v1` the store lacks (the nightly run then keeps them current) |
| `uv run python tools/etf_spreads.py` | the spread model against consolidated quotes on those ETFs (`CALIBRATION.md` §1; no returns read) |
| `pusula data crypto` | every USDT spot pair Binance has listed, dead ones included (~7 min the first time, no key); the nightly run extends it, and a Binance outage skips only that family. A crypto session is complete at 00:00 UTC of the next day, so its books run a session behind the others at 01:00 Istanbul time, which the dead-man check allows |
| `uv run python tools/crypto_spreads.py`, `crypto_report.py` | the spread fit on live books; post-hoc intervals for `us-crypto-v1` |
| `pusula data option-chain` | one-off: every option quote DoltHub has, with greeks, for expiries within 200 days (~1.5 hours, 1.4 GB); read by `pusula optquote` |
| `pusula data record-book --hours N [--symbols ...]` | record Binance's order book as it happens (100 ms diffs, every trade, a snapshot every 30 minutes; ~30 MB an hour for six pairs). Binance keeps no history of it: only what is recorded exists. Run it for as long as you can; a crash loses seconds, but a sleeping machine or a lost network loses everything until it returns (the first eight-hour run holds 3.4 hours). The gate needs 480 symbol-hours: 80 hours of six pairs |
| `uv run python tools/impact_report.py` | write `docs/IMPACT_CALIBRATION.md`: Y from tick data, sweeps against the cost model (needs Alpaca, ~8 min) |
| `uv run python tools/surface_report.py` | write `docs/SURFACE_REPORT.md`: SVI surfaces against the quotes, what one ATM volatility cost |
| `pusula optquote register / run / report` | the `us-optquote-v1` family (fetches unadjusted bars on first use, ~5 min) |
| `pusula micro register / run [--rerun-reason ...]` | the `mm-crypto-v1` family on whatever has been recorded; repeat it as the recording grows |
| `pusula data quality` | print the quality report and the data digest |
| `pusula study register --family us-equity-v1` | pre-register a family (before running it) |
| `pusula study run --family us-equity-v1 --workers 12` | run base / lag+1 / 2x-cost for every trial, apply the gate, append to the ledger, write `docs/RESULTS_<family>.md` (refuses edited trials) |
| `pusula study report --family <f>` | regenerate `docs/RESULTS_<f>.md` from the ledger |
| `pusula options register / run / report` | the `us-options-v1` family |
| `pusula forward` | nightly: update bars, implied vols, fundamentals and short-sale volume, advance the shadow books of every family in `FORWARD_FAMILIES` (us-equity-v2, us-equity-v3, us-optsignal-v1, us-fundamental-v1, us-shortsale-v1, us-shortsale-v2, us-trend-v1, us-crossasset-v1, us-crypto-v1) by the new sessions, write `dashboard.json`, `daily_report.md` and `decisions/<date>.md` per family |
| `pusula forward --family us-equity-v2 --no-update --rehearse-from 2026-08-03 --root <dir>` | rehearsal: replay the forward loop from an earlier date into a separate directory (labelled REHEARSAL everywhere; never a forward record) |
| `pusula execute --family F --book B [--send] [--adopt] [--capital-fraction x]` | mirror one forward book in the Alpaca paper account. Without `--send` it only prints the orders. It scales the book's latest target to the account's equity, refuses a stale or halted book, a session already sent, a second book on one account, or an account holding positions it did not open (`--adopt` overrides that one). `tools/nightly.ps1 -Mirror family/book` does it each night; off by default. Paper fills are simulated by the broker and say little about impact |
| `pusula execute --all [--send] [--adopt]` | every open book together in the paper account: each gets an equal share of the capital, targets are added symbol by symbol and the net is sent. Longs in fractions, shorts in whole shares and only what the broker lends, a position changing sign through zero first (the far side the next night). Crypto books too, netted in dollars, on the coins Alpaca lists (25 of the top 100, 71% of their volume) and long only: what cannot be held stays in cash, and the plan says how much. Each run first reports how the last plan filled (`fills-<session>.json`) and how the account has done against the books' mix (`tracking.json`). The account is a test of the plumbing and of the mix: each strategy's record is its own book. `tools/register_tasks.ps1 -Mirror all` makes the nightly run do this; the first plan went out on 2026-10-31. Never for a live account |
| `pusula execute --all --capital-per-book 100000 [--reset-record] [--send]` | give each book a fixed amount in place of an equal share. It needs that amount times every open book (crypto included) in the account, and refuses otherwise, which `pusula health` reports: new books opening can make an account too small. The paper balance is set only in Alpaca's dashboard, by resetting the account; after that, `--reset-record` moves the old plans aside and starts a new record. `tools/register_tasks.ps1 -Mirror all -CapitalPerBook 100000` makes it nightly |
| `pusula execute ... --live --send` | refused unless the trial passed the gate in the ledger, its forward record passed with stresses, `PUSULA_LIVE_ARM` names this book, and `PUSULA_LIVE_MAX_CAPITAL` and separate live keys (`PUSULA_ALPACA_LIVE_KEY`, `PUSULA_ALPACA_LIVE_SECRET`) are set. No flag bypasses this. Today every book fails the first condition |
| `pusula health [--notify [--digest]]` | dead-man check: is every family's snapshot current, does every journal and the research ledger verify, is a book halted, what did the daily report warn about. Writes `$PUSULA_HOME/health.json`; exits 1 when something is critical. `--notify` sends alerts and pings the heartbeat |
| `pusula review [--family <f>] [--stress --workers 8]` | the forward record of each book against the gate: sessions so far and needed, Sharpe, z against the holdout's claim, Sharpe difference to the equal-weight book with an interval, a verdict from 63 sessions on. Writes `review.md` and `review.json` next to the books. `--stress` re-runs the research engine over the forward window: lag and cost stresses, and the drift between paper book and engine |
| `pusula explain --family <f> --book <b> --security <id or ticker>` | a book's history in one security and where it stands on the trial's signals |
| `pusula app [--forward-root <dir>]` | the same dashboard as a desktop program: its own window (WebView2, part of Windows 11), a server on a free loopback port for as long as the window is open. Keys: 1-5 views, `[` `]` family, up and down book, R refresh; it follows the nightly run by itself. Read-only. A failure goes to `$PUSULA_HOME/logs/app.err.log` and a message box |
| `powershell -File tools/install_shortcut.ps1` | a "Pusula" shortcut on the desktop that opens it without a console (again after moving the project) |
| `pusula web --port 8780 [--forward-root <dir>]` | read-only dashboard on http://127.0.0.1:8780 (family selector; `--forward-root` shows a rehearsal; health, forward review, last session's trades and why) |

## Container

The image is a batch runner (`data`, `study`, `forward`); the dashboard binds 127.0.0.1 by
design and runs on the host. First run in a fresh volume:

```bash
docker build -t pusula .
docker run --rm --env-file .env -v pusula-data:/data pusula data fetch      # ~3 min
docker run --rm -v pusula-data:/data pusula data iv-update                  # bootstraps IV (~200 MB)
docker run --rm --env-file .env -v pusula-data:/data -v pusula-ledger:/app/ledger   -v pusula-docs:/app/docs pusula forward --family all
```

The Dockerfile's steps were replayed in a clean folder (locked, no dev packages; `study report`,
`forward` without an implied-vol store, IV bootstrap from DoltHub); the image itself has not been
built on this machine, because Docker was not running.

## Nightly forward run

The forward books advance only on completed sessions; `last_completed_session()` treats a
session as complete 30 minutes after the 16:00 ET close: 23:30 Istanbul time while New York is
on summer time, 00:30 in winter (Istanbul stays on UTC+3). The Windows scheduled task
`pusula-forward` runs `tools/nightly.ps1` at 01:00 Tuesday to Saturday, which covers both, for
the Monday to Friday sessions; a missed run (machine off or asleep) starts when the machine is
next available, and the next run picks up every session it missed (observed 2026-10-27: asleep
at 01:00, woke at 21:01, the task started at 21:07 by itself). The task does not wake the
machine; `New-ScheduledTaskSettingsSet -WakeToRun` would, if the books should advance at
01:00 regardless. Logs:
`$PUSULA_HOME/logs/nightly-<stamp>.{out,err}.log`. Registration (an operator action; run it
again after moving the repository, because a task keeps the path it was registered with and
fails silently once the script is gone):

```powershell
powershell -ExecutionPolicy Bypass -File tools\register_tasks.ps1
```

It registers `pusula-forward` (01:00 Tuesday to Saturday) and `pusula-health` (12:00 daily).
Check them with `Get-ScheduledTaskInfo pusula-forward` (`LastTaskResult` 0) and the newest log,
or with `pusula health`.

### Knowing that it runs

The books stood still from 2026-10-27 to 2026-10-30 and nothing said so: every scheduled run got
HTTP 403 (a bare `end` date reaches into the free plan's 15-minute SIP embargo on the evening of
the session; `data.alpaca.end_param`), and then the task's script had moved. So:

* `nightly.ps1` ends with `pusula review` and `pusula health --notify --forward-exit <code>`;
* `pusula-health` runs the check on its own at noon, for the night the nightly task did not start;
* books more than one completed session behind are critical, as are a halted book, a journal or
  ledger that does not verify, a missing snapshot and a failed night; the daily report's warnings
  (stale feed, a signal missing from a book, a held name with no bar or a move beyond 25%, an
  optimiser failure) are warnings.

Alerts are opt-in, from the environment or the dotenv file:

| variable | effect |
|---|---|
| `PUSULA_TELEGRAM_TOKEN`, `PUSULA_TELEGRAM_CHAT_ID` | an alert is sent when it appears, every 24 hours while it stays open, and once when it resolves (state in `$PUSULA_HOME/alert_state.json`; a failed send is retried by the next run; the token never reaches a log) |
| the same, with `--digest` | one short summary per session (books, traded, halted, best and worst, open alerts), which the nightly run asks for: a night without one did not finish (state in `$PUSULA_HOME/digest_state.json`) |
| `PUSULA_HEARTBEAT_URL` | pinged when nothing is critical. Only something outside this machine can notice that the machine is off: point a dead-man service (healthchecks.io and the like) at it |

The run is idempotent: a session already processed is never processed again (`PaperEngine.step`
refuses), and a book resumes only if its journal head matches its saved state.

Before a study runs, `pusula study run` checks that every trial in the code hashes to its
registered configuration and refuses otherwise (an edited trial must be registered under a new
name).

The option-implied family reads the implied-vol store, which the nightly run extends from
DoltHub. If DoltHub's SQL API fails that night (it timed out on 2026-10-27 and, before the
fix, stopped every book: the head-commit lookup sorted the whole commit log and hit the
server's ~56 s deadline; it now reads the branch table, under a second), only the
option-implied family sits the night out, the others advance and the run exits 1. If the feed stops, IV older than 7 days is treated as missing: `iv_rv_spread` then has
no score, its names become ineligible and the book unwinds; `iv_rv_plus_momentum` degrades to
its momentum half (the composite averages the signals that exist). The daily report shows,
for every optimised book, the share of index members each of its signals scores on the last
session, and warns when one falls below 50% ("the book is trading without it") and when implied
vols (7 days) or FINRA short volume (4 days, families that read it) stop arriving. The fundamentals family reads the SEC store, which the nightly
run refreshes (~500 filing-index requests, then company facts only for companies with a new
10-Q/10-K or earnings release). If SEC is unreachable, the other books still advance, the
fundamentals family sits the night out (it is not traded on a stale store) and the run exits 1,
so the scheduled task's `LastTaskResult` shows it; the next good night replays the missed
sessions with exactly what was known on each (events carry acceptance times). The short-sale
family reads FINRA's daily file, published about 17:15 ET; in winter the 01:00 run (17:00 EST)
can start before the day's file exists. That costs nothing: a file is used only from the next
session's close, the missing newest file is not recorded as missing, and the next run fetches
it. If FINRA is unreachable, only the short-sale families sit the night out and the run exits 1,
as for SEC. Orders reach a broker only after a deliberate code change for a gate-promoted
strategy.

## Failure handling

| symptom | meaning | action |
|---|---|---|
| `journal head does not match the saved state` | the journal or the state file was edited or partially written | do not delete either; inspect the last journal lines; the journal is the source of truth |
| `ledger on disk diverged` / `chain broken` | the research ledger was edited | restore it from version control; never "fix" a line |
| `re-downloaded [...]` after `data update` | a split or dividend rescaled history | expected; the next research run will show a new data digest |
| `KILL SWITCH` in a book's journal | the book hit its drawdown or daily-loss limit | stays halted until an operator decides; there is no automatic reset |
| optimiser `solver_failures > 0` in a result | an optimisation failed and the book kept its portfolio | reported in the result; investigate before trusting the trial |

## Orders

Nothing sends orders to a broker by default. `execution.broker.AlpacaPaperBroker` is dry-run
unless constructed with `dry_run=False`; wiring it to a book is a deliberate code change for a
gate-promoted strategy with a passing forward record, not a configuration switch.
