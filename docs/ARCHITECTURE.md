# Architecture

## One loop, two ledgers

```
                 point-in-time data                    decisions                       effects
  ┌────────────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────┐
  │ universe  Wikipedia change log →    │   │ alpha     causal signals     │   │ research   Backtester    │
  │           membership spells         │   │           (look-ahead tested)│   │            float shares  │
  │ store     Alpaca SIP daily bars,    │──▶│ ml        walk-forward,      │──▶│            fast, many    │
  │           as-of entity mapping,     │   │           purged + embargoed │   │            trials        │
  │           manifest + SHA-256        │   │ risk      factor model + VRA │   │                          │
  │ quality   pointwise repairs only    │   │ portfolio cost-aware cvxpy   │   │ production PaperEngine   │
  │ factors   Fama-French 5 + MOM, rf   │   │           optimiser (DPP)    │   │            integer shares│
  │ iv        30-day ATM IV, DoltHub    │   │ gate      pre-trade limits   │   │            micro-dollars │
  │ fundamentals SEC EDGAR XBRL + filing│   │                              │   │            orders, broker│
  │           index, PIT to acceptance  │   │                              │   │                          │
  └────────────────────────────────────┘   └──────────────────────────────┘   │                          │
                                                                              │            hash journal  │
                                                                              └──────────────────────────┘
```

Both engines call the **same** `Strategy.decide(DecisionContext)` and the **same** risk gate.
The context is built from rows `<= t` only. `tests/test_parity.py` requires the two engines to
agree on daily returns to within share rounding (max 5 bp, correlation > 0.999), and the
journal alone must rebuild the production book exactly.

## Layers

`tests/test_architecture.py` enforces that a module imports only from its own layer or lower:

```
core → stats → data → alpha → risk → execution → portfolio → ml → engine → options → research → live → web → cli
```

| layer | responsibility | key files |
|---|---|---|
| core | hashing (canonical JSON, SHA-256), exact money (micro-dollars), NYSE calendar, clock, errors | `hashing.py`, `money.py`, `calendar.py` |
| stats | PSR/DSR/MinTRL, Newey-West, HAC OLS, stationary bootstrap, Holm/BH, purged & CPCV splits, PBO | `sharpe.py`, `inference.py`, `pbo.py`, `cv.py` |
| data | Alpaca client, PIT universe, bar store + manifest, quality, French factors, IV, SEC EDGAR client and point-in-time fundamentals (security master, XBRL facts, earnings releases), option-surface features, FINRA daily short-sale volume, synthetic market | `pipeline.py`, `universe.py`, `store.py`, `sec.py`, `fundamentals.py`, `optskew.py`, `shortvol.py`, `synthetic.py` |
| alpha | causal operators, signal library, composite, IC evaluation, look-ahead checker | `signals.py`, `ops.py`, `lookahead.py` |
| risk | statistical factor model with VRA, bias statistics, VaR/ES, stress scenarios, pre-trade gate, kill switch | `model.py`, `validate.py`, `limits.py` |
| execution | cost model (calibrated spread, √impact, fees, borrow), order state machine, brokers | `costs.py`, `orders.py`, `broker.py` |
| portfolio | decision interface, optimiser, hierarchical risk parity, strategies (optimised alpha, trend heritage, forecast risk parity, equal weight) | `optimizer.py`, `hrp.py`, `strategies.py` |
| ml | walk-forward ridge / gradient boosting alpha combination | `walkforward.py` |
| engine | research backtester, exact ledger, journal, paper engine, attribution | `backtest.py`, `paper.py`, `ledger.py` |
| options | Black-Scholes-Merton, IV inversion, American trees, SVI smiles, volatility surfaces, strategies at real quotes, call overlay | `black.py`, `american.py`, `svi.py`, `surface.py`, `quoted.py`, `overlay.py` |
| micro | local order book from diffs, queue-position market maker, impact estimators | `book.py`, `maker.py`, `impact.py` |
| research | pre-registration ledger, gate, study runner, report, families, exploratory robustness | `registry.py`, `gates.py`, `study.py`, `robustness.py` |
| live | forward shadow books (one per registered trial), daily report, decision log, mirroring a book in a broker account, forward review against the gate, signal record, dead-man check, alerts and digest | `forward.py`, `report.py`, `decisions.py`, `execute.py`, `review.py`, `signal_record.py`, `health.py`, `notify.py` |
| web | read-only dashboard, in a browser or as a desktop program | `app.py`, `index.html`, `desktop.py` |

## Decisions that matter

* **Timing.** A decision at the close of session `t` executes at session `t + lag` (VWAP by
  default, open/close selectable). Signal evaluation uses the same convention: the forward
  return of row `t` starts at the execution price of `t + 1`, never at the close of `t`.
* **Survivorship.** Membership spells are rebuilt from the dated change log; a removed member
  is priced `asof` its last index day, so a reused ticker cannot leak a different company's
  history. Security ids are `TICKER@spell-start` and never change.
* **Corporate actions.** Prices are split- and dividend-adjusted (`adjustment=all`); the nightly
  update re-requests the last stored week and re-downloads a security whose overlap moved
  (a new split rescales all history). Spin-offs are not adjusted by the vendor and appear as
  extreme moves in the quality report (AA 2016-11-01, TGNA 2017-06-01).
* **Delisting.** Positions in a security whose prices end are closed at its last close minus
  the spread on the next session.
* **Costs.** Calibrated spread model (docs/CALIBRATION.md), square-root impact, SEC fee on
  sells, commission, 5% ADV participation cap (unfilled remainder expires), borrow on shorts,
  margin spread on debit balances, risk-free interest on cash.
* **Determinism.** No wall clock below `live`; solver warm starts are disabled (a resumed book
  reproduced an uninterrupted one only after this: `tests/test_determinism.py`).
* **Fail loudly.** No broad `except`, no silent `pass` (architecture test). A rejected target
  keeps the current portfolio and is journalled; a solver failure is counted and reported.

## Data locations

`PUSULA_HOME` (default `~/.pusula`: outside OneDrive, which locks files mid-sync and broke
atomic renames during development, and outside `AppData`, which Windows redirects to a private
copy for processes started from an MSIX-packaged app, so the nightly scheduled task could not
see a store built from such a terminal) holds `data/` (bars, manifest, universe
HTML snapshot, spread samples, factors, IV, and `fundamentals/`: the security master with how
each spell got its CIK, distilled XBRL facts, the filing index with acceptance times, the
per-company events derived from them, and a manifest with the SHA-256 of every SEC response;
and `optskew/`: daily per-stock option-surface features reduced from DoltHub's strike-level
chains while streaming, with the SHA-256 of the bytes read; and `shortvol/`: FINRA's daily
short-sale and total volume for every symbol, with a per-file SHA-256 chain and the list of
sessions without a file; a session's file is used from the next session's close, because FINRA
publishes it after the close. FINRA names a stock by the ticker it had that day, so a renamed
member is read under its earlier tickers (`src/pusula/data/ticker_history.py`, each entry found
and re-checked by volume co-movement in `tools/shortvol_report.py`), and a row counts only on a
session on which the security has a price)
and `forward/` (one directory per shadow book with `journal.jsonl` and `state.json`, plus
`dashboard.json`). The research ledger lives in the repository (`ledger/experiments.jsonl`) so
that it is versioned with the code, as are the hand-verified CIK overrides
(`src/pusula/data/cik_overrides.py`).

## Fundamentals: when is a number known?

An XBRL fact carries the accession it was reported in; the accession's EDGAR acceptance time
(UTC, to the second) maps to the first session that closes after it, and that session is the
first on which a decision may use the number. Values are as known at that time (a later
restatement never reaches earlier sessions); concept tag chains are resolved within each
filing; Q4 is the fiscal year minus the nine-month year-to-date. Earnings-release dates come
from 8-K Item 2.02 acceptance times, not from period ends or filing dates. Details and
coverage: [FUNDAMENTALS_DATA.md](FUNDAMENTALS_DATA.md).
