# Research protocol

The protocol exists because the easiest person to fool is the author. Every rule below is
enforced in code; none relies on remembering it.

## 1. Before data: register

`pusula study register --family <f>` writes each trial to `ledger/experiments.jsonl`:
hypothesis (one sentence, falsifiable), the full configuration and its SHA-256, gate thresholds,
development and holdout windows, the SHA-256 of every source file. Records are hash-chained;
editing any line breaks `Registry.verify()` (the web view shows the chain status).

* A name is bound to its configuration hash. Changing a parameter after seeing results
  requires a **new name**, which is a new trial.
* Every registered configuration counts in the deflated Sharpe ratio of its family, whether or
  not its result was looked at. Benchmarks are flagged and do not count.

What may be done with real data **before** registering: data-quality checks, the spread
calibration (quotes, not returns) and the risk-model bias test (forecast calibration, not
strategy returns). What may not: looking at any signal's IC or any strategy's returns.

## 2. Run

`pusula study run --family <f>` runs every trial three times:

| variant | change | purpose |
|---|---|---|
| base | as registered | the result |
| lag2 | execution one session later | an edge that dies with one day of delay was never tradable |
| cost2 | every cost component doubled | the impact coefficient is a literature value, not a fit |

Results are appended with the data-manifest digest and the source manifest captured at the
start of the run. A family that already has results can only be re-run with
`--rerun-reason`, which is recorded.

## 3. Promote (all must hold on the holdout window)

| check | threshold | why |
|---|---|---|
| deflated Sharpe ratio (excess returns) | ≥ 0.95 | corrects for the number of registered trials in the programme, the dispersion of their Sharpe ratios, and non-normal returns |
| Newey-West t of mean daily return | ≥ 2.0 | daily strategy returns are autocorrelated |
| FF5 + momentum alpha t (HAC) | ≥ 2.0 | a factor premium available for a few basis points is not alpha |
| lag stress | Sharpe(lag+1) ≥ 50% of Sharpe, and > 0 | NihaiAlgoritma's exec-lag lesson |
| cost stress | NW t with 2× costs ≥ 1.0 | "mean > 0" was too weak (meridian v1) |
| PBO (family, CSCV) | ≤ 0.5 | is the best configuration in-sample usually above median out of sample? |
| maximum drawdown | ≤ 25% | |
| stability | ≥ 60% of calendar years positive | |

A check whose input is missing (stress not run, factors unavailable) **fails**.

## 2b. The code that runs is the code that was registered

`study run` recomputes every trial's configuration hash from the code and refuses to run if it
differs from the registered one. Fields added to `Trial` later are left out of the hashed form at
their defaults, so earlier registrations stay valid. `study reevaluate` recomputes the deflated
Sharpe ratio and the verdict from the statistics stored in the ledger when the *evaluation*
method changes (it requires `--rerun-reason`).

## 3b. Re-design after a result: a new version, and it pays for the old one

When a result reveals a *design* flaw (not a bug in measurement), the family is never edited.
A new version is registered (`us-equity-v2`), its rationale is written next to it before it
runs, and the deflated Sharpe of every version counts the trials of all versions of the
programme (`Registry.program_trials("us-equity-")`). A *measurement* bug (the code computed a
statistic wrongly) is fixed in code and the same registered configuration is re-run with
`--rerun-reason`; both runs stay in the ledger.

This happened once already. `us-equity-v1`'s first run reported holdout Sharpe ratios of 12,
12 and 83. Two causes: Sharpe-type statistics were computed on raw instead of excess returns
(measurement bug: fixed, v1 re-run), and research backtests used the production kill switch,
which latches and left three books in T-bills from 2020 on (design flaw: v2 runs research
without the latch; production books keep it).

## 4. After promotion: forward

Every registered trial, promoted or not, also runs as a forward paper book from the first
session after its registration (`pusula forward`). This record is out of sample by
construction and cannot be back-filled. A strategy is eligible for capital only after its
forward record passes the same gate on its own; until then even a promoted strategy trades
paper. `pusula review` applies that gate to the record: no verdict below 63 sessions, then every
check, with the deflated Sharpe counting the whole programme; the lag and cost stresses come from
`--stress`, and without it they fail like any other missing check. It also reports how many
sessions the record needs to mean anything (MinTRL at the Sharpe the holdout claimed).

## 4a. What it takes to trade real money

`pusula execute` can mirror a book, or all of them netted (`--all`), in the paper account on the
operator's word, promoted or not: paper costs nothing and proves the plumbing. A netted account says
nothing about any one strategy, and no result is ever read from it. A live account is refused by code (`live/execute.py`,
`live_refusals`) unless the trial passed this gate in the ledger, its forward record passed it on its
own with the stresses run, the operator armed exactly that book, and a capital limit and separate
keys are set. The refusal lists what is missing. There is no flag around it, and the thresholds are not
lowered to reach it: a strategy that cannot pass is a strategy that should not be funded.

## 4b. Exploration is allowed, and it cannot promote anything

`tools/robustness.py` re-runs registered trials with neighbouring parameters and with 1e-9 noise on
prices (`docs/ROBUSTNESS.md`), and `pusula review` reports each signal's IC since registration. Neither
is a gate input. A neighbouring parameter is not a trial: if one looks better, it is registered under
a new name and pays for itself in the deflated Sharpe. A family on a fixed list of funds registers
that list with its configuration (`StudySpec.universe`), so the list is covered by the hash; a family
with its own factors registers those too (`StudySpec.factor_names`): the alpha check of a crypto
family would mean nothing on US equity factors.

A run may be repeated for a defect, with the reason in the ledger, and both results stay on the chain.
That happened twice on 2026-10-30, both found by reading the first results: reused crypto symbols
appearing as returns (`us-crypto-v1`), and Saturday-dated option snapshots leaving a book in cash
(`us-optquote-v1`). Neither fix looks at returns: each is a rule about the data.

## Lessons carried over from NihaiAlgoritma, and where they live now

| lesson (NihaiAlgoritma CLAUDE.md / Tur reports) | enforcement here |
|---|---|
| today's index members are survivors (ACI DERS #1, PEAD 1.57 → 1.05) | `data/universe.py` spells + `asof` entity mapping |
| a one-bar execution lag kills most TA edges | `lag2` variant + gate |
| compare against a fixed baseline, not against nothing | equal-weight benchmark in every family; FF5+MOM alpha |
| number of tries inflates Sharpe | DSR with the registered trial count |
| live and backtest code diverged (Tur-101) | one `Strategy` for both engines + `tests/test_parity.py` |
| equity curve mixed resets with P&L (B-15) | capital events separate from P&L; exact identity |
| fail-soft hid broken states (Tur-65) | no broad except (architecture test) |
| a portfolio filled in alphabetical order (BULGU-101-1) | deterministic, score-ordered selection with explicit tie-breaks |
| complexity before measurable edge (282 M model) | ML must beat the linear composite through the same gate |
