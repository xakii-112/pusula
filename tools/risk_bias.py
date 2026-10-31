"""Risk-model calibration, part 3: optimisation bias (CALIBRATION.md §3).

An optimiser loads on the directions whose risk the model under-states, so a model that is
unbiased on random portfolios can still be biased on the portfolios that are actually held.
This measures the bias statistic of optimised portfolios built from pure-noise alphas (no
return forecast is used: calibration, not research) for a fixed list of candidate model
configurations, on the development window (to choose) and on 2023-2026 (to confirm):

* closed-form portfolios (``risk.validate``): dollar-neutral Sigma^-1 alpha, minimum variance;
* the production long-short optimiser (``research.families.LS``: dollar and beta neutral,
  1.5% names, 10% ex-ante vol cap) with noise alphas of production scale
  (IC 0.02 x forecast vol x sqrt(21) x N(0,1)), no costs, from flat.

    uv run python tools/risk_bias.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# fixed before any real-data result was seen
CONFIGS: dict[str, dict[str, object]] = {
    "adopted": {},
    "shrink_0.30": {"specific_shrinkage": 0.30},
    "shrink_0.50": {"specific_shrinkage": 0.50},
    "factors_20": {"n_factors": 20},
    "eigen_1.2": {"eigen_adjust": 1.2},
    "eigen_1.2_shrink_0.30": {"eigen_adjust": 1.2, "specific_shrinkage": 0.30},
    # added after the first six were measured (min-variance bias 2.3, residual correlation
    # within industries +0.05 / +0.09 for 2 / 3-digit SIC), fixed before these two were run
    "industry_sic2": {"industry_min_size": 5, "_sic_digits": 2},
    "industry_sic3": {"industry_min_size": 5, "_sic_digits": 3},
}
WINDOWS = {"dev": ("2017-01-03", "2022-12-30"), "holdout": ("2023-01-03", "2026-10-22")}
STEP, N_ALPHA = 10, 5


def _summary(b: list[float]) -> dict[str, float]:
    arr = np.asarray(b)
    b2 = np.maximum(arr**2, 1e-12)
    return {
        "bias": float(arr.std(ddof=1)),
        "band": float(np.sqrt(2.0 / arr.size)),
        "n": int(arr.size),
        "q": float(np.mean(b2 - np.log(b2))),
    }


def _ls_bias(
    returns: np.ndarray,
    members: np.ndarray,
    spec: object,
    lo: int,
    hi: int,
    groups: np.ndarray | None,
) -> dict[str, float]:
    """The production long-short optimiser with zero costs and a flat start reduces to:
    max alpha'w - lambda h risk(w) s.t. |w| <= 1.5%, gross <= 2, sum w = 0, |beta'w| <= 0.05,
    sqrt(risk) <= 10%/sqrt(252). Built per date with the model's own factor count (the
    production optimiser's compiled DPP form fixes it; padded to 200 factors its parameter
    mapping needed 13.5 GB). Only alpha is a parameter."""
    import cvxpy as cp

    from pusula.research.families import LS
    from pusula.risk.model import estimate

    rng = np.random.default_rng(2)
    out, vols = [], []
    assert LS.vol_target_daily is not None
    assert LS.beta_max is not None
    for t in range(lo, hi - 1, STEP):
        fr = estimate(returns, t, members[t], spec, groups)  # type: ignore[arg-type]
        if fr is None:
            continue
        nxt = returns[t + 1, fr.cols]
        ok = np.isfinite(nxt)
        L = np.linalg.cholesky(fr.F + 1e-12 * np.eye(fr.F.shape[0]))
        bt = L.T @ fr.B.T  # k x n
        sd = np.sqrt(fr.d)
        ew = np.where(ok, 1.0 / ok.sum(), 0.0)
        beta = fr.beta(ew)
        ub = np.where(ok, LS.max_weight, 0.0)
        a = cp.Parameter(fr.n)
        w = cp.Variable(fr.n)
        risk = cp.sum_squares(bt @ w) + cp.sum_squares(cp.multiply(sd, w))
        prob = cp.Problem(
            cp.Maximize(a @ w - LS.risk_aversion * 21.0 * risk),
            [
                cp.abs(w) <= ub,
                cp.norm1(w) <= LS.max_gross,
                cp.sum(w) == 0,
                cp.abs(beta @ w) <= LS.beta_max,
                cp.norm(cp.hstack([bt @ w, cp.multiply(sd, w)]), 2) <= LS.vol_target_daily,
            ],
        )
        sig = fr.total_vol()
        for _ in range(N_ALPHA):
            a.value = np.where(ok, 0.02 * sig * np.sqrt(21.0) * rng.normal(size=fr.n), 0.0)
            prob.solve(solver=cp.CLARABEL, warm_start=False)
            if w.value is None:
                continue
            wv = np.asarray(w.value)
            s = fr.vol(wv, annualise=False)
            if s > 0:
                out.append(float(wv @ np.where(ok, nxt, 0.0)) / s)
                vols.append(s * np.sqrt(252.0))
    return {**_summary(out), "mean_ex_ante_vol": float(np.mean(vols))}


def run(name: str) -> dict[str, object]:
    from pusula.data.pipeline import research_panel
    from pusula.risk.model import RiskModelSpec, simple_returns
    from pusula.risk.validate import bias_statistics

    p = research_panel().panel
    returns = simple_returns(p["close"])
    members = p["in_universe"] > 0
    cfg = dict(CONFIGS[name])
    digits = cfg.pop("_sic_digits", None)
    groups = None
    if digits is not None:
        from pusula.data import fundamentals as F
        from pusula.data.settings import data_dir

        m = F.read_master(data_dir())
        sic = {str(s): c for s, c in zip(m["security_id"], m["sic"], strict=True)}
        div = 10 ** (4 - int(digits))  # type: ignore[call-overload]
        groups = np.array(
            [
                int(sic[s]) // div if str(sic.get(s)) not in ("<NA>", "None") else -1
                for s in p.symbols
            ],
            dtype=np.int64,
        )
    spec = replace(RiskModelSpec(n_factors=12, window=252), **cfg)  # type: ignore[arg-type]
    res: dict[str, object] = {}
    for w, (a, b) in WINDOWS.items():
        lo, hi = p.asof_index(a), p.asof_index(b) + 1
        reps = bias_statistics(
            returns, members, spec, start=lo, step=STEP, n_random=20, end=hi, groups=groups
        )
        res[w] = {
            r.portfolio: {"bias": r.bias, "band": r.band, "n": r.n, "q": r.q_stat} for r in reps
        }
        res[w]["ls_optimizer_random_alpha"] = _ls_bias(returns, members, spec, lo, hi, groups)  # type: ignore[index]
    print(f"[risk_bias] {name} done", flush=True)
    return {name: res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="configurations to (re)run")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"  # one BLAS thread per worker process
    out = ROOT / "var" / "risk_bias.json"
    out.parent.mkdir(exist_ok=True)
    results: dict[str, object] = (
        json.loads(out.read_text(encoding="utf-8")) if out.exists() and args.only else {}
    )
    with ProcessPoolExecutor(args.workers) as ex:
        for part in ex.map(run, args.only or list(CONFIGS)):
            results.update(part)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for w in WINDOWS:
        print(f"\n== {w}")
        ports = list(next(iter(results.values()))[w])  # type: ignore[index]
        print(f"{'config':24s}" + "".join(f"{p[:14]:>15s}" for p in ports))
        for name, r in results.items():
            print(f"{name:24s}" + "".join(f"{r[w][p]['bias']:15.3f}" for p in ports))  # type: ignore[index]


if __name__ == "__main__":
    main()
