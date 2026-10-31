"""Out-of-sample validation of risk forecasts (bias statistics).

For a portfolio with forecast daily vol s_t (made at the close of t) and realised next-day
return r_{t+1}, the standardised outcome b_t = r_{t+1} / s_t has unit variance if the
forecast is unbiased. The bias statistic is the standard deviation of b over the test period;
its 95% band under a correct model is roughly 1 +/- sqrt(2 / T). Values well above 1 mean risk
is under-forecast (the dangerous direction: positions too large); below 1, over-forecast.

Random portfolios test the model where nobody is looking for its errors. Optimised portfolios
are where it matters: an optimiser loads on the directions whose risk the model under-states
(Shepard 2009; Menchero, Wang & Orr 2011). ``optimized_random_alpha`` (dollar-neutral
Sigma^-1 alpha for pure-noise alphas) and ``min_variance`` (fully invested Sigma^-1 1) measure
that optimisation bias; they use no return forecast, so they are calibration, not research.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pusula.risk.model import FactorRisk, RiskModelSpec, estimate

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BiasReport:
    portfolio: str
    bias: float
    band: float
    n: int
    q_stat: float  # mean of (b^2 - ln b^2): penalises both directions (Patton 2011 QLIKE-type)


def solve_sigma(fr: FactorRisk, x: FloatArray) -> FloatArray:
    """Sigma^-1 x by the Woodbury identity (O(n k^2); the n x n matrix is never formed)."""
    dinv_x = x / fr.d
    db = fr.B / fr.d[:, None]
    k = fr.F.shape[0]
    m = np.linalg.inv(fr.F + 1e-14 * np.eye(k)) + fr.B.T @ db
    return dinv_x - db @ np.linalg.solve(m, fr.B.T @ dinv_x)


def mean_variance_weights(fr: FactorRisk, alpha: FloatArray, *, neutral: bool) -> FloatArray:
    """Unconstrained mean-variance weights Sigma^-1 (alpha - c 1): dollar-neutral (c chosen so
    the weights sum to zero, gross 1) or, with ``neutral=False``, fully invested (sum 1)."""
    a = solve_sigma(fr, alpha)
    o = solve_sigma(fr, np.ones(fr.n))
    if neutral:
        w = a - (a.sum() / o.sum()) * o
        return w / np.abs(w).sum()
    return a / a.sum()


def bias_statistics(
    returns: FloatArray,
    members: npt.NDArray[np.bool_],
    spec: RiskModelSpec,
    start: int,
    step: int = 5,
    n_random: int = 20,
    seed: int = 0,
    end: int | None = None,
    groups: npt.NDArray[np.int64] | None = None,
) -> list[BiasReport]:
    """Bias statistics for equal-weight, random long-only, random long-short, single-name and
    optimised (random-alpha mean-variance, minimum-variance) portfolios, re-estimating the
    model every ``step`` sessions from ``start`` to ``end`` (exclusive; default: the end)."""
    rng = np.random.default_rng(seed)
    rng_alpha = np.random.default_rng(seed + 1)  # own stream: the random portfolios above stay
    # exactly those of the published calibration (CALIBRATION.md §2)
    T = returns.shape[0] if end is None else min(end, returns.shape[0])
    outcomes: dict[str, list[float]] = {
        "equal_weight": [],
        "random_long": [],
        "random_long_short": [],
        "single_name": [],
        "optimized_random_alpha": [],
        "min_variance": [],
    }
    for t in range(start, T - 1, step):
        m = members[t]
        fr = estimate(returns, t, m, spec, groups)
        if fr is None:
            continue
        nxt = returns[t + 1, fr.cols]
        ok = np.isfinite(nxt)
        if ok.sum() < 20:
            continue

        realised = np.where(ok, nxt, 0.0)

        def push(
            name: str, w: FloatArray, fr: FactorRisk = fr, realised: FloatArray = realised
        ) -> None:
            s = fr.vol(w, annualise=False)
            if s > 0:
                outcomes[name].append(float(w @ realised) / s)

        n = fr.n
        ew = np.where(ok, 1.0, 0.0)
        push("equal_weight", ew / ew.sum())
        for _ in range(n_random):
            w = np.where(ok, rng.random(n), 0.0)
            push("random_long", w / w.sum())
            ls = np.where(ok, rng.normal(size=n), 0.0)
            push("random_long_short", ls / np.abs(ls).sum())
            j = int(rng.choice(np.nonzero(ok)[0]))
            single = np.zeros(n)
            single[j] = 1.0
            push("single_name", single)
            alpha = np.where(ok, rng_alpha.normal(size=n), 0.0)
            push("optimized_random_alpha", mean_variance_weights(fr, alpha, neutral=True))
        push("min_variance", mean_variance_weights(fr, np.ones(n), neutral=False))
    out = []
    for name, b in outcomes.items():
        arr = np.asarray(b)
        if arr.size < 10:
            continue
        b2 = np.maximum(arr**2, 1e-12)
        out.append(
            BiasReport(
                name,
                float(arr.std(ddof=1)),
                float(np.sqrt(2.0 / arr.size)),
                int(arr.size),
                float(np.mean(b2 - np.log(b2))),
            )
        )
    return out
