"""Promotion gate: the conditions a strategy must meet, out of sample, to be traded.

Thresholds are part of the pre-registration and are fixed before results exist. A check whose
input is missing or non-finite **fails**: a stress test that was not run is not a pass.

Checks (all on the holdout window unless stated):

* ``dsr``          deflated Sharpe ratio, trials = every registered config in the family
* ``t_nw``         Newey-West t-statistic of the mean daily net return
* ``alpha_t``      t-statistic of the intercept vs. Fama-French 5 + momentum (HAC): is it more
                   than cheaply available factor premia?
* ``lag_stress``   executing one session later keeps >= ``lag_retention`` of the Sharpe ratio
                   (and stays positive): edges that vanish with a day's delay were never
                   tradable (NihaiAlgoritma's exec-lag lesson)
* ``cost_stress``  with costs doubled, the NW t-stat stays >= ``cost_stress_t`` (one-sided);
                   "mean > 0" was too weak (meridian v1 lesson)
* ``pbo``          probability of backtest overfitting of the family (full sample) <= max
* ``drawdown``     maximum drawdown <= max
* ``stability``    share of calendar years (dev + holdout) with positive net return >= min
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class GateSpec:
    min_dsr: float = 0.95
    min_t_nw: float = 2.0
    min_alpha_t: float = 2.0
    lag_retention: float = 0.5
    cost_stress_t: float = 1.0
    max_pbo: float = 0.5
    max_drawdown: float = 0.25
    min_positive_years: float = 0.6

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    promoted: bool
    checks: tuple[Check, ...]

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def as_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "failed": self.failed,
            "checks": [
                {
                    "name": c.name,
                    "value": c.value if math.isfinite(c.value) else None,
                    "threshold": c.threshold,
                    "passed": c.passed,
                    "note": c.note,
                }
                for c in self.checks
            ],
        }


def _ge(name: str, v: float, thr: float, note: str = "") -> Check:
    return Check(name, v, thr, bool(math.isfinite(v) and v >= thr), note)


def _le(name: str, v: float, thr: float, note: str = "") -> Check:
    return Check(name, v, thr, bool(math.isfinite(v) and v <= thr), note)


def evaluate(
    spec: GateSpec,
    *,
    dsr: float,
    t_nw: float,
    alpha_t: float,
    sharpe: float,
    sharpe_lag2: float,
    t_nw_cost2: float,
    pbo: float,
    max_dd: float,
    positive_year_share: float,
) -> Verdict:
    retention = sharpe_lag2 / sharpe if math.isfinite(sharpe) and sharpe > 0 else math.nan
    lag_ok = math.isfinite(retention) and retention >= spec.lag_retention and sharpe_lag2 > 0
    checks = (
        _ge("dsr", dsr, spec.min_dsr),
        _ge("t_nw", t_nw, spec.min_t_nw),
        _ge("alpha_t", alpha_t, spec.min_alpha_t, "intercept vs FF5+MOM, HAC"),
        Check(
            "lag_stress",
            retention,
            spec.lag_retention,
            bool(lag_ok),
            f"Sharpe lag2 {sharpe_lag2:.2f}",
        ),
        _ge("cost_stress", t_nw_cost2, spec.cost_stress_t, "NW t with 2x costs"),
        _le("pbo", pbo, spec.max_pbo, "family CSCV"),
        _le("drawdown", max_dd, spec.max_drawdown),
        _ge("stability", positive_year_share, spec.min_positive_years, "share of positive years"),
    )
    return Verdict(all(c.passed for c in checks), checks)
