"""Where did the money come from?

* Factor attribution: regression of daily net excess returns on Fama-French 5 + momentum with
  HAC errors. The intercept is the alpha that known factor premia do not explain.
* Cost attribution: gross return, trading costs, borrow and financing, per year.
* Calendar-year table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from pusula.engine.backtest import BacktestResult
from pusula.stats.inference import OLSResult, ols_hac

FACTORS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    alpha_annual: float
    alpha_t: float
    betas: dict[str, float]
    tstats: dict[str, float]
    r2: float
    n: int


def factor_attribution(
    returns: npt.NDArray[np.float64],
    dates: npt.NDArray[np.datetime64],
    factors: pd.DataFrame,
    names: tuple[str, ...] = FACTORS,
) -> FactorAttribution:
    f = factors.reindex(pd.DatetimeIndex(dates.astype("datetime64[ns]")))
    rf = f["rf"].to_numpy()
    y = returns - rf
    X = f[list(names)].to_numpy()
    res: OLSResult = ols_hac(y, X, names=names)
    if not np.isfinite(res.coef).all():
        return FactorAttribution(math.nan, math.nan, {}, {}, math.nan, res.n)
    return FactorAttribution(
        alpha_annual=float(res.coef[0] * 252),
        alpha_t=float(res.t[0]),
        betas={k: float(res.coef[i + 1]) for i, k in enumerate(names)},
        tstats={k: float(res.t[i + 1]) for i, k in enumerate(names)},
        r2=res.r2,
        n=res.n,
    )


def yearly_table(res: BacktestResult) -> pd.DataFrame:
    idx = pd.DatetimeIndex(res.dates.astype("datetime64[ns]"))
    eq = pd.Series(res.equity, index=idx)
    prev = eq.shift(1)
    df = pd.DataFrame(
        {
            "net": res.returns,
            "gross": res.gross_returns,
            "cost": res.costs / prev.to_numpy(),
            "borrow": res.borrow / prev.to_numpy(),
            "turnover": res.turnover,
        },
        index=idx,
    ).iloc[1:]
    g = df.groupby(pd.DatetimeIndex(df.index).year)
    out = pd.DataFrame(
        {
            "net": g["net"].apply(lambda x: float(np.prod(1 + x) - 1)),
            "gross": g["gross"].apply(lambda x: float(np.prod(1 + x) - 1)),
            "cost": g["cost"].sum(),
            "borrow": g["borrow"].sum(),
            "turnover": g["turnover"].sum(),
            "sharpe": g["net"].apply(
                lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std() > 0 else np.nan
            ),
        }
    )
    out.index.name = "year"
    return out
