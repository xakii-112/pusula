"""Heteroskedasticity- and autocorrelation-consistent inference.

Daily strategy returns and daily information coefficients are autocorrelated (overlapping
holding periods, slow signals). Ordinary t-statistics then overstate significance; every
t-statistic in pusula is Newey-West unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sps


def nw_lags(n: int) -> int:
    """Newey-West (1994) plug-in bandwidth floor(4 (n/100)^(2/9))."""
    return max(0, math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _long_run_variance(u: npt.NDArray[np.float64], lags: int) -> float:
    n = u.size
    s = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(u[k:] @ u[:-k]) / n
    return s


@dataclass(frozen=True, slots=True)
class MeanTest:
    mean: float
    se: float
    t: float
    p: float
    n: int
    lags: int


def nw_mean_test(x: npt.ArrayLike, lags: int | None = None) -> MeanTest:
    """t-test of mean(x) = 0 with a Bartlett-kernel HAC standard error."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n < 3:
        return MeanTest(math.nan, math.nan, math.nan, math.nan, n, 0)
    L = nw_lags(n) if lags is None else lags
    mu = float(a.mean())
    lrv = _long_run_variance(a - mu, L)
    se = math.sqrt(max(lrv, 0.0) / n)
    t = mu / se if se > 0 else math.nan
    p = float(2.0 * sps.t.sf(abs(t), df=n - 1)) if math.isfinite(t) else math.nan
    return MeanTest(mu, se, t, p, n, L)


@dataclass(frozen=True, slots=True)
class OLSResult:
    coef: npt.NDArray[np.float64]
    se: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    r2: float
    n: int
    lags: int
    names: tuple[str, ...]

    def get(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return float(self.coef[i]), float(self.t[i])


def ols_hac(
    y: npt.ArrayLike,
    X: npt.ArrayLike,
    names: tuple[str, ...] | None = None,
    add_const: bool = True,
    lags: int | None = None,
) -> OLSResult:
    """OLS with Newey-West covariance. Rows with any non-finite value are dropped."""
    yv = np.asarray(y, dtype=np.float64)
    Xm = np.asarray(X, dtype=np.float64)
    if Xm.ndim == 1:
        Xm = Xm[:, None]
    ok = np.isfinite(yv) & np.all(np.isfinite(Xm), axis=1)
    yv, Xm = yv[ok], Xm[ok]
    if add_const:
        Xm = np.column_stack([np.ones(yv.size), Xm])
    n, k = Xm.shape
    labels = names if names is not None else tuple(f"x{i}" for i in range(k - add_const))
    if add_const:
        labels = ("const", *labels)
    if len(labels) != k:
        raise ValueError("names length does not match regressors")
    if n <= k:
        nan = np.full(k, np.nan)
        return OLSResult(nan, nan, nan, math.nan, n, 0, labels)
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ yv
    resid = yv - Xm @ beta
    L = nw_lags(n) if lags is None else lags
    Xu = Xm * resid[:, None]
    S = Xu.T @ Xu
    for j in range(1, min(L, n - 1) + 1):
        w = 1.0 - j / (L + 1.0)
        G = Xu[j:].T @ Xu[:-j]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else math.nan
    return OLSResult(beta, se, t, r2, n, L, labels)
