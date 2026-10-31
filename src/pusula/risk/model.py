"""Statistical factor risk model.

    Sigma = B F B' + diag(d)

estimated at each decision date from trailing daily returns of the point-in-time universe:

1. returns are scaled by each name's EWMA volatility so that PCA finds common *correlation*
   structure rather than the most volatile names;
2. the first ``k`` principal components of the scaled returns are the factors; loadings are
   mapped back to return space ``B = diag(sigma) V``;
3. factor covariance ``F`` is an EWMA covariance of the factor returns;
4. specific variances ``d`` are EWMA variances of the residuals, shrunk towards the
   cross-sectional median (noisy individual estimates are the main source of optimiser error);
5. a volatility-regime adjustment (VRA, as in commercial equity risk models) rescales the
   whole covariance by the recent cross-sectional bias of standardised returns, so the model
   reacts to a market-wide volatility shift within days instead of a half-life;
6. optionally (``eigen_adjust``), the eigen-adjustment of Menchero, Wang & Orr (2011): a
   sample covariance under-states the variance of its own smallest eigenportfolios, exactly
   the directions an optimiser loads on. Factor returns are simulated from the estimated
   covariance, re-estimated with the same weights, and the ratio of true to estimated
   variance of each simulated eigenportfolio scales the matching eigenvalue
   (``gamma = a (v - 1) + 1``, Barra's empirical scaling ``a``).

Everything uses returns up to and including the decision date only. ``risk.validate`` checks
the forecasts out of sample with bias statistics, the standard test for risk models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RiskModelSpec:
    window: int = 252
    n_factors: int = 12
    vol_halflife: float = 42.0
    factor_halflife: float = 84.0
    specific_halflife: float = 42.0
    specific_shrinkage: float = 0.10
    min_coverage: float = 0.8
    clip_sigma: float = 6.0  # winsorise standardised returns at this many sigmas
    vra_halflife: float | None = 21.0  # None disables the volatility-regime adjustment
    vra_split: bool = (
        False  # separate factor/specific adjustments over-corrected factor risk (CALIBRATION.md)
    )
    # Additions after the first calibration default to off, so that every registered family
    # keeps its configuration (and hash); new families opt in (CALIBRATION.md §3).
    eigen_adjust: float | None = None  # Barra scaling a of the eigen-adjustment; None = off
    eigen_sims: int = 100
    # industry factors on the residuals: groups (e.g. SIC codes) are passed to ``estimate``;
    # a group needs at least this many names to get its own factor. None = off.
    industry_min_size: int | None = None


@dataclass(frozen=True, slots=True)
class FactorRisk:
    """Risk forecast for one date over the securities ``cols`` (indices into the panel)."""

    cols: npt.NDArray[np.intp]
    B: FloatArray  # n x k loadings (daily return units)
    F: FloatArray  # k x k factor covariance (daily)
    d: FloatArray  # n specific variances (daily)

    @property
    def n(self) -> int:
        return int(self.cols.size)

    def covariance(self) -> FloatArray:
        return self.B @ self.F @ self.B.T + np.diag(self.d)

    def variance(self, w: FloatArray) -> float:
        y = self.B.T @ w
        return float(y @ self.F @ y + (self.d * w * w).sum())

    def vol(self, w: FloatArray, annualise: bool = True) -> float:
        v = np.sqrt(max(self.variance(w), 0.0))
        return float(v * np.sqrt(252.0)) if annualise else float(v)

    def total_vol(self) -> FloatArray:
        """Per-name daily total volatility forecast."""
        return np.sqrt(np.einsum("ik,kl,il->i", self.B, self.F, self.B) + self.d)

    def beta(self, market_w: FloatArray) -> FloatArray:
        """Forecast beta of each name to the portfolio ``market_w``."""
        y = self.B.T @ market_w
        cov = self.B @ (self.F @ y) + self.d * market_w
        return cov / self.variance(market_w)

    def factor_exposure(self, w: FloatArray) -> FloatArray:
        return self.B.T @ w

    def marginal_contribution(self, w: FloatArray) -> FloatArray:
        """Euler risk contributions: w_i (Sigma w)_i / sigma_p; sums to sigma_p (daily)."""
        sig_w = self.B @ (self.F @ (self.B.T @ w)) + self.d * w
        sp = np.sqrt(self.variance(w))
        return w * sig_w / sp if sp > 0 else np.zeros_like(w)


def _ewm_weights(n: int, halflife: float) -> FloatArray:
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    return w / w.sum()


def estimate(
    returns: FloatArray,
    t: int,
    members: npt.NDArray[np.bool_],
    spec: RiskModelSpec,
    groups: npt.NDArray[np.int64] | None = None,
) -> FactorRisk | None:
    """Fit the model at decision row ``t`` from ``returns[t - window + 1 : t + 1]``.

    ``returns`` is T x N daily simple returns (NaN = no data); ``members`` the N-vector of names
    eligible at ``t``. Names with less than ``min_coverage`` of the window are excluded.
    ``groups`` (N industry codes, -1 unknown) is used only with ``spec.industry_min_size``.
    Returns ``None`` if the window is not yet full.
    """
    lo = t - spec.window + 1
    if lo < 0:
        return None
    R = returns[lo : t + 1]
    cover = np.isfinite(R).mean(axis=0)
    cols = np.nonzero(members & (cover >= spec.min_coverage))[0]
    if cols.size < spec.n_factors + 10:
        return None
    X = R[:, cols]
    W = X.shape[0]
    w_vol = _ewm_weights(W, spec.vol_halflife)
    filled = np.where(np.isfinite(X), X, 0.0)
    obs = np.isfinite(X)
    var = (w_vol[:, None] * filled**2).sum(0) / np.maximum((w_vol[:, None] * obs).sum(0), 1e-12)
    sigma = np.sqrt(np.maximum(var, 1e-10))
    Z = np.clip(filled / sigma, -spec.clip_sigma, spec.clip_sigma)  # missing -> 0 (mean)
    # PCA of standardised returns (time-weighted so the model reacts to regime changes)
    sw = np.sqrt(_ewm_weights(W, spec.factor_halflife) * W)
    _, _, Vt = np.linalg.svd(Z * sw[:, None], full_matrices=False)
    k = min(spec.n_factors, Vt.shape[0])
    V = Vt[:k].T  # n x k
    fz = Z @ V  # W x k factor returns in standardised space (orthogonal projection)
    rz = Z - fz @ V.T  # standardised residuals
    expo = V  # n x k exposures in standardised space
    if spec.industry_min_size is not None and groups is not None:
        # Industry factors: each day's mean standardised residual of a group; correlation the
        # statistical factors leave inside an industry would otherwise count as specific risk
        g = groups[cols]
        labels, counts = np.unique(g[g >= 0], return_counts=True)
        labels = labels[counts >= spec.industry_min_size]
        if labels.size:
            H = (g[:, None] == labels[None, :]).astype(np.float64)  # n x m membership
            find = rz @ (H / H.sum(0))  # W x m industry factor returns
            rz = rz - find @ H.T
            fz = np.hstack([fz, find])
            expo = np.hstack([V, H])
    B = sigma[:, None] * expo
    w_f = _ewm_weights(W, spec.factor_halflife)
    fm = (w_f[:, None] * fz).sum(0)
    fc = fz - fm
    F = (w_f[:, None] * fc).T @ fc
    if spec.eigen_adjust is not None:
        F = eigen_adjusted(F, w_f, spec.eigen_adjust, spec.eigen_sims)
    resid = rz * sigma  # back to return units
    w_s = _ewm_weights(W, spec.specific_halflife)
    sv = (w_s[:, None] * np.where(obs, resid, 0.0) ** 2).sum(0) / np.maximum(
        (w_s[:, None] * obs).sum(0), 1e-12
    )
    med = float(np.median(sv))
    d = (1.0 - spec.specific_shrinkage) * sv + spec.specific_shrinkage * med
    if spec.vra_halflife is not None:
        if spec.vra_split:
            F = F * _vra(fz, spec)
            d = d * _vra(np.where(obs, resid, np.nan), spec)
        else:
            scale = _vra(X, spec)
            F, d = F * scale, d * scale
    return FactorRisk(cols.astype(np.intp), B, F, np.maximum(d, 1e-10))


def eigen_adjusted(F: FloatArray, weights: FloatArray, a: float, sims: int) -> FloatArray:
    """Menchero, Wang & Orr (2011) eigen-adjusted covariance.

    For each of ``sims`` simulations, factor returns are drawn from N(0, F) for the same number
    of observations, their covariance is estimated with the same ``weights``, and for every
    simulated eigenportfolio the true variance (under F) is compared with the estimated one.
    The mean ratio v_k, scaled by ``a``, multiplies the k-th eigenvalue (sorted ascending in
    both, so the smallest estimated eigenvalue is matched with its own bias). Seeded: the
    result depends on its inputs only (determinism rule)."""
    lam0, U0 = np.linalg.eigh(F)
    lam0 = np.maximum(lam0, 1e-18)
    k, W = F.shape[0], weights.size
    rng = np.random.default_rng(12345)
    ratio = np.zeros(k)
    for _ in range(sims):
        f = rng.standard_normal((W, k)) * np.sqrt(lam0)  # eigenbasis: covariance diag(lam0)
        fc = f - (weights[:, None] * f).sum(0)
        Fm = (weights[:, None] * fc).T @ fc
        lam_m, Um = np.linalg.eigh(Fm)
        true = np.einsum("ik,i,ik->k", Um, lam0, Um)
        ratio += true / np.maximum(lam_m, 1e-18)
    v = np.sqrt(ratio / sims)
    gamma = a * (v - 1.0) + 1.0
    return (U0 * (gamma**2 * lam0)) @ U0.T


def _vra(X: FloatArray, spec: RiskModelSpec) -> float:
    """Variance multiplier from the recent cross-sectional bias of one-step-ahead standardised
    returns: each day's returns are divided by the EWMA vol known the day before."""
    lam = 0.5 ** (1.0 / spec.vol_halflife)
    obs = np.isfinite(X)
    x2 = np.where(obs, X, 0.0) ** 2
    W = X.shape[0]
    var = np.full(X.shape[1], np.nan)
    b2 = np.full(W, np.nan)
    warm = 20
    min_names = min(20, X.shape[1] - 1)
    for s_ in range(W):
        if s_ >= warm:
            with np.errstate(invalid="ignore", divide="ignore"):
                z2 = x2[s_] / var
            ok = obs[s_] & np.isfinite(z2) & (var > 0)
            if ok.sum() > min_names:
                b2[s_] = float(np.mean(np.minimum(z2[ok], spec.clip_sigma**2)))
        prev = np.where(np.isfinite(var), var, x2[s_])
        var = np.where(obs[s_], lam * prev + (1 - lam) * x2[s_], prev)
    ok = np.isfinite(b2)
    if ok.sum() < 10:
        return 1.0
    w = _ewm_weights(int(ok.sum()), spec.vra_halflife or 21.0)
    return float(np.clip(w @ b2[ok], 0.25, 4.0))


def simple_returns(close: FloatArray) -> FloatArray:
    out = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = close[1:] / close[:-1] - 1.0
    return out
