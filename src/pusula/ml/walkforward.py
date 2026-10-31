"""Machine-learned alpha combination, trained walk-forward with purging.

The model learns to map today's standardised signals to the cross-sectional rank of the
forward return. It is refit every ``refit_every`` sessions on an expanding window that ends
``horizon + lag + embargo`` sessions before the refit date, so every training label was fully
realised (and tradeable) before the model is used. Predictions for the following block are
made with that frozen model.

The output is just another score panel; it goes through the same optimiser, costs and gate as
the hand-built composite, and it must beat the composite to justify its complexity. In
NihaiAlgoritma a 282 M-parameter network did not; here the default models are deliberately
small (ridge, shallow boosted trees).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pusula.alpha import ops
from pusula.alpha.combine import SignalSpec, feature_stack
from pusula.alpha.evaluate import forward_returns
from pusula.alpha.signals import universe_mask
from pusula.data.panel import FloatArray, Panel


class Regressor(Protocol):
    def fit(self, X: FloatArray, y: FloatArray) -> object: ...

    def predict(self, X: FloatArray) -> FloatArray: ...


@dataclass
class Ridge:
    alpha: float = 10.0
    coef_: FloatArray | None = None
    intercept_: float = 0.0

    def fit(self, X: FloatArray, y: FloatArray) -> Ridge:
        mu = X.mean(axis=0)
        Xc = X - mu
        self.coef_ = np.linalg.solve(
            Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - y.mean())
        )
        self.intercept_ = float(y.mean() - mu @ self.coef_)
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        assert self.coef_ is not None
        return np.asarray(X @ self.coef_ + self.intercept_, dtype=np.float64)


def make_model(kind: str, seed: int = 0) -> Regressor:
    if kind == "ridge":
        return Ridge(alpha=100.0)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model: Regressor = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=400,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        return model
    raise KeyError(f"unknown model kind {kind!r}")


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    features: tuple[SignalSpec, ...]
    model: str = "ridge"  # ridge | gbm | ensemble
    horizon: int = 5
    lag: int = 1
    embargo: int = 5
    refit_every: int = 63
    min_train: int = 504
    sample_every: int = 5  # use every k-th date in training (labels overlap anyway)
    beta_neutral: bool = True
    seed: int = 0


@dataclass
class WalkForwardResult:
    scores: FloatArray
    refit_rows: list[int] = field(default_factory=list)
    train_rows: list[int] = field(default_factory=list)
    coefs: list[FloatArray] = field(default_factory=list)


def _xy(F: FloatArray, y: FloatArray, rows: npt.NDArray[np.intp]) -> tuple[FloatArray, FloatArray]:
    X = F[rows].reshape(-1, F.shape[2])
    yy = y[rows].reshape(-1)
    ok = np.isfinite(yy) & np.all(np.isfinite(X), axis=1)
    return X[ok], yy[ok]


def walk_forward_scores(spec: WalkForwardSpec, panel: Panel, origin: int = 0) -> WalkForwardResult:
    """ML score panel; rows before the first refit are NaN.

    ``origin`` anchors the refit schedule (refits at ``origin + min_train + j * refit_every``)
    so that running on a truncated panel reproduces the same models: required for the
    look-ahead test and for live/backtest parity.
    """
    mask = universe_mask(panel)
    F = feature_stack(spec.features, panel, spec.beta_neutral)
    fwd = forward_returns(panel, spec.horizon, spec.lag)
    y = ops.cs_gauss(fwd, mask)
    T = panel.n_dates
    scores = np.full((T, panel.n_symbols), np.nan)
    out = WalkForwardResult(scores)
    purge = spec.horizon + spec.lag + spec.embargo
    first = origin + spec.min_train
    for tau in range(first, T, spec.refit_every):
        end = tau - purge  # exclusive: labels of rows < end are realised by tau
        rows = np.arange(origin, end, spec.sample_every)
        if rows.size < 20:
            continue
        X, yy = _xy(F, y, rows)
        if X.shape[0] < 1000:
            continue
        kinds = ["ridge", "gbm"] if spec.model == "ensemble" else [spec.model]
        block = slice(tau, min(tau + spec.refit_every, T))
        Fb = F[block]
        flat = Fb.reshape(-1, Fb.shape[2])
        ok = np.all(np.isfinite(flat), axis=1)
        pred = np.zeros(flat.shape[0])
        for kind in kinds:
            m = make_model(kind, spec.seed)
            m.fit(X, yy)
            p = np.full(flat.shape[0], np.nan)
            p[ok] = m.predict(flat[ok])
            pz = ops.cs_zscore(
                p.reshape(Fb.shape[0], Fb.shape[1]), mask[block] if mask is not None else None
            )
            pred = pred + np.where(np.isfinite(pz), pz, 0.0).reshape(-1) / len(kinds)
            if isinstance(m, Ridge) and m.coef_ is not None:
                out.coefs.append(m.coef_.copy())
        pred_m = pred.reshape(Fb.shape[0], Fb.shape[1])
        pred_m[~ok.reshape(Fb.shape[0], Fb.shape[1])] = np.nan
        if mask is not None:
            pred_m = np.where(mask[block] > 0, pred_m, np.nan)
        scores[block] = ops.cs_zscore(pred_m, mask[block] if mask is not None else None)
        out.refit_rows.append(tau)
        out.train_rows.append(int(X.shape[0]))
    return out


def ml_alpha_fn(
    spec: WalkForwardSpec, origin_date: str | None = None
) -> Callable[[Panel], FloatArray]:
    """Adapter for :class:`pusula.portfolio.strategies.OptimizedAlpha` (``alpha_fn``)."""

    def fn(panel: Panel) -> FloatArray:
        origin = (
            0
            if origin_date is None
            else int(np.searchsorted(panel.dates, np.datetime64(origin_date, "D")))
        )
        return walk_forward_scores(spec, panel, origin).scores

    return fn


def feature_names(specs: Sequence[SignalSpec]) -> list[str]:
    return [s.name for s in specs]
