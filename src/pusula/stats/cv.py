"""Cross-validation for overlapping financial labels.

A label formed at ``t`` over ``(t, t + h]`` overlaps every label formed in ``(t - h, t + h)``.
Training on those neighbours leaks the test labels into the fit. Purging removes training
observations whose label window intersects the test window; the embargo additionally drops
observations right after the test window (serial correlation of features).
López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 and 12.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]


def purge_train(
    train: IntArray, test_lo: int, test_hi: int, horizon: int, embargo: int
) -> IntArray:
    """Drop training indices whose label window ``[i, i + horizon]`` overlaps
    ``[test_lo, test_hi]``, plus an embargo of ``embargo`` observations after the test block."""
    keep = (train + horizon < test_lo) | (train > test_hi + embargo)
    return train[keep]


def purged_kfold(
    n: int, k: int, horizon: int, embargo: int = 0
) -> Iterator[tuple[IntArray, IntArray]]:
    """Contiguous K-fold with purging and embargo (``horizon`` in observations)."""
    if k < 2 or n < k:
        raise ValueError("need 2 <= k <= n")
    bounds = np.linspace(0, n, k + 1).astype(int)
    all_idx = np.arange(n)
    for j in range(k):
        lo, hi = int(bounds[j]), int(bounds[j + 1]) - 1
        test = all_idx[lo : hi + 1]
        train = np.concatenate([all_idx[:lo], all_idx[hi + 1 :]])
        yield purge_train(train, lo, hi, horizon, embargo), test


def walk_forward(
    n: int,
    first_train: int,
    step: int,
    horizon: int,
    embargo: int = 0,
    expanding: bool = True,
    window: int | None = None,
) -> Iterator[tuple[IntArray, IntArray]]:
    """Expanding (or rolling) walk-forward splits; the test block always follows training.

    Training ends ``horizon + embargo`` observations before the test block starts so that no
    training label reaches into the test period.
    """
    start = first_train
    while start < n:
        stop = min(start + step, n)
        train_end = start - horizon - embargo  # exclusive
        lo = 0 if expanding or window is None else max(0, train_end - window)
        if train_end > lo:
            yield np.arange(lo, train_end), np.arange(start, stop)
        start = stop


def cpcv_splits(
    n: int, n_groups: int, n_test_groups: int, horizon: int, embargo: int = 0
) -> Iterator[tuple[IntArray, IntArray]]:
    """Combinatorial purged CV: every choice of ``n_test_groups`` out of ``n_groups`` blocks."""
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    all_idx = np.arange(n)
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n, dtype=bool)
        for g in combo:
            test_mask[bounds[g] : bounds[g + 1]] = True
        train = all_idx[~test_mask]
        for g in combo:
            train = purge_train(train, int(bounds[g]), int(bounds[g + 1]) - 1, horizon, embargo)
        yield train, all_idx[test_mask]
