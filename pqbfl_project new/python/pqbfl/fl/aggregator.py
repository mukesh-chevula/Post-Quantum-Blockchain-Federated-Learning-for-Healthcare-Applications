from __future__ import annotations

import numpy as np

from pqbfl.fl.model import LogisticModel


def fedavg(models: list[tuple[LogisticModel, int]]) -> LogisticModel:
    if not models:
        raise ValueError("no models")

    total = sum(n for _, n in models)
    if total <= 0:
        raise ValueError("total weight must be > 0")

    d = models[0][0].w.shape[0]
    w = 0.0
    b = 0.0

    w_sum = None
    for m, n in models:
        if m.w.shape[0] != d:
            raise ValueError("dimension mismatch")
        coef = n / total
        if w_sum is None:
            w_sum = coef * m.w
        else:
            w_sum = w_sum + coef * m.w
        b += coef * m.b

    return LogisticModel(w=w_sum, b=b)


def coord_median(models: list[tuple[LogisticModel, int]]) -> LogisticModel:
    """Coordinate-wise median aggregator (unweighted).

    Useful as a simple robust alternative to FedAvg.
    """

    if not models:
        raise ValueError("no models")

    w_stack = np.stack([m.w for m, _ in models], axis=0)
    b_stack = np.array([m.b for m, _ in models], dtype=np.float64)
    return LogisticModel(w=np.median(w_stack, axis=0), b=float(np.median(b_stack)))


def trimmed_mean(models: list[tuple[LogisticModel, int]], *, trim_ratio: float = 0.1) -> LogisticModel:
    """Coordinate-wise trimmed mean aggregator (unweighted).

    trim_ratio=0.1 trims 10% lowest and 10% highest values per coordinate.
    """

    if not models:
        raise ValueError("no models")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5)")

    w_stack = np.stack([m.w for m, _ in models], axis=0)
    b_stack = np.array([m.b for m, _ in models], dtype=np.float64)

    n = w_stack.shape[0]
    k = int(np.floor(trim_ratio * n))
    if k == 0:
        return LogisticModel(w=np.mean(w_stack, axis=0), b=float(np.mean(b_stack)))
    if 2 * k >= n:
        raise ValueError("trim_ratio too large for number of models")

    w_sorted = np.sort(w_stack, axis=0)
    b_sorted = np.sort(b_stack)
    w_trim = w_sorted[k : n - k, :]
    b_trim = b_sorted[k : n - k]
    return LogisticModel(w=np.mean(w_trim, axis=0), b=float(np.mean(b_trim)))
