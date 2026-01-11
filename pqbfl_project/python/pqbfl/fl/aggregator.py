from __future__ import annotations

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
