from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticModel:
    w: np.ndarray  # shape (d,)
    b: float

    @staticmethod
    def init(d: int, seed: int = 0) -> "LogisticModel":
        rng = np.random.default_rng(seed)
        w = rng.normal(scale=0.01, size=(d,)).astype(np.float64)
        b = 0.0
        return LogisticModel(w=w, b=b)

    def copy(self) -> "LogisticModel":
        return LogisticModel(w=self.w.copy(), b=float(self.b))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.int64)

    def train_sgd(self, x: np.ndarray, y: np.ndarray, *, lr: float = 0.1, epochs: int = 1, batch_size: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                xb = x[batch]
                yb = y[batch]

                p = self.predict_proba(xb)
                grad_w = (xb.T @ (p - yb)) / xb.shape[0]
                grad_b = float(np.mean(p - yb))

                self.w -= lr * grad_w
                self.b -= lr * grad_b

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, w=self.w, b=np.array([self.b], dtype=np.float64))
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> "LogisticModel":
        buf = io.BytesIO(data)
        npz = np.load(buf)
        w = npz["w"].astype(np.float64)
        b = float(npz["b"][0])
        return LogisticModel(w=w, b=b)


def accuracy(model: LogisticModel, x: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(x)
    return float(np.mean(pred == y))
