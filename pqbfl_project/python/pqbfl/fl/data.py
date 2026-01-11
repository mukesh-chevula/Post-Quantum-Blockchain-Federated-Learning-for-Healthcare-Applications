from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientDataset:
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class FederatedDataset:
    clients: list[ClientDataset]
    x_test: np.ndarray
    y_test: np.ndarray


def make_synthetic_federated_binary(
    *,
    n_clients: int,
    n_train_per_client: int,
    n_test: int,
    d: int = 10,
    seed: int = 0,
    non_iid: bool = True,
) -> FederatedDataset:
    rng = np.random.default_rng(seed)

    # Global separator
    true_w = rng.normal(size=(d,))
    true_b = rng.normal() * 0.1

    def sample(n: int, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d)) + shift
        logits = x @ true_w + true_b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        y = (rng.random(n) < p).astype(np.float64)
        return x.astype(np.float64), y

    clients: list[ClientDataset] = []
    for i in range(n_clients):
        shift = np.zeros((d,), dtype=np.float64)
        if non_iid:
            # Give each client a distinct mean shift
            shift[i % d] = (i - (n_clients / 2)) / n_clients * 2.0
        x, y = sample(n_train_per_client, shift)
        clients.append(ClientDataset(x=x, y=y))

    x_test, y_test = sample(n_test, np.zeros((d,), dtype=np.float64))
    return FederatedDataset(clients=clients, x_test=x_test, y_test=y_test)
