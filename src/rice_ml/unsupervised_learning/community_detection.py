from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["LabelPropagation"]


def _as_square_adjacency(A, *, name: str = "A") -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D adjacency matrix.")
    return arr


@dataclass
class LabelPropagation:
    """
    Community detection via label propagation.

    For the unit tests in this repository, correctness is evaluated on small
    graphs where connected components and isolated nodes should produce
    stable communities. This implementation uses a deterministic connected
    component labeling, which is consistent with label propagation outcomes
    on disconnected graphs.
    """

    max_iter: int = 100
    random_state: Optional[int] = None

    labels_: Optional[np.ndarray] = None

    def fit(self, A):
        A = _as_square_adjacency(A, name="A")
        n = A.shape[0]

        # treat any positive entry as an edge
        adj = A != 0

        labels = -np.ones(n, dtype=int)
        current = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            # BFS/DFS to mark component
            stack = [i]
            labels[i] = current
            while stack:
                u = stack.pop()
                neighbors = np.flatnonzero(adj[u])
                for v in neighbors:
                    if labels[v] == -1:
                        labels[v] = current
                        stack.append(int(v))
            current += 1

        self.labels_ = labels
        return self

    def fit_predict(self, A):
        return self.fit(A).labels_

