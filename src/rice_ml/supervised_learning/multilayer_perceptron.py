from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["MultilayerPerceptron"]


def _as2d_float(X, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as1d_int(y, *, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr.astype(int, copy=False)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class MultilayerPerceptron:
    """
    Simple feedforward MLP for binary classification.

    This is a small educational implementation intended to satisfy the
    unit-test API: fit / predict / score with binary labels {0,1}.
    """

    hidden_layers: List[int]
    learning_rate: float = 0.1
    max_iter: int = 1000
    random_state: Optional[int] = None

    # learned params
    weights_: Optional[List[np.ndarray]] = None
    biases_: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        if not isinstance(self.hidden_layers, list) or len(self.hidden_layers) == 0:
            raise ValueError("hidden_layers must be a non-empty list of positive ints.")
        if any(int(h) <= 0 for h in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive ints.")
        self.hidden_layers = [int(h) for h in self.hidden_layers]
        self.learning_rate = float(self.learning_rate)
        self.max_iter = int(self.max_iter)

    def _init_params(self, n_features: int):
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features] + self.hidden_layers + [1]
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier-like scaling for tanh
            scale = np.sqrt(1.0 / fan_in)
            W = rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=float)
            weights.append(W)
            biases.append(b)
        self.weights_ = weights
        self.biases_ = biases

    def fit(self, X, y):
        X = _as2d_float(X, name="X")
        y = _as1d_int(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])) and not np.array_equal(classes, np.array([0])) and not np.array_equal(classes, np.array([1])):
            raise ValueError("MultilayerPerceptron supports only binary labels {0,1}.")

        n_samples, n_features = X.shape
        self._init_params(n_features)

        Y = y.reshape(-1, 1).astype(float)
        lr = float(self.learning_rate)

        for _ in range(self.max_iter):
            # forward pass
            activations: List[np.ndarray] = [X]
            pre_acts: List[np.ndarray] = []

            A = X
            for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
                Z = A @ W + b
                pre_acts.append(Z)
                A = np.tanh(Z)
                activations.append(A)

            # output layer
            W_out, b_out = self.weights_[-1], self.biases_[-1]
            Z_out = A @ W_out + b_out
            pre_acts.append(Z_out)
            Y_hat = _sigmoid(Z_out)

            # backward pass (binary cross-entropy with sigmoid)
            dZ = (Y_hat - Y) / n_samples  # (n,1)

            grads_W: List[np.ndarray] = [None] * len(self.weights_)
            grads_b: List[np.ndarray] = [None] * len(self.biases_)

            # output gradients
            grads_W[-1] = activations[-1].T @ dZ
            grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ W_out.T

            # hidden layers backprop
            for layer in range(len(self.hidden_layers) - 1, -1, -1):
                Z = pre_acts[layer]
                dZ_h = dA * (1.0 - np.tanh(Z) ** 2)
                grads_W[layer] = activations[layer].T @ dZ_h
                grads_b[layer] = np.sum(dZ_h, axis=0, keepdims=True)
                dA = dZ_h @ self.weights_[layer].T

            # gradient step
            for i in range(len(self.weights_)):
                self.weights_[i] -= lr * grads_W[i]
                self.biases_[i] -= lr * grads_b[i]

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights_ is None or self.biases_ is None:
            raise TypeError("Model is not fitted yet.")
        X = _as2d_float(X, name="X")
        A = X
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            A = np.tanh(A @ W + b)
        Z_out = A @ self.weights_[-1] + self.biases_[-1]
        return _sigmoid(Z_out).reshape(-1)

    def predict(self, X) -> np.ndarray:
        # tests expect TypeError before fit
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y) -> float:
        y = _as1d_int(y, name="y")
        preds = self.predict(X)
        return float(np.mean(preds == y))

