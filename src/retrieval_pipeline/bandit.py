from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


def l2_norm_safe(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


@dataclass
class Arm:
    chunk_id: str
    feature: np.ndarray
    retrieval_score: float


class LinUCB:
    """Shared linear UCB over dynamic arms with contextual arm features."""

    def __init__(self, dim: int, alpha: float = 1.0, l2_lambda: float = 1.0):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.l2_lambda = float(l2_lambda)

        self.A = np.eye(self.dim, dtype=np.float64) * self.l2_lambda
        self.b = np.zeros(self.dim, dtype=np.float64)

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def score(self, x: np.ndarray) -> float:
        x = x.astype(np.float64)
        theta = self.theta
        A_inv_x = np.linalg.solve(self.A, x)
        exploit = float(theta @ x)
        explore = self.alpha * float(np.sqrt(max(0.0, x @ A_inv_x)))
        return exploit + explore

    def choose_index(self, features: List[np.ndarray]) -> int:
        if not features:
            raise ValueError("No features to choose from.")
        scores = [self.score(x) for x in features]
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, reward: float) -> None:
        x = x.astype(np.float64)
        r = float(reward)
        self.A += np.outer(x, x)
        self.b += r * x

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "alpha": self.alpha,
            "l2_lambda": self.l2_lambda,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinUCB":
        model = cls(dim=int(payload["dim"]), alpha=float(payload["alpha"]), l2_lambda=float(payload["l2_lambda"]))
        model.A = np.asarray(payload["A"], dtype=np.float64)
        model.b = np.asarray(payload["b"], dtype=np.float64)
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> "LinUCB":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)


def retrieval_greedy_index(arms: List[Arm]) -> int:
    return int(np.argmax([a.retrieval_score for a in arms]))


def epsilon_greedy_choice(best_idx: int, n_arms: int, epsilon: float, rng: np.random.Generator) -> tuple[int, float]:
    if n_arms <= 0:
        raise ValueError("n_arms must be > 0")

    eps = float(max(0.0, min(1.0, epsilon)))
    explore = bool(rng.random() < eps)

    if explore:
        idx = int(rng.integers(0, n_arms))
    else:
        idx = int(best_idx)

    if idx == best_idx:
        propensity = (1.0 - eps) + eps / n_arms
    else:
        propensity = eps / n_arms
    return idx, float(propensity)


def ridge_fit(X: np.ndarray, y: np.ndarray, reg: float = 1.0) -> np.ndarray:
    d = X.shape[1]
    lhs = X.T @ X + np.eye(d) * float(reg)
    rhs = X.T @ y
    return np.linalg.solve(lhs, rhs)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w
