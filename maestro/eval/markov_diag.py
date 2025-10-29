"""Markovity diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class Transition:
    state: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    next_state: Dict[str, np.ndarray]


def _flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state["g_data"], state["g_model"], state["g_progress"]], axis=0)


def compute_markov_diagnostics(transitions: List[Transition]) -> Dict[str, float]:
    if len(transitions) < 2:
        return {"r2": 0.0}
    X = np.stack([_flatten_state(t.state) for t in transitions[:-1]])
    y = np.stack([_flatten_state(t.next_state) for t in transitions[:-1]])
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    residual = ((y - preds) ** 2).sum()
    total = ((y - y.mean(axis=0)) ** 2).sum()
    r2 = 1.0 - residual / max(total, 1e-8)
    return {"r2": float(max(min(r2, 1.0), -1.0))}
