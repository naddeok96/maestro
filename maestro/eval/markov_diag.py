"""Markovity diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


@dataclass
class Transition:
    state: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    next_state: Dict[str, np.ndarray]


def _flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state["g_data"], state["g_model"], state["g_progress"]], axis=0)


def _flatten_action(action: Dict[str, np.ndarray]) -> np.ndarray:
    if not action:
        return np.zeros(0, dtype=np.float32)
    w = np.asarray(action.get("w", []), dtype=np.float32).ravel()
    eta = np.asarray(action.get("eta", []), dtype=np.float32).ravel()
    u = np.asarray(action.get("u", []), dtype=np.float32).ravel()
    return np.concatenate([w, eta, u], axis=0)


def compute_markov_diagnostics(transitions: List[Transition]) -> Dict[str, float]:
    """Return linear R², ΔR² when adding (S_{t-1}, a_{t-1}), and MLP R²."""
    if len(transitions) < 3:
        return {"r2": 0.0, "linear_r2": 0.0, "delta_r2": 0.0, "mlp_r2": 0.0}
    S_t = np.stack([_flatten_state(t.state) for t in transitions[1:-1]])
    S_tp1 = np.stack([_flatten_state(t.next_state) for t in transitions[1:-1]])
    S_tm1 = np.stack([_flatten_state(t_prev.state) for t_prev in transitions[0:-2]])
    A_tm1 = np.stack([_flatten_action(t_prev.action) for t_prev in transitions[0:-2]])
    lin = LinearRegression().fit(S_t, S_tp1)
    pred_lin = lin.predict(S_t)
    resid = ((S_tp1 - pred_lin) ** 2).sum()
    tot = ((S_tp1 - S_tp1.mean(axis=0)) ** 2).sum()
    linear_r2 = float(np.clip(1.0 - resid / max(tot, 1e-8), -1.0, 1.0))
    X_big = np.concatenate([S_t, S_tm1, A_tm1], axis=1)
    lin_big = LinearRegression().fit(X_big, S_tp1)
    pred_big = lin_big.predict(X_big)
    resid_big = ((S_tp1 - pred_big) ** 2).sum()
    linear_r2_big = float(np.clip(1.0 - resid_big / max(tot, 1e-8), -1.0, 1.0))
    delta_r2 = max(0.0, linear_r2_big - linear_r2)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", max_iter=500, random_state=0)
    mlp.fit(S_t, S_tp1)
    pred_mlp = mlp.predict(S_t)
    resid_mlp = ((S_tp1 - pred_mlp) ** 2).sum()
    mlp_r2 = float(np.clip(1.0 - resid_mlp / max(tot, 1e-8), -1.0, 1.0))
    return {"r2": linear_r2, "linear_r2": linear_r2, "delta_r2": delta_r2, "mlp_r2": mlp_r2}
