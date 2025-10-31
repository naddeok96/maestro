"""Markovity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import torch
from torch import nn


@dataclass
class Transition:
    state: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    next_state: Dict[str, np.ndarray]
    episode_id: Optional[int | str] = None


def _flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [state["g_data"], state["g_model"], state["g_progress"]], axis=0
    )


def _flatten_action(action: Dict[str, np.ndarray]) -> np.ndarray:
    if not action:
        return np.zeros(0, dtype=np.float32)
    w = np.asarray(action.get("w", []), dtype=np.float32).ravel()
    eta = np.asarray(action.get("eta", []), dtype=np.float32).ravel()
    u = np.asarray(action.get("u", []), dtype=np.float32).ravel()
    return np.concatenate([w, eta, u], axis=0)


def _group_transitions_by_episode(
    transitions: Sequence[Transition],
) -> List[List[Transition]]:
    """Group transitions by ``episode_id`` preserving order of appearance."""

    if not transitions:
        return []

    episodes: Dict[object, List[Transition]] = {}
    order: List[object] = []
    for idx, transition in enumerate(transitions):
        episode_key = transition.episode_id
        if episode_key is None:
            # Fallback to a unique pseudo-identifier to avoid accidental mixing.
            episode_key = -(idx + 1)
        if episode_key not in episodes:
            episodes[episode_key] = []
            order.append(episode_key)
        episodes[episode_key].append(transition)
    return [episodes[key] for key in order]


def split_transitions_by_episode(
    transitions: Sequence[Transition],
    ratio: float = 0.8,
    random_state: Optional[int] = 0,
) -> Tuple[List[Transition], List[Transition]]:
    """Split transitions into train/test sets grouped by episode.

    The split avoids leaking information across episodes by ensuring entire
    episodes end up exclusively in either train or test. When the input lacks
    ``episode_id`` metadata or contains too few episodes, the function falls
    back to a chronological split while keeping behaviour deterministic.
    """

    if not transitions:
        return [], []

    ratio = float(np.clip(ratio, 0.0, 1.0))
    grouped = _group_transitions_by_episode(transitions)
    if not grouped:
        return [], []

    if len(grouped) == 1:
        split_idx = max(1, int(round(len(transitions) * ratio)))
        split_idx = min(split_idx, len(transitions) - 1) if len(transitions) > 1 else 0
        return list(transitions[:split_idx]), list(transitions[split_idx:])

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(grouped))
    rng.shuffle(indices)
    split_point = int(np.floor(len(grouped) * ratio))
    split_point = min(max(split_point, 1), len(grouped) - 1)
    train_groups = [grouped[i] for i in indices[:split_point]]
    test_groups = [grouped[i] for i in indices[split_point:]]

    train = [transition for group in train_groups for transition in group]
    test = [transition for group in test_groups for transition in group]
    return train, test


def _prepare_supervised_arrays(
    episodes: Sequence[Sequence[Transition]],
) -> Dict[str, np.ndarray]:
    """Construct flattened arrays for supervised learning tasks."""

    current_states: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    hist_current_states: List[np.ndarray] = []
    hist_prev_states: List[np.ndarray] = []
    hist_prev_actions: List[np.ndarray] = []
    hist_next_states: List[np.ndarray] = []

    for episode in episodes:
        for idx, transition in enumerate(episode):
            current = _flatten_state(transition.state)
            nxt = _flatten_state(transition.next_state)
            current_states.append(current)
            next_states.append(nxt)

            if idx == 0:
                continue

            prev_transition = episode[idx - 1]
            hist_current_states.append(current)
            hist_prev_states.append(_flatten_state(prev_transition.state))
            hist_prev_actions.append(_flatten_action(prev_transition.action))
            hist_next_states.append(nxt)

    def _to_array(values: List[np.ndarray]) -> np.ndarray:
        if not values:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    result = {
        "current_states": _to_array(current_states),
        "next_states": _to_array(next_states),
        "hist_current_states": _to_array(hist_current_states),
        "hist_prev_states": _to_array(hist_prev_states),
        "hist_prev_actions": _to_array(hist_prev_actions),
        "hist_next_states": _to_array(hist_next_states),
    }
    return result


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_true.shape[0] < 2:
        return 0.0
    try:
        return float(r2_score(y_true, y_pred))
    except ValueError:
        return 0.0


def _infer_feature_slices(state: Dict[str, np.ndarray]) -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    offset = 0
    for key in ("g_data", "g_model", "g_progress"):
        values = np.asarray(state.get(key, []), dtype=np.float32).ravel()
        length = int(values.size)
        slices[key] = slice(offset, offset + length)
        offset += length
    return slices


def _compute_per_feature_r2_from_arrays(
    train_states: np.ndarray,
    train_targets: np.ndarray,
    test_states: np.ndarray,
    test_targets: np.ndarray,
    feature_slices: Dict[str, slice],
) -> Dict[str, float]:
    """Compute R² scores per feature block using separate linear models."""

    per_feature: Dict[str, float] = {}
    for name, slc in feature_slices.items():
        if slc.stop - slc.start <= 0:
            per_feature[f"{name}_r2"] = 0.0
            continue
        y_train = train_targets[:, slc] if train_targets.size else np.zeros((0, 0))
        y_test = test_targets[:, slc] if test_targets.size else np.zeros((0, 0))
        if y_train.shape[0] == 0 or y_test.shape[0] < 2:
            per_feature[f"{name}_r2"] = 0.0
            continue
        model = LinearRegression().fit(train_states, y_train)
        preds = model.predict(test_states)
        per_feature[f"{name}_r2"] = _safe_r2_score(y_test, preds)
    return per_feature


def compute_per_feature_r2(
    transitions: Sequence[Transition],
    ratio: float = 0.8,
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """Public helper to compute per-feature R² on a held-out split."""

    if len(transitions) < 2:
        return {"g_data_r2": 0.0, "g_model_r2": 0.0, "g_progress_r2": 0.0}

    train_transitions, test_transitions = split_transitions_by_episode(
        transitions, ratio=ratio, random_state=random_state
    )
    if not train_transitions or not test_transitions:
        return {"g_data_r2": 0.0, "g_model_r2": 0.0, "g_progress_r2": 0.0}

    feature_slices = _infer_feature_slices(train_transitions[0].state)
    train_episodes = _group_transitions_by_episode(train_transitions)
    test_episodes = _group_transitions_by_episode(test_transitions)

    train_arrays = _prepare_supervised_arrays(train_episodes)
    test_arrays = _prepare_supervised_arrays(test_episodes)

    S_train = train_arrays["current_states"]
    Y_train = train_arrays["next_states"]
    S_test = test_arrays["current_states"]
    Y_test = test_arrays["next_states"]

    if S_train.size == 0 or S_test.size == 0:
        return {"g_data_r2": 0.0, "g_model_r2": 0.0, "g_progress_r2": 0.0}

    return _compute_per_feature_r2_from_arrays(
        S_train, Y_train, S_test, Y_test, feature_slices
    )


class _GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        return self.readout(outputs)


def _compute_gru_history_r2(
    train_episodes: Sequence[Sequence[Transition]],
    test_episodes: Sequence[Sequence[Transition]],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
) -> float:
    if not train_episodes or not test_episodes:
        return 0.0

    train_sequences = []
    test_sequences = []

    for episode in train_episodes:
        inputs = [_flatten_state(t.state) for t in episode]
        targets = [_flatten_state(t.next_state) for t in episode]
        if not inputs:
            continue
        train_sequences.append(
            (
                torch.tensor(np.asarray(inputs, dtype=np.float32)),
                torch.tensor(np.asarray(targets, dtype=np.float32)),
            )
        )

    for episode in test_episodes:
        inputs = [_flatten_state(t.state) for t in episode]
        targets = [_flatten_state(t.next_state) for t in episode]
        if not inputs:
            continue
        test_sequences.append(
            (
                torch.tensor(np.asarray(inputs, dtype=np.float32)),
                torch.tensor(np.asarray(targets, dtype=np.float32)),
            )
        )

    if not train_sequences or not test_sequences:
        return 0.0

    input_dim = train_sequences[0][0].shape[-1]
    output_dim = train_sequences[0][1].shape[-1]

    device = torch.device("cpu")
    model = _GRURegressor(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    torch.manual_seed(0)
    for epoch in range(max(1, epochs)):
        total_loss = 0.0
        for inputs, targets in train_sequences:
            inputs = inputs.to(device).unsqueeze(0)
            targets = targets.to(device).unsqueeze(0)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        if total_loss == 0.0:
            break

    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in test_sequences:
            outputs = model(inputs.to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
            all_true.append(targets.numpy())
            all_pred.append(outputs)

    if not all_true or not all_pred:
        return 0.0

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return _safe_r2_score(y_true, y_pred)


def compute_markov_diagnostics(transitions: List[Transition]) -> Dict[str, float]:
    """Enhanced Markov diagnostics with episode-aware splits and per-feature R²."""

    if len(transitions) < 3:
        return {
            "r2": 0.0,
            "linear_r2": 0.0,
            "delta_r2": 0.0,
            "mlp_r2": 0.0,
            "gru_history_r2": 0.0,
            "linear_history_r2": 0.0,
        }

    train_transitions, test_transitions = split_transitions_by_episode(transitions)

    if not train_transitions or not test_transitions:
        return {
            "r2": 0.0,
            "linear_r2": 0.0,
            "delta_r2": 0.0,
            "mlp_r2": 0.0,
            "gru_history_r2": 0.0,
            "linear_history_r2": 0.0,
        }

    train_episodes = _group_transitions_by_episode(train_transitions)
    test_episodes = _group_transitions_by_episode(test_transitions)

    train_arrays = _prepare_supervised_arrays(train_episodes)
    test_arrays = _prepare_supervised_arrays(test_episodes)

    S_train = train_arrays["current_states"]
    Y_train = train_arrays["next_states"]
    S_test = test_arrays["current_states"]
    Y_test = test_arrays["next_states"]

    if S_train.size == 0 or S_test.size == 0:
        return {
            "r2": 0.0,
            "linear_r2": 0.0,
            "delta_r2": 0.0,
            "mlp_r2": 0.0,
            "gru_history_r2": 0.0,
            "linear_history_r2": 0.0,
        }

    feature_slices = _infer_feature_slices(train_transitions[0].state)

    lin = LinearRegression().fit(S_train, Y_train)
    pred_lin = lin.predict(S_test)
    linear_r2 = _safe_r2_score(Y_test, pred_lin)

    hist_current_train = train_arrays["hist_current_states"]
    hist_prev_train = train_arrays["hist_prev_states"]
    hist_action_train = train_arrays["hist_prev_actions"]
    hist_next_train = train_arrays["hist_next_states"]

    hist_current_test = test_arrays["hist_current_states"]
    hist_prev_test = test_arrays["hist_prev_states"]
    hist_action_test = test_arrays["hist_prev_actions"]
    hist_next_test = test_arrays["hist_next_states"]

    linear_history_r2 = 0.0
    if hist_current_train.size > 0 and hist_current_test.size > 0:
        base_lin_hist = LinearRegression().fit(hist_current_train, hist_next_train)
        base_pred_hist = base_lin_hist.predict(hist_current_test)
        linear_history_r2 = _safe_r2_score(hist_next_test, base_pred_hist)

        X_train_hist = np.concatenate(
            [hist_current_train, hist_prev_train, hist_action_train], axis=1
        )
        X_test_hist = np.concatenate(
            [hist_current_test, hist_prev_test, hist_action_test], axis=1
        )
        lin_hist = LinearRegression().fit(X_train_hist, hist_next_train)
        pred_hist = lin_hist.predict(X_test_hist)
        history_r2 = _safe_r2_score(hist_next_test, pred_hist)
        delta_r2 = max(0.0, history_r2 - linear_history_r2)
    else:
        history_r2 = 0.0
        delta_r2 = 0.0

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128), activation="relu", max_iter=500, random_state=0
    )
    mlp.fit(S_train, Y_train)
    pred_mlp = mlp.predict(S_test)
    mlp_r2 = _safe_r2_score(Y_test, pred_mlp)

    per_feature = _compute_per_feature_r2_from_arrays(
        S_train, Y_train, S_test, Y_test, feature_slices
    )

    gru_history_r2 = _compute_gru_history_r2(
        train_episodes, test_episodes, hidden_dim=128, epochs=100, learning_rate=1e-3
    )

    return {
        "r2": linear_r2,
        "linear_r2": linear_r2,
        "delta_r2": delta_r2,
        "mlp_r2": mlp_r2,
        "gru_history_r2": gru_history_r2,
        "linear_history_r2": linear_history_r2,
        **per_feature,
    }
