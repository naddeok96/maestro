from __future__ import annotations

import numpy as np

from maestro.baselines.stateful_schedulers import create_scheduler


def _dummy_metrics(values: list[float]) -> dict:
    return {
        f"d{i}": {"accuracy": val}
        for i, val in enumerate(values)
    }


def test_uniform_scheduler_returns_uniform_action() -> None:
    sched = create_scheduler("uniform", ["d0", "d1"], (1e-4, 1e-2), 10)
    sched.start_episode({}, np.zeros((2, 8), dtype=np.float32))
    action, _, _, info = sched.act({}, np.zeros((2, 8), dtype=np.float32))
    assert np.allclose(action["w"], np.array([0.5, 0.5], dtype=np.float32))
    assert info["chosen_dataset"] == -1.0


def test_easy_to_hard_advances_datasets() -> None:
    sched = create_scheduler("easy_to_hard", ["a", "b", "c"], (1e-4, 1e-2), 3)
    sched.start_episode({}, np.zeros((3, 8), dtype=np.float32))
    action, _, _, info = sched.act({}, np.zeros((3, 8), dtype=np.float32))
    assert info["chosen_dataset"] == 0
    sched.step_index = 2
    action, _, _, info = sched.act({}, np.zeros((3, 8), dtype=np.float32))
    assert info["chosen_dataset"] == 2


def test_greedy_scheduler_targets_lowest_accuracy() -> None:
    sched = create_scheduler("greedy", ["d0", "d1", "d2"], (1e-4, 1e-2), 5)
    sched.start_episode({}, np.zeros((3, 8), dtype=np.float32), _dummy_metrics([0.2, 0.4, 0.1]))
    action, _, _, info = sched.act({}, np.zeros((3, 8), dtype=np.float32))
    assert info["chosen_dataset"] == 2


def test_linucb_updates_internal_means() -> None:
    sched = create_scheduler("bandit_linucb", ["d0", "d1"], (1e-4, 1e-2), 4)
    sched.start_episode({}, np.zeros((2, 8), dtype=np.float32))
    action, _, _, info = sched.act({}, np.zeros((2, 8), dtype=np.float32))
    idx = int(info["chosen_dataset"])
    sched.update(0.0, {"dataset_metrics": _dummy_metrics([0.3, 0.7])})
    assert getattr(sched, "counts")[idx] == 1
    assert np.isclose(getattr(sched, "means")[idx], [0.3, 0.7][idx])


def test_thompson_sampling_updates_posteriors() -> None:
    sched = create_scheduler("bandit_thompson", ["d0", "d1"], (1e-4, 1e-2), 4)
    sched.start_episode({}, np.zeros((2, 8), dtype=np.float32))
    _, _, _, info = sched.act({}, np.zeros((2, 8), dtype=np.float32))
    idx = int(info["chosen_dataset"])
    alpha_before = getattr(sched, "alpha")[idx]
    beta_before = getattr(sched, "beta")[idx]
    sched.update(0.0, {"dataset_metrics": _dummy_metrics([0.2, 0.8])})
    assert getattr(sched, "alpha")[idx] != alpha_before or getattr(sched, "beta")[idx] != beta_before


def test_create_scheduler_invalid_method_raises() -> None:
    try:
        create_scheduler("unknown", ["d0"], (1e-4, 1e-2), 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown method")
