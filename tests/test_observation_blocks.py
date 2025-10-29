from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig


def make_env() -> MaestroEnv:
    datasets = build_from_config("configs/tasks/classification.yaml", seed=0)
    config = MaestroEnvConfig(
        datasets=datasets,
        horizon=4,
        batch_size=8,
        initial_budget=64,
        probe_size=16,
        grad_project_dim=32,
        grad_ema_beta=0.9,
        grad_norm_alpha=0.9,
        eta_min=1e-5,
        eta_max=1e-2,
        weight_decay=1e-4,
        momentum=0.9,
        seed=0,
    )
    return MaestroEnv(config)


def test_progress_updates_with_learning_rate():
    env = make_env()
    obs, _ = env.reset()
    action_low = {"w": np.ones(len(env.datasets)) / len(env.datasets), "eta": np.array([1e-5]), "u": np.array([0.2])}
    obs_low, _, _, _, _ = env.step(action_low)
    action_high = {"w": np.ones(len(env.datasets)) / len(env.datasets), "eta": np.array([1e-2]), "u": np.array([0.2])}
    obs_high, _, _, _, _ = env.step(action_high)
    assert obs_high["g_progress"][2] > obs_low["g_progress"][2]
