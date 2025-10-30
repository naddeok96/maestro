from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import TeacherPolicy


def make_env() -> MaestroEnv:
    datasets = build_from_config("configs/tasks/classification.yaml", seed=0)
    config = MaestroEnvConfig(
        datasets=datasets,
        horizon=5,
        batch_size=16,
        initial_budget=128,
        probe_size=32,
        grad_project_dim=64,
        grad_ema_beta=0.9,
        grad_norm_alpha=0.9,
        eta_min=1e-5,
        eta_max=1e-2,
        weight_decay=1e-4,
        momentum=0.9,
        seed=0,
    )
    return MaestroEnv(config)


def test_env_step_shapes():
    env = make_env()
    obs, _ = env.reset()
    assert obs["g_data"].shape == (8,)
    action = {
        "w": np.ones(len(env.datasets)) / len(env.datasets),
        "eta": np.array([1e-3]),
        "u": np.array([0.5]),
    }
    next_obs, reward, done, _, info = env.step(action)
    assert next_obs["g_model"].shape == (6,)
    assert info["usage"] > 0
    assert env.budget.remaining < env.config.initial_budget
