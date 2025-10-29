from __future__ import annotations

import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig


def test_env_step_and_budget():
    curriculum = SyntheticCurriculum(seed=123)
    env = MaestroEnv(MaestroEnvConfig(curriculum=curriculum, max_steps=5, budget=1.0))
    obs, info = env.reset()
    assert obs.shape == (env.action_space.n * 3,)
    total_cost = 0.0
    terminated = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        total_cost += info["cost"]
        if terminated:
            break
    assert terminated
    assert env.budget.remaining <= 1.0
    assert total_cost >= 0.0


def test_render_returns_string():
    env = MaestroEnv(MaestroEnvConfig(curriculum=SyntheticCurriculum(seed=0)))
    env.reset()
    output = env.render()
    assert isinstance(output, str)
    assert "acc" in output
