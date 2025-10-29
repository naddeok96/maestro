from __future__ import annotations

import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOAgent


def test_debug_rollout_runs():
    curriculum = SyntheticCurriculum(seed=10)
    env = MaestroEnv(MaestroEnvConfig(curriculum=curriculum, max_steps=10, budget=3.0))
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    obs, _ = env.reset()
    rewards = []
    for _ in range(5):
        action, logp, value = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated:
            break
    assert len(rewards) > 0
    assert np.isfinite(rewards).all()
