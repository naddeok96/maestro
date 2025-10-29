"""Transfer evaluation harness."""
from __future__ import annotations

from typing import Dict

from maestro.envs.maestro_env import MaestroEnv
from maestro.policy.ppo import TeacherPolicy


def evaluate_transfer(env: MaestroEnv, policy: TeacherPolicy, steps: int) -> Dict[str, float]:
    obs, _ = env.reset()
    descriptors = env.last_per_dataset_descriptors
    total_reward = 0.0
    for _ in range(steps):
        action, _, _ = policy.act(obs, descriptors)
        obs, reward, done, _, info = env.step(action)
        descriptors = env.last_per_dataset_descriptors
        total_reward += reward
        if done:
            break
    return {"macro_accuracy": info.get("macro_accuracy", 0.0), "return": total_reward}
