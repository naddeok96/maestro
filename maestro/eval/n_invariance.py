"""Number-of-datasets invariance evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from maestro.envs.maestro_env import MaestroEnv
from maestro.policy.ppo import TeacherPolicy


def evaluate_permutations(env: MaestroEnv, policy: TeacherPolicy, permutations: List[List[int]]) -> Dict[str, float]:
    results = []
    for perm in permutations:
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        permuted = descriptors[perm]
        action, _, _, _ = policy.act(obs, permuted)
        obs, reward, done, _, info = env.step(action)
        results.append(info["macro_accuracy"])
    return {"mean_macro": float(np.mean(results)), "std_macro": float(np.std(results))}
