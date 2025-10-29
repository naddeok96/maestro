"""Evaluation script for Maestro policies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from maestro.baselines.random_policy import RandomPolicy
from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.diagnostics import compute_markov_diagnostic, summarize_rollout
from maestro.utils.logging import JSONLogger


def run_rollout(env: MaestroEnv, policy, steps: int) -> list[dict]:
    obs, info = env.reset()
    history: list[dict] = []
    for _ in range(steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        history.append({"reward": reward, "info": info})
        if terminated:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies in Maestro")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("runs/eval/metrics.jsonl"))
    args = parser.parse_args()

    curriculum = SyntheticCurriculum()
    env = MaestroEnv(MaestroEnvConfig(curriculum=curriculum))
    policy = RandomPolicy(env.action_space)
    logger = JSONLogger(args.out)

    for ep in range(args.episodes):
        history = run_rollout(env, policy, args.steps)
        metrics = summarize_rollout(history)
        markov = compute_markov_diagnostic(metrics.rewards)
        logger.log(
            {
                "episode": ep,
                "cumulative_reward": metrics.cumulative_reward,
                "average_cost": metrics.average_cost,
                "average_accuracy": metrics.average_accuracy,
                "markov_auto_corr": markov,
            }
        )
        print(
            f"Episode {ep}: reward={metrics.cumulative_reward:.2f} accuracy={metrics.average_accuracy:.2f}"
        )

    logger.flush()


if __name__ == "__main__":
    main()
