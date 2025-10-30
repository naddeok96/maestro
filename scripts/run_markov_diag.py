"""Run Markov diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.markov_diag import Transition, compute_markov_diagnostics
from maestro.policy.ppo import TeacherPolicy
from maestro.utils.config import load_config


def build_env(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
    datasets = build_from_config(task_cfg, seed)
    env_config = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
    )
    return MaestroEnv(env_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )

    diagnostics_per_task: Dict[str, Dict[str, float]] = {}
    task_list: Iterable[str] = config.get("eval_tasks", config["tasks"])
    base_seed = config.get("seed", 0)
    for index, task_cfg in enumerate(task_list):
        env_seed = base_seed + index * 19
        env = build_env(config, task_cfg, env_seed)
        transitions: List[Transition] = []
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        for _ in range(config["horizon"]):
            action, _, _, _ = policy.act(obs, descriptors)
            next_obs, reward, done, _, _ = env.step(action)
            transitions.append(
                Transition(state=obs, action=action, next_state=next_obs)
            )
            obs = next_obs
            descriptors = env.last_per_dataset_descriptors
            if done:
                break
        diagnostics_per_task[Path(task_cfg).stem] = compute_markov_diagnostics(
            transitions
        )
        env.close()
    print(diagnostics_per_task)


if __name__ == "__main__":
    main()
