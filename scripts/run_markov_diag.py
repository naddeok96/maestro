"""Run Markov diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.markov_diag import Transition, compute_markov_diagnostics
from maestro.policy.ppo import TeacherPolicy

from .run_meta_train import load_config


def build_env(config: Dict[str, Any]) -> MaestroEnv:
    datasets = []
    seed = config.get("seed", 0)
    for task_cfg in config["tasks"]:
        datasets.extend(build_from_config(task_cfg, seed))
        seed += 9
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
        seed=config.get("seed", 0),
    )
    return MaestroEnv(env_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config)
    context_dim = 8 + 6 + 11
    policy = TeacherPolicy(descriptor_dim=8, context_dim=context_dim, eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]))

    transitions: List[Transition] = []
    obs, _ = env.reset()
    descriptors = env.last_per_dataset_descriptors
    for _ in range(config["horizon"]):
        action, _, _ = policy.act(obs, descriptors)
        next_obs, reward, done, _, _ = env.step(action)
        transitions.append(Transition(state=obs, action=action, next_state=next_obs))
        obs = next_obs
        descriptors = env.last_per_dataset_descriptors
        if done:
            break
    diagnostics = compute_markov_diagnostics(transitions)
    print(diagnostics)


if __name__ == "__main__":
    main()
