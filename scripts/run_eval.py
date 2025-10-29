"""Run evaluation for a trained teacher."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import TeacherPolicy
from maestro.utils.serialization import load_checkpoint

from .run_meta_train import load_config


def build_env_from_tasks(config: Dict[str, Any], task_list) -> MaestroEnv:
    datasets = []
    seed = config.get("seed", 0)
    for task_cfg in task_list:
        datasets.extend(build_from_config(task_cfg, seed))
        seed += 11
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
    parser = argparse.ArgumentParser(description="Evaluate a trained teacher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env_from_tasks(config, config.get("eval_tasks", config["tasks"]))
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )

    obs, _ = env.reset()
    descriptors = env.last_per_dataset_descriptors
    action, _, _ = policy.act(obs, descriptors)
    obs, reward, done, _, info = env.step(action)
    print({"macro_accuracy": info.get("macro_accuracy", 0.0), "return": reward})


if __name__ == "__main__":
    main()
