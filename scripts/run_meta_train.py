"""Run MAESTRO meta-training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOTeacher, PPOConfig, TeacherPolicy
from maestro.utils import RunPaths
from maestro.utils.logging import MetricsLogger


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    if "defaults" in cfg:
        merged: Dict[str, Any] = {}
        for default in cfg.pop("defaults"):
            default_path = (path.parent / default).resolve()
            merged = _merge_dicts(merged, load_config(default_path))
        cfg = _merge_dicts(merged, cfg)
    return cfg


def build_env(config: Dict[str, Any]) -> MaestroEnv:
    datasets: List = []
    seed = config.get("seed", 0)
    for task_cfg in config["tasks"]:
        datasets.extend(build_from_config(task_cfg, seed))
        seed += 13
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
    parser = argparse.ArgumentParser(description="Run MAESTRO meta-training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config)
    context_dim = 8 + 6 + 11
    policy = TeacherPolicy(descriptor_dim=8, context_dim=context_dim, eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]))
    ppo = PPOTeacher(policy, PPOConfig(learning_rate=config["ppo"]["learning_rate"]))

    output_dir = RunPaths(Path(config["logging"]["output_dir"]).parent, Path(config["logging"]["output_dir"]).name).resolve()
    logger = MetricsLogger(output_dir)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "config": config}))
        return

    for episode in range(config["run"]["total_episodes"]):
        stats = ppo.train_episode(env, config["horizon"])
        stats["episode"] = episode
        logger.log_row(stats)
    logger.flush_json()


if __name__ == "__main__":
    main()
