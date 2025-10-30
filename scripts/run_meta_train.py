"""Run MAESTRO meta-training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOTeacher, PPOConfig, TeacherPolicy
from maestro.utils import RunPaths
from maestro.utils.config import load_config
from maestro.utils.logging import MetricsLogger


def build_env_for_task(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
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
    parser = argparse.ArgumentParser(description="Run MAESTRO meta-training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    ppo = PPOTeacher(policy, PPOConfig(learning_rate=config["ppo"]["learning_rate"]))

    output_dir = RunPaths(Path(config["logging"]["output_dir"]).parent, Path(config["logging"]["output_dir"]).name).resolve()
    logger = MetricsLogger(output_dir)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "config": config}))
        return

    tasks = config["tasks"]
    base_seed = config.get("seed", 0)
    for episode in range(config["run"]["total_episodes"]):
        task_cfg = tasks[episode % len(tasks)]
        env_seed = base_seed + episode * 31
        env = build_env_for_task(config, task_cfg, env_seed)
        stats = ppo.train_episode(env, config["horizon"])
        stats["episode"] = episode
        stats["task"] = Path(task_cfg).stem
        logger.log_row(stats)
        env.close()
    logger.flush_json()


if __name__ == "__main__":
    main()
