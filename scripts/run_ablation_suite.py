#!/usr/bin/env python
"""Run a suite of observation ablations for the teacher policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.transfer import evaluate_transfer
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils.config import load_config


def _build_env(
    config: Dict[str, object],
    task_cfg: str,
    seed: int,
    ablations: Dict[str, bool] | None,
) -> MaestroEnv:
    datasets = build_from_config(task_cfg, seed)
    env_cfg = MaestroEnvConfig(
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
        ablations=ablations,
    )
    return MaestroEnv(env_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate observation ablations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ablations"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks: Iterable[str] = config.get("tasks", [])
    tasks = list(tasks)
    if not tasks:
        raise ValueError("No tasks found in config")
    task_cfg = tasks[min(args.task_index, len(tasks) - 1)]

    flag_space: List[Dict[str, bool]] = [
        {},
        {"drop_grad_cosine": True},
        {"drop_progress_block": True},
        {"drop_model_block": True},
        {"drop_data_block": True},
    ]

    results = []
    for flags in flag_space:
        policy = TeacherPolicy(
            descriptor_dim=8,
            g_model_dim=6,
            g_progress_dim=11,
            eta_bounds=(
                config["optimizer"]["eta_min"],
                config["optimizer"]["eta_max"],
            ),
        )

        if args.checkpoint and args.checkpoint.exists():
            state = torch.load(args.checkpoint, map_location="cpu")
            policy.load_state_dict(state.get("policy", state))
        elif args.train_episodes > 0:
            ppo = PPOTeacher(policy, PPOConfig(**config.get("ppo", {})))
            for episode in range(args.train_episodes):
                env_train = _build_env(
                    config,
                    task_cfg,
                    seed + episode * 13,
                    flags if flags else None,
                )
                try:
                    ppo.train_episode(env_train, config["horizon"])
                finally:
                    env_train.close()

        env_eval = _build_env(config, task_cfg, seed + 777, flags if flags else None)
        try:
            stats = evaluate_transfer(env_eval, policy, steps=args.steps)
        finally:
            env_eval.close()

        row = {"flags": json.dumps(flags, sort_keys=True)}
        row.update(stats)
        results.append(row)

    df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(json.dumps({"csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
