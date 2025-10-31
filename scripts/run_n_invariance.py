#!/usr/bin/env python
"""Evaluate N-invariance of a teacher policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.n_invariance import evaluate_permutations
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.wandb import init_wandb_run, log_metrics


def _build_env(
    config: dict,
    task_cfg_path: str,
    seed: int,
    num_datasets: int,
) -> MaestroEnv:
    datasets = build_from_config(task_cfg_path, seed, num_datasets=num_datasets)
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
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on N=train_num_datasets and evaluate permutation robustness at eval_num_datasets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-num-datasets", type=int, default=3)
    parser.add_argument("--eval-num-datasets", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=32)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/n_invariance"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks found in config")
    task_cfg = tasks[min(args.task_index, len(tasks) - 1)]

    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    ppo = PPOTeacher(policy, PPOConfig(**config.get("ppo", {})))

    run_name = f"n_invariance_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(
        run_name,
        config={
            "config": config,
            "task_cfg": task_cfg,
            "train_num_datasets": args.train_num_datasets,
            "eval_num_datasets": args.eval_num_datasets,
            "train_episodes": args.train_episodes,
            "permutations": args.permutations,
        },
    )

    try:
        if args.checkpoint and args.checkpoint.exists():
            state = torch.load(args.checkpoint, map_location="cpu")
            policy.load_state_dict(state.get("policy", state))
        else:
            for episode in range(max(0, args.train_episodes)):
                env = _build_env(
                    config,
                    task_cfg,
                    seed + episode * 31,
                    num_datasets=args.train_num_datasets,
                )
                try:
                    train_stats = ppo.train_episode(env, config["horizon"])
                finally:
                    env.close()
                train_stats = {f"train/{k}": v for k, v in train_stats.items()}
                train_stats["train/episode"] = episode
                log_metrics(train_stats)

        eval_env = _build_env(
            config,
            task_cfg,
            seed + 999,
            num_datasets=args.eval_num_datasets,
        )
        try:
            rng = np.random.default_rng(seed + 123)
            n = args.eval_num_datasets
            perms: List[List[int]] = [
                rng.permutation(n).tolist() for _ in range(args.permutations)
            ]
            stats = evaluate_permutations(eval_env, policy, perms)
        finally:
            eval_env.close()
        log_metrics({f"eval/{k}": v for k, v in stats.items()})
    finally:
        wandb_run.finish()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "train_num_datasets": args.train_num_datasets,
        "eval_num_datasets": args.eval_num_datasets,
        "permutations": args.permutations,
        "mean_macro": stats["mean_macro"],
        "sigma_macro": stats["std_macro"],
    }
    output_path = args.output_dir / "n_invariance.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
