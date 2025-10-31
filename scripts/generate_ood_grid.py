#!/usr/bin/env python
"""Generate an out-of-distribution grid over noise and imbalance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.ood_grid import evaluate_ood_grid
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils.config import load_config


def _build_env(
    config: dict,
    task_cfg_path: str,
    seed: int,
    *,
    overrides: dict | None = None,
) -> MaestroEnv:
    datasets = build_from_config(task_cfg_path, seed, overrides=overrides or {})
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


def _parse_list(arg: str) -> List[float]:
    return [float(item) for item in arg.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policy robustness over a grid of noise and imbalance values."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--noise", type=str, default="0.0,0.1,0.3")
    parser.add_argument("--imbalance", type=str, default="0.0,0.4,0.6")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ood_grid"),
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

    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )

    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(state.get("policy", state))
    elif args.train_episodes > 0:
        ppo = PPOTeacher(policy, PPOConfig(**config.get("ppo", {})))
        for episode in range(args.train_episodes):
            env = _build_env(config, task_cfg, seed + episode * 37)
            try:
                ppo.train_episode(env, config["horizon"])
            finally:
                env.close()

    noise_vals = _parse_list(args.noise)
    imbalance_vals = _parse_list(args.imbalance)

    rows = []
    for i, noise in enumerate(noise_vals):
        for j, imbalance in enumerate(imbalance_vals):
            env = _build_env(
                config,
                task_cfg,
                seed + 1000 + i * 100 + j,
                overrides={"noise": float(noise), "imbalance": float(imbalance)},
            )
            try:
                stats = evaluate_ood_grid([env], policy, steps=args.steps)
            finally:
                env.close()
            rows.append(
                {
                    "noise": float(noise),
                    "imbalance": float(imbalance),
                    "mean_macro": stats["mean_macro"],
                    "std_macro": stats["std_macro"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["noise", "imbalance"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ood_grid.csv"
    df.to_csv(csv_path, index=False)

    pivot = df.pivot(index="noise", columns="imbalance", values="mean_macro")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(
        ticks=range(len(pivot.columns)),
        labels=[f"{col:.2f}" for col in pivot.columns],
    )
    plt.yticks(
        ticks=range(len(pivot.index)),
        labels=[f"{row:.2f}" for row in pivot.index],
    )
    plt.xlabel("imbalance")
    plt.ylabel("noise")
    plt.title("OOD Macro Accuracy")
    plt.colorbar()
    plt.tight_layout()
    heatmap_path = args.output_dir / "ood_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    print(json.dumps({"csv": str(csv_path), "heatmap": str(heatmap_path)}, indent=2))


if __name__ == "__main__":
    main()
