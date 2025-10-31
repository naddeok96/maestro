"""Run comparative experiments across teacher baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from maestro.baselines import BaselineScheduler, create_scheduler
from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils import RunPaths
from maestro.utils.config import load_config
from maestro.utils.logging import MetricsLogger


METHOD_CHOICES = [
    "ppo",
    "uniform",
    "easy_to_hard",
    "greedy",
    "bandit_linucb",
    "bandit_thompson",
    "pbt",
    "bohb",
]


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
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_config)


def get_output_directory(method: str, args: argparse.Namespace) -> Path:
    base_dir = Path(args.output_dir) if args.output_dir else Path("outputs/comparative")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_paths = RunPaths(base_dir, f"{method}_{timestamp}")
    return run_paths.resolve()


def _baseline_episode(
    env: MaestroEnv,
    scheduler: BaselineScheduler,
    horizon: int,
) -> Dict[str, float]:
    observation, _ = env.reset()
    descriptors = env.last_per_dataset_descriptors
    scheduler.start_episode(
        observation,
        descriptors,
        dataset_metrics={name: {"accuracy": 0.0} for name in scheduler.dataset_names},
    )
    total_reward = 0.0
    macro_accuracy = 0.0
    usages: List[float] = []
    etas: List[float] = []
    usage_fraction: List[float] = []

    for _ in range(horizon):
        action, _, _, action_info = scheduler.act(observation, descriptors)
        next_obs, reward, terminated, truncated, info = env.step(action)
        combined_info = dict(info)
        combined_info.update(action_info)
        scheduler.update(reward, combined_info)
        observation = next_obs
        descriptors = env.last_per_dataset_descriptors
        total_reward += float(reward)
        macro_accuracy = float(info.get("macro_accuracy", macro_accuracy))
        usages.append(float(info.get("usage", 0.0)))
        etas.append(float(action["eta"][0]))
        usage_fraction.append(float(action["u"][0]))
        if terminated or truncated:
            break
    env.close()

    avg_usage = float(np.mean(usages)) if usages else 0.0
    avg_eta = float(np.mean(etas)) if etas else 0.0
    avg_u = float(np.mean(usage_fraction)) if usage_fraction else 0.0
    return {
        "return": total_reward,
        "macro_accuracy": macro_accuracy,
        "avg_usage": avg_usage,
        "avg_eta": avg_eta,
        "avg_u": avg_u,
        "num_steps": float(len(usages)),
    }


def run_baseline(
    method: str,
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Iterable[int],
) -> None:
    tasks: Iterable[str] = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in config")
    baseline_cfg = config.get("baselines", {})
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger(seed_dir)
        torch.manual_seed(seed)
        total_episodes = config.get("run", {}).get("total_episodes", 1)
        for episode in tqdm(range(total_episodes), desc=f"seed={seed}"):
            task_cfg = tasks[episode % len(tasks)]
            env_seed = seed + episode * 31
            env = build_env_for_task(config, task_cfg, env_seed)
            scheduler = create_scheduler(
                method,
                [spec.name for spec in env.config.datasets],
                (config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
                config["horizon"],
                usage=config.get("baseline_usage", 0.1),
                method_kwargs=baseline_cfg.get(method, {}),
            )
            stats = _baseline_episode(env, scheduler, config["horizon"])
            stats.update(
                {
                    "episode": episode,
                    "task": Path(task_cfg).stem,
                    "seed": seed,
                    "method": method,
                }
            )
            logger.log_row(stats)
        logger.flush_json()


def run_ppo(
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Iterable[int],
    deterministic_eval: bool,
) -> None:
    tasks: Iterable[str] = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in config")
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger(seed_dir)
        torch.manual_seed(seed)
        policy = TeacherPolicy(
            descriptor_dim=8,
            g_model_dim=6,
            g_progress_dim=11,
            eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
        )
        ppo_cfg = PPOConfig(**config.get("ppo", {}))
        ppo = PPOTeacher(policy, ppo_cfg)
        total_episodes = config.get("run", {}).get("total_episodes", 1)
        checkpoint_interval = config.get("run", {}).get("checkpoint_interval", 50)
        checkpoint_path = seed_dir / "policy.pt"
        best_return: Optional[float] = None

        for episode in tqdm(range(total_episodes), desc=f"seed={seed}"):
            task_cfg = tasks[episode % len(tasks)]
            env_seed = seed + episode * 31
            env = build_env_for_task(config, task_cfg, env_seed)
            stats = ppo.train_episode(env, config["horizon"])
            env.close()
            stats.update(
                {
                    "episode": episode,
                    "task": Path(task_cfg).stem,
                    "seed": seed,
                    "method": "ppo",
                }
            )
            logger.log_row(stats)
            current_return = stats["return"]
            is_best = best_return is None or current_return > best_return
            if is_best:
                best_return = current_return
            should_checkpoint = (episode + 1) % checkpoint_interval == 0 or is_best
            if should_checkpoint:
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optim": ppo.optim.state_dict(),
                        "config": config,
                        "episode": episode + 1,
                        "best_return": best_return,
                        "lambda_cmdp": ppo.lambda_cmdp,
                    },
                    checkpoint_path,
                )
        logger.flush_json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparative baselines")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="For PPO, use deterministic policy during eval checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds = args.seeds or [config.get("seed", 0)]
    output_dir = get_output_directory(args.method, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": str(args.config),
        "method": args.method,
        "seeds": list(seeds),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.method == "ppo":
        run_ppo(config, output_dir, seeds, args.deterministic_eval)
    else:
        run_baseline(args.method, config, output_dir, seeds)


if __name__ == "__main__":
    main()
