"""Run MAESTRO meta-training with PPO."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils import RunPaths
from maestro.utils.config import load_config
from maestro.utils.logging import MetricsLogger
from maestro.utils.wandb import init_wandb_run, log_checkpoint, log_metrics


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


def get_output_directory(
    config: Dict[str, Any], output_override: Optional[Path] = None
) -> Path:
    logging_cfg = config.get("logging", {})
    run_id = config.get("run", {}).get("id", "debug")
    base_dir = (
        Path(output_override)
        if output_override is not None
        else Path(logging_cfg.get("output_dir", "outputs"))
    )
    if base_dir.name == run_id:
        run_paths = RunPaths(base_dir.parent, run_id)
    else:
        run_paths = RunPaths(base_dir, run_id)
    return run_paths.resolve()


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return torch.load(handle, map_location="cpu")


def run_meta_training(
    config_path: Path,
    *,
    dry_run: bool = False,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    resume: Optional[Path] = None,
    deterministic_eval: bool = False,
) -> Path:
    """Execute PPO meta-training and return the resolved output directory."""

    config = load_config(config_path)
    if dry_run:
        print(json.dumps({"status": "dry_run", "config": config}, indent=2))
        return get_output_directory(config, output_dir)

    run_seed = seed if seed is not None else config.get("seed", 0)
    torch.manual_seed(run_seed)

    resolved_output = get_output_directory(config, output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(resolved_output)
    run_name = f"ppo_meta_train_{run_seed}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(run_name, config={"config": config, "seed": run_seed})

    ppo_cfg = PPOConfig(**config.get("ppo", {}))
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    if deterministic_eval:
        policy.eval()
    ppo = PPOTeacher(policy, ppo_cfg)

    start_episode = 0
    best_return: Optional[float] = None
    checkpoint_path = resolved_output / "policy.pt"
    if resume and resume.exists():
        ckpt = load_checkpoint(resume)
        policy.load_state_dict(ckpt["policy"])
        ppo.optim.load_state_dict(ckpt["optim"])
        ppo.lambda_cmdp = ckpt.get("lambda_cmdp", 0.0)
        start_episode = ckpt.get("episode", 0)
        best_return = ckpt.get("best_return")

    tasks: Iterable[str] = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in config")

    total_episodes = config.get("run", {}).get("total_episodes", 1)
    checkpoint_interval = config.get("run", {}).get("checkpoint_interval", 50)

    try:
        for episode in range(start_episode, total_episodes):
            task_cfg = tasks[episode % len(tasks)]
            env_seed = run_seed + episode * 31
            env = build_env_for_task(config, task_cfg, env_seed)
            stats = ppo.train_episode(env, config["horizon"])
            env.close()
            stats["episode"] = episode
            stats["task"] = Path(task_cfg).stem
            logger.log_row(stats)
            log_metrics(stats)

            current_return = stats["return"]
            is_best = best_return is None or current_return > best_return
            if is_best:
                best_return = current_return

            should_checkpoint = (episode + 1) % checkpoint_interval == 0 or is_best
            if should_checkpoint:
                save_checkpoint(
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
                log_checkpoint(checkpoint_path, resolved_output)
    finally:
        logger.flush_json()
        wandb_run.finish()

    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAESTRO meta-training")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the training config"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use deterministic policy for periodic evaluations",
    )
    args = parser.parse_args()

    run_meta_training(
        args.config,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        deterministic_eval=args.deterministic_eval,
    )


if __name__ == "__main__":
    main()
