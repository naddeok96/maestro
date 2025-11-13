"""Run MAESTRO meta-training with PPO."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from maestro.baselines.stateful_schedulers import create_scheduler
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


def _clone_observation(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(val, copy=True) for key, val in obs.items()}


def _make_scheduler(
    method_name: str,
    dataset_names: Sequence[str],
    eta_bounds: tuple[float, float],
    horizon: int,
    usage: float,
):
    alias = {
        "uniform": "uniform",
        "easy_to_hard": "easy_to_hard",
        "greedy": "greedy",
        "linucb": "bandit_linucb",
    }
    key = method_name.lower()
    if key not in alias:
        raise ValueError(
            f"Unsupported BC baseline '{method_name}'. "
            "Expected one of: uniform, easy_to_hard, greedy, linucb."
        )
    return create_scheduler(alias[key], dataset_names, eta_bounds, horizon, usage=usage)


def bc_warm_start(
    policy: TeacherPolicy,
    config: Dict[str, Any],
    task_cfgs: Sequence[str],
    seed: int,
    *,
    baseline: str,
    usage_target: float,
    episodes: int,
    epochs: int,
    learning_rate: float,
) -> None:
    """Run a short BC warm-start on policy heads using baseline trajectories."""

    if episodes <= 0 or not task_cfgs:
        return

    policy_device = policy.device
    horizon = int(config["horizon"])
    eta_bounds = (config["optimizer"]["eta_min"], config["optimizer"]["eta_max"])
    collected: List[Dict[str, torch.Tensor | Dict[str, np.ndarray]]] = []
    rng = np.random.default_rng(seed)
    original_mode = policy.training
    policy.train()

    for episode_idx in range(episodes):
        task_cfg = task_cfgs[episode_idx % len(task_cfgs)]
        env_seed = seed + 101 * (episode_idx + 1)
        env = build_env_for_task(config, task_cfg, env_seed)
        try:
            obs, _ = env.reset()
            descriptors = env.last_per_dataset_descriptors
            dataset_names = [spec.name for spec in env.config.datasets]
            scheduler = _make_scheduler(
                baseline,
                dataset_names,
                eta_bounds,
                horizon,
                usage=usage_target,
            )
            scheduler.start_episode(
                obs,
                descriptors,
                dataset_metrics={name: {"accuracy": 0.0} for name in dataset_names},
            )

            for _ in range(horizon):
                action, _, _, _ = scheduler.act(obs, descriptors)
                w_target = torch.as_tensor(
                    np.array(action["w"], copy=True),
                    dtype=torch.float32,
                    device=policy_device,
                )
                eta_target = torch.as_tensor(
                    float(action["eta"][0]), dtype=torch.float32, device=policy_device
                )
                u_target = torch.as_tensor(
                    float(action["u"][0]), dtype=torch.float32, device=policy_device
                )
                collected.append(
                    {
                        "observation": _clone_observation(obs),
                        "descriptors": np.array(descriptors, copy=True),
                        "w": w_target,
                        "eta": eta_target,
                        "u": u_target,
                    }
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                scheduler.update(
                    float(reward), {"dataset_metrics": info.get("dataset_metrics", {})}
                )
                obs = next_obs
                descriptors = env.last_per_dataset_descriptors
                if terminated or truncated:
                    break
        finally:
            env.close()

    if not collected:
        if not original_mode:
            policy.eval()
        return

    optimizer = torch.optim.Adam(
        policy.policy_heads.parameters(), lr=float(learning_rate)
    )
    batch_size = 128
    for epoch_idx in range(max(1, epochs)):
        perm = rng.permutation(len(collected))
        for start in range(0, len(perm), batch_size):
            indices = perm[start : start + batch_size]
            losses: List[torch.Tensor] = []
            for idx in indices:
                sample = collected[idx]
                obs_np = sample["observation"]
                desc_np = sample["descriptors"]
                _, _, _, encoded, context = policy._prepare_inputs(
                    obs_np, desc_np  # type: ignore[arg-type]
                )
                outputs = policy.policy_heads(encoded, context)
                alpha = F.softplus(outputs.mixture_logits) + policy.dirichlet_eps
                w_pred = alpha / alpha.sum()
                eta_pred = policy._bound_eta(outputs.lr_logit)
                u_pred = torch.sigmoid(outputs.usage_logit)
                loss = (
                    F.mse_loss(w_pred, sample["w"])
                    + 0.25 * F.mse_loss(eta_pred, sample["eta"])
                    + 0.25 * F.mse_loss(u_pred, sample["u"])
                )
                losses.append(loss)
            if not losses:
                continue
            loss_value = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            log_metrics({"bc_loss": float(loss_value.detach().cpu())})

    log_metrics({"bc_samples": float(len(collected))})
    if not original_mode:
        policy.eval()


def _lerp(episode: int, warmup: int, start: float, end: float) -> float:
    if warmup <= 0:
        return float(end)
    frac = min(max(episode, 0) / float(warmup), 1.0)
    return float(start * (1.0 - frac) + end * frac)


def _apply_exploration_schedule(
    ppo_config: PPOConfig,
    schedule_cfg: Dict[str, Any],
    episode: int,
    defaults: Dict[str, float],
) -> Dict[str, float]:
    warmup = int(schedule_cfg.get("warmup_episodes", 0))
    entropy_mix = _lerp(
        episode,
        warmup,
        float(schedule_cfg.get("entropy_mix_warmup", defaults["entropy_mix"])),
        float(schedule_cfg.get("entropy_mix_final", defaults["entropy_mix"])),
    )
    entropy_u = _lerp(
        episode,
        warmup,
        float(schedule_cfg.get("entropy_u_warmup", defaults["entropy_u"])),
        float(schedule_cfg.get("entropy_u_final", defaults["entropy_u"])),
    )
    barrier_u = _lerp(
        episode,
        warmup,
        float(schedule_cfg.get("barrier_u_warmup", defaults["barrier_u"])),
        float(schedule_cfg.get("barrier_u_final", defaults["barrier_u"])),
    )
    ppo_config.entropy_coef_mix = entropy_mix
    ppo_config.entropy_coef_u = entropy_u
    ppo_config.barrier_kappa_prime = barrier_u
    return {
        "entropy_mix": entropy_mix,
        "entropy_u": entropy_u,
        "barrier_u": barrier_u,
    }


def run_meta_training(
    config_path: Path,
    *,
    dry_run: bool = False,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    resume: Optional[Path] = None,
    deterministic_eval: bool = False,
    bc_warm_start_flag: bool = False,
    bc_episodes: int = 2,
    bc_baseline: str = "uniform",
    bc_usage: float = 0.4,
    bc_epochs: int = 2,
) -> Path:
    """Execute PPO meta-training and return the resolved output directory."""

    config = load_config(config_path)
    tasks: List[str] = list(config.get("tasks", []))
    if dry_run:
        print(json.dumps({"status": "dry_run", "config": config}, indent=2))
        return get_output_directory(config, output_dir)
    if bc_warm_start_flag and not (0.0 < bc_usage < 1.0):
        raise ValueError("--bc-usage must be within (0, 1)")

    run_seed = seed if seed is not None else config.get("seed", 0)
    torch.manual_seed(run_seed)

    resolved_output = get_output_directory(config, output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(resolved_output)
    run_name = f"ppo_meta_train_{run_seed}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(run_name, config={"config": config, "seed": run_seed})

    ppo_cfg = PPOConfig(**config.get("ppo", {}))
    teacher_cfg = config.get("teacher", {})
    mixture_bias_init = teacher_cfg.get("mixture_bias_init")
    mixture_bias_init = (
        None if mixture_bias_init is None else float(mixture_bias_init)
    )
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
        mixture_bias_init=mixture_bias_init,
    )
    if bc_warm_start_flag:
        task_subset = tasks[: max(1, min(len(tasks), 2))]
        bc_warm_start(
            policy,
            config,
            task_subset,
            run_seed,
            baseline=bc_baseline,
            usage_target=bc_usage,
            episodes=bc_episodes,
            epochs=bc_epochs,
            learning_rate=ppo_cfg.learning_rate,
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

    if not tasks:
        raise ValueError("No tasks specified in config")

    total_episodes = config.get("run", {}).get("total_episodes", 1)
    checkpoint_interval = config.get("run", {}).get("checkpoint_interval", 50)
    schedule_defaults = {
        "entropy_mix": ppo_cfg.entropy_coef_mix,
        "entropy_u": ppo_cfg.entropy_coef_u,
        "barrier_u": ppo_cfg.barrier_kappa_prime,
    }
    exploration_cfg = config.get("ppo_exploration")

    try:
        for episode in range(start_episode, total_episodes):
            task_cfg = tasks[episode % len(tasks)]
            if exploration_cfg:
                schedule_vals = _apply_exploration_schedule(
                    ppo.config, exploration_cfg, episode, schedule_defaults
                )
                log_metrics(
                    {
                        "sched/entropy_mix": schedule_vals["entropy_mix"],
                        "sched/entropy_u": schedule_vals["entropy_u"],
                        "sched/barrier_u": schedule_vals["barrier_u"],
                    }
                )
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
    parser.add_argument(
        "--bc-warm-start",
        action="store_true",
        help="Run a short behavior-cloning warm start before PPO",
    )
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=2,
        help="Number of episodes to collect for BC warm start",
    )
    parser.add_argument(
        "--bc-baseline",
        type=str,
        default="uniform",
        choices=["uniform", "easy_to_hard", "greedy", "linucb"],
        help="Baseline scheduler used to synthesize BC targets",
    )
    parser.add_argument(
        "--bc-usage",
        type=float,
        default=0.4,
        help="Target usage fraction when creating BC targets (0, 1)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=2,
        help="Supervised epochs to fit policy heads during BC warm start",
    )
    args = parser.parse_args()

    run_meta_training(
        args.config,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        deterministic_eval=args.deterministic_eval,
        bc_warm_start_flag=args.bc_warm_start,
        bc_episodes=args.bc_episodes,
        bc_baseline=args.bc_baseline,
        bc_usage=args.bc_usage,
        bc_epochs=args.bc_epochs,
    )


if __name__ == "__main__":
    main()
