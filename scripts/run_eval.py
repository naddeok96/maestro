"""Run evaluation for a trained teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.serialization import load_checkpoint


def build_env_from_task(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained teacher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    if args.checkpoint is not None:
        state = load_checkpoint(args.checkpoint)
        policy.load_state_dict(state.get("policy", state))

    task_list: Iterable[str] = config.get("eval_tasks", config["tasks"])
    base_seed = config.get("seed", 0)
    results = {}
    for index, task_cfg in enumerate(task_list):
        env_seed = base_seed + index * 17
        env = build_env_from_task(config, task_cfg, env_seed)
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        total_reward = 0.0
        info = {}
        for _ in range(args.steps):
            action, _, _, _ = policy.act(obs, descriptors)
            obs, reward, done, _, info = env.step(action)
            descriptors = env.last_per_dataset_descriptors
            total_reward += reward
            if done:
                break
        results[Path(task_cfg).stem] = {
            "macro_accuracy": info.get("macro_accuracy", 0.0),
            "return": total_reward,
        }
        env.close()
    print(results)


if __name__ == "__main__":
    main()
