"""Meta-training script for Maestro PPO agent."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOAgent, PPOConfig
from maestro.utils.logging import JSONLogger


def rollout(env: MaestroEnv, agent: PPOAgent, horizon: int) -> dict[str, np.ndarray]:
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(horizon):
        action, logp, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)
        val_buf.append(value)
        logp_buf.append(logp)
        done = float(terminated or truncated)
        done_buf.append(done)
        obs = next_obs
        if terminated:
            obs, _ = env.reset()
    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "act": np.asarray(act_buf, dtype=np.int64),
        "rew": np.asarray(rew_buf, dtype=np.float32),
        "val": np.asarray(val_buf, dtype=np.float32),
        "logp": np.asarray(logp_buf, dtype=np.float32),
        "done": np.asarray(done_buf, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-train PPO agent on Maestro")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--logdir", type=Path, default=Path("runs/meta_train"))
    args = parser.parse_args()

    curriculum = SyntheticCurriculum()
    env = MaestroEnv(MaestroEnvConfig(curriculum=curriculum))
    agent = PPOAgent(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    logger = JSONLogger(args.logdir / "metrics.jsonl")

    for episode in range(args.episodes):
        trajectory = rollout(env, agent, args.horizon)
        advantages, returns = agent.compute_gae(
            trajectory["rew"], trajectory["val"], trajectory["done"], agent.config.gamma, agent.config.lam
        )
        trajectory["adv"] = advantages
        trajectory["ret"] = returns
        metrics = agent.update(trajectory)
        logger.log({"episode": episode, **metrics, "reward_sum": float(np.sum(trajectory["rew"]))})
        print(f"Episode {episode}: reward_sum={np.sum(trajectory['rew']):.2f}")

    logger.flush()


if __name__ == "__main__":
    main()
