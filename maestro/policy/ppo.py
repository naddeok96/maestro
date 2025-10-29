"""Lightweight PPO implementation with DeepSets style heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepSetEncoder(nn.Module):
    """Permutation invariant encoder using DeepSets pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, set_size, feat)
        phi_x = self.phi(x)
        pooled = phi_x.mean(dim=1)
        return self.rho(pooled)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        set_size = action_dim
        feature_dim = observation_dim // set_size
        self.encoder = DeepSetEncoder(feature_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, dim = obs.shape
        # reshape into set representation
        set_size = self.policy_head[-1].out_features
        feature_dim = dim // set_size
        set_obs = obs.view(batch, set_size, feature_dim)
        latent = self.encoder(set_obs)
        logits = self.policy_head(latent)
        values = self.value_head(latent).squeeze(-1)
        return logits, values


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 4
    batch_size: int = 32


class PPOAgent:
    """Minimal PPO trainer that works with :class:`MaestroEnv`."""

    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.device = torch.device("cpu")
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, trajectories: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.tensor(trajectories["obs"], dtype=torch.float32, device=self.device)
        act = torch.tensor(trajectories["act"], dtype=torch.int64, device=self.device)
        adv = torch.tensor(trajectories["adv"], dtype=torch.float32, device=self.device)
        logp_old = torch.tensor(trajectories["logp"], dtype=torch.float32, device=self.device)
        ret = torch.tensor(trajectories["ret"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        metrics: dict[str, float] = {}
        for _ in range(self.config.train_iters):
            logits, value = self.model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            loss_v = ((value - ret) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = loss_pi + 0.5 * loss_v - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics["loss_pi"] = float(loss_pi.item())
        metrics["loss_v"] = float(loss_v.item())
        metrics["entropy"] = float(entropy.item())
        return metrics

    @staticmethod
    def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t]
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values
        return adv, ret
