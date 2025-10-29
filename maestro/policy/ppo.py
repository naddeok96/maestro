"""Minimal PPO implementation for MAESTRO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from maestro.envs.maestro_env import MaestroEnv

from .deepsets import DeepSetsEncoder
from .policy_heads import PolicyHeads


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class TeacherPolicy(nn.Module):
    def __init__(self, descriptor_dim: int, context_dim: int, eta_bounds: tuple[float, float]):
        super().__init__()
        self.encoder = DeepSetsEncoder(input_dim=descriptor_dim, phi_dim=64, rho_dim=64)
        self.policy_heads = PolicyHeads(
            descriptor_dim=64 + context_dim,
            context_dim=64 + context_dim,
            eta_bounds=eta_bounds,
        )
        self.value_head = nn.Sequential(
            nn.Linear(64 + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observation: Dict[str, np.ndarray], descriptors: np.ndarray):
        g_data = torch.from_numpy(observation["g_data"]).float()
        g_model = torch.from_numpy(observation["g_model"]).float()
        g_progress = torch.from_numpy(observation["g_progress"]).float()
        context = torch.cat([g_data, g_model, g_progress], dim=0)
        descriptors_tensor = torch.from_numpy(descriptors).float()
        summary, encoded = self.encoder(descriptors_tensor)
        repeated_context = torch.cat([summary, context], dim=0)
        expanded = torch.cat([encoded, context.expand(encoded.size(0), -1)], dim=-1)
        mixture, eta, usage = self.policy_heads(expanded, repeated_context)
        value = self.value_head(repeated_context)
        return mixture, eta, usage, value

    def act(self, observation: Dict[str, np.ndarray], descriptors: np.ndarray):
        mixture, eta, usage, value = self.forward(observation, descriptors)
        action = {
            "w": mixture.detach().cpu().numpy(),
            "eta": np.array([eta.item()], dtype=np.float32),
            "u": np.array([usage.item()], dtype=np.float32),
        }
        log_prob = torch.tensor(0.0)
        return action, log_prob, value


class PPOTeacher:
    def __init__(self, policy: TeacherPolicy, config: PPOConfig):
        self.policy = policy
        self.config = config
        self.optim = Adam(self.policy.parameters(), lr=config.learning_rate)

    def train_episode(self, env: MaestroEnv, horizon: int) -> Dict[str, float]:
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        ep_reward = 0.0
        for _ in range(horizon):
            action, _, _ = self.policy.act(obs, descriptors)
            obs, reward, done, _, info = env.step(action)
            descriptors = env.last_per_dataset_descriptors
            ep_reward += reward
            if done:
                break
        return {"return": ep_reward}
