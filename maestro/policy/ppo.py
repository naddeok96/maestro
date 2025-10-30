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
    """Policy network that operates on grouped MAESTRO observations.

    The network mirrors the specification from the paper: per-dataset descriptors
    are encoded with a DeepSets encoder, mean pooled to obtain a permutation
    invariant summary, and combined with the global model/progress features to
    form the context consumed by the shared policy heads.
    """

    def __init__(
        self,
        descriptor_dim: int,
        g_model_dim: int,
        g_progress_dim: int,
        eta_bounds: tuple[float, float],
        phi_dim: int = 64,
        rho_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = DeepSetsEncoder(input_dim=descriptor_dim, phi_dim=phi_dim, rho_dim=rho_dim)
        context_dim = rho_dim + descriptor_dim + g_model_dim + g_progress_dim
        self.policy_heads = PolicyHeads(
            descriptor_dim=phi_dim,
            context_dim=context_dim,
            eta_bounds=eta_bounds,
        )
        self.value_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observation: Dict[str, np.ndarray], descriptors: np.ndarray):
        g_data = torch.as_tensor(observation["g_data"], dtype=torch.float32)
        g_model = torch.as_tensor(observation["g_model"], dtype=torch.float32)
        g_progress = torch.as_tensor(observation["g_progress"], dtype=torch.float32)
        descriptors_tensor = torch.as_tensor(descriptors, dtype=torch.float32)

        summary, encoded = self.encoder(descriptors_tensor)

        # Build the global context: DeepSets summary + grouped observation blocks.
        context = torch.cat([summary, g_data, g_model, g_progress], dim=0)

        mixture, eta, usage = self.policy_heads(encoded, context)
        value = self.value_head(context)
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
        last_macro_accuracy = 0.0
        for _ in range(horizon):
            action, _, _ = self.policy.act(obs, descriptors)
            obs, reward, done, _, info = env.step(action)
            descriptors = env.last_per_dataset_descriptors
            ep_reward += reward
            last_macro_accuracy = float(info.get("macro_accuracy", last_macro_accuracy))
            if done:
                break
        return {"return": ep_reward, "macro_accuracy": last_macro_accuracy}
