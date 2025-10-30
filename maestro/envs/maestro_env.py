"""Gymnasium environment for MAESTRO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from maestro.datasets import DatasetSpec
from maestro.envs.budgets import BudgetManager
from maestro.envs.observations import Observation, ObservationBuilder
from maestro.envs.student_runner import SegmentOutput, StudentRunner
from maestro.students import build_student
from maestro.utils import OptimizerSettings, seed_everything


@dataclass
class MaestroEnvConfig:
    datasets: List[DatasetSpec]
    horizon: int
    batch_size: int
    initial_budget: int
    probe_size: int
    grad_project_dim: int
    grad_ema_beta: float
    grad_norm_alpha: float
    eta_min: float
    eta_max: float
    weight_decay: float
    momentum: float
    seed: int = 0


class MaestroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MaestroEnvConfig):
        super().__init__()
        self.config = config
        seed_everything(config.seed)
        self.device = torch.device("cpu")
        self.student = build_student(config.datasets).to(self.device)
        self.budget = BudgetManager(total_budget=config.initial_budget)
        self.runner = StudentRunner(
            student=self.student,
            datasets=config.datasets,
            batch_size=config.batch_size,
            probe_size=config.probe_size,
            grad_project_dim=config.grad_project_dim,
            grad_ema_beta=config.grad_ema_beta,
            grad_norm_alpha=config.grad_norm_alpha,
            seed=config.seed,
            device=self.device,
        )
        self.observation_builder = ObservationBuilder(config.datasets, float(config.initial_budget))
        self.current_step = 0
        self.previous_macro = 0.0
        self.last_observation: Optional[Observation] = None
        self.datasets = config.datasets
        obs_dim = 8
        self.observation_space = spaces.Dict(
            {
                "g_data": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "g_model": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "g_progress": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "w": spaces.Box(low=0.0, high=1.0, shape=(len(self.datasets),), dtype=np.float32),
                "eta": spaces.Box(low=config.eta_min, high=config.eta_max, shape=(1,), dtype=np.float32),
                "u": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _initial_observation(self) -> Observation:
        dummy_output = SegmentOutput(
            dataset_metrics={spec.name: {"accuracy": 0.0, "loss": 0.0} for spec in self.datasets},
            macro_accuracy=0.0,
            train_loss=0.0,
            val_loss=0.0,
            grad_projection=torch.zeros(self.config.grad_project_dim),
            grad_ema=torch.zeros(self.config.grad_project_dim),
            grad_cosine=0.0,
            grad_norm=0.0,
            grad_norm_ema=0.0,
            param_change=0.0,
            descriptors={spec.name: torch.zeros(8) for spec in self.datasets},
            usage=0,
            batches=0,
            lr=self.config.eta_min,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            mixture=[1.0 / len(self.datasets)] * len(self.datasets),
        )
        observation = self.observation_builder.build(
            student=self.student,
            step_index=0,
            horizon=self.config.horizon,
            remaining_budget=self.budget.remaining,
            segment=dummy_output,
        )
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            seed_everything(seed)
        self.student = build_student(self.config.datasets).to(self.device)
        self.runner = StudentRunner(
            student=self.student,
            datasets=self.config.datasets,
            batch_size=self.config.batch_size,
            probe_size=self.config.probe_size,
            grad_project_dim=self.config.grad_project_dim,
            grad_ema_beta=self.config.grad_ema_beta,
            grad_norm_alpha=self.config.grad_norm_alpha,
            seed=self.config.seed,
            device=self.device,
        )
        self.budget.reset()
        self.observation_builder.reset()
        self.current_step = 0
        self.previous_macro = 0.0
        observation = self._initial_observation()
        self.last_observation = observation
        return self._to_gym_obs(observation), {}

    def step(self, action: Dict[str, np.ndarray]):
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction("Action outside bounds")
        mixture = np.array(action["w"], dtype=np.float32)
        if mixture.sum() <= 0:
            mixture = np.ones_like(mixture) / len(mixture)
        mixture = mixture / mixture.sum()
        eta = float(np.clip(action["eta"][0], self.config.eta_min, self.config.eta_max))
        usage_fraction = float(np.clip(action["u"][0], 0.0, 1.0))
        available_examples = int(self.budget.remaining)
        usage_examples = int(usage_fraction * available_examples)
        desired_batches = (
            int(np.ceil(usage_examples / self.config.batch_size)) if usage_examples > 0 else 0
        )
        # hard cap by what's left in the budget; zero batches allowed when insufficient budget
        max_batches = int(available_examples // self.config.batch_size)
        batches = min(desired_batches, max_batches)
        settings = OptimizerSettings(
            learning_rate=eta,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
        )
        segment = self.runner.run_segment(mixture, batches, settings)
        self.budget.consume(segment.usage)
        self.current_step += 1
        observation = self.observation_builder.build(
            student=self.student,
            step_index=self.current_step,
            horizon=self.config.horizon,
            remaining_budget=self.budget.remaining,
            segment=segment,
        )
        reward = segment.macro_accuracy - self.previous_macro
        self.previous_macro = segment.macro_accuracy
        no_batches_left = max_batches == 0
        terminated = self.budget.is_depleted
        truncated = (self.current_step >= self.config.horizon) or no_batches_left
        self.last_observation = observation
        info = {
            "macro_accuracy": segment.macro_accuracy,
            "dataset_metrics": segment.dataset_metrics,
            "usage": segment.usage,
            "cost": segment.usage,
        }
        return self._to_gym_obs(observation), reward, terminated, truncated, info

    def _to_gym_obs(self, observation: Observation) -> Dict[str, np.ndarray]:
        return {
            "g_data": observation.g_data.astype(np.float32),
            "g_model": observation.g_model.astype(np.float32),
            "g_progress": observation.g_progress.astype(np.float32),
        }

    @property
    def last_per_dataset_descriptors(self) -> np.ndarray:
        if self.last_observation is None:
            return np.zeros((len(self.datasets), 8), dtype=np.float32)
        return self.last_observation.descriptors

    def render(self):
        if self.last_observation is None:
            return ""
        metrics = [f"{name}: {acc:.2f}" for name, acc in zip(self.last_observation.dataset_names, self.last_observation.descriptors[:, 0])]
        return " | ".join(metrics)
