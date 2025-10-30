"""PPO implementation tailored for the MAESTRO teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet, Normal
from torch.nn import functional as F
from torch.optim import Adam

from maestro.envs.maestro_env import MaestroEnv

from .deepsets import DeepSetsEncoder
from .policy_heads import PolicyHeadOutput, PolicyHeads


@dataclass
class PPOConfig:
    """Configuration for PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    minibatch_size: int = 64
    rollout_length: int = 256
    epochs: int = 4
    entropy_coef_mix: float = 0.01
    entropy_coef_u: float = 0.01
    barrier_kappa: float = 1e-4
    barrier_kappa_prime: float = 1e-4
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    cmdp_lambda_lr: float = 1e-3
    cmdp_target_fraction: float = 0.8


@dataclass
class RolloutStep:
    observation: Dict[str, torch.Tensor]
    descriptors: torch.Tensor
    action: Dict[str, torch.Tensor]
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    usage: torch.Tensor


class RolloutBuffer:
    """Simple list-based rollout buffer for PPO."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.steps: List[RolloutStep] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.steps)

    def _stack_tensor(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tuple(values)).to(self.device)

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        if not self.steps:
            self.advantages = torch.zeros(0, device=self.device)
            self.returns = torch.zeros(0, device=self.device)
            return
        rewards = self._stack_tensor([s.reward for s in self.steps])
        values = self._stack_tensor([s.value for s in self.steps])
        dones = self._stack_tensor([s.done for s in self.steps])
        values = torch.cat([values, last_value.unsqueeze(0)], dim=0)

        advantages = torch.zeros(len(self.steps), device=self.device)
        gae = torch.zeros(1, device=self.device)
        for step in reversed(range(len(self.steps))):
            non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
            )
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        self.advantages = advantages
        self.returns = returns

    def iter_minibatches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterable[List[int]]:
        indices = torch.arange(len(self.steps), device=self.device)
        if shuffle and len(indices) > 0:
            indices = indices[torch.randperm(len(indices))]
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size].tolist()


class TeacherPolicy(nn.Module):
    """Teacher policy with stochastic actions suitable for PPO."""

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
        self.encoder = DeepSetsEncoder(
            input_dim=descriptor_dim, phi_dim=phi_dim, rho_dim=rho_dim
        )
        context_dim = rho_dim + descriptor_dim + g_model_dim + g_progress_dim
        self.policy_heads = PolicyHeads(
            descriptor_dim=phi_dim,
            context_dim=context_dim,
            eta_bounds=eta_bounds,
        )
        self.value_head = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.eta_bounds = eta_bounds
        self.dirichlet_eps = 1e-3
        self.eta_log_std = nn.Parameter(torch.zeros(1))
        self.u_log_std = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self.eta_log_std.device

    def _prepare_inputs(
        self, observation: Dict[str, np.ndarray], descriptors: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_data = torch.as_tensor(
            observation["g_data"], dtype=torch.float32, device=self.device
        )
        g_model = torch.as_tensor(
            observation["g_model"], dtype=torch.float32, device=self.device
        )
        g_progress = torch.as_tensor(
            observation["g_progress"], dtype=torch.float32, device=self.device
        )
        descriptors_tensor = torch.as_tensor(
            descriptors, dtype=torch.float32, device=self.device
        )
        summary, encoded = self.encoder(descriptors_tensor)
        context = torch.cat([summary, g_data, g_model, g_progress], dim=0)
        return g_data, g_model, g_progress, encoded, context

    def _build_distributions(
        self, outputs: PolicyHeadOutput
    ) -> tuple[Dirichlet, Normal, Normal, torch.Tensor]:
        alpha = F.softplus(outputs.mixture_logits) + self.dirichlet_eps
        dirichlet = Dirichlet(alpha)
        eta_mean = self._bound_eta(outputs.lr_logit)
        eta_dist = Normal(eta_mean, self.eta_log_std.exp())
        usage_mean = torch.sigmoid(outputs.usage_logit)
        usage_dist = Normal(usage_mean, self.u_log_std.exp())
        return dirichlet, eta_dist, usage_dist, alpha

    def _bound_eta(self, raw: torch.Tensor) -> torch.Tensor:
        eta_min, eta_max = self.eta_bounds
        return eta_min + (eta_max - eta_min) * torch.sigmoid(raw)

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[
        Dict[str, np.ndarray], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
    ]:
        _, _, _, encoded, context = self._prepare_inputs(observation, descriptors)
        outputs = self.policy_heads(encoded, context)
        value = self.value_head(context).squeeze(-1)

        if deterministic:
            mixture = torch.softmax(outputs.mixture_logits, dim=-1)
            eta = torch.tensor(
                (self.eta_bounds[0] + self.eta_bounds[1]) * 0.5, device=self.device
            )
            usage = torch.tensor(0.5, device=self.device)
            action_np = {
                "w": mixture.detach().cpu().numpy(),
                "eta": np.array([eta.item()], dtype=np.float32),
                "u": np.array([usage.item()], dtype=np.float32),
            }
            zero = torch.zeros((), device=self.device)
            action_info = {
                "w": mixture.detach(),
                "eta": eta.detach(),
                "u": usage.detach(),
                "eta_sample": eta.detach(),
                "u_sample": usage.detach(),
            }
            return action_np, zero, value, action_info

        dirichlet, eta_dist, usage_dist, _ = self._build_distributions(outputs)
        mixture = dirichlet.rsample()
        eta_sample = eta_dist.rsample()
        usage_sample = usage_dist.rsample()

        eta = torch.clamp(
            eta_sample, self.eta_bounds[0] + 1e-6, self.eta_bounds[1] - 1e-6
        )
        usage = torch.clamp(usage_sample, 1e-6, 1.0 - 1e-6)

        log_prob = (
            dirichlet.log_prob(mixture)
            + eta_dist.log_prob(eta_sample)
            + usage_dist.log_prob(usage_sample)
        )

        action_np = {
            "w": mixture.detach().cpu().numpy(),
            "eta": np.array([eta.detach().cpu().item()], dtype=np.float32),
            "u": np.array([usage.detach().cpu().item()], dtype=np.float32),
        }
        action_info = {
            "w": mixture.detach(),
            "eta": eta.detach(),
            "u": usage.detach(),
            "eta_sample": eta_sample.detach(),
            "u_sample": usage_sample.detach(),
        }
        return action_np, log_prob.detach(), value.detach(), action_info

    def evaluate_actions(
        self,
        batch_observations: (
            Sequence[Dict[str, np.ndarray]] | Sequence[Dict[str, torch.Tensor]]
        ),
        batch_descriptors: Sequence[np.ndarray] | Sequence[torch.Tensor],
        batch_actions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies_mix: List[torch.Tensor] = []
        entropies_eta: List[torch.Tensor] = []
        entropies_u: List[torch.Tensor] = []
        eta_preds: List[torch.Tensor] = []
        u_preds: List[torch.Tensor] = []

        for obs, desc, w, eta_sample, u_sample in zip(
            batch_observations,
            batch_descriptors,
            batch_actions["w"],
            batch_actions["eta_sample"],
            batch_actions["u_sample"],
        ):
            if isinstance(obs["g_data"], torch.Tensor):
                obs_t = {
                    "g_data": obs["g_data"].detach().cpu().numpy(),
                    "g_model": obs["g_model"].detach().cpu().numpy(),
                    "g_progress": obs["g_progress"].detach().cpu().numpy(),
                }
            else:
                obs_t = obs
            if isinstance(desc, torch.Tensor):
                desc_arr = desc.detach().cpu().numpy()
            else:
                desc_arr = desc
            _, _, _, encoded, context = self._prepare_inputs(obs_t, desc_arr)
            outputs = self.policy_heads(encoded, context)
            dirichlet, eta_dist, usage_dist, _ = self._build_distributions(outputs)
            lp = (
                dirichlet.log_prob(w)
                + eta_dist.log_prob(eta_sample)
                + usage_dist.log_prob(u_sample)
            )
            log_probs.append(lp)
            values.append(self.value_head(context).squeeze(-1))
            entropies_mix.append(dirichlet.entropy())
            entropies_eta.append(eta_dist.entropy())
            entropies_u.append(usage_dist.entropy())
            eta_preds.append(self._bound_eta(outputs.lr_logit))
            u_preds.append(torch.sigmoid(outputs.usage_logit))

        return {
            "log_prob": torch.stack(log_probs),
            "values": torch.stack(values),
            "entropy_mix": torch.stack(entropies_mix),
            "entropy_eta": torch.stack(entropies_eta),
            "entropy_u": torch.stack(entropies_u),
            "eta_pred": torch.stack(eta_preds),
            "u_pred": torch.stack(u_preds),
        }

    def value(
        self, observation: Dict[str, np.ndarray], descriptors: np.ndarray
    ) -> torch.Tensor:
        _, _, _, encoded, context = self._prepare_inputs(observation, descriptors)
        return self.value_head(context).squeeze(-1)


class PPOTeacher:
    """Driver that performs PPO updates for the teacher policy."""

    def __init__(self, policy: TeacherPolicy, config: PPOConfig):
        self.policy = policy
        self.config = config
        self.optim = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.lambda_cmdp = 0.0

    def _to_tensor_obs(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(val, dtype=torch.float32, device=self.policy.device)
            for key, val in observation.items()
        }

    def train_episode(self, env: MaestroEnv, horizon: int) -> Dict[str, float]:
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        buffer = RolloutBuffer(self.policy.device)

        total_reward = 0.0
        macro_acc = 0.0
        episode_usage = 0.0

        for _ in range(horizon):
            action_np, log_prob, value, action_info = self.policy.act(
                obs, descriptors, deterministic=False
            )
            torch_obs = self._to_tensor_obs(obs)
            torch_desc = torch.as_tensor(
                descriptors, dtype=torch.float32, device=self.policy.device
            )
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            cost = float(info.get("cost", 0.0))
            penalty = self.lambda_cmdp * cost
            reward_adj = reward - penalty

            total_reward += reward
            macro_acc = float(info.get("macro_accuracy", macro_acc))
            episode_usage += cost

            buffer.add(
                RolloutStep(
                    observation=torch_obs,
                    descriptors=torch_desc,
                    action={
                        "w": action_info["w"].to(self.policy.device),
                        "eta": action_info["eta"].to(self.policy.device),
                        "u": action_info["u"].to(self.policy.device),
                        "eta_sample": action_info["eta_sample"].to(self.policy.device),
                        "u_sample": action_info["u_sample"].to(self.policy.device),
                    },
                    log_prob=log_prob.to(self.policy.device),
                    value=value.to(self.policy.device),
                    reward=torch.tensor(
                        reward_adj, dtype=torch.float32, device=self.policy.device
                    ),
                    done=torch.tensor(
                        float(terminated),
                        dtype=torch.float32,
                        device=self.policy.device,
                    ),
                    usage=torch.tensor(
                        cost, dtype=torch.float32, device=self.policy.device
                    ),
                )
            )

            obs = next_obs
            descriptors = env.last_per_dataset_descriptors

            if terminated or truncated:
                break

        if terminated:
            last_value = torch.zeros(1, device=self.policy.device)
        else:
            last_value = self.policy.value(obs, descriptors).detach().unsqueeze(0)

        buffer.compute_returns_and_advantages(
            last_value.squeeze(0), self.config.gamma, self.config.gae_lambda
        )

        if buffer.advantages is not None and buffer.advantages.numel() > 0:
            advantages = buffer.advantages
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        else:
            advantages = torch.zeros(0, device=self.policy.device)

        # Prepare tensors for training
        log_probs_old = (
            torch.stack([step.log_prob for step in buffer.steps])
            if buffer.steps
            else torch.zeros(0)
        )
        values_old = (
            torch.stack([step.value for step in buffer.steps])
            if buffer.steps
            else torch.zeros(0)
        )

        for _ in range(self.config.epochs):
            for batch_indices in buffer.iter_minibatches(self.config.minibatch_size):
                if not batch_indices:
                    continue
                obs_batch = [buffer.steps[i].observation for i in batch_indices]
                desc_batch = [buffer.steps[i].descriptors for i in batch_indices]
                actions_batch = {
                    key: torch.stack(
                        [buffer.steps[i].action[key] for i in batch_indices]
                    )
                    for key in ("w", "eta", "u", "eta_sample", "u_sample")
                }
                old_logp = torch.stack(
                    [buffer.steps[i].log_prob for i in batch_indices]
                )
                old_values = torch.stack([buffer.steps[i].value for i in batch_indices])
                adv_batch = advantages[batch_indices]
                returns_batch = buffer.returns[batch_indices]

                eval_results = self.policy.evaluate_actions(
                    obs_batch, desc_batch, actions_batch
                )
                logp = eval_results["log_prob"]
                values_pred = eval_results["values"]
                entropy_mix = eval_results["entropy_mix"]
                entropy_eta = eval_results["entropy_eta"]
                entropy_u = eval_results["entropy_u"]

                ratio = torch.exp(logp - old_logp)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                )
                policy_loss = -torch.min(
                    ratio * adv_batch, clipped_ratio * adv_batch
                ).mean()

                value_loss = F.mse_loss(values_pred, returns_batch)

                entropy_term = (
                    self.config.entropy_coef_mix * entropy_mix.mean()
                    + self.config.entropy_coef_u
                    * (entropy_eta.mean() + entropy_u.mean())
                )

                eta_vals = actions_batch["eta"]
                u_vals = actions_batch["u"]
                eta_min, eta_max = self.policy.eta_bounds
                barrier_eta = -torch.log(eta_vals - eta_min + 1e-6) - torch.log(
                    eta_max - eta_vals + 1e-6
                )
                barrier_u = -torch.log(u_vals + 1e-6) - torch.log(1.0 - u_vals + 1e-6)
                barrier = (
                    self.config.barrier_kappa * barrier_eta.mean()
                    + self.config.barrier_kappa_prime * barrier_u.mean()
                )

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - entropy_term
                    + barrier
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optim.step()

        initial_budget = float(env.config.initial_budget)
        target = self.config.cmdp_target_fraction * initial_budget
        self.lambda_cmdp = max(
            0.0,
            self.lambda_cmdp + self.config.cmdp_lambda_lr * (episode_usage - target),
        )

        avg_eta = (
            torch.stack([step.action["eta"] for step in buffer.steps]).mean().item()
            if buffer.steps
            else 0.0
        )
        avg_u = (
            torch.stack([step.action["u"] for step in buffer.steps]).mean().item()
            if buffer.steps
            else 0.0
        )
        avg_usage = (
            torch.stack([step.usage for step in buffer.steps]).mean().item()
            if buffer.steps
            else 0.0
        )

        return {
            "return": total_reward,
            "macro_accuracy": macro_acc,
            "lambda_cmdp": self.lambda_cmdp,
            "avg_eta": avg_eta,
            "avg_u": avg_u,
            "avg_usage": avg_usage,
            "num_steps": float(len(buffer.steps)),
        }
