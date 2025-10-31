"""Stateful baseline schedulers for comparative evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

@dataclass
class PBTConfig:
    population_size: int = 4
    exploit_interval: int = 20
    perturb_scale: float = 0.15


@dataclass
class BOHBConfig:
    random_trials: int = 5
    top_k: int = 3
    bandwidth: float = 20.0


BaselineAction = Tuple[Dict[str, np.ndarray], torch.Tensor, torch.Tensor, Dict[str, float]]


@dataclass
class BaselineConfig:
    """Configuration parameters shared by baselines."""

    eta_bounds: Tuple[float, float]
    default_usage: float = 0.1
    method_kwargs: Optional[Dict[str, float]] = None


class BaselineScheduler:
    """Base class for deterministic curriculum schedulers."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
    ) -> None:
        self.dataset_names = list(dataset_names)
        self.num_datasets = len(self.dataset_names)
        self.config = config
        self.eta_value = sum(config.eta_bounds) * 0.5
        self.usage_fraction = float(np.clip(config.default_usage, 1e-6, 1.0))
        self.step_index = 0
        self._last_choice: Optional[int] = None
        self._metrics: Dict[str, float] = {
            name: 0.0 for name in self.dataset_names
        }

    # ------------------------------------------------------------------
    # Episode lifecycle helpers
    # ------------------------------------------------------------------
    def start_episode(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
        dataset_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Reset internal counters before an episode begins."""

        self.step_index = 0
        self._last_choice = None
        if dataset_metrics is not None:
            self._update_metrics(dataset_metrics)
        else:
            self._metrics = {name: 0.0 for name in self.dataset_names}

    # ------------------------------------------------------------------
    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def update(self, reward: float, info: Dict[str, float]) -> None:
        """Update internal statistics after observing an environment step."""

        dataset_metrics = info.get("dataset_metrics")
        if dataset_metrics is not None:
            self._update_metrics(dataset_metrics)
        self.step_index += 1

    # ------------------------------------------------------------------
    def _update_metrics(self, dataset_metrics: Dict[str, Dict[str, float]]) -> None:
        for name in self.dataset_names:
            metrics = dataset_metrics.get(name)
            if metrics is None:
                continue
            self._metrics[name] = float(metrics.get("accuracy", 0.0))

    # ------------------------------------------------------------------
    def _build_action(
        self,
        mixture: np.ndarray,
        eta_override: Optional[float] = None,
        usage_override: Optional[float] = None,
        chosen_idx: Optional[int] = None,
    ) -> BaselineAction:
        eta_value = float(
            np.clip(
                self.eta_value if eta_override is None else eta_override,
                self.config.eta_bounds[0],
                self.config.eta_bounds[1],
            )
        )
        usage_value = float(
            np.clip(
                self.usage_fraction if usage_override is None else usage_override,
                0.0,
                1.0,
            )
        )
        mixture = np.asarray(mixture, dtype=np.float32)
        if mixture.sum() <= 0:
            mixture = np.ones_like(mixture) / max(1, len(mixture))
        mixture = mixture / mixture.sum()
        action = {
            "w": mixture.astype(np.float32),
            "eta": np.array([eta_value], dtype=np.float32),
            "u": np.array([usage_value], dtype=np.float32),
        }
        info: Dict[str, float] = {
            "chosen_dataset": float(chosen_idx) if chosen_idx is not None else -1.0
        }
        zero = torch.zeros((), dtype=torch.float32)
        return action, zero, zero, info


class UniformScheduler(BaselineScheduler):
    """Uniform sampling over datasets."""

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        mixture = np.ones(self.num_datasets, dtype=np.float32)
        return self._build_action(mixture)


class EasyToHardScheduler(BaselineScheduler):
    """Curriculum that sweeps from the first dataset to the last."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
        total_steps: int,
    ) -> None:
        super().__init__(dataset_names, config)
        self.total_steps = max(1, total_steps)

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        fraction = self.step_index / self.total_steps
        index = min(self.num_datasets - 1, int(fraction * self.num_datasets))
        mixture = np.zeros(self.num_datasets, dtype=np.float32)
        mixture[index] = 1.0
        return self._build_action(mixture, chosen_idx=index)


class GreedyAccuracyScheduler(BaselineScheduler):
    """Always focus on the lowest accuracy dataset."""

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        accuracies = np.array([self._metrics[name] for name in self.dataset_names])
        index = int(np.argmin(accuracies)) if accuracies.size else 0
        mixture = np.zeros(self.num_datasets, dtype=np.float32)
        mixture[index] = 1.0
        return self._build_action(mixture, chosen_idx=index)


class LinUCBScheduler(BaselineScheduler):
    """LinUCB-style bandit using per-dataset accuracies as rewards."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
    ) -> None:
        super().__init__(dataset_names, config)
        kwargs = config.method_kwargs or {}
        self.alpha = float(kwargs.get("alpha", 1.0))
        self.counts = np.zeros(self.num_datasets, dtype=np.float32)
        self.means = np.zeros(self.num_datasets, dtype=np.float32)
        self.total_count = 0

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        # optimism for unseen arms
        exploration = np.where(
            self.counts > 0,
            self.alpha
            * np.sqrt(np.log(max(1, self.total_count + 1)) / (self.counts + 1e-8)),
            np.inf,
        )
        scores = self.means + exploration
        index = int(np.argmax(scores)) if scores.size else 0
        mixture = np.zeros(self.num_datasets, dtype=np.float32)
        mixture[index] = 1.0
        self._last_choice = index
        return self._build_action(mixture, chosen_idx=index)

    def update(self, reward: float, info: Dict[str, float]) -> None:
        super().update(reward, info)
        if self._last_choice is None:
            return
        dataset_metrics = info.get("dataset_metrics")
        if dataset_metrics is None:
            return
        name = self.dataset_names[self._last_choice]
        acc = float(dataset_metrics.get(name, {}).get("accuracy", 0.0))
        self.total_count += 1
        self.counts[self._last_choice] += 1
        count = self.counts[self._last_choice]
        prev_mean = self.means[self._last_choice]
        self.means[self._last_choice] = prev_mean + (acc - prev_mean) / count


class ThompsonSamplingScheduler(BaselineScheduler):
    """Thompson sampling with Beta posteriors over accuracies."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
    ) -> None:
        super().__init__(dataset_names, config)
        kwargs = config.method_kwargs or {}
        prior = float(kwargs.get("prior", 1.0))
        self.alpha = np.full(self.num_datasets, prior, dtype=np.float32)
        self.beta = np.full(self.num_datasets, prior, dtype=np.float32)
        self._rng = np.random.default_rng(int(kwargs.get("seed", 0)))
        self._last_choice: Optional[int] = None

    def act(
        self,
        observation: Dict[str, np.ndarray],
        descriptors: np.ndarray,
    ) -> BaselineAction:
        samples = self._rng.beta(self.alpha, self.beta)
        index = int(np.argmax(samples)) if samples.size else 0
        mixture = np.zeros(self.num_datasets, dtype=np.float32)
        mixture[index] = 1.0
        self._last_choice = index
        return self._build_action(mixture, chosen_idx=index)

    def update(self, reward: float, info: Dict[str, float]) -> None:
        super().update(reward, info)
        if self._last_choice is None:
            return
        dataset_metrics = info.get("dataset_metrics")
        if dataset_metrics is None:
            return
        name = self.dataset_names[self._last_choice]
        acc = float(dataset_metrics.get(name, {}).get("accuracy", 0.0))
        acc = float(np.clip(acc, 1e-6, 1.0 - 1e-6))
        self.alpha[self._last_choice] += acc
        self.beta[self._last_choice] += 1.0 - acc


class PopulationBasedTrainingScheduler(BaselineScheduler):
    """Population-based training baseline for curriculum control."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
        pbt_config: Optional[PBTConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset_names, config)
        self.pbt_config = pbt_config or PBTConfig()
        self._rng = np.random.default_rng(seed)
        self._population = [self._sample_candidate() for _ in range(self.pbt_config.population_size)]
        self._active_index = 0
        self._updates_since_exploit = 0

    def _sample_candidate(self) -> Dict[str, np.ndarray]:
        mixture = self._rng.dirichlet(np.ones(self.num_datasets, dtype=np.float64)).astype(np.float32)
        eta = float(self._rng.uniform(*self.config.eta_bounds))
        usage = float(np.clip(self.config.default_usage + self._rng.normal(0.0, 0.05), 0.05, 0.9))
        return {"mixture": mixture, "eta": eta, "usage": usage, "score": 0.0}

    def start_episode(self, observation, descriptors, dataset_metrics=None) -> None:  # type: ignore[override]
        super().start_episode(observation, descriptors, dataset_metrics)
        for candidate in self._population:
            candidate["score"] = 0.0
        self._updates_since_exploit = 0

    def act(self, observation, descriptors):  # type: ignore[override]
        idx = self.step_index % len(self._population)
        self._active_index = idx
        candidate = self._population[idx]
        return self._build_action(
            candidate["mixture"],
            eta_override=candidate["eta"],
            usage_override=candidate["usage"],
        )

    def update(self, reward: float, info: Dict[str, float]) -> None:  # type: ignore[override]
        super().update(reward, info)
        candidate = self._population[self._active_index]
        candidate["score"] += float(reward)
        self._updates_since_exploit += 1
        if self._updates_since_exploit >= self.pbt_config.exploit_interval:
            self._exploit_and_explore()
            self._updates_since_exploit = 0

    def _exploit_and_explore(self) -> None:
        scores = np.array([cand["score"] for cand in self._population])
        order = np.argsort(-scores)
        top_half = order[: max(1, len(order) // 2)]
        bottom_half = order[len(order) // 2 :]
        for idx in bottom_half:
            source = self._population[int(self._rng.choice(top_half))]
            cloned = {
                "mixture": self._mutate_mixture(source["mixture"].copy()),
                "eta": self._mutate_scalar(source["eta"], self.config.eta_bounds),
                "usage": float(np.clip(self._mutate_scalar(source["usage"], (0.05, 0.9)), 0.05, 0.9)),
                "score": 0.0,
            }
            self._population[idx] = cloned

    def _mutate_mixture(self, mixture: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self.pbt_config.perturb_scale, size=mixture.shape)
        mixture = np.clip(mixture + noise, 1e-3, None)
        mixture /= mixture.sum()
        return mixture.astype(np.float32)

    def _mutate_scalar(self, value: float, bounds: Sequence[float]) -> float:
        perturbed = value * float(1.0 + self._rng.normal(0.0, self.pbt_config.perturb_scale))
        return float(np.clip(perturbed, bounds[0], bounds[1]))


class BOHBScheduler(BaselineScheduler):
    """BOHB-style scheduler sampling mixtures from a KDE over top performers."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        config: BaselineConfig,
        bohb_config: Optional[BOHBConfig] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset_names, config)
        self.bohb_config = bohb_config or BOHBConfig()
        self._rng = np.random.default_rng(seed)
        self._history: list[Dict[str, np.ndarray | float]] = []
        self._pending: Optional[Dict[str, np.ndarray | float]] = None

    def act(self, observation, descriptors):  # type: ignore[override]
        if len(self._history) < self.bohb_config.random_trials:
            mixture = self._rng.dirichlet(np.ones(self.num_datasets, dtype=np.float64)).astype(np.float32)
        else:
            top = sorted(self._history, key=lambda entry: float(entry["score"]), reverse=True)[
                : self.bohb_config.top_k
            ]
            mean = np.mean([entry["mixture"] for entry in top], axis=0)
            mixture = self._rng.dirichlet(mean * self.bohb_config.bandwidth + 1e-3).astype(np.float32)
        eta = float(self._rng.uniform(*self.config.eta_bounds))
        usage = float(np.clip(self.config.default_usage + self._rng.normal(0.0, 0.05), 0.05, 0.9))
        self._pending = {"mixture": mixture, "eta": eta, "usage": usage, "score": 0.0}
        return self._build_action(mixture, eta_override=eta, usage_override=usage)

    def update(self, reward: float, info: Dict[str, float]) -> None:  # type: ignore[override]
        super().update(reward, info)
        if self._pending is None:
            return
        self._pending["score"] = float(self._pending.get("score", 0.0) + reward)
        self._history.append(self._pending)
        self._pending = None


def create_scheduler(
    method: str,
    dataset_names: Sequence[str],
    eta_bounds: Tuple[float, float],
    horizon: int,
    usage: float = 0.1,
    method_kwargs: Optional[Dict[str, float]] = None,
) -> BaselineScheduler:
    """Factory for constructing schedulers by name."""

    config = BaselineConfig(
        eta_bounds=eta_bounds,
        default_usage=usage,
        method_kwargs=method_kwargs,
    )
    method = method.lower()
    if method == "uniform":
        return UniformScheduler(dataset_names, config)
    if method in {"easy_to_hard", "easy-to-hard"}:
        return EasyToHardScheduler(dataset_names, config, total_steps=horizon)
    if method == "greedy":
        return GreedyAccuracyScheduler(dataset_names, config)
    if method in {"bandit_linucb", "linucb"}:
        return LinUCBScheduler(dataset_names, config)
    if method in {"bandit_thompson", "thompson"}:
        return ThompsonSamplingScheduler(dataset_names, config)
    if method == "pbt":
        kwargs = method_kwargs or {}
        seed = int(kwargs.get("seed", 0))
        cfg_kwargs = {k: kwargs[k] for k in ("population_size", "exploit_interval", "perturb_scale") if k in kwargs}
        pbt_cfg = PBTConfig(**cfg_kwargs) if cfg_kwargs else None
        return PopulationBasedTrainingScheduler(dataset_names, config, pbt_config=pbt_cfg, seed=seed)
    if method == "bohb":
        kwargs = method_kwargs or {}
        seed = int(kwargs.get("seed", 0))
        cfg_kwargs = {k: kwargs[k] for k in ("random_trials", "top_k", "bandwidth") if k in kwargs}
        bohb_cfg = BOHBConfig(**cfg_kwargs) if cfg_kwargs else None
        return BOHBScheduler(dataset_names, config, bohb_config=bohb_cfg, seed=seed)
    raise ValueError(f"Unknown baseline method: {method}")
