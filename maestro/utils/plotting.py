"""Plotting utilities for experiment figures."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from maestro.eval.diagnostics import RolloutMetrics


def plot_learning_curve(metrics: Iterable[RolloutMetrics], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cumulative_rewards = [m.cumulative_reward for m in metrics]
    averages = [m.average_accuracy for m in metrics]
    fig, ax1 = plt.subplots()
    ax1.plot(cumulative_rewards, label="Cumulative reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax2 = ax1.twinx()
    ax2.plot(averages, color="tab:orange", label="Accuracy")
    ax2.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
