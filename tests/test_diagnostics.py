from __future__ import annotations

import numpy as np

from maestro.eval.diagnostics import compute_markov_diagnostic, summarize_rollout


def test_markov_diagnostic_bounds():
    series = np.array([0.0, 1.0, 0.5, 0.5, 1.0])
    value = compute_markov_diagnostic(series)
    assert -1.0 <= value <= 1.0


def test_summarize_rollout_extracts_metrics():
    history = [
        {"reward": 1.0, "info": {"cost": 0.1, "accuracy": 0.8}},
        {"reward": 0.5, "info": {"cost": 0.2, "accuracy": 0.7}},
    ]
    metrics = summarize_rollout(history)
    assert metrics.cumulative_reward == 1.5
    assert metrics.average_cost == 0.15
    assert metrics.average_accuracy == 0.75
