"""Statistical helper functions for reporting uncertainty."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for paired samples.

    Parameters
    ----------
    a, b:
        Arrays containing paired measurements (e.g. per-seed macro accuracies).
    n:
        Number of bootstrap resamples.
    seed:
        Seed for the RNG controlling the resampling process.

    Returns
    -------
    mean_diff, (low, high)
        The mean difference ``a - b`` and the 95% confidence interval obtained
        from the bootstrap distribution.
    """

    if a.shape != b.shape:
        raise ValueError("Arrays must be paired and share the same shape")
    if a.ndim != 1:
        raise ValueError("Bootstrap expects one-dimensional inputs")

    rng = np.random.default_rng(seed)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    if diffs.size == 0:
        return mean_diff, (mean_diff, mean_diff)
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        samples[i] = float(np.mean(diffs[idx]))
    low, high = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    return mean_diff, (low, high)


def permutation_test(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
    """Return a two-sided permutation test p-value for paired data."""

    if a.shape != b.shape:
        raise ValueError("Arrays must be paired and share the same shape")
    if a.ndim != 1:
        raise ValueError("Permutation test expects one-dimensional inputs")
    rng = np.random.default_rng(seed)
    diffs = a - b
    observed = abs(float(np.mean(diffs)))
    if diffs.size == 0:
        return 1.0
    count = 0
    for _ in range(n):
        signs = rng.choice([-1.0, 1.0], size=diffs.size)
        permuted = diffs * signs
        if abs(float(np.mean(permuted))) >= observed:
            count += 1
    return (count + 1) / (n + 1)


__all__ = ["paired_bootstrap", "permutation_test"]

