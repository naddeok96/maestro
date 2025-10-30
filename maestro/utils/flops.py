"""Utility for estimating forward/backward FLOPs."""

from __future__ import annotations

import contextlib
import time
from typing import Optional

import torch

try:  # pragma: no cover - optional dependency
    from ptflops import get_model_complexity_info
except Exception:  # pragma: no cover
    get_model_complexity_info = None  # type: ignore


def estimate_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> float:
    """Estimate the number of FLOPs for a forward+backward pass.

    Falls back to measuring execution time when ptflops is not available.
    """

    if get_model_complexity_info is not None:
        with contextlib.redirect_stdout(None):
            macs, _ = get_model_complexity_info(
                model, input_shape, as_strings=False, print_per_layer_stat=False
            )
        return float(macs * 2.0)

    # Fallback: time a single forward/backward pass on CPU and approximate FLOPs
    device = next(model.parameters()).device
    if hasattr(model, "embedding") and isinstance(model.embedding, torch.nn.Embedding):
        dummy = torch.randint(
            0,
            max(model.embedding.num_embeddings - 1, 1),
            (1, *input_shape),
            device=device,
        )
    else:
        dummy = torch.randn(1, *input_shape, device=device)
    start = time.perf_counter()
    output = model(dummy)
    loss = output.sum()
    loss.backward()
    elapsed = time.perf_counter() - start
    # Assume ~1e9 FLOPs per second as a rough proxy on CPU
    return max(elapsed, 1e-6) * 1e9
