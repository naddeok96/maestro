"""Utility for estimating forward/backward FLOPs."""

from __future__ import annotations

import contextlib
import time
from typing import Tuple

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from ptflops import get_model_complexity_info
except Exception:  # pragma: no cover
    get_model_complexity_info = None  # type: ignore


def _has_embedding(m: nn.Module) -> bool:
    return any(isinstance(mod, nn.Embedding) for mod in m.modules())


def estimate_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> float:
    """Estimate the number of FLOPs for a forward+backward pass.

    Falls back to measuring execution time when ptflops is not available.
    """
    if get_model_complexity_info is not None:
        try:
            # Provide an input_constructor so ptflops uses the correct dtype/shape.
            uses_emb = _has_embedding(model)

            def input_constructor(input_res: Tuple[int, ...]):
                if uses_emb:
                    # Try to infer vocab size from a common attribute; fall back to a safe bound.
                    vocab = 10_000
                    for mod in model.modules():
                        if isinstance(mod, nn.Embedding):
                            vocab = max(1, getattr(mod, "num_embeddings", vocab))
                            break
                    x = torch.randint(0, vocab, (1, *input_res), dtype=torch.long)
                else:
                    x = torch.randn(1, *input_res)
                return (x,)

            with contextlib.redirect_stdout(None):
                macs, _ = get_model_complexity_info(
                    model,
                    input_shape,
                    as_strings=False,
                    print_per_layer_stat=False,
                    input_constructor=input_constructor,
                )
            # Count forward MACs and double for backward as a cheap proxy.
            return float(macs * 2.0)
        except Exception:
            # If ptflops chokes (e.g., unusual forward signature), fall back below.
            pass

    # Fallback: time a single forward/backward pass on CPU and approximate FLOPs
    device = next(model.parameters()).device
    if _has_embedding(model):
        dummy = torch.randint(
            0,
            # Try to get vocab size from any embedding module
            max(
                next((getattr(mod, "num_embeddings", 10_000) for mod in model.modules() if isinstance(mod, nn.Embedding)), 10_000) - 1,
                1,
            ),
            (1, *input_shape),
            device=device,
            dtype=torch.long,
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
