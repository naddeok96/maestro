from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from maestro.envs.metrics import ece_5, macro_average


def test_macro_average():
    metrics = {"a": 0.5, "b": 0.75}
    assert macro_average(metrics) == 0.625


def test_ece_5_zero_for_perfect_calibration():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    assert ece_5(logits, targets) < 1e-6
