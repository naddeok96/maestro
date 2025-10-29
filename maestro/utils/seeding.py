"""Global seeding utilities."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
