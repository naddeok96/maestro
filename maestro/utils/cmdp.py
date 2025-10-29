"""Utilities for enforcing CMDP budgets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetManager:
    """Tracks a single scalar budget for CMDP style constraints."""

    total_budget: float

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        self.remaining = float(self.total_budget)

    def reset(self) -> None:
        self.remaining = float(self.total_budget)

    def consume(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Cannot consume negative budget")
        self.remaining = max(0.0, self.remaining - amount)

    @property
    def is_depleted(self) -> bool:
        return self.remaining <= 1e-6
