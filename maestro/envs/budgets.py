"""Budget handling for the CMDP formulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetManager:
    total_budget: int

    def __post_init__(self) -> None:
        self.remaining = float(self.total_budget)

    def reset(self) -> None:
        self.remaining = float(self.total_budget)

    @property
    def is_depleted(self) -> bool:
        return self.remaining <= 0.0

    def consume(self, amount: float) -> None:
        self.remaining = max(0.0, self.remaining - amount)

    def fraction_remaining(self) -> float:
        if self.total_budget <= 0:
            return 0.0
        return float(self.remaining / self.total_budget)
