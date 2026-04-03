from __future__ import annotations

import torch


def local_monotonicity_loss(
    earlier_values: torch.Tensor,
    later_values: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    if earlier_values.shape != later_values.shape:
        raise ValueError("earlier_values and later_values must have the same shape")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")
    return torch.relu(later_values - earlier_values + margin).mean()
