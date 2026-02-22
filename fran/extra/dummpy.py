"""Experimental placeholders for importance-sampling work.

This module intentionally avoids running training code at import time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        self.pre_activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        self.pre_activation = x
        x = self.relu(x)
        x = self.fc2(x)
        return x


def compute_importance_score(model: SimpleModel, loss: torch.Tensor, rho: float = 1.0) -> float:
    _ = loss  # Kept for compatibility with earlier prototype signature.
    if model.pre_activation is None or model.pre_activation.grad is None:
        return 0.0
    activation_deriv = (model.pre_activation > 0).float()
    gradient_term = activation_deriv * model.pre_activation.grad
    return float(rho * torch.norm(gradient_term, p=2).item() ** 2)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = weights / (weights.sum() + 1e-12)
    return float(1.0 / (np.square(weights).sum() + 1e-12))
