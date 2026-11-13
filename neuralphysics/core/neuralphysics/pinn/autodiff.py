"""Automatic-differentiation helpers for PDE residuals."""
from __future__ import annotations

import torch
from torch import Tensor


def grad(u: Tensor, x: Tensor) -> Tensor:
    """Row-wise gradient du/dx for a batch.

    ``u`` has shape (N, 1) and row i depends only on row i of ``x`` (N, d), so
    differentiating the sum gives each sample's own gradient. The graph is kept
    so the result can be differentiated again.
    """
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
