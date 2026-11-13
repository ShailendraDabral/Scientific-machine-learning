"""Fully connected networks for physics-informed training."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class Sine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sine": Sine,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "relu": nn.ReLU,  # second derivative is zero almost everywhere: kept to demonstrate that failure
}


class MLP(nn.Module):
    """Plain multilayer perceptron with Xavier-initialised weights."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int = 1, activation: str = "tanh"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(ACTIVATIONS[activation]())
        self.net = nn.Sequential(*layers)
        self.layer_sizes = dims

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PINN(nn.Module):
    """u_theta(x): a problem's input features followed by an MLP."""

    def __init__(self, problem, hidden: list[int], activation: str = "tanh", dtype=torch.float64):
        super().__init__()
        self.problem = problem
        in_dim = problem.feature_dim or problem.dim
        self.mlp = MLP(in_dim, hidden, 1, activation).to(dtype)
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.problem.features(x.to(self.dtype)))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
