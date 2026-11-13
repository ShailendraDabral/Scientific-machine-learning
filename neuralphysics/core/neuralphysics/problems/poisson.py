"""Poisson equation on the unit square."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid
from ..pinn.autodiff import grad
from ..solvers import poisson2d_dirichlet
from .base import ParamSpec, Problem, to_numpy

PI = float(np.pi)


class Poisson2D(Problem):
    """Laplacian(u) = f on (0, 1)^2 with Dirichlet data.

    Manufactured solution: u = A sin(m pi x) sin(n pi y) + B (x^2 - y^2).
    The second term is harmonic, so it changes the boundary data g = B (x^2 - y^2)
    without changing the source f = -A (m^2 + n^2) pi^2 sin(m pi x) sin(n pi y).
    """

    name = "poisson_2d"
    title = "Poisson equation"
    equation = r"\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)"
    coords = ("x", "y")
    time_dependent = False
    analytical_method = "Manufactured solution (closed form)"
    numerical_method = "Five-point finite differences, sparse direct solve (h = 1/160)"
    eval_shape = (81, 81)
    PARAMS = (
        ParamSpec("mode_m", "Source mode m (x)", "pde", 1, kind="int", min=1, max=3, step=1),
        ParamSpec("mode_n", "Source mode n (y)", "pde", 1, kind="int", min=1, max=3, step=1),
        ParamSpec("amplitude", "Source amplitude A", "pde", 1.0, min=0.1, max=2.0, step=0.1),
        ParamSpec("boundary_amplitude", "Boundary data B in g = B (x² − y²)", "boundary", 0.0,
                  min=-1.0, max=1.0, step=0.1),
    )

    def bounds(self):
        return np.zeros(2), np.ones(2)

    def _mode(self, x: Tensor) -> Tensor:
        return (self.p["amplitude"] * torch.sin(self.p["mode_m"] * PI * x[:, 0:1])
                * torch.sin(self.p["mode_n"] * PI * x[:, 1:2]))

    def source(self, x: Tensor) -> Tensor:
        return -(self.p["mode_m"] ** 2 + self.p["mode_n"] ** 2) * PI**2 * self._mode(x)

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        u_yy = grad(du[:, 1:2], x)[:, 1:2]
        return u_xx + u_yy - self.source(x)

    def boundary_value(self, x):
        return self.p["boundary_amplitude"] * (x[:, 0:1] ** 2 - x[:, 1:2] ** 2)

    def exact(self, x):
        return self._mode(x) + self.boundary_value(x)

    def numerical(self, grid: Grid) -> np.ndarray:
        x, _ = grid.axes
        sx = 2
        n = (len(x) - 1) * sx + 1
        xs = np.linspace(x[0], x[-1], n)
        xx, yy = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u = poisson2d_dirichlet(to_numpy(self.source, pts).reshape(n, n),
                                to_numpy(self.boundary_value, pts).reshape(n, n), xs[1] - xs[0])
        return u[::sx, ::sx]
