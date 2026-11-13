"""Heat equation in one and two space dimensions."""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import simpson
from torch import Tensor

from ..grid import Grid
from ..pinn.autodiff import grad
from ..solvers import heat1d_crank_nicolson, heat2d_crank_nicolson
from .base import ParamSpec, Problem, to_numpy

PI = float(np.pi)


class Heat1D(Problem):
    """u_t = alpha u_xx on (0, 1) with fixed end temperatures.

    The initial profile is the steady linear profile joining the two end
    temperatures plus a shape that vanishes at both ends, so initial and
    boundary data are compatible.
    """

    name = "heat_1d"
    title = "1D heat equation"
    equation = r"\frac{\partial u}{\partial t} = \alpha\,\frac{\partial^2 u}{\partial x^2}"
    coords = ("x", "t")
    analytical_method = "Fourier sine series (80 modes, coefficients by Simpson quadrature)"
    numerical_method = "Crank-Nicolson finite differences (dx = 1/400, dt = T/1000)"
    eval_shape = (101, 101)
    n_series = 80
    PARAMS = (
        ParamSpec("alpha", "Diffusivity α", "pde", 0.1, min=0.01, max=1.0, step=0.01),
        ParamSpec("t_final", "Final time T", "pde", 1.0, min=0.1, max=2.0, step=0.1),
        ParamSpec("ic", "Initial profile", "initial", "two_modes", kind="choice",
                  choices=("sine", "two_modes", "gaussian")),
        ParamSpec("ic_mode", "Sine mode k (profile 'sine')", "initial", 1, kind="int", min=1, max=4, step=1),
        ParamSpec("ic_amplitude", "Initial amplitude", "initial", 1.0, min=0.1, max=2.0, step=0.1),
        ParamSpec("u_left", "Left temperature u(0, t)", "boundary", 0.0, min=-1.0, max=1.0, step=0.1),
        ParamSpec("u_right", "Right temperature u(1, t)", "boundary", 0.0, min=-1.0, max=1.0, step=0.1),
    )

    def __init__(self, **params):
        super().__init__(**params)
        xs = np.linspace(0.0, 1.0, 4001)
        shape = to_numpy(self._shape, xs[:, None]).ravel()
        k = np.arange(1, self.n_series + 1)
        # b_k = 2 int_0^1 shape(x) sin(k pi x) dx
        self._b = 2 * simpson(shape[None, :] * np.sin(PI * k[:, None] * xs[None, :]), x=xs, axis=1)

    def bounds(self):
        return np.array([0.0, 0.0]), np.array([1.0, self.p["t_final"]])

    def _shape(self, x: Tensor) -> Tensor:
        a, kind = self.p["ic_amplitude"], self.p["ic"]
        if kind == "sine":
            return a * torch.sin(self.p["ic_mode"] * PI * x)
        if kind == "two_modes":
            return a * (torch.sin(PI * x) + 0.5 * torch.sin(3 * PI * x))
        return a * torch.exp(-((x - 0.5) ** 2) / (2 * 0.08**2))  # tails are 3e-9 at the ends

    def _steady(self, x: Tensor) -> Tensor:
        return self.p["u_left"] + (self.p["u_right"] - self.p["u_left"]) * x

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        return du[:, 1:2] - self.p["alpha"] * u_xx

    def initial_value(self, x):
        return self._steady(x[:, 0:1]) + self._shape(x[:, 0:1])

    def boundary_value(self, x):
        return self._steady(x[:, 0:1])

    def exact(self, x):
        xs, t = x[:, 0:1], x[:, 1:2]
        k = torch.arange(1, self.n_series + 1, dtype=x.dtype)[None, :]
        b = torch.as_tensor(self._b, dtype=x.dtype)[None, :]
        series = (b * torch.exp(-self.p["alpha"] * (k * PI) ** 2 * t) * torch.sin(k * PI * xs)).sum(1, keepdim=True)
        return self._steady(xs) + series

    def numerical(self, grid: Grid) -> np.ndarray:
        x, t = grid.axes
        sx, st = 4, 10
        xs = np.linspace(x[0], x[-1], (len(x) - 1) * sx + 1)
        n_steps = (len(t) - 1) * st
        u0 = to_numpy(self.initial_value, np.stack([xs, np.zeros_like(xs)], axis=1)).ravel()
        sol = heat1d_crank_nicolson(u0, self.p["alpha"], xs[1] - xs[0], (t[-1] - t[0]) / n_steps, n_steps,
                                    self.p["u_left"], self.p["u_right"], store_every=st)
        return sol[::sx, :]


class Heat2D(Problem):
    """u_t = alpha (u_xx + u_yy) on the unit square with a constant boundary temperature."""

    name = "heat_2d"
    title = "2D heat equation"
    equation = (r"\frac{\partial u}{\partial t} = \alpha\left(\frac{\partial^2 u}{\partial x^2}"
                r" + \frac{\partial^2 u}{\partial y^2}\right)")
    coords = ("x", "y", "t")
    analytical_method = "Separation of variables (closed form)"
    numerical_method = "Crank-Nicolson, five-point Laplacian, sparse LU (h = 1/160, dt = T/200)"
    eval_shape = (41, 41, 21)
    PARAMS = (
        ParamSpec("alpha", "Diffusivity α", "pde", 0.05, min=0.01, max=0.5, step=0.01),
        ParamSpec("t_final", "Final time T", "pde", 1.0, min=0.1, max=2.0, step=0.1),
        ParamSpec("mode_m", "Initial mode m (x)", "initial", 1, kind="int", min=1, max=3, step=1),
        ParamSpec("mode_n", "Initial mode n (y)", "initial", 1, kind="int", min=1, max=3, step=1),
        ParamSpec("ic_amplitude", "Initial amplitude", "initial", 1.0, min=0.1, max=2.0, step=0.1),
        ParamSpec("u_boundary", "Boundary temperature", "boundary", 0.0, min=-1.0, max=1.0, step=0.1),
    )

    def bounds(self):
        return np.zeros(3), np.array([1.0, 1.0, self.p["t_final"]])

    def _mode(self, x: Tensor) -> Tensor:
        return (self.p["ic_amplitude"] * torch.sin(self.p["mode_m"] * PI * x[:, 0:1])
                * torch.sin(self.p["mode_n"] * PI * x[:, 1:2]))

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        u_yy = grad(du[:, 1:2], x)[:, 1:2]
        return du[:, 2:3] - self.p["alpha"] * (u_xx + u_yy)

    def initial_value(self, x):
        return self.p["u_boundary"] + self._mode(x)

    def boundary_value(self, x):
        return torch.full_like(x[:, 0:1], self.p["u_boundary"])

    def exact(self, x):
        rate = self.p["alpha"] * (self.p["mode_m"] ** 2 + self.p["mode_n"] ** 2) * PI**2
        return self.p["u_boundary"] + torch.exp(-rate * x[:, 2:3]) * self._mode(x)

    def numerical(self, grid: Grid) -> np.ndarray:
        x, y, t = grid.axes
        sx, st = 4, 10
        n = (len(x) - 1) * sx + 1
        xs = np.linspace(x[0], x[-1], n)
        xx, yy = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)
        u0 = to_numpy(self.initial_value, pts).reshape(n, n)
        n_steps = (len(t) - 1) * st
        sol = heat2d_crank_nicolson(u0, self.p["alpha"], xs[1] - xs[0], (t[-1] - t[0]) / n_steps, n_steps,
                                    self.p["u_boundary"], store_every=st)
        return sol[::sx, ::sx, :]
