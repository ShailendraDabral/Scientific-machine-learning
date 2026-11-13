"""Viscous Burgers equation: the Dirichlet benchmark and a periodic variant."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid
from ..pinn.autodiff import grad
from ..solvers import burgers_cole_hopf, burgers_etdrk4
from .base import ParamSpec, Problem

PI = float(np.pi)


class Burgers1D(Problem):
    """u_t + u u_x = nu u_xx on (-1, 1), u(x, 0) = -A sin(pi x), u(+-1, t) = 0.

    The initial condition is odd and 2-periodic and the equation preserves both
    properties, so the solution of the periodic problem vanishes at x = +-1 and
    is also the solution of this Dirichlet problem. That is what allows a
    periodic spectral solver and the whole-line Cole-Hopf formula to be used.
    """

    name = "burgers_1d"
    title = "Viscous Burgers equation"
    equation = (r"\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}"
                r" = \nu\,\frac{\partial^2 u}{\partial x^2}")
    coords = ("x", "t")
    analytical_method = "Cole-Hopf transform, integrals by trapezoid quadrature (2001 nodes)"
    numerical_method = "Fourier pseudo-spectral (2048 modes, 2/3 dealiasing) with ETDRK4, dt = 5e-4"
    eval_shape = (257, 101)
    PARAMS = (
        ParamSpec("nu", "Viscosity ν", "pde", 0.01 / PI, min=0.003, max=0.1, step=0.001),
        ParamSpec("ic_amplitude", "Initial amplitude A", "initial", 1.0, min=0.2, max=1.0, step=0.1),
    )

    def bounds(self):
        return np.array([-1.0, 0.0]), np.array([1.0, 1.0])

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        return du[:, 1:2] + u * du[:, 0:1] - self.p["nu"] * u_xx

    def initial_value(self, x):
        return -self.p["ic_amplitude"] * torch.sin(PI * x[:, 0:1])

    def boundary_value(self, x):
        return torch.zeros_like(x[:, 0:1])

    def exact_on_grid(self, grid: Grid) -> np.ndarray:
        a = self.p["ic_amplitude"]
        return burgers_cole_hopf(
            grid.axes[0], grid.axes[1], self.p["nu"],
            u0=lambda y: -a * np.sin(PI * y),
            phi=lambda y: a * (np.cos(PI * y) - 1.0) / PI,
            phi_range=2 * a / PI,
        )

    def numerical(self, grid: Grid) -> np.ndarray:
        x, t = grid.axes
        sx, st = 8, 20
        n = (len(x) - 1) * sx
        xs = -1.0 + 2.0 * np.arange(n) / n
        n_steps = (len(t) - 1) * st
        sol = burgers_etdrk4(-self.p["ic_amplitude"] * np.sin(PI * xs), self.p["nu"], 2.0,
                             (t[-1] - t[0]) / n_steps, n_steps, store_every=st)  # (n_store, n)
        sol = sol[:, ::sx].T
        return np.concatenate([sol, sol[:1, :]], axis=0)  # x = 1 by periodicity


class BurgersPeriodic(Problem):
    """u_t + u u_x = nu u_xx on the periodic unit interval with an arbitrary band-limited u0.

    Used to compare a per-instance PINN with neural operators on identical
    inputs. The initial condition is given by samples on a uniform grid and is
    evaluated exactly anywhere through its Fourier series. The network input is
    (cos 2 pi x, sin 2 pi x, t), which makes every prediction exactly periodic,
    so no boundary loss is needed.
    """

    name = "burgers_periodic"
    title = "Periodic Burgers equation"
    equation = Burgers1D.equation
    coords = ("x", "t")
    has_boundary = False
    feature_dim = 3
    analytical_method = None
    numerical_method = "Fourier pseudo-spectral with ETDRK4"
    PARAMS = (
        ParamSpec("nu", "Viscosity ν", "pde", 0.05, min=0.005, max=0.5, step=0.005),
        ParamSpec("t_final", "Final time T", "pde", 1.0, min=0.1, max=2.0, step=0.1),
    )

    def __init__(self, u0_samples: np.ndarray, **params):
        super().__init__(**params)
        self.u0_samples = np.asarray(u0_samples, dtype=np.float64)
        n = len(self.u0_samples)
        coeff = np.fft.rfft(self.u0_samples) / n
        self._k = torch.arange(len(coeff), dtype=torch.float64)
        self._re = torch.as_tensor(coeff.real.copy())
        self._im = torch.as_tensor(coeff.imag.copy())
        self._weight = torch.full((len(coeff),), 2.0, dtype=torch.float64)
        self._weight[0] = 1.0
        if n % 2 == 0:
            self._weight[-1] = 1.0

    def bounds(self):
        return np.array([0.0, 0.0]), np.array([1.0, self.p["t_final"]])

    def features(self, x):
        return torch.cat([torch.cos(2 * PI * x[:, 0:1]), torch.sin(2 * PI * x[:, 0:1]),
                          2 * x[:, 1:2] / self.p["t_final"] - 1], dim=1)

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        return du[:, 1:2] + u * du[:, 0:1] - self.p["nu"] * u_xx

    def initial_value(self, x: Tensor) -> Tensor:
        ang = 2 * PI * x[:, 0:1].to(torch.float64) * self._k[None, :]
        val = (self._weight * (self._re * torch.cos(ang) - self._im * torch.sin(ang))).sum(1, keepdim=True)
        return val.to(x.dtype)
