"""One-dimensional wave equation on a string with fixed ends."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ..grid import Grid
from ..metrics import stats
from ..pinn.autodiff import grad
from ..solvers import wave1d_leapfrog
from .base import ParamSpec, Problem, to_numpy

PI = float(np.pi)


class Wave1D(Problem):
    """u_tt = c^2 u_xx on (0, 1), u = 0 at both ends.

    Initial displacement is a sum of two sine modes; the initial velocity is a
    multiple of the first mode. Both give a closed-form standing-wave solution.
    """

    name = "wave_1d"
    title = "1D wave equation"
    equation = r"\frac{\partial^2 u}{\partial t^2} = c^2\,\frac{\partial^2 u}{\partial x^2}"
    coords = ("x", "t")
    analytical_method = "Standing-wave superposition (closed form)"
    numerical_method = "Explicit leapfrog central differences (dx = 1/1000, Courant number <= 0.5)"
    eval_shape = (101, 101)
    PARAMS = (
        ParamSpec("c", "Wave speed c", "pde", 1.0, min=0.5, max=2.0, step=0.1),
        ParamSpec("t_final", "Final time T", "pde", 1.0, min=0.5, max=2.0, step=0.1),
        ParamSpec("k1", "First mode k₁", "initial", 1, kind="int", min=1, max=3, step=1),
        ParamSpec("a1", "First mode amplitude", "initial", 1.0, min=0.0, max=1.5, step=0.1),
        ParamSpec("k2", "Second mode k₂", "initial", 2, kind="int", min=1, max=4, step=1),
        ParamSpec("a2", "Second mode amplitude", "initial", 0.5, min=0.0, max=1.5, step=0.1),
        ParamSpec("v_amplitude", "Initial velocity amplitude", "initial", 0.0, min=-3.0, max=3.0, step=0.5),
    )

    def bounds(self):
        return np.array([0.0, 0.0]), np.array([1.0, self.p["t_final"]])

    def residual(self, u_fn, x):
        u = u_fn(x)
        du = grad(u, x)
        u_xx = grad(du[:, 0:1], x)[:, 0:1]
        u_tt = grad(du[:, 1:2], x)[:, 1:2]
        return u_tt - self.p["c"] ** 2 * u_xx

    def initial_value(self, x):
        xs = x[:, 0:1]
        return (self.p["a1"] * torch.sin(self.p["k1"] * PI * xs)
                + self.p["a2"] * torch.sin(self.p["k2"] * PI * xs))

    def initial_velocity(self, x: Tensor) -> Tensor:
        return self.p["v_amplitude"] * torch.sin(self.p["k1"] * PI * x[:, 0:1])

    def boundary_value(self, x):
        return torch.zeros_like(x[:, 0:1])

    def exact(self, x):
        xs, t = x[:, 0:1], x[:, 1:2]
        c, k1, k2 = self.p["c"], self.p["k1"], self.p["k2"]
        w1, w2 = c * k1 * PI, c * k2 * PI
        return (torch.sin(k1 * PI * xs) * (self.p["a1"] * torch.cos(w1 * t)
                                           + self.p["v_amplitude"] / w1 * torch.sin(w1 * t))
                + self.p["a2"] * torch.sin(k2 * PI * xs) * torch.cos(w2 * t))

    def losses(self, model, pts):
        out = super().losses(model, pts)
        xi = pts["initial"]
        u_t = grad(model(xi), xi)[:, 1:2]
        out["initial_velocity"] = (u_t - self.initial_velocity(xi)).pow(2).mean()
        return out

    def loss_terms(self):
        return super().loss_terms() + ["initial_velocity"]

    def numerical(self, grid: Grid) -> np.ndarray:
        x, t = grid.axes
        sx = 10
        xs = np.linspace(x[0], x[-1], (len(x) - 1) * sx + 1)
        dx = xs[1] - xs[0]
        t_span = t[-1] - t[0]
        st = math.ceil(self.p["c"] * t_span / (0.5 * dx) / (len(t) - 1))
        n_steps = (len(t) - 1) * st
        pts = np.stack([xs, np.zeros_like(xs)], axis=1)
        sol = wave1d_leapfrog(to_numpy(self.initial_value, pts).ravel(),
                              to_numpy(self.initial_velocity, pts).ravel(),
                              self.p["c"], dx, t_span / n_steps, n_steps, store_every=st)
        return sol[::sx, :]

    def extra_condition_errors(self, model, grid: Grid):
        x = grid.axes[0]
        pts = torch.tensor(np.stack([x, np.zeros_like(x)], axis=1), requires_grad=True)
        u_t = grad(model(pts), pts)[:, 1:2]
        err = (u_t - self.initial_velocity(pts)).detach().numpy()
        return {"initial_velocity": stats(err)}
