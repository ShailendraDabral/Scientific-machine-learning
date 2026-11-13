"""Reference solvers and analytical solutions are checked against each other."""
import numpy as np
import pytest
import torch

from neuralphysics.metrics import error_stats
from neuralphysics.pinn import residual_on_grid
from neuralphysics.problems import REGISTRY, Problem, make_problem
from neuralphysics.solvers import heat1d_crank_nicolson, wave1d_leapfrog, poisson2d_dirichlet, burgers_etdrk4


@pytest.mark.parametrize("name", [n for n, c in REGISTRY.items() if c.exact is not Problem.exact])
def test_exact_solution_satisfies_pde(name):
    """Autodiff residual of the closed-form solution vanishes to round-off."""
    problem = make_problem(name)
    res = residual_on_grid(problem, problem.exact, problem.eval_grid())
    scale = np.abs(problem.exact_on_grid(problem.eval_grid())).max() * 100
    assert np.abs(res).max() < 1e-9 * scale


@pytest.mark.parametrize("name,tol", [("heat_1d", 1e-5), ("heat_2d", 1e-4), ("wave_1d", 1e-4),
                                      ("burgers_1d", 1e-4), ("poisson_2d", 1e-4)])
def test_numerical_matches_analytical(name, tol):
    problem = make_problem(name)
    grid = problem.eval_grid()
    err = error_stats(problem.numerical(grid), problem.exact_on_grid(grid))
    assert err["rel_l2"] < tol, err


def test_heat_general_boundary_and_profile():
    problem = make_problem("heat_1d", ic="gaussian", u_left=0.5, u_right=-0.3, alpha=0.2)
    grid = problem.eval_grid()
    assert error_stats(problem.numerical(grid), problem.exact_on_grid(grid))["rel_l2"] < 1e-4


def _order(errors):
    return np.log2(errors[0] / errors[1])


def test_crank_nicolson_second_order():
    errs = []
    for n in (20, 40):
        x = np.linspace(0, 1, n + 1)
        sol = heat1d_crank_nicolson(np.sin(np.pi * x), 0.1, 1 / n, 1 / n, n)
        errs.append(np.abs(sol[:, -1] - np.exp(-0.1 * np.pi**2) * np.sin(np.pi * x)).max())
    assert 1.8 < _order(errs) < 2.2


def test_leapfrog_second_order():
    errs = []
    for n in (40, 80):
        x = np.linspace(0, 1, n + 1)
        steps = round(0.8 / (0.5 / n))  # T = 0.8: away from the turning point, where the error is not cancelled
        sol = wave1d_leapfrog(np.sin(2 * np.pi * x), np.zeros(n + 1), 1.0, 1 / n, 0.5 / n, steps)
        errs.append(np.abs(sol[:, -1] - np.sin(2 * np.pi * x) * np.cos(2 * np.pi * 0.8)).max())
    assert 1.8 < _order(errs) < 2.2


def test_poisson_second_order():
    errs = []
    for n in (16, 32):
        x = np.linspace(0, 1, n + 1)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        exact = np.sin(np.pi * xx) * np.sin(2 * np.pi * yy) + 0.3 * (xx**2 - yy**2)
        f = -5 * np.pi**2 * np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)
        errs.append(np.abs(poisson2d_dirichlet(f, exact, 1 / n) - exact).max())
    assert 1.8 < _order(errs) < 2.2


def test_etdrk4_fourth_order_in_time():
    n = 128
    x = np.arange(n) / n
    u0 = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    ref = burgers_etdrk4(u0, 0.05, 1.0, 1 / 3200, 1600)[-1]
    errs = [np.abs(burgers_etdrk4(u0, 0.05, 1.0, 0.5 / s, s)[-1] - ref).max() for s in (40, 80)]
    assert 3.5 < _order(errs) < 4.5


def test_burgers_travelling_wave_residual():
    """The Burgers residual code is exact on the travelling-wave solution."""
    problem = make_problem("burgers_1d", nu=0.05)
    nu, a, c = 0.05, 0.8, 0.3
    exact = lambda x: c - a * torch.tanh(a * (x[:, 0:1] - c * x[:, 1:2]) / (2 * nu))
    res = residual_on_grid(problem, exact, problem.eval_grid())
    assert np.abs(res).max() < 1e-10
