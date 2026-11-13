"""Conventional finite-difference solvers (NumPy and SciPy only).

Every routine is second-order accurate in space and time. Each returns the
solution on the solver grid at the stored time levels, so callers can subsample
it onto an evaluation grid without interpolation.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu, spsolve


def heat1d_crank_nicolson(
    u0: np.ndarray,
    alpha: float,
    dx: float,
    dt: float,
    n_steps: int,
    left: float = 0.0,
    right: float = 0.0,
    store_every: int = 1,
) -> np.ndarray:
    """Crank-Nicolson for u_t = alpha u_xx with Dirichlet values ``left`` and ``right``.

    Returns an array of shape (nx, n_steps // store_every + 1).
    """
    u = np.array(u0, dtype=np.float64)
    u[0], u[-1] = left, right
    n = len(u) - 2
    r = alpha * dt / dx**2
    # (I - r/2 D) u^{n+1} = (I + r/2 D) u^n, D = tridiag(1, -2, 1)
    ab = np.zeros((3, n))
    ab[0, 1:] = -r / 2
    ab[1, :] = 1 + r
    ab[2, :-1] = -r / 2
    out = [u.copy()]
    for step in range(1, n_steps + 1):
        rhs = u[1:-1] + (r / 2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        rhs[0] += (r / 2) * left
        rhs[-1] += (r / 2) * right
        u[1:-1] = solve_banded((1, 1), ab, rhs)
        if step % store_every == 0:
            out.append(u.copy())
    return np.stack(out, axis=1)


def _laplacian_2d(n: int, h: float) -> sp.csc_matrix:
    """Five-point Laplacian on an n x n interior grid (C-ordered unknowns)."""
    d = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / h**2
    eye = sp.identity(n)
    return (sp.kron(d, eye) + sp.kron(eye, d)).tocsc()


def heat2d_crank_nicolson(
    u0: np.ndarray,
    alpha: float,
    h: float,
    dt: float,
    n_steps: int,
    boundary: float = 0.0,
    store_every: int = 1,
) -> np.ndarray:
    """Crank-Nicolson for u_t = alpha (u_xx + u_yy) on a square, constant Dirichlet value.

    ``u0`` has shape (n, n) including boundary nodes. Returns (n, n, n_store).
    The implicit system is factorised once with a sparse LU decomposition.
    """
    u = np.array(u0, dtype=np.float64)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = boundary
    n = u.shape[0] - 2
    lap = _laplacian_2d(n, h)
    eye = sp.identity(n * n, format="csc")
    lhs = splu((eye - (alpha * dt / 2) * lap).tocsc())
    rhs_op = (eye + (alpha * dt / 2) * lap).tocsr()
    # Work with w = u - boundary, which has homogeneous Dirichlet data.
    w = (u[1:-1, 1:-1] - boundary).ravel()
    out = [u.copy()]
    for step in range(1, n_steps + 1):
        w = lhs.solve(rhs_op @ w)
        if step % store_every == 0:
            u = np.full_like(u, boundary)
            u[1:-1, 1:-1] = w.reshape(n, n) + boundary
            out.append(u)
    return np.stack(out, axis=2)


def wave1d_leapfrog(
    u0: np.ndarray,
    v0: np.ndarray,
    c: float,
    dx: float,
    dt: float,
    n_steps: int,
    store_every: int = 1,
) -> np.ndarray:
    """Explicit central-difference (leapfrog) scheme for u_tt = c^2 u_xx, fixed ends.

    Stable for the Courant number c dt / dx <= 1. Returns (nx, n_store).
    """
    r2 = (c * dt / dx) ** 2
    if r2 > 1.0 + 1e-12:
        raise ValueError(f"CFL violated: (c dt/dx)^2 = {r2:.3f} > 1")
    prev = np.array(u0, dtype=np.float64)
    prev[0] = prev[-1] = 0.0
    lap = np.zeros_like(prev)
    lap[1:-1] = prev[2:] - 2 * prev[1:-1] + prev[:-2]
    # Second-order start from the Taylor expansion with u_tt = c^2 u_xx.
    cur = prev + dt * np.asarray(v0, dtype=np.float64) + 0.5 * r2 * lap
    cur[0] = cur[-1] = 0.0
    out = [prev.copy()]
    if store_every == 1:
        out.append(cur.copy())
    for step in range(2, n_steps + 1):
        nxt = np.zeros_like(cur)
        nxt[1:-1] = 2 * cur[1:-1] - prev[1:-1] + r2 * (cur[2:] - 2 * cur[1:-1] + cur[:-2])
        prev, cur = cur, nxt
        if step % store_every == 0:
            out.append(cur.copy())
    return np.stack(out, axis=1)


def poisson2d_dirichlet(f: np.ndarray, g: np.ndarray, h: float) -> np.ndarray:
    """Five-point solve of Laplacian(u) = f on a square with Dirichlet data.

    ``f`` and ``g`` have shape (n, n) on the full grid; only the boundary entries
    of ``g`` are used. Returns u on the full grid.
    """
    u = np.array(g, dtype=np.float64)
    n = u.shape[0] - 2
    rhs = np.array(f[1:-1, 1:-1], dtype=np.float64)
    rhs[0, :] -= u[0, 1:-1] / h**2
    rhs[-1, :] -= u[-1, 1:-1] / h**2
    rhs[:, 0] -= u[1:-1, 0] / h**2
    rhs[:, -1] -= u[1:-1, -1] / h**2
    u[1:-1, 1:-1] = spsolve(_laplacian_2d(n, h), rhs.ravel()).reshape(n, n)
    return u
