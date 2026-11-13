"""Exact solution of the viscous Burgers equation through the Cole-Hopf transform.

For u_t + u u_x = nu u_xx on the real line with initial data u0,

    u(x, t) = int u0(y) K(x, y, t) dy / int K(x, y, t) dy,
    K = exp( -(x - y)^2 / (4 nu t) - Phi(y) / (2 nu) ),   Phi(y) = int_0^y u0(s) ds.

The integrals are analytic in y and decay like a Gaussian, so the trapezoid rule
on a truncated interval converges exponentially. Exponents are shifted by their
maximum before exponentiation to avoid overflow at small viscosity.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x and 2.x


def burgers_cole_hopf(
    x: np.ndarray,
    t: np.ndarray,
    nu: float,
    u0: Callable[[np.ndarray], np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
    phi_range: float,
    n_quad: int = 2001,
) -> np.ndarray:
    """Evaluate the exact solution on the tensor grid x by t. Returns (len(x), len(t)).

    ``phi_range`` is max(Phi) - min(Phi); it sets how far the quadrature window
    must reach for the neglected tails to be below 1e-17 of the peak.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((len(x), len(t)))
    for j, tj in enumerate(np.asarray(t, dtype=np.float64)):
        if tj <= 0.0:
            out[:, j] = u0(x)
            continue
        half_width = np.sqrt(4 * nu * tj * (phi_range / (2 * nu) + 40.0))
        eta = np.linspace(-half_width, half_width, n_quad)
        y = x[:, None] - eta[None, :]
        expo = -(eta[None, :] ** 2) / (4 * nu * tj) - phi(y) / (2 * nu)
        expo -= expo.max(axis=1, keepdims=True)
        w = np.exp(expo)
        out[:, j] = _trapezoid(u0(y) * w, eta, axis=1) / _trapezoid(w, eta, axis=1)
    return out
