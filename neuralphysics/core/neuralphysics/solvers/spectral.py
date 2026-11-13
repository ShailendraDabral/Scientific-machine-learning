"""Fourier pseudo-spectral solver for the periodic viscous Burgers equation.

Time stepping is ETDRK4 (Cox and Matthews), with the coefficients evaluated by
contour integration as in Kassam and Trefethen (2005) to avoid cancellation.
"""
from __future__ import annotations

import numpy as np


def burgers_etdrk4(
    u0: np.ndarray,
    nu: float,
    length: float,
    dt: float,
    n_steps: int,
    store_every: int = 1,
    dealias: bool = True,
) -> np.ndarray:
    """Solve u_t + u u_x = nu u_xx with period ``length``.

    ``u0`` has shape (..., N): samples on the uniform periodic grid (right
    end point excluded). Leading axes are a batch. Returns (..., n_store, N).
    """
    u0 = np.asarray(u0, dtype=np.float64)
    n = u0.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=length / n)  # angular wavenumbers
    lin = -nu * k**2
    e_full = np.exp(dt * lin)
    e_half = np.exp(dt * lin / 2)

    m = 32  # contour points
    roots = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * lin[:, None] + roots[None, :]
    q = dt * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
    f1 = dt * np.real(np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3, axis=1))
    f2 = dt * np.real(np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * lr - lr**2 + np.exp(lr) * (4 - lr)) / lr**3, axis=1))

    g = -0.5j * k
    mask = np.ones_like(k)
    if dealias:
        mask[k > (2.0 / 3.0) * k.max()] = 0.0  # 2/3 rule

    def nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(v_hat * mask, n=n, axis=-1)
        return g * mask * np.fft.rfft(u * u, axis=-1)

    v = np.fft.rfft(u0, axis=-1)
    out = [u0.copy()]
    for step in range(1, n_steps + 1):
        nv = nonlinear(v)
        a = e_half * v + q * nv
        na = nonlinear(a)
        b = e_half * v + q * na
        nb = nonlinear(b)
        c = e_half * a + q * (2 * nb - nv)
        nc = nonlinear(c)
        v = e_full * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
        if step % store_every == 0:
            out.append(np.fft.irfft(v, n=n, axis=-1))
    return np.stack(out, axis=-2)
