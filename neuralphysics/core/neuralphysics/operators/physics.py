"""PDE residual of a gridded space-time field, used as the physics loss of PINO.

For a neural operator the output is a field on a grid, so derivatives are taken
on the grid: exactly in x with the Fourier transform (the problem is periodic)
and with fourth-order central differences in t.

The residual is evaluated on time levels 3 .. nt-3. Levels whose stencil would
touch t = 0 are excluded on purpose: the high Fourier modes of the initial
condition decay faster than one time step, so a finite difference across the
first levels measures that unresolved transient and not a violation of the PDE.
The initial condition is constrained separately. The residual of the
*reference* data under this discretisation is reported with every experiment as
the floor below which a physics residual is indistinguishable from
discretisation error.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def burgers_residual_grid(u: Tensor, nu: float, dt: float) -> Tensor:
    """Residual u_t + u u_x - nu u_xx for u of shape (B, nx, nt). Returns (B, nx, nt - 5)."""
    nx = u.shape[1]
    k = 2 * math.pi * torch.fft.fftfreq(nx, d=1.0 / nx, device=u.device).to(u.dtype)
    k_odd = k.clone()
    if nx % 2 == 0:
        k_odd[nx // 2] = 0.0  # the Nyquist mode has no well-defined odd derivative
    u_hat = torch.fft.fft(u, dim=1)
    u_x = torch.fft.ifft(1j * k_odd[None, :, None] * u_hat, dim=1).real
    u_xx = torch.fft.ifft(-(k[None, :, None] ** 2) * u_hat, dim=1).real
    u_t = (-u[:, :, 5:] + 8 * u[:, :, 4:-1] - 8 * u[:, :, 2:-3] + u[:, :, 1:-4]) / (12 * dt)
    return u_t + (u * u_x - nu * u_xx)[:, :, 3:-2]
