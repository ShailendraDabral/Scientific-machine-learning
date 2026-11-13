"""Neural operator architectures: DeepONet and the Fourier Neural Operator.

Both map an input function sampled on nx points to the solution on an
(nx, nt) space-time grid. A physics-informed neural operator (PINO) is the FNO
below trained with the additional PDE-residual loss in ``physics.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..pinn.networks import MLP

TWO_PI = 2 * math.pi


class DeepONet(nn.Module):
    """G(u)(y) = sum_k b_k(u(x_1..x_m)) t_k(y) + b_0   (Lu et al. 2021).

    The branch network encodes the input function from its sensor values; the
    trunk network encodes the query coordinate y = (x, t). The trunk sees
    (cos 2 pi x, sin 2 pi x, t) so that predictions are periodic in x.
    """

    def __init__(self, n_sensors: int, width: int = 128, depth: int = 3, basis: int = 64,
                 activation: str = "gelu", t_final: float = 1.0):
        super().__init__()
        self.branch = MLP(n_sensors, [width] * depth, basis, activation)
        self.trunk = MLP(3, [width] * depth, basis, activation)
        self.bias = nn.Parameter(torch.zeros(1))
        self.t_final = t_final

    def trunk_features(self, x: Tensor, t: Tensor) -> Tensor:
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        return torch.stack([torch.cos(TWO_PI * xx), torch.sin(TWO_PI * xx), 2 * tt / self.t_final - 1],
                           dim=-1).reshape(-1, 3)

    def forward(self, u0: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """u0: (B, nx) sensor values. Returns (B, len(x), len(t)); x and t may be any query grid."""
        b = self.branch(u0)  # (B, p)
        tr = torch.tanh(self.trunk(self.trunk_features(x, t)))  # (P, p)
        return (b @ tr.T + self.bias).reshape(u0.shape[0], len(x), len(t))


class SpectralConv2d(nn.Module):
    """Fourier layer kernel: multiply the lowest Fourier modes by learned complex weights.

    (K v)(x) = F^-1( R . F(v) )(x), with R of shape (in, out, modes_x, modes_t)
    for the positive and the negative x-frequencies. The weights live in
    Fourier space, not on a grid, which is what makes the layer independent of
    the sampling resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_t: int):
        super().__init__()
        self.out_channels, self.modes_x, self.modes_t = out_channels, modes_x, modes_t
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_t)
        self.w_pos = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def forward(self, v: Tensor) -> Tensor:
        nx, nt = v.shape[-2:]
        mx, mt = min(self.modes_x, nx // 2), min(self.modes_t, nt // 2 + 1)
        v_hat = torch.fft.rfft2(v, norm="forward")  # 'forward' norm keeps amplitudes resolution-independent
        out = torch.zeros(v.shape[0], self.out_channels, nx, nt // 2 + 1, dtype=torch.cfloat, device=v.device)
        out[:, :, :mx, :mt] = torch.einsum("bixy,ioxy->boxy", v_hat[:, :, :mx, :mt], self.w_pos[:, :, :mx, :mt])
        out[:, :, -mx:, :mt] = torch.einsum("bixy,ioxy->boxy", v_hat[:, :, -mx:, :mt], self.w_neg[:, :, :mx, :mt])
        return torch.fft.irfft2(out, s=(nx, nt), norm="forward")


class FNO2d(nn.Module):
    """Fourier Neural Operator on the space-time grid (Li et al. 2021).

    v_0 = P(u0(x), cos 2 pi x, sin 2 pi x, t)
    v_{l+1} = sigma( W_l v_l + F^-1( R_l . F(v_l) ) )
    u = Q(v_L)

    Time is not periodic, so the time axis is zero-padded before the Fourier
    layers and cropped afterwards.
    """

    def __init__(self, modes_x: int = 10, modes_t: int = 6, width: int = 20, n_layers: int = 4,
                 t_final: float = 1.0, pad_fraction: float = 0.25):
        super().__init__()
        self.t_final, self.pad_fraction = t_final, pad_fraction
        self.lift = nn.Linear(4, width)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes_x, modes_t) for _ in range(n_layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.project = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, u0: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """u0: (B, nx) on the grid ``x``. Returns (B, nx, len(t))."""
        batch, nx, nt = u0.shape[0], len(x), len(t)
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        chans = torch.stack([u0[:, :, None].expand(batch, nx, nt),
                             torch.cos(TWO_PI * xx).expand(batch, nx, nt),
                             torch.sin(TWO_PI * xx).expand(batch, nx, nt),
                             (2 * tt / self.t_final - 1).expand(batch, nx, nt)], dim=-1)
        v = self.lift(chans).permute(0, 3, 1, 2)  # (B, C, nx, nt)
        pad = int(round(self.pad_fraction * nt))
        v = F.pad(v, (0, pad))
        for i, (spec, loc) in enumerate(zip(self.spectral, self.local)):
            v = spec(v) + loc(v)
            if i < len(self.spectral) - 1:
                v = F.gelu(v)
        v = v[..., :nt].permute(0, 2, 3, 1)
        return self.project(v).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Number of real-valued trainable parameters (complex weights count twice)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
