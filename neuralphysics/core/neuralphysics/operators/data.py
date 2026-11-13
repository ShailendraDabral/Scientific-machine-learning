"""Operator-learning datasets for the periodic viscous Burgers equation.

The learned operator maps an initial condition to the whole space-time solution,

    G : u0(x)  ->  u(x, t),   x in [0, 1) periodic,  t in [0, T].

Inputs are band-limited Gaussian random fields. Outputs come from the
pseudo-spectral ETDRK4 reference solver, never from a network.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..solvers import burgers_etdrk4


@dataclass(frozen=True)
class OperatorDataConfig:
    n_train: int = 256
    n_test: int = 64
    n_ood: int = 64
    nx: int = 64  # spatial points, periodic (right end excluded)
    nt: int = 32  # time levels including t = 0 and t = T
    nu: float = 0.03
    t_final: float = 1.0
    length_scale: float = 0.10  # correlation length of training and test inputs
    ood_length_scale: float = 0.06  # rougher inputs for the out-of-distribution set
    amplitude: float = 0.6  # standard deviation of the inputs
    max_mode: int = 12  # band limit of the inputs
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_grf(n: int, nx: int, length_scale: float, amplitude: float, max_mode: int,
               rng: np.random.Generator) -> np.ndarray:
    """Zero-mean periodic Gaussian random field with a squared-exponential spectrum.

    u0(x) = sum_k s_k (a_k cos 2 pi k x + b_k sin 2 pi k x),  a_k, b_k ~ N(0, 1),
    s_k^2 proportional to exp(-(2 pi k l)^2 / 2), normalised so that Var[u0] = amplitude^2.
    """
    k = np.arange(1, max_mode + 1)
    s2 = np.exp(-0.5 * (2 * np.pi * k * length_scale) ** 2)
    s = amplitude * np.sqrt(s2 / s2.sum())
    x = np.arange(nx) / nx
    a = rng.standard_normal((n, max_mode)) * s
    b = rng.standard_normal((n, max_mode)) * s
    phase = 2 * np.pi * k[None, :] * x[:, None]  # (nx, K)
    return a @ np.cos(phase).T + b @ np.sin(phase).T


def solve_burgers(u0: np.ndarray, cfg: OperatorDataConfig, refine: int = 4, max_dt: float = 1e-3) -> np.ndarray:
    """Reference solution for inputs of shape (n, nx); returns (n, nx, nt)."""
    n, nx = u0.shape
    # Band-limited inputs are upsampled exactly by zero-padding their spectrum.
    fine = np.fft.irfft(np.fft.rfft(u0, axis=1), n=nx * refine, axis=1) * refine
    store = math.ceil(cfg.t_final / (cfg.nt - 1) / max_dt)
    n_steps = (cfg.nt - 1) * store
    sol = burgers_etdrk4(fine, cfg.nu, 1.0, cfg.t_final / n_steps, n_steps, store_every=store)
    return np.transpose(sol[:, :, ::refine], (0, 2, 1))


@dataclass
class OperatorDataset:
    config: OperatorDataConfig
    x: np.ndarray
    t: np.ndarray
    u0: dict[str, np.ndarray]  # split -> (n, nx)
    u: dict[str, np.ndarray]  # split -> (n, nx, nt)


@lru_cache(maxsize=4)
def generate_dataset(cfg: OperatorDataConfig) -> OperatorDataset:
    """Train, test and out-of-distribution splits from independent random streams."""
    rngs = np.random.default_rng(cfg.seed).spawn(3)
    sizes = {"train": cfg.n_train, "test": cfg.n_test, "ood": cfg.n_ood}
    scales = {"train": cfg.length_scale, "test": cfg.length_scale, "ood": cfg.ood_length_scale}
    u0 = {s: sample_grf(sizes[s], cfg.nx, scales[s], cfg.amplitude, cfg.max_mode, rng)
          for s, rng in zip(sizes, rngs)}
    u = {s: solve_burgers(u0[s], cfg) for s in sizes}
    return OperatorDataset(cfg, np.arange(cfg.nx) / cfg.nx, np.linspace(0.0, cfg.t_final, cfg.nt), u0, u)
