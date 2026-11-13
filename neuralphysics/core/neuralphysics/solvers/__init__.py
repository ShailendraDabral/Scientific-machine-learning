"""Reference solvers: conventional numerics that involve no neural network."""
from .cole_hopf import burgers_cole_hopf
from .finite_difference import (
    heat1d_crank_nicolson,
    heat2d_crank_nicolson,
    poisson2d_dirichlet,
    wave1d_leapfrog,
)
from .spectral import burgers_etdrk4

__all__ = [
    "burgers_cole_hopf",
    "burgers_etdrk4",
    "heat1d_crank_nicolson",
    "heat2d_crank_nicolson",
    "poisson2d_dirichlet",
    "wave1d_leapfrog",
]
