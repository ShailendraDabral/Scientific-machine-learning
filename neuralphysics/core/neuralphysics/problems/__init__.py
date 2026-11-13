"""Catalogue of differential equations."""
from __future__ import annotations

from typing import Any

from .base import ParamSpec, Problem
from .burgers import Burgers1D, BurgersPeriodic
from .heat import Heat1D, Heat2D
from .poisson import Poisson2D
from .wave import Wave1D

REGISTRY: dict[str, type[Problem]] = {
    cls.name: cls for cls in (Heat1D, Heat2D, Wave1D, Burgers1D, Poisson2D)
}


def make_problem(name: str, **params: Any) -> Problem:
    """Instantiate a catalogue problem by name with validated parameters."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown problem {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


__all__ = ["REGISTRY", "make_problem", "ParamSpec", "Problem", "Heat1D", "Heat2D", "Wave1D",
           "Burgers1D", "BurgersPeriodic", "Poisson2D"]
