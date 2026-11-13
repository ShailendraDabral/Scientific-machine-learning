"""Tensor-product evaluation grids."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A tensor-product grid. Field arrays are indexed in the order of ``names``."""

    names: tuple[str, ...]
    axes: tuple[np.ndarray, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    def points(self) -> np.ndarray:
        """All grid points as an (N, d) array, C-ordered to match ``reshape(shape)``."""
        mesh = np.meshgrid(*self.axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def axis(self, name: str) -> np.ndarray:
        return self.axes[self.names.index(name)]
