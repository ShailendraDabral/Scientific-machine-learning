"""Problem definition shared by every differential equation in the catalogue.

A ``Problem`` owns the mathematics of one equation: its domain, PDE residual,
initial and boundary data, analytical solution (where one exists) and a
conventional numerical solver. Physics-informed training and all error metrics
are built on this interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import qmc
from torch import Tensor

from ..grid import Grid
from ..metrics import stats


@dataclass(frozen=True)
class ParamSpec:
    """One user-adjustable parameter. ``group`` is 'pde', 'initial' or 'boundary'."""

    name: str
    label: str
    group: str
    default: Any
    kind: str = "float"  # 'float' | 'int' | 'choice'
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()

    def validate(self, value: Any) -> Any:
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.name} must be one of {self.choices}, got {value!r}")
            return value
        value = int(value) if self.kind == "int" else float(value)
        if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
            raise ValueError(f"{self.name}={value} is outside [{self.min}, {self.max}]")
        return value


def to_numpy(fn: Callable[[Tensor], Tensor], points: np.ndarray) -> np.ndarray:
    """Evaluate a torch function on NumPy points in float64."""
    with torch.no_grad():
        return fn(torch.as_tensor(points, dtype=torch.float64)).numpy()


class Problem:
    """Base class. Subclasses define the equation; this class supplies the generic parts."""

    name = ""
    title = ""
    equation = ""  # LaTeX
    coords: tuple[str, ...] = ()
    time_dependent = True  # if True the last coordinate is time
    has_boundary = True  # False for periodic problems whose network is periodic by construction
    analytical_method: str | None = None
    numerical_method = ""
    eval_shape: tuple[int, ...] = ()
    PARAMS: tuple[ParamSpec, ...] = ()

    def __init__(self, **params: Any):
        specs = {s.name: s for s in self.PARAMS}
        unknown = set(params) - set(specs)
        if unknown:
            raise ValueError(f"Unknown parameters for {self.name}: {sorted(unknown)}")
        self.p = {n: s.validate(params.get(n, s.default)) for n, s in specs.items()}

    # ----- domain ---------------------------------------------------------
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return len(self.coords)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(self.dim - 1 if self.time_dependent else self.dim))

    feature_dim: int | None = None  # None means: use the normalised coordinates

    def features(self, x: Tensor) -> Tensor:
        """Network input features. Default: coordinates scaled to [-1, 1]."""
        lo, hi = self.bounds()
        lo_t = torch.as_tensor(lo, dtype=x.dtype)
        hi_t = torch.as_tensor(hi, dtype=x.dtype)
        return 2 * (x - lo_t) / (hi_t - lo_t) - 1

    # ----- mathematics (torch, differentiable) -----------------------------
    def residual(self, u_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
        """PDE residual at points ``x`` (requires_grad must be set). Shape (N, 1)."""
        raise NotImplementedError

    def initial_value(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def boundary_value(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def exact(self, x: Tensor) -> Tensor | None:
        """Analytical solution, or None when the equation has none in closed form."""
        return None

    # ----- physics-informed loss -------------------------------------------
    def losses(self, model: Callable[[Tensor], Tensor], pts: dict[str, Tensor]) -> dict[str, Tensor]:
        """Mean squared PDE residual and condition misfits, one entry per loss term."""
        out = {"pde": self.residual(model, pts["interior"]).pow(2).mean()}
        if "boundary" in pts:
            xb = pts["boundary"]
            out["boundary"] = (model(xb) - self.boundary_value(xb)).pow(2).mean()
        if "initial" in pts:
            xi = pts["initial"]
            out["initial"] = (model(xi) - self.initial_value(xi)).pow(2).mean()
        return out

    def loss_terms(self) -> list[str]:
        terms = ["pde"]
        if self.has_boundary:
            terms.append("boundary")
        if self.time_dependent:
            terms.append("initial")
        return terms

    def sample(self, n_interior: int, n_boundary: int, n_initial: int, seed: int = 0) -> dict[str, np.ndarray]:
        """Collocation points: Latin hypercube in the interior, uniform on faces and at t = 0."""
        lo, hi = self.bounds()
        d = self.dim
        rng = np.random.default_rng(seed + 1)
        out = {"interior": qmc.scale(qmc.LatinHypercube(d, seed=seed).random(n_interior), lo, hi)}
        if self.has_boundary:
            per_face = max(1, n_boundary // (2 * len(self.spatial_axes)))
            faces = []
            for axis in self.spatial_axes:
                for value in (lo[axis], hi[axis]):
                    pts = lo + (hi - lo) * rng.random((per_face, d))
                    pts[:, axis] = value
                    faces.append(pts)
            out["boundary"] = np.concatenate(faces)
        if self.time_dependent:
            pts = lo + (hi - lo) * rng.random((max(1, n_initial), d))
            pts[:, -1] = lo[-1]
            out["initial"] = pts
        return out

    # ----- reference solutions on a grid -----------------------------------
    def eval_grid(self) -> Grid:
        lo, hi = self.bounds()
        axes = tuple(np.linspace(lo[i], hi[i], n) for i, n in enumerate(self.eval_shape))
        return Grid(self.coords, axes)

    def exact_on_grid(self, grid: Grid) -> np.ndarray | None:
        with torch.no_grad():
            u = self.exact(torch.as_tensor(grid.points(), dtype=torch.float64))
        return None if u is None else u.numpy().reshape(grid.shape)

    def numerical(self, grid: Grid) -> np.ndarray:
        """Conventional numerical solution sampled on ``grid``."""
        raise NotImplementedError

    # ----- condition errors of any gridded solution -------------------------
    def condition_errors(self, values: np.ndarray, grid: Grid) -> dict[str, dict[str, float]]:
        """Misfit of a gridded solution on the boundary faces and on the initial slice."""
        pts = grid.points().reshape(*grid.shape, self.dim)
        out: dict[str, dict[str, float]] = {}
        if self.has_boundary:
            errs = []
            for axis in self.spatial_axes:
                for idx in (0, -1):
                    p = np.take(pts, idx, axis=axis).reshape(-1, self.dim)
                    v = np.take(values, idx, axis=axis).ravel()
                    errs.append(v - to_numpy(self.boundary_value, p).ravel())
            out["boundary"] = stats(np.concatenate(errs))
        if self.time_dependent:
            p = np.take(pts, 0, axis=self.dim - 1).reshape(-1, self.dim)
            v = np.take(values, 0, axis=self.dim - 1).ravel()
            out["initial"] = stats(v - to_numpy(self.initial_value, p).ravel())
        return out

    def extra_condition_errors(self, model: Callable[[Tensor], Tensor], grid: Grid) -> dict[str, dict[str, float]]:
        """Errors of derivative-type conditions, which need the network itself."""
        return {}

    # ----- description for user interfaces ----------------------------------
    def describe(self) -> dict[str, Any]:
        lo, hi = self.bounds()
        return {
            "name": self.name,
            "title": self.title,
            "equation": self.equation,
            "coords": list(self.coords),
            "time_dependent": self.time_dependent,
            "lower": [float(v) for v in lo],
            "upper": [float(v) for v in hi],
            "eval_shape": list(self.eval_shape),
            "loss_terms": self.loss_terms(),
            "analytical_method": self.analytical_method,
            "numerical_method": self.numerical_method,
            "params": [asdict(s) for s in self.PARAMS],
        }
