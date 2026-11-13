"""Physics-informed training: Adam followed by optional L-BFGS.

The trainer is a plain loop with two hooks so that it can be driven from a
notebook, the command line or a streaming service without modification:

* ``on_event(event)`` receives ``metrics`` and ``snapshot`` dictionaries.
* ``control()`` is called between iterations. It may block (pause) and returns
  False to stop training early.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..grid import Grid
from .networks import PINN

Event = dict[str, Any]


@dataclass
class PINNConfig:
    hidden_layers: int = 4
    neurons: int = 32
    activation: str = "tanh"
    learning_rate: float = 1e-3
    epochs: int = 3000  # full-batch Adam steps
    lbfgs_iterations: int = 500  # 0 disables the second-order phase
    n_collocation: int = 5000
    n_boundary: int = 400
    n_initial: int = 400
    w_pde: float = 1.0
    w_boundary: float = 10.0
    w_initial: float = 10.0
    lr_decay: float = 0.9  # learning-rate factor applied every 1000 epochs
    seed: int = 0
    dtype: str = "float64"
    log_every: int = 25
    snapshot_every: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_on_grid(model: PINN, grid: Grid, batch: int = 16384) -> np.ndarray:
    """Network prediction on a grid, as a float64 array shaped like the grid."""
    pts = torch.as_tensor(grid.points(), dtype=model.dtype)
    with torch.no_grad():
        out = torch.cat([model(pts[i:i + batch]) for i in range(0, len(pts), batch)])
    return out.double().numpy().reshape(grid.shape)


def residual_on_grid(problem, model: Callable, grid: Grid, dtype=torch.float64, batch: int = 8192) -> np.ndarray:
    """Pointwise PDE residual of ``model`` on a grid, computed by automatic differentiation."""
    pts = torch.as_tensor(grid.points(), dtype=dtype)
    chunks = []
    for i in range(0, len(pts), batch):
        x = pts[i:i + batch].clone().requires_grad_(True)
        chunks.append(problem.residual(model, x).detach())
    return torch.cat(chunks).double().numpy().reshape(grid.shape)


class PINNTrainer:
    """Trains one network on one problem with fixed collocation points."""

    def __init__(self, problem, config: PINNConfig, grid: Grid | None = None):
        self.problem = problem
        self.cfg = config
        self.dtype = getattr(torch, config.dtype)
        torch.manual_seed(config.seed)
        self.model = PINN(problem, [config.neurons] * config.hidden_layers, config.activation, self.dtype)
        sampled = problem.sample(config.n_collocation, config.n_boundary, config.n_initial, config.seed)
        self.points_np = sampled
        self.pts = {k: torch.tensor(v, dtype=self.dtype, requires_grad=True) for k, v in sampled.items()}
        self.grid = grid if grid is not None else problem.eval_grid()
        self.weights = {"pde": config.w_pde, "boundary": config.w_boundary,
                        "initial": config.w_initial, "initial_velocity": config.w_initial}
        self.history: list[Event] = []
        self.step = 0
        self.stopped = False
        self.elapsed = 0.0
        self._last_snapshot = 0

    def loss(self) -> tuple[torch.Tensor, dict[str, float]]:
        terms = self.problem.losses(self.model, self.pts)
        total = sum(self.weights[name] * value for name, value in terms.items())
        values = {name: float(value.detach()) for name, value in terms.items()}
        values["total"] = float(total.detach())
        return total, values

    def train(self, on_event: Callable[[Event], None] | None = None,
              control: Callable[[], bool] | None = None) -> list[Event]:
        cfg = self.cfg
        emit = on_event or (lambda event: None)
        proceed = control or (lambda: True)
        start = time.perf_counter()

        def report(phase: str, values: dict[str, float], lr: float, force_snapshot: bool = False) -> None:
            if not np.isfinite(values["total"]):
                raise FloatingPointError(f"Loss became non-finite at step {self.step}")
            event = {"type": "metrics", "step": self.step, "phase": phase, "losses": values,
                     "lr": lr, "elapsed": time.perf_counter() - start}
            self.history.append(event)
            emit(event)
            if force_snapshot or self.step - self._last_snapshot >= cfg.snapshot_every:
                self._last_snapshot = self.step
                emit({"type": "snapshot", "step": self.step, "prediction": predict_on_grid(self.model, self.grid)})

        # ---- phase 1: Adam ------------------------------------------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=cfg.lr_decay)
        _, values = self.loss()
        report("adam", values, cfg.learning_rate, force_snapshot=True)
        for _ in range(cfg.epochs):
            if not proceed():
                self.stopped = True
                break
            optimizer.zero_grad()
            total, values = self.loss()
            total.backward()
            optimizer.step()
            scheduler.step()
            self.step += 1
            if self.step % cfg.log_every == 0 or self.step == cfg.epochs:
                _, values = self.loss()
                report("adam", values, scheduler.get_last_lr()[0])

        # ---- phase 2: L-BFGS ----------------------------------------------
        if cfg.lbfgs_iterations > 0 and not self.stopped:
            burst = 10
            lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=burst, max_eval=burst * 2,
                                      history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14,
                                      line_search_fn="strong_wolfe")
            last: dict[str, float] = {}

            def closure():
                lbfgs.zero_grad()
                total, vals = self.loss()
                total.backward()
                last.update(vals)
                return total

            done = 0
            base = self.step
            while done < cfg.lbfgs_iterations:
                if not proceed():
                    self.stopped = True
                    break
                lbfgs.step(closure)
                n_iter = lbfgs.state[lbfgs._params[0]]["n_iter"]
                if n_iter == done:  # line search made no progress: converged
                    break
                done = n_iter
                self.step = base + done
                _, values = self.loss()
                report("lbfgs", values, 1.0)

        emit({"type": "snapshot", "step": self.step, "prediction": predict_on_grid(self.model, self.grid)})
        self.elapsed = time.perf_counter() - start
        return self.history
