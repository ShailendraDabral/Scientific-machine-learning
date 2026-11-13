"""One experiment = one equation solved three ways, with every error measured.

The three solutions are kept strictly apart and labelled by ``kind``:

* ``analytical`` - closed form or exact integral representation (when it exists)
* ``numerical``  - a conventional solver (finite differences or spectral)
* ``learned``    - a physics-informed neural network trained with PyTorch
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .grid import Grid
from .metrics import error_stats, stats
from .pinn import PINNConfig, PINNTrainer, predict_on_grid, residual_on_grid
from .problems import Problem, make_problem

#: Training presets tuned so each experiment finishes in a few minutes on a laptop CPU.
PRESETS: dict[str, dict[str, Any]] = {
    "heat_1d": dict(hidden_layers=4, neurons=32, epochs=2000, lbfgs_iterations=500, n_collocation=4000),
    "heat_2d": dict(hidden_layers=4, neurons=48, epochs=2000, lbfgs_iterations=500, n_collocation=8000,
                    n_boundary=1600, n_initial=800),
    "wave_1d": dict(hidden_layers=4, neurons=64, epochs=3000, lbfgs_iterations=1500, n_collocation=6000),
    "burgers_1d": dict(hidden_layers=6, neurons=32, epochs=3000, lbfgs_iterations=3000, n_collocation=10000,
                       n_boundary=200, n_initial=200),
    "poisson_2d": dict(hidden_layers=4, neurons=32, epochs=1500, lbfgs_iterations=500, n_collocation=4000,
                       n_boundary=800),
}


def preset_config(problem_name: str, **overrides: Any) -> PINNConfig:
    return PINNConfig(**{**PRESETS.get(problem_name, {}), **overrides})


@dataclass
class Solution:
    kind: str  # 'analytical' | 'numerical' | 'learned'
    method: str
    values: np.ndarray
    seconds: float = 0.0


@dataclass
class ExperimentResult:
    problem: dict[str, Any]
    params: dict[str, Any]
    config: dict[str, Any]
    grid: Grid
    analytical: Solution | None
    numerical: Solution
    learned: Solution
    learned_residual: np.ndarray
    metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    n_parameters: int = 0
    stopped_early: bool = False

    def summary(self) -> dict[str, Any]:
        """Everything except the arrays, ready for JSON."""
        sols = {s.kind: {"method": s.method, "seconds": round(s.seconds, 3)}
                for s in (self.analytical, self.numerical, self.learned) if s is not None}
        return {"problem": self.problem["name"], "title": self.problem["title"], "params": self.params,
                "config": self.config, "grid_shape": list(self.grid.shape), "solutions": sols,
                "n_parameters": self.n_parameters, "stopped_early": self.stopped_early,
                "final_losses": self.history[-1]["losses"] if self.history else {},
                "metrics": self.metrics}

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {f"axis_{n}": a for n, a in zip(self.grid.names, self.grid.axes)}
        for sol in (self.analytical, self.numerical, self.learned):
            if sol is not None:
                arrays[sol.kind] = sol.values
        arrays["learned_pde_residual"] = self.learned_residual
        np.savez_compressed(out / f"{self.problem['name']}.npz", **arrays)
        (out / f"{self.problem['name']}.json").write_text(json.dumps(self.summary(), indent=2))
        return out

    def report(self) -> str:
        """Plain-text table that keeps the three kinds of solution apart."""
        def fmt(v):
            return "      n/a" if v is None else f"{v:9.2e}"

        lines = [f"{self.problem['title']}  [{self.problem['name']}]  params={self.params}",
                 f"{'solution':<11}{'PDE res rms':>12}{'boundary max':>13}{'initial max':>12}"
                 f"{'relL2 vs ana':>13}{'relL2 vs num':>13}  method"]
        for sol in (self.analytical, self.numerical, self.learned):
            if sol is None:
                lines.append(f"{'analytical':<11}  none available for this equation")
                continue
            m = self.metrics[sol.kind]
            cond = m.get("conditions", {})
            lines.append(
                f"{sol.kind:<11}"
                f"{fmt(m.get('pde_residual', {}).get('rms')):>12}"
                f"{fmt(cond.get('boundary', {}).get('max')):>13}"
                f"{fmt(cond.get('initial', {}).get('max')):>12}"
                f"{fmt(m.get('error_vs_analytical', {}).get('rel_l2')):>13}"
                f"{fmt(m.get('error_vs_numerical', {}).get('rel_l2')):>13}  {sol.method}")
        return "\n".join(lines)


class Experiment:
    """Builds the analytical and numerical references, trains the PINN and measures everything."""

    def __init__(self, problem: str | Problem, params: dict[str, Any] | None = None,
                 config: PINNConfig | dict[str, Any] | None = None):
        self.problem = make_problem(problem, **(params or {})) if isinstance(problem, str) else problem
        if isinstance(config, PINNConfig):
            self.cfg = config
        else:
            self.cfg = preset_config(self.problem.name, **(config or {}))
        self.grid = self.problem.eval_grid()
        self.analytical: Solution | None = None
        self.numerical: Solution | None = None

    def compute_references(self) -> None:
        t0 = time.perf_counter()
        exact = self.problem.exact_on_grid(self.grid)
        t1 = time.perf_counter()
        if exact is not None:
            self.analytical = Solution("analytical", self.problem.analytical_method or "", exact, t1 - t0)
        numerical = self.problem.numerical(self.grid)
        self.numerical = Solution("numerical", self.problem.numerical_method, numerical, time.perf_counter() - t1)

    @property
    def reference(self) -> Solution:
        """The most trustworthy available solution: analytical if it exists, else numerical."""
        if self.numerical is None:
            self.compute_references()
        return self.analytical or self.numerical  # type: ignore[return-value]

    def run(self, on_event: Callable[[dict[str, Any]], None] | None = None,
            control: Callable[[], bool] | None = None) -> ExperimentResult:
        reference = self.reference
        trainer = PINNTrainer(self.problem, self.cfg, self.grid)
        self.trainer = trainer

        def forward(event: dict[str, Any]) -> None:
            if event["type"] == "snapshot":
                err = error_stats(event["prediction"], reference.values)
                event = {**event, "abs_error": np.abs(event["prediction"] - reference.values),
                         "rel_l2": err["rel_l2"], "max_abs": err["max_abs"], "reference_kind": reference.kind}
            if on_event:
                on_event(event)

        trainer.train(forward, control)
        return self.evaluate(trainer)

    def evaluate(self, trainer: PINNTrainer) -> ExperimentResult:
        problem, grid = self.problem, self.grid
        prediction = predict_on_grid(trainer.model, grid)
        residual = residual_on_grid(problem, trainer.model, grid, trainer.dtype)
        cfg = self.cfg
        learned = Solution(
            "learned",
            f"PINN, MLP {cfg.hidden_layers}x{cfg.neurons} {cfg.activation}, Adam {cfg.epochs} + L-BFGS "
            f"{cfg.lbfgs_iterations}, {cfg.n_collocation} collocation points, {cfg.dtype}",
            prediction, trainer.elapsed)

        metrics: dict[str, Any] = {}
        if self.analytical is not None:
            m: dict[str, Any] = {"conditions": problem.condition_errors(self.analytical.values, grid)}
            if type(problem).exact is not Problem.exact:  # differentiable closed form: check it solves the PDE
                m["pde_residual"] = stats(residual_on_grid(problem, problem.exact, grid))
            metrics["analytical"] = m
        m = {"conditions": problem.condition_errors(self.numerical.values, grid)}
        if self.analytical is not None:
            m["error_vs_analytical"] = error_stats(self.numerical.values, self.analytical.values)
        metrics["numerical"] = m
        m = {"pde_residual": stats(residual),
             "conditions": {**problem.condition_errors(prediction, grid),
                            **problem.extra_condition_errors(trainer.model, grid)},
             "error_vs_numerical": error_stats(prediction, self.numerical.values)}
        if self.analytical is not None:
            m["error_vs_analytical"] = error_stats(prediction, self.analytical.values)
        metrics["learned"] = m

        return ExperimentResult(
            problem=problem.describe(), params=dict(problem.p), config=cfg.to_dict(), grid=grid,
            analytical=self.analytical, numerical=self.numerical, learned=learned, learned_residual=residual,
            metrics=metrics, history=trainer.history, n_parameters=trainer.model.n_parameters(),
            stopped_early=trainer.stopped)


def run_experiment(name: str, params: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
                   on_event: Callable[[dict[str, Any]], None] | None = None) -> ExperimentResult:
    """Convenience wrapper: build, train and evaluate one catalogue experiment."""
    return Experiment(name, params, config).run(on_event)
