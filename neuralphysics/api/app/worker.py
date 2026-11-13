"""Training worker: runs in its own operating-system process.

The worker turns a validated request into calls on the scientific core and
puts plain dictionaries on a queue. Events of type ``field`` carry a NumPy
array; everything else is JSON-serialisable.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _control(events: dict[str, Any]) -> Callable[[], bool]:
    def proceed() -> bool:
        while events["pause"].is_set() and not events["stop"].is_set():
            time.sleep(0.05)
        return not events["stop"].is_set()
    return proceed


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _field(emit, name: str, array: np.ndarray, step: int = 0) -> None:
    emit({"type": "field", "name": name, "step": int(step), "array": np.asarray(array, dtype=np.float32)})


def run_pinn(request: dict[str, Any], emit, proceed) -> dict[str, Any]:
    from neuralphysics.experiments import Experiment

    exp = Experiment(request["problem"], request["params"], request["config"])
    exp.compute_references()
    grid = exp.grid
    emit({"type": "setup", "grid": {"names": list(grid.names), "axes": [a.tolist() for a in grid.axes]},
          "reference_kind": exp.reference.kind, "problem": exp.problem.describe(),
          "methods": {"analytical": exp.problem.analytical_method, "numerical": exp.problem.numerical_method}})
    if exp.analytical is not None:
        _field(emit, "analytical", exp.analytical.values)
    _field(emit, "numerical", exp.numerical.values)

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "snapshot":
            _field(emit, "prediction", event["prediction"], event["step"])
            _field(emit, "abs_error", event["abs_error"], event["step"])
            emit({"type": "snapshot", "step": event["step"], "rel_l2": event["rel_l2"],
                  "max_abs": event["max_abs"], "reference_kind": event["reference_kind"]})
        else:
            emit(event)

    result = exp.run(on_event, proceed)
    _field(emit, "points", exp.trainer.points_np["interior"])
    _field(emit, "residual", result.learned_residual, exp.trainer.step)
    return {"summary": result.summary()}


def run_operator(request: dict[str, Any], emit, proceed, run_dir: str) -> dict[str, Any]:
    import torch
    from neuralphysics.operators import OperatorConfig, OperatorDataConfig, OperatorTrainer, generate_dataset

    data_cfg = OperatorDataConfig(**request["data"])
    cfg = OperatorConfig(**request["config"])
    cfg.monitor_index = min(cfg.monitor_index, data_cfg.n_test - 1)
    ds = generate_dataset(data_cfg)
    trainer = OperatorTrainer(ds, cfg)
    emit({"type": "setup", "grid": {"names": ["x", "t"], "axes": [ds.x.tolist(), ds.t.tolist()]},
          "monitor_index": cfg.monitor_index, "reference_residual_rms": trainer.reference_residual_rms(),
          "methods": {"numerical": "Fourier pseudo-spectral ETDRK4 reference solver"}})
    for split in ("test", "ood"):
        _field(emit, f"u0_{split}", ds.u0[split])
        _field(emit, f"u_{split}", ds.u[split])

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "snapshot":
            _field(emit, "prediction", event["prediction"], event["step"])
            _field(emit, "abs_error", event["abs_error"], event["step"])
        else:
            emit(event)

    trainer.train(on_event, proceed)
    ev = trainer.evaluate()
    for split in ("test", "ood"):
        _field(emit, f"pred_{split}", ev["predictions"][split])
        _field(emit, f"abserr_{split}", np.abs(ev["predictions"][split] - ds.u[split]))
    torch.save({"state": trainer.model.state_dict(), "config": cfg.to_dict(), "data": data_cfg.to_dict()},
               Path(run_dir) / "model.pt")
    ev.pop("predictions")
    return {"summary": {"metrics": ev, "config": cfg.to_dict(), "data": data_cfg.to_dict(),
                        "reference_residual_rms": trainer.reference_residual_rms(),
                        "stopped_early": trainer.stopped}}


def run_benchmark_job(request: dict[str, Any], emit, proceed) -> dict[str, Any]:
    from neuralphysics.operators import OperatorConfig, OperatorDataConfig
    from neuralphysics.operators.benchmark import PINN_INSTANCE_CONFIG, run_benchmark
    from neuralphysics.pinn import PINNConfig

    data_cfg = OperatorDataConfig(**request["data"])
    cfgs = {m: OperatorConfig(model=m, epochs=request["deeponet_epochs" if m == "deeponet" else "epochs"])
            for m in ("deeponet", "fno", "pino")}
    pinn_cfg = PINNConfig(**{**PINN_INSTANCE_CONFIG, "epochs": request["pinn_epochs"],
                             "lbfgs_iterations": request["pinn_lbfgs_iterations"]})
    emit({"type": "setup", "grid": None, "methods": {"numerical": "Fourier pseudo-spectral ETDRK4 reference solver"}})
    result = run_benchmark(data_cfg, cfgs, pinn_cfg, request["n_pinn_instances"],
                           lambda e: None if e["type"] == "snapshot" else emit(e), proceed)
    arrays = result.pop("arrays")
    x, t = result.pop("x"), result.pop("t")
    for name, arr in arrays.items():
        _field(emit, f"bench_{name}", arr)
        if name not in ("u0", "reference"):
            _field(emit, f"bench_err_{name}", np.abs(arr - arrays["reference"]))
    return {"summary": {**result, "grid": {"names": ["x", "t"], "axes": [x.tolist(), t.tolist()]}}}


def worker_main(kind: str, request: dict[str, Any], queue, events: dict[str, Any], run_dir: str,
                threads: int | None = None) -> None:
    """Process entry point. Always ends by putting a ``done`` or ``error`` event on the queue."""
    def emit(event: dict[str, Any]) -> None:
        queue.put(event if event.get("type") == "field" else _jsonable(event))

    try:
        import torch
        if threads:
            torch.set_num_threads(threads)
        proceed = _control(events)
        if kind == "pinn":
            out = run_pinn(request, emit, proceed)
        elif kind == "operator":
            out = run_operator(request, emit, proceed, run_dir)
        elif kind == "benchmark":
            out = run_benchmark_job(request, emit, proceed)
        else:
            raise ValueError(f"Unknown run kind {kind!r}")
        emit({"type": "done", "stopped": bool(events["stop"].is_set()), **out})
    except Exception as exc:  # noqa: BLE001 - every failure must reach the client
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
