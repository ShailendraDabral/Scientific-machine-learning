"""Like-for-like comparison of DeepONet, FNO, PINO and a conventional PINN.

Fairness rules, all recorded in the returned ``settings``:

* The three operators train on the *same* training set (same inputs, same
  reference solutions, same seed) and are scored on the same test set.
* The PINN has no training set: it is trained from scratch on each test input,
  using only the PDE and the initial condition. It is scored on the first K
  test inputs, and the operators' errors on exactly those K inputs are reported
  beside it.
* Every prediction is compared with the same pseudo-spectral reference solution
  on the same space-time grid, and the PDE residual of every prediction is
  measured with the same grid discretisation.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from ..grid import Grid
from ..pinn import PINNConfig, PINNTrainer, predict_on_grid
from ..problems import BurgersPeriodic
from .data import OperatorDataConfig, generate_dataset
from .physics import burgers_residual_grid
from .trainer import OperatorConfig, OperatorTrainer

LABELS = {"deeponet": "DeepONet", "fno": "FNO", "pino": "PINO"}

#: DeepONet is far cheaper per epoch than a Fourier model, so it is given more epochs.
DEFAULT_EPOCHS = {"deeponet": 300, "fno": 40, "pino": 40}

#: Per-instance PINN settings used for the comparison.
PINN_INSTANCE_CONFIG = dict(hidden_layers=4, neurons=32, activation="tanh", epochs=1500, lbfgs_iterations=300,
                            n_collocation=3000, n_initial=256, n_boundary=0, snapshot_every=10**9)


def train_pinn_instance(u0: np.ndarray, x: np.ndarray, t: np.ndarray, nu: float, t_final: float,
                        config: PINNConfig, control: Callable[[], bool] | None = None):
    """Solve one initial-value problem with a PINN. Returns (prediction (nx, nt), seconds, n_parameters)."""
    problem = BurgersPeriodic(u0, nu=nu, t_final=t_final)
    trainer = PINNTrainer(problem, config, Grid(("x", "t"), (x, t)))
    trainer.train(control=control)
    return predict_on_grid(trainer.model, trainer.grid), trainer.elapsed, trainer.model.n_parameters()


def run_benchmark(data_cfg: OperatorDataConfig | None = None,
                  operator_cfgs: dict[str, OperatorConfig] | None = None,
                  pinn_cfg: PINNConfig | None = None, n_pinn_instances: int = 3,
                  on_event: Callable[[dict[str, Any]], None] | None = None,
                  control: Callable[[], bool] | None = None) -> dict[str, Any]:
    data_cfg = data_cfg or OperatorDataConfig()
    operator_cfgs = operator_cfgs or {m: OperatorConfig(model=m, epochs=DEFAULT_EPOCHS[m])
                                      for m in ("deeponet", "fno", "pino")}
    pinn_cfg = pinn_cfg or PINNConfig(**PINN_INSTANCE_CONFIG)
    emit = on_event or (lambda e: None)
    proceed = control or (lambda: True)
    ds = generate_dataset(data_cfg)
    k = min(n_pinn_instances, data_cfg.n_test)
    dt = float(ds.t[1] - ds.t[0])
    stages = [*operator_cfgs, "pinn"]

    def grid_residual(pred: np.ndarray) -> float:
        r = burgers_residual_grid(torch.as_tensor(pred, dtype=torch.float64), data_cfg.nu, dt)
        return float(r.pow(2).mean().sqrt())

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"u0": ds.u0["test"][:k], "reference": ds.u["test"][:k]}
    stopped = False
    for i, (name, cfg) in enumerate(operator_cfgs.items()):
        emit({"type": "stage", "stage": name, "label": LABELS.get(name, name), "index": i, "total": len(stages)})
        trainer = OperatorTrainer(ds, cfg)
        trainer.train(lambda e, name=name: emit({**e, "model": name}), control)
        if trainer.stopped:
            stopped = True
            break
        ev = trainer.evaluate()
        same = np.asarray(ev["test"]["per_sample_rel_l2"][:k])
        arrays[name] = ev["predictions"]["test"][:k]
        rows.append({
            "model": LABELS.get(name, name), "kind": "operator", "n_parameters": ev["n_parameters"],
            "offline_training_seconds": ev["train_seconds"],
            "test_rel_l2_mean": ev["test"]["rel_l2_mean"], "test_rel_l2_std": ev["test"]["rel_l2_std"],
            "ood_rel_l2_mean": ev["ood"]["rel_l2_mean"],
            "same_instances_rel_l2": float(same.mean()), "per_instance_rel_l2": same.tolist(),
            "seconds_per_new_input": ev["test"]["inference_ms_per_sample"] / 1000,
            "pde_residual_rms": ev["test"]["pde_residual_rms"],
        })

    if not stopped:
        emit({"type": "stage", "stage": "pinn", "label": "PINN (per instance)", "index": len(stages) - 1,
              "total": len(stages)})
        preds, errs, secs, n_params = [], [], [], 0
        for j in range(k):
            if not proceed():
                stopped = True
                break
            pred, seconds, n_params = train_pinn_instance(ds.u0["test"][j], ds.x, ds.t, data_cfg.nu,
                                                         data_cfg.t_final, pinn_cfg, control)
            ref = ds.u["test"][j]
            err = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
            preds.append(pred), errs.append(err), secs.append(seconds)
            emit({"type": "pinn_instance", "instance": j, "rel_l2": err, "seconds": seconds})
        if preds:
            arrays["pinn"] = np.stack(preds)
            rows.append({
                "model": "PINN (per instance)", "kind": "pinn", "n_parameters": n_params,
                "offline_training_seconds": 0.0, "test_rel_l2_mean": None, "test_rel_l2_std": None,
                "ood_rel_l2_mean": None, "same_instances_rel_l2": float(np.mean(errs)),
                "per_instance_rel_l2": errs, "seconds_per_new_input": float(np.mean(secs)),
                "pde_residual_rms": grid_residual(np.stack(preds)),
            })

    return {
        "settings": {
            "task": "Periodic viscous Burgers: u0(x) -> u(x, t) on the space-time grid",
            "reference_solver": "Fourier pseudo-spectral ETDRK4 (4x finer in x, dt <= 1e-3), subsampled",
            "data": data_cfg.to_dict(),
            "operators": {n: c.to_dict() for n, c in operator_cfgs.items()},
            "pinn": pinn_cfg.to_dict(),
            "pinn_instances": list(range(k)),
            "error_metric": "relative L2 over the full (nx, nt) grid against the reference solver",
            "residual_metric": "RMS of u_t + u u_x - nu u_xx on the grid (spectral in x, 4th-order central differences in t, levels 3..nt-3)",
            "torch": torch.__version__, "threads": torch.get_num_threads(),
        },
        "reference_residual_rms": grid_residual(ds.u["test"]),
        "rows": rows, "arrays": arrays, "stopped_early": stopped, "x": ds.x, "t": ds.t,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
