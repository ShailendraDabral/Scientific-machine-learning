"""Training and evaluation of neural operators (DeepONet, FNO, PINO)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .data import OperatorDataset
from .models import DeepONet, FNO2d, count_parameters
from .physics import burgers_residual_grid

Event = dict[str, Any]


@dataclass
class OperatorConfig:
    model: str = "fno"  # 'deeponet' | 'fno' | 'pino'
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 2e-3
    lr_final_fraction: float = 0.05  # cosine decay to this fraction of the initial rate
    # FNO and PINO
    width: int = 20
    modes_x: int = 10
    modes_t: int = 6
    n_layers: int = 4
    # DeepONet
    branch_width: int = 128
    depth: int = 3
    basis: int = 64
    # PINO: total = w_data * data + w_pde * mean(residual^2) + w_ic * mean((u(x,0) - u0)^2)
    w_data: float = 1.0
    w_pde: float = 0.05
    w_ic: float = 5.0
    seed: int = 0
    snapshot_every: int = 2
    monitor_index: int = 0  # test sample whose prediction is streamed during training

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model(cfg: OperatorConfig, dataset: OperatorDataset) -> nn.Module:
    dc = dataset.config
    if cfg.model == "deeponet":
        return DeepONet(dc.nx, cfg.branch_width, cfg.depth, cfg.basis, t_final=dc.t_final)
    if cfg.model in ("fno", "pino"):
        return FNO2d(cfg.modes_x, cfg.modes_t, cfg.width, cfg.n_layers, t_final=dc.t_final)
    raise ValueError(f"Unknown operator model {cfg.model!r}")


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample relative L2 error over the space-time grid. Shape (B,)."""
    diff = (pred - target).flatten(1).norm(dim=1)
    return diff / target.flatten(1).norm(dim=1)


class OperatorTrainer:
    """Mini-batch Adam training of one operator model on one dataset."""

    def __init__(self, dataset: OperatorDataset, config: OperatorConfig):
        self.ds, self.cfg = dataset, config
        torch.manual_seed(config.seed)
        self.model = build_model(config, dataset)
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        self.x, self.t = f32(dataset.x), f32(dataset.t)
        self.u0 = {s: f32(v) for s, v in dataset.u0.items()}
        self.u = {s: f32(v) for s, v in dataset.u.items()}
        self.dt = float(dataset.t[1] - dataset.t[0])
        self.history: list[Event] = []
        self.elapsed = 0.0
        self.stopped = False

    # ----- loss ---------------------------------------------------------------
    def loss(self, u0: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        pred = self.model(u0, self.x, self.t)
        data = relative_l2(pred, target).mean()
        terms = {"data": float(data.detach())}
        total = self.cfg.w_data * data if self.cfg.model == "pino" else data
        if self.cfg.model == "pino":
            pde = burgers_residual_grid(pred, self.ds.config.nu, self.dt).pow(2).mean()
            ic = (pred[:, :, 0] - u0).pow(2).mean()
            total = total + self.cfg.w_pde * pde + self.cfg.w_ic * ic
            terms.update(pde=float(pde.detach()), initial=float(ic.detach()))
        terms["total"] = float(total.detach())
        return total, terms

    @torch.no_grad()
    def predict(self, u0: Tensor, batch: int = 64) -> Tensor:
        self.model.eval()
        out = torch.cat([self.model(u0[i:i + batch], self.x, self.t) for i in range(0, len(u0), batch)])
        self.model.train()
        return out

    # ----- training -------------------------------------------------------------
    def train(self, on_event: Callable[[Event], None] | None = None,
              control: Callable[[], bool] | None = None) -> list[Event]:
        cfg = self.cfg
        emit = on_event or (lambda e: None)
        proceed = control or (lambda: True)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs), eta_min=cfg.learning_rate * cfg.lr_final_fraction)
        gen = torch.Generator().manual_seed(cfg.seed)
        n = len(self.u0["train"])
        start = time.perf_counter()

        def snapshot(epoch: int) -> None:
            i = cfg.monitor_index
            pred = self.predict(self.u0["test"][i:i + 1])[0].numpy()
            emit({"type": "snapshot", "step": epoch, "prediction": pred,
                  "abs_error": np.abs(pred - self.ds.u["test"][i])})

        snapshot(0)
        for epoch in range(1, cfg.epochs + 1):
            sums: dict[str, float] = {}
            batches = 0
            for idx in torch.randperm(n, generator=gen).split(cfg.batch_size):
                if not proceed():
                    self.stopped = True
                    break
                opt.zero_grad()
                total, terms = self.loss(self.u0["train"][idx], self.u["train"][idx])
                total.backward()
                opt.step()
                batches += 1
                for k, v in terms.items():
                    sums[k] = sums.get(k, 0.0) + v
            if self.stopped:
                break
            sched.step()
            test_err = float(relative_l2(self.predict(self.u0["test"]), self.u["test"]).mean())
            event = {"type": "metrics", "step": epoch, "phase": "adam",
                     "losses": {k: v / batches for k, v in sums.items()}, "test_rel_l2": test_err,
                     "lr": sched.get_last_lr()[0], "elapsed": time.perf_counter() - start}
            self.history.append(event)
            emit(event)
            if epoch % cfg.snapshot_every == 0 or epoch == cfg.epochs:
                snapshot(epoch)
        self.elapsed = time.perf_counter() - start
        return self.history

    # ----- evaluation -------------------------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        """Errors on the test and out-of-distribution sets, physics residual and inference time."""
        out: dict[str, Any] = {"model": self.cfg.model, "n_parameters": count_parameters(self.model),
                               "train_seconds": self.elapsed, "predictions": {}}
        for split in ("test", "ood"):
            t0 = time.perf_counter()
            pred = self.predict(self.u0[split])
            seconds = time.perf_counter() - t0
            err = relative_l2(pred, self.u[split]).numpy()
            res = burgers_residual_grid(pred, self.ds.config.nu, self.dt)
            out[split] = {"rel_l2_mean": float(err.mean()), "rel_l2_std": float(err.std()),
                          "rel_l2_max": float(err.max()), "per_sample_rel_l2": err.tolist(),
                          "pde_residual_rms": float(res.pow(2).mean().sqrt()),
                          "initial_error_rms": float((pred[:, :, 0] - self.u0[split]).pow(2).mean().sqrt()),
                          "inference_ms_per_sample": 1000 * seconds / len(pred)}
            out["predictions"][split] = pred.numpy()
        return out

    def reference_residual_rms(self, split: str = "test") -> float:
        """Residual of the reference data under the grid discretisation (the physics-loss floor)."""
        return float(burgers_residual_grid(self.u[split], self.ds.config.nu, self.dt).pow(2).mean().sqrt())


def evaluate_new_input(checkpoint: str, length_scale: float, amplitude: float, seed: int) -> dict[str, Any]:
    """Apply a saved operator to a freshly sampled input and compare with the reference solver.

    ``checkpoint`` is a file written by ``torch.save({"state", "config", "data"})``.
    """
    from .data import OperatorDataConfig, generate_dataset, sample_grf, solve_burgers

    saved = torch.load(checkpoint, weights_only=True)
    data_cfg = OperatorDataConfig(**saved["data"])
    ds = generate_dataset(data_cfg)
    model = build_model(OperatorConfig(**saved["config"]), ds)
    model.load_state_dict(saved["state"])
    model.eval()
    u0 = sample_grf(1, data_cfg.nx, length_scale, amplitude, data_cfg.max_mode, np.random.default_rng(seed))
    reference = solve_burgers(u0, data_cfg)[0]
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        pred = model(f32(u0), f32(ds.x), f32(ds.t))[0].double().numpy()
    return {"u0": u0[0].tolist(), "reference": reference.tolist(), "prediction": pred.tolist(),
            "abs_error": np.abs(pred - reference).tolist(), "x": ds.x.tolist(), "t": ds.t.tolist(),
            "rel_l2": float(np.linalg.norm(pred - reference) / np.linalg.norm(reference)),
            "length_scale": length_scale, "amplitude": amplitude, "seed": seed}
