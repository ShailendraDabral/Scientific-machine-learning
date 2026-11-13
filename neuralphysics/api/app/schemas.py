"""Request models. Bounds protect the machine: nothing here can start an unbounded job."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Activation = Literal["tanh", "sine", "gelu", "silu", "softplus", "relu"]


class PINNSettings(BaseModel):
    hidden_layers: int = Field(4, ge=1, le=8)
    neurons: int = Field(32, ge=4, le=128)
    activation: Activation = "tanh"
    learning_rate: float = Field(1e-3, ge=1e-5, le=1e-1)
    epochs: int = Field(2000, ge=0, le=20000)
    lbfgs_iterations: int = Field(0, ge=0, le=5000)
    n_collocation: int = Field(4000, ge=100, le=20000)
    n_boundary: int = Field(400, ge=0, le=4000)
    n_initial: int = Field(400, ge=1, le=4000)
    w_pde: float = Field(1.0, ge=0, le=1000)
    w_boundary: float = Field(10.0, ge=0, le=1000)
    w_initial: float = Field(10.0, ge=0, le=1000)
    seed: int = Field(0, ge=0, le=10**6)
    dtype: Literal["float32", "float64"] = "float64"
    log_every: int = Field(25, ge=1, le=1000)
    snapshot_every: int = Field(100, ge=10, le=5000)


class PINNRunRequest(BaseModel):
    kind: Literal["pinn"] = "pinn"
    problem: str
    params: dict[str, Any] = {}
    config: PINNSettings = PINNSettings()


class OperatorData(BaseModel):
    n_train: int = Field(256, ge=16, le=2000)
    n_test: int = Field(64, ge=8, le=256)
    n_ood: int = Field(64, ge=8, le=256)
    nx: Literal[32, 64, 128] = 64
    nt: Literal[16, 32, 64] = 32
    nu: float = Field(0.03, ge=0.01, le=0.5)
    t_final: float = Field(1.0, ge=0.2, le=2.0)
    length_scale: float = Field(0.10, ge=0.04, le=0.5)
    ood_length_scale: float = Field(0.06, ge=0.04, le=0.5)
    amplitude: float = Field(0.6, ge=0.1, le=2.0)
    max_mode: int = Field(12, ge=1, le=15)
    seed: int = Field(0, ge=0, le=10**6)


class OperatorSettings(BaseModel):
    model: Literal["deeponet", "fno", "pino"] = "fno"
    epochs: int = Field(40, ge=1, le=1000)
    batch_size: int = Field(32, ge=4, le=256)
    learning_rate: float = Field(2e-3, ge=1e-5, le=1e-1)
    width: int = Field(20, ge=8, le=64)
    modes_x: int = Field(10, ge=2, le=16)
    modes_t: int = Field(6, ge=2, le=16)
    n_layers: int = Field(4, ge=1, le=6)
    branch_width: int = Field(128, ge=16, le=256)
    depth: int = Field(3, ge=1, le=6)
    basis: int = Field(64, ge=8, le=256)
    w_data: float = Field(1.0, ge=0, le=100)
    w_pde: float = Field(0.05, ge=0, le=100)
    w_ic: float = Field(5.0, ge=0, le=100)
    seed: int = Field(0, ge=0, le=10**6)
    snapshot_every: int = Field(2, ge=1, le=50)
    monitor_index: int = Field(0, ge=0, le=255)


class OperatorRunRequest(BaseModel):
    kind: Literal["operator"] = "operator"
    data: OperatorData = OperatorData()
    config: OperatorSettings = OperatorSettings()


class BenchmarkRunRequest(BaseModel):
    kind: Literal["benchmark"] = "benchmark"
    data: OperatorData = OperatorData()
    epochs: int = Field(40, ge=1, le=500)  # FNO and PINO
    deeponet_epochs: int = Field(300, ge=1, le=2000)
    n_pinn_instances: int = Field(3, ge=1, le=8)
    pinn_epochs: int = Field(1500, ge=100, le=10000)
    pinn_lbfgs_iterations: int = Field(300, ge=0, le=2000)


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "stop"]


class EvaluateRequest(BaseModel):
    """A fresh random input for a trained operator: tests generalisation beyond the stored sets."""
    length_scale: float = Field(0.10, ge=0.04, le=0.5)
    amplitude: float = Field(0.6, ge=0.1, le=2.0)
    seed: int = Field(0, ge=0, le=10**6)
