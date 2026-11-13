"""Neural operators: data generation, architecture properties and the physics residual."""
import numpy as np
import pytest
import torch

from neuralphysics.operators import (DeepONet, FNO2d, OperatorConfig, OperatorDataConfig, OperatorTrainer,
                                     SpectralConv2d, burgers_residual_grid, generate_dataset, sample_grf)
from neuralphysics.problems import BurgersPeriodic

SMALL = OperatorDataConfig(n_train=48, n_test=16, n_ood=16)


def test_grf_statistics():
    u0 = sample_grf(4000, 64, 0.1, 0.6, 12, np.random.default_rng(0))
    assert abs(u0.std() - 0.6) < 0.02 and abs(u0.mean()) < 0.02


def test_dataset_is_deterministic_and_shared():
    a = generate_dataset(OperatorDataConfig(n_train=8, n_test=8, n_ood=8))
    b = generate_dataset(OperatorDataConfig(n_train=8, n_test=8, n_ood=8))
    assert a is b  # one cached dataset object: every model sees identical arrays
    assert a.u["train"].shape == (8, 64, 32)
    np.testing.assert_allclose(a.u["test"][:, :, 0], a.u0["test"], atol=1e-12)


def test_reference_data_satisfies_discrete_pde():
    ds = generate_dataset(SMALL)
    u = torch.as_tensor(ds.u["test"])
    res = burgers_residual_grid(u, SMALL.nu, float(ds.t[1] - ds.t[0]))
    u_t = (u[:, :, 4:] - u[:, :, 2:-2]) / (2 * float(ds.t[1] - ds.t[0]))
    assert res.pow(2).mean().sqrt() < 0.03 * u_t.pow(2).mean().sqrt()


def test_spectral_layer_is_resolution_invariant():
    torch.manual_seed(0)
    layer = SpectralConv2d(1, 1, modes_x=6, modes_t=4)

    def field(nx, nt):
        x = torch.arange(nx) / nx
        t = torch.arange(nt) / nt
        xx, tt = torch.meshgrid(x, t, indexing="ij")
        return (torch.cos(2 * torch.pi * 2 * xx) * torch.sin(2 * torch.pi * tt) + 0.3 * torch.sin(2 * torch.pi * 3 * xx))[None, None]

    coarse, fine = layer(field(32, 16)), layer(field(64, 32))
    assert torch.allclose(fine[:, :, ::2, ::2], coarse, atol=1e-5)


def test_models_accept_other_resolutions():
    u0 = torch.randn(3, 64)
    x, t = torch.arange(64) / 64, torch.linspace(0, 1, 32)
    assert FNO2d()(u0, x, t).shape == (3, 64, 32)
    assert DeepONet(64)(u0, x, t).shape == (3, 64, 32)
    assert DeepONet(64)(u0, torch.arange(128) / 128, torch.linspace(0, 1, 50)).shape == (3, 128, 50)


def test_periodic_burgers_initial_condition_is_exact():
    ds = generate_dataset(SMALL)
    problem = BurgersPeriodic(ds.u0["test"][0], nu=SMALL.nu)
    pts = torch.tensor(np.stack([ds.x, np.zeros_like(ds.x)], axis=1))
    np.testing.assert_allclose(problem.initial_value(pts).numpy().ravel(), ds.u0["test"][0], atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["deeponet", "fno", "pino"])
def test_operator_training_reduces_test_error(model):
    data = OperatorDataConfig(n_train=128, n_test=16, n_ood=16)
    epochs = 60 if model == "deeponet" else 15  # DeepONet is much cheaper per epoch
    trainer = OperatorTrainer(generate_dataset(data), OperatorConfig(model=model, epochs=epochs))
    history = trainer.train()
    assert history[-1]["test_rel_l2"] < 0.5
    assert history[-1]["test_rel_l2"] < 0.6 * history[0]["test_rel_l2"]
    ev = trainer.evaluate()
    assert len(ev["test"]["per_sample_rel_l2"]) == data.n_test
