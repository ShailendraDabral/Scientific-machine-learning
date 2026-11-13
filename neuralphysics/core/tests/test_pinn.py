"""Physics-informed training: hooks, event stream and convergence."""
import numpy as np
import pytest

from neuralphysics.experiments import Experiment
from neuralphysics.pinn import PINNConfig, PINNTrainer
from neuralphysics.problems import make_problem

FAST = dict(epochs=30, lbfgs_iterations=0, n_collocation=300, n_boundary=80, n_initial=80, log_every=10, snapshot_every=10)


def test_event_stream_and_shapes():
    exp = Experiment("heat_1d", config=FAST)
    events = []
    result = exp.run(events.append)
    kinds = {e["type"] for e in events}
    assert kinds == {"metrics", "snapshot"}
    snap = [e for e in events if e["type"] == "snapshot"][-1]
    assert snap["prediction"].shape == exp.grid.shape == snap["abs_error"].shape
    assert snap["reference_kind"] == "analytical"
    assert set(result.metrics) == {"analytical", "numerical", "learned"}
    assert {"pde", "boundary", "initial", "total"} <= set(result.history[-1]["losses"])
    for kind in ("analytical", "numerical", "learned"):
        assert getattr(result, kind).kind == kind


def test_control_can_stop_training():
    calls = {"n": 0}

    def control():
        calls["n"] += 1
        return calls["n"] <= 5

    trainer = PINNTrainer(make_problem("poisson_2d"), PINNConfig(**{**FAST, "epochs": 100}))
    trainer.train(control=control)
    assert trainer.stopped and trainer.step == 5


def test_wave_has_velocity_condition():
    result = Experiment("wave_1d", params={"v_amplitude": 1.0}, config=FAST).run()
    assert "initial_velocity" in result.history[-1]["losses"]
    assert "initial_velocity" in result.metrics["learned"]["conditions"]


def test_parameter_validation():
    with pytest.raises(ValueError):
        make_problem("heat_1d", alpha=50.0)
    with pytest.raises(ValueError):
        make_problem("heat_1d", unknown=1.0)


@pytest.mark.slow
def test_poisson_pinn_converges():
    result = Experiment("poisson_2d", config=dict(epochs=600, lbfgs_iterations=200, n_collocation=1500)).run()
    assert result.metrics["learned"]["error_vs_analytical"]["rel_l2"] < 2e-2
    assert np.isfinite(result.learned_residual).all()
