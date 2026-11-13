"""Change PDE parameters, initial conditions and boundary conditions.

    python examples/02_change_the_physics.py
"""
from neuralphysics.experiments import Experiment
from neuralphysics.problems import REGISTRY

for name, cls in REGISTRY.items():
    print(name, {spec.name: spec.default for spec in cls.PARAMS})

fast = {"epochs": 800, "lbfgs_iterations": 200, "n_collocation": 2000}

# Heat: a Gaussian pulse between a warm left end and a cold right end.
heat = Experiment("heat_1d", params={"ic": "gaussian", "u_left": 0.5, "u_right": -0.3, "alpha": 0.2}, config=fast).run()
print(heat.report(), "\n")

# Poisson: the harmonic term B (x^2 - y^2) changes the boundary data but not the source.
poisson = Experiment("poisson_2d", params={"mode_m": 2, "boundary_amplitude": 0.5}, config=fast).run()
print(poisson.report())
