"""One equation, three solutions: analytical, numerical and learned, with every error measured.

    python examples/01_three_solutions.py
"""
from neuralphysics.experiments import Experiment

# A short run. Remove `config` to use the full preset.
experiment = Experiment("heat_1d", config={"epochs": 800, "lbfgs_iterations": 200, "n_collocation": 2000})
result = experiment.run()

print(result.report())
for solution in (result.analytical, result.numerical, result.learned):
    print(f"{solution.kind:<11} array {solution.values.shape}, computed in {solution.seconds:.2f} s")

learned = result.metrics["learned"]
print("PDE residual (rms, max):", learned["pde_residual"])
print("Boundary error:", learned["conditions"]["boundary"])
print("Initial error:", learned["conditions"]["initial"])
print("Error vs analytical:", learned["error_vs_analytical"])
