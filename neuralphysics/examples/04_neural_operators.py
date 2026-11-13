"""Train an FNO on solver-generated data, then test it on inputs it has never seen.

    python examples/04_neural_operators.py
"""
import numpy as np

from neuralphysics.operators import OperatorConfig, OperatorDataConfig, OperatorTrainer, generate_dataset

data = OperatorDataConfig(n_train=128)          # periodic Burgers: u0(x) -> u(x, t)
dataset = generate_dataset(data)                # reference solutions from the ETDRK4 spectral solver
print("inputs", dataset.u0["train"].shape, "outputs", dataset.u["train"].shape)

trainer = OperatorTrainer(dataset, OperatorConfig(model="fno", epochs=20))   # or "deeponet", "pino"
trainer.train(lambda e: e["type"] == "metrics" and e["step"] % 5 == 0
              and print(f"epoch {e['step']:3d}  test relative L2 {e['test_rel_l2']:.4f}"))

metrics = trainer.evaluate()
for split in ("test", "ood"):
    m = metrics[split]
    print(f"{split:<5} relative L2 {m['rel_l2_mean']:.4f} +- {m['rel_l2_std']:.4f}, "
          f"PDE residual {m['pde_residual_rms']:.3f}, {m['inference_ms_per_sample']:.2f} ms per input")
print("residual floor of the reference data:", round(trainer.reference_residual_rms(), 4))
print("worst test sample:", int(np.argmax(metrics["test"]["per_sample_rel_l2"])))
