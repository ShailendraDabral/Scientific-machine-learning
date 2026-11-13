# Example experiments

Seven experiments you can run in the browser or from Python, each with what to change and what to look
for. Runnable scripts for the Python versions are in [`examples/`](../examples). All numbers quoted
below were measured with the default seeds; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

In the browser, experiments 1 to 5 are in the **PINN Training Lab** and 6 to 7 in the **Neural Operator
Lab**. In demo mode they replay recordings made with the presets.

## 1. One equation, three solutions

**Run.** PINN lab, *1D heat equation*, press Train. Python: `examples/01_three_solutions.py`.

**Look for.** The predicted map converges to the analytical map while the absolute-error map fades. The
dashed line in the loss plot marks the switch from Adam to L-BFGS, where all loss terms drop sharply.
The final table has one row per kind of solution:

| Solution | PDE residual (rms) | Boundary error (max) | Rel. L2 vs analytical |
| --- | --- | --- | --- |
| analytical | 1.6e-16 | 3.1e-16 | |
| numerical (Crank-Nicolson) | | 0 | 5.4e-6 |
| learned (PINN, 4 × 32) | 7.4e-3 | 1.9e-3 | 6.8e-4 |

The conventional solver is about a hundred times more accurate than the network here, in a thousandth
of the time. That is typical for low-dimensional forward problems and is shown on purpose.

## 2. Boundary and initial conditions

**Run.** *1D heat equation*, set *Initial profile* to `gaussian`, *Left temperature* to 0.5 and *Right
temperature* to −0.3. Python: `examples/02_change_the_physics.py`.

**Look for.** The solution relaxes to the straight line joining the two end temperatures. The analytical
solution adapts automatically: its Fourier coefficients are recomputed by quadrature for the new
profile.

## 3. Activation functions

**Run.** *Poisson equation*, train once with `tanh` and once with `relu`.

**Look for.** With `relu` the PDE-residual loss barely moves while the boundary loss falls. A ReLU
network is piecewise linear, so its second derivatives are zero almost everywhere and the Laplacian in
the residual carries no gradient.

## 4. A steep front

**Run.** *Viscous Burgers equation* with the preset (several minutes on a laptop CPU).

**Look for.** A sharp front forms at $x = 0$ around $t = 0.4$. Drag the training timeline: the network
finds the smooth parts first and the front last, and the error map concentrates on the front. Raising
the viscosity to 0.05 makes the problem easy; compare the error curves.

| Solution | Rel. L2 vs analytical (Cole-Hopf) |
| --- | --- |
| numerical (spectral ETDRK4) | 3.6e-10 |
| learned (PINN, 6 × 32) | 5.6e-4 |

## 5. Network size and collocation points

**Run.** *1D wave equation*. Train with 2 layers of 16 neurons, then with the preset (4 × 64). Then
keep the preset and reduce the collocation points from 6000 to 500.

**Look for.** The small network cannot represent the two-mode standing wave, and its loss plateaus. With
few collocation points the training loss still falls but the prediction error stops improving: the
network satisfies the PDE at the sampled points and not between them.

## 6. Learning an operator

**Run.** Operator lab, train *FNO*, then *DeepONet*, then *PINO* on the same dataset. Python:
`examples/04_neural_operators.py`.

**Look for.** The prediction for one test input sharpens epoch by epoch. After training, browse the test
set and the out-of-distribution set, and generate new inputs with a shorter length scale: error grows as
inputs get rougher than anything in the training set.

| Model | Test rel. L2 | Out-of-distribution rel. L2 | Parameters |
| --- | --- | --- | --- |
| DeepONet | 0.103 | 0.217 | 91,393 |
| FNO | 0.021 | 0.051 | 387,189 |
| PINO | 0.032 | 0.060 | 387,189 |

At these settings the physics loss did not improve on plain FNO. The weights $\lambda_p$ and $\lambda_0$
are exposed so you can explore that trade-off; they were not tuned.

## 7. Operators against a conventional PINN

**Run.** Operator lab, *Run comparison*. Python: `python scripts/run_benchmark.py`.

**Look for.** The table separates two costs: training done once (operators) and work per new input
(a forward pass for operators, a full optimisation for the PINN).

| Method | Rel. L2 on the same 3 inputs | Time per new input | Offline training |
| --- | --- | --- | --- |
| DeepONet | 0.119 | under 1 ms | 13 s |
| FNO | 0.025 | about 1 ms | 36 s |
| PINO | 0.038 | about 1 ms | 37 s |
| PINN, trained per input | 0.004 | 22 s | none |

The PINN is the most accurate and by far the slowest per input. An operator pays off when many inputs
must be solved.

**Fairness rules** (recorded in the `settings` block saved with every comparison):

- the three operators train on the same training set and are scored on the same test set;
- the PINN sees no solution data and is trained from scratch on each of the first K test inputs;
- the operators' errors on exactly those K inputs are reported beside the PINN's;
- every prediction is compared with the same reference solution on the same grid, and every PDE
  residual is measured with the same discretisation.
