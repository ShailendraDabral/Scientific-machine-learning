# Model and system architecture

Part 1 describes the neural networks. Part 2 describes how the software is put together.

## Part 1: Models

### PINN (`core/neuralphysics/pinn/networks.py`)

A fully connected network maps a point of the domain to the solution value there.

```
(x, t)  ->  scale to [-1, 1]^d  ->  [Linear -> activation] x L  ->  Linear  ->  u
```

| Element | Detail |
| --- | --- |
| Input | The $d$ coordinates of the problem, scaled affinely to $[-1,1]^d$ |
| Hidden layers | $L$ layers of $n$ neurons; user-adjustable, 1 to 8 layers and 4 to 128 neurons |
| Activation | `tanh` (default), `sine`, `gelu`, `silu`, `softplus`, `relu` |
| Output | One linear unit |
| Initialisation | Xavier normal weights, zero biases |
| Parameters | $(d+1)n + (L-1)(n+1)n + (n+1)$ |

`relu` is offered to demonstrate a failure: its second derivative is zero almost everywhere, so a
second-order PDE residual carries no gradient signal through it.

Presets used for the published results:

| Experiment | Layers × neurons | Parameters | Adam epochs | L-BFGS iterations | Collocation points |
| --- | --- | --- | --- | --- | --- |
| `heat_1d` | 4 × 32 | 3,297 | 2,000 | 500 | 4,000 |
| `heat_2d` | 4 × 48 | 7,297 | 2,000 | 500 | 8,000 |
| `wave_1d` | 4 × 64 | 12,737 | 3,000 | 1,500 | 6,000 |
| `burgers_1d` | 6 × 32 | 5,409 | 3,000 | 3,000 | 10,000 |
| `poisson_2d` | 4 × 32 | 3,297 | 1,500 | 500 | 4,000 |

For the periodic Burgers problem used in the operator comparison the network input is
$(\cos 2\pi x,\ \sin 2\pi x,\ 2t/T-1)$. Every prediction is then exactly periodic in $x$, so no boundary
loss is needed.

### DeepONet (`operators/models.py`, class `DeepONet`)

```math
G_\theta(u_0)(x,t) = \sum_{k=1}^{p} b_k\big(u_0(x_1),\dots,u_0(x_m)\big)\; t_k(x,t) + b_0
```

```
u0 at m = 64 sensors  ->  branch MLP: 64 -> 128 -> 128 -> 128 -> p         = b   (p = 64)
(cos 2πx, sin 2πx, t) ->  trunk  MLP:  3 -> 128 -> 128 -> 128 -> p -> tanh = t
prediction on the grid = b · tᵀ + b0
```

| Element | Detail |
| --- | --- |
| Branch | MLP on the sensor values, GELU activations |
| Trunk | MLP on the query coordinate, GELU activations, `tanh` on its output |
| Output | Inner product of the two $p$-vectors plus a scalar bias |
| Parameters | 91,393 at the defaults |
| Query points | Any $(x,t)$: the trunk is a function of continuous coordinates |
| Input sampling | Fixed: the branch expects the input at the same 64 sensors |

### Fourier Neural Operator (`operators/models.py`, classes `SpectralConv2d`, `FNO2d`)

```math
v_0 = P\big(u_0(x),\ \cos 2\pi x,\ \sin 2\pi x,\ t\big), \qquad
v_{l+1} = \sigma\Big(W_l v_l + \mathcal{F}^{-1}\big(R_l\cdot\mathcal{F}(v_l)\big)\Big), \qquad u = Q(v_L)
```

```
4 input channels on the (x, t) grid
  -> lift P: Linear 4 -> 20
  -> zero-pad the time axis by 25 %
  -> 4 Fourier layers: [spectral convolution + 1x1 convolution] -> GELU (none after the last)
  -> crop the padding
  -> project Q: Linear 20 -> 64 -> GELU -> Linear 64 -> 1
```

| Element | Detail |
| --- | --- |
| Spectral convolution | 2D real FFT; the lowest 10 modes in $x$ (positive and negative) and 6 in $t$ are multiplied by learned complex weights of shape (in, out, 10, 6); all other modes are set to zero; inverse FFT |
| Local path $W_l$ | A 1×1 convolution added to the spectral path |
| Time padding | Time is not periodic, so the axis is zero-padded before the Fourier layers and cropped after |
| Normalisation | FFTs use the `forward` norm, which makes mode amplitudes independent of the grid size |
| Parameters | 387,189 real parameters at the defaults (complex weights count twice) |
| Resolution | The weights live in Fourier space, so one layer gives the same values on any grid that resolves the retained modes; this is a unit test |

### Physics-informed neural operator (`operators/physics.py`, `operators/trainer.py`)

PINO is the FNO above with a different loss: the data misfit plus the PDE residual and the
initial-condition misfit of every predicted field (weights $\lambda_d = 1$, $\lambda_p = 0.05$,
$\lambda_0 = 5$). The residual is computed on the output grid, spectrally in $x$ and with fourth-order
differences in $t$. The discretisation and its limits are in
[MATHEMATICS.md](MATHEMATICS.md#operator-learning).

### How the four methods differ

| | PINN | DeepONet | FNO | PINO |
| --- | --- | --- | --- | --- |
| Learns | One solution | An operator | An operator | An operator |
| Training signal | PDE and conditions | Solver data | Solver data | Solver data and PDE |
| New input function | Retrain from scratch | One forward pass | One forward pass | One forward pass |
| Output | Any point | Any point | A grid | A grid |

## Part 2: System

```
frontend/   React, TypeScript, Tailwind; Plotly.js (2D), Three.js (3D)      renders only
    |  REST: JSON configurations and commands
    |  WebSocket: JSON metrics, binary float32 fields
api/        FastAPI: validation, job manager, streaming                     orchestrates only
    |  one operating-system process per run; multiprocessing queue and events
core/       neuralphysics: problems, solvers, pinn, operators, metrics      all the mathematics
```

**Rules that the code enforces.**

- `core` imports no web framework and can be used from a notebook or the command line.
- `api` contains no numerical code. It validates requests against bounded Pydantic models, starts a
  worker process per run and relays what the worker emits.
- `frontend` computes no solution, residual or error. It arranges numbers for display (reshaping,
  colour mapping, camera).

**Life of a run.**

1. The browser posts a configuration to `POST /api/runs`. The request is validated, including problem
   parameters, before any process starts.
2. The job manager starts `worker_main` with the `spawn` start method and records the run.
3. The worker builds the problem, computes the analytical and numerical references, and trains. It puts
   `metrics` dictionaries and `field` arrays on a multiprocessing queue.
4. A drain thread moves events onto the asyncio loop. JSON events are stored for replay; for each field
   the newest binary frame is kept.
5. A WebSocket client first receives everything recorded so far, then the live stream. A page reload
   loses nothing. A slow client may skip field frames but never metrics.
6. `pause`, `resume` and `stop` set multiprocessing events that the training loop checks between
   iterations. L-BFGS runs in bursts of 10 iterations so it can be paused too.
7. At the end the worker evaluates all error measures and sends a `done` event with the summary.

**Core layout.**

```
core/neuralphysics/
  problems/    base.py (Problem, ParamSpec), heat.py, wave.py, burgers.py, poisson.py
  solvers/     finite_difference.py, spectral.py (ETDRK4), cole_hopf.py
  pinn/        networks.py, autodiff.py, trainer.py
  operators/   data.py, models.py, physics.py, trainer.py, benchmark.py
  experiments.py   one equation solved three ways, with every error measured
  metrics.py, grid.py, cli.py
```

**Adding an equation.** Subclass `Problem`, define `bounds`, `residual`, `initial_value`,
`boundary_value`, `numerical` and, if one exists, `exact`; list its adjustable parameters in `PARAMS`;
add the class to `REGISTRY` in `problems/__init__.py`. The API catalogue and the front-end forms are
generated from the class, so no other file changes.

**Demo mode.** The demo build swaps the transport layer of the front end (`src/api.ts`) for a replayer
(`src/demo.ts`) that reads recorded message streams. The recordings are made by
`scripts/record_demo.py`, which calls the same worker functions as the live service.
