# Reproducibility

Every number in this repository can be regenerated with the commands below. This page states what is
fixed, what is not, and what agreement to expect.

## What is fixed

| Source of variation | How it is controlled |
| --- | --- |
| Network initialisation | `torch.manual_seed(config.seed)`, default 0 |
| Collocation points | Latin hypercube and uniform samples seeded with `config.seed` |
| Operator datasets | `numpy.random.default_rng(seed).spawn(3)`: independent streams for train, test and out-of-distribution |
| Mini-batch order | A `torch.Generator` seeded with the operator `config.seed` |
| Problem and training settings | Stored in full in every result file (`params`, `config`, `settings`) |
| Reference solutions | Deterministic solvers with fixed resolutions |

## What is not bit-for-bit reproducible

Floating-point results of multi-threaded linear algebra depend on the CPU, the number of threads and the
library versions. Training amplifies those differences, most of all in L-BFGS, whose line search can
take a different path after a change in the last digit. Expect:

- analytical and numerical solutions: identical to about 1e-12;
- trained networks: the same order of magnitude of error, typically within a factor of two;
- wall-clock times: machine dependent.

For the closest match install the exact versions in `requirements-lock.txt` and fix the thread count,
for example `OMP_NUM_THREADS=2`. The published numbers were produced on Linux x86-64, Python 3.13,
2 CPU threads, CPU-only PyTorch.

## Regenerate the results

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e core -e api

python -m neuralphysics.cli run all --out results      # five experiments; about 25 minutes on 2 cores
python scripts/run_benchmark.py results                # operator comparison; about 3 minutes
```

Each experiment writes `results/<name>.json` (settings, methods, all error measures, final losses) and
`results/<name>.npz` (grid axes and the `analytical`, `numerical` and `learned` arrays, plus the learned
PDE residual). The comparison writes `operator_benchmark.json` and `.npz`.

## Expected values

PINN experiments, presets, seed 0:

| Experiment | Numerical vs analytical (rel. L2) | Learned vs analytical (rel. L2) | Learned PDE residual (rms) | Learned boundary error (max) |
| --- | --- | --- | --- | --- |
| `heat_1d` | 5.4e-6 | 6.8e-4 | 7.4e-3 | 1.9e-3 |
| `heat_2d` | 1.3e-5 | 1.3e-3 | 5.3e-3 | 4.8e-3 |
| `wave_1d` | 2.0e-6 | 3.8e-3 | 1.4e-2 | 1.7e-2 |
| `burgers_1d` | 3.6e-10 | 5.6e-4 | 3.3e-3 | 1.9e-4 |
| `poisson_2d` | 3.2e-5 | 5.2e-4 | 5.2e-3 | 1.3e-3 |

Operator comparison, default settings, seed 0 (relative L2):

| Method | Same 3 test inputs | Full test set | Out of distribution |
| --- | --- | --- | --- |
| DeepONet | 0.119 | 0.103 | 0.217 |
| FNO | 0.025 | 0.021 | 0.051 |
| PINO | 0.038 | 0.032 | 0.060 |
| PINN per input | 0.004 | | |

Residual floor of the reference data: 5.8e-3.

## Tests that guard the mathematics

```bash
cd core && pytest            # 29 tests, about 40 seconds
cd api && pytest             # 3 tests, including a real streamed run with pause and resume
```

| Test | Guarantees |
| --- | --- |
| `test_exact_solution_satisfies_pde` | The autodiff residual of each closed-form solution is at round-off, so residual code and exact solution agree |
| `test_burgers_travelling_wave_residual` | The Burgers residual is exact on an independent exact solution |
| `test_numerical_matches_analytical` | Every solver agrees with its analytical solution to its documented tolerance |
| `test_*_second_order`, `test_etdrk4_fourth_order_in_time` | Each scheme converges at its theoretical order |
| `test_spectral_layer_is_resolution_invariant` | A Fourier layer gives the same values on a coarse and a fine grid |
| `test_reference_data_satisfies_discrete_pde` | The reference data satisfy the discretised PDE used in the PINO loss to within 3 % of the rms of $u_t$ |
| `test_dataset_is_deterministic_and_shared` | Every model in a comparison receives the identical dataset object |

## The demo recordings

The demo replays real runs. To regenerate them and rebuild the demo file:

```bash
python scripts/record_demo.py                      # writes frontend/public/demo/*.json
cd frontend && npm install && npm run build:demo   # writes frontend/dist-demo/index.html
cp dist-demo/index.html ../demo/index.html
```

`record_demo.py` calls the same worker functions as the live service and stores the messages they emit,
with their timestamps. Two things differ from a live run, both for size: arrays are stored as float16
(about three significant digits, for display only; the summaries hold the full-precision metrics), and
stacked per-sample fields are truncated to the first 24 samples. Replays longer than 30 seconds are
time-compressed. Each recording carries its full request and the date it was made.

## Reporting a discrepancy

Include the `.json` result file, the output of `pip freeze`, the CPU model and the thread count.
