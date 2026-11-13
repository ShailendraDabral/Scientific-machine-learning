# API documentation

The service listens on port 8000. Interactive OpenAPI documentation is served at `/docs`, and the schema
at `/openapi.json`. This page is the narrative reference.

## REST endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Liveness and the number of active runs |
| `GET /api/problems` | Equation catalogue with parameter schemas and training presets |
| `GET /api/operators/defaults` | Default dataset, operator and comparison settings |
| `POST /api/runs` | Start a run (`201`); body is one of the three request types below |
| `GET /api/runs` | All runs, without summaries |
| `GET /api/runs/{id}` | State, request, recorded events and summary of one run |
| `POST /api/runs/{id}/control` | Body `{"action": "pause" \| "resume" \| "stop"}` |
| `GET /api/runs/{id}/fields/{name}` | Newest binary frame of a field |
| `POST /api/runs/{id}/evaluate` | Apply a finished operator run to a new random input |
| `DELETE /api/runs/{id}` | Stop the run if active and remove it (`204`) |

Errors: `404` unknown run or field, `409` evaluate on a run that is not a finished operator run, `422`
invalid request or parameter out of bounds, `429` too many active runs.

### Start a PINN run

```bash
curl -X POST localhost:8000/api/runs -H 'Content-Type: application/json' -d '{
  "kind": "pinn",
  "problem": "heat_1d",
  "params": {"alpha": 0.2, "ic": "gaussian", "u_left": 0.5},
  "config": {"hidden_layers": 4, "neurons": 32, "activation": "tanh", "learning_rate": 0.001,
             "epochs": 2000, "lbfgs_iterations": 500, "n_collocation": 4000}
}'
```

`problem` is a name from the catalogue. `params` may set any parameter listed for that problem; omitted
ones take their defaults.

| `config` field | Default | Bounds |
| --- | --- | --- |
| `hidden_layers` | 4 | 1 to 8 |
| `neurons` | 32 | 4 to 128 |
| `activation` | `tanh` | `tanh`, `sine`, `gelu`, `silu`, `softplus`, `relu` |
| `learning_rate` | 0.001 | 1e-5 to 0.1 |
| `epochs` | 2000 | 0 to 20,000 |
| `lbfgs_iterations` | 0 | 0 to 5,000 |
| `n_collocation` | 4000 | 100 to 20,000 |
| `n_boundary`, `n_initial` | 400 | up to 4,000 |
| `w_pde`, `w_boundary`, `w_initial` | 1, 10, 10 | 0 to 1,000 |
| `seed` | 0 | |
| `dtype` | `float64` | `float32`, `float64` |
| `log_every`, `snapshot_every` | 25, 100 | steps between metrics and between field snapshots |

### Start an operator run

```json
{"kind": "operator",
 "data": {"n_train": 256, "n_test": 64, "n_ood": 64, "nx": 64, "nt": 32, "nu": 0.03,
          "length_scale": 0.1, "ood_length_scale": 0.06, "amplitude": 0.6, "seed": 0},
 "config": {"model": "fno", "epochs": 40, "learning_rate": 0.002, "batch_size": 32,
            "width": 20, "modes_x": 10, "modes_t": 6, "n_layers": 4}}
```

`model` is `deeponet`, `fno` or `pino`. DeepONet uses `branch_width`, `depth` and `basis`; PINO adds
`w_data`, `w_pde` and `w_ic`.

### Start a comparison

```json
{"kind": "benchmark", "data": {}, "epochs": 40, "deeponet_epochs": 300,
 "n_pinn_instances": 3, "pinn_epochs": 1500, "pinn_lbfgs_iterations": 300}
```

### Evaluate a trained operator on a new input

```bash
curl -X POST localhost:8000/api/runs/<id>/evaluate -H 'Content-Type: application/json' \
     -d '{"length_scale": 0.08, "amplitude": 0.6, "seed": 1}'
```

Returns `u0`, `reference`, `prediction`, `abs_error`, `x`, `t` as nested lists and `rel_l2`.

## WebSocket stream

`WS /ws/runs/{id}` sends a `hello` message, then every binary frame and JSON event recorded so far, then
live messages until `done` or `error`, and closes.

### JSON messages

| `type` | Fields | Sent |
| --- | --- | --- |
| `hello` | `run` | Once, on connection |
| `setup` | `grid` (`names`, `axes`), `reference_kind`, `methods`, `problem` | Once, before training |
| `metrics` | `step`, `phase` (`adam`, `lbfgs`), `losses` (`total`, `pde`, `boundary`, `initial`, ...), `lr`, `elapsed`; operators add `test_rel_l2`; comparisons add `model` | Every `log_every` steps or every epoch |
| `snapshot` | `step`, `rel_l2`, `max_abs`, `reference_kind` | With each prediction frame (PINN runs) |
| `status` | `state` | After pause and resume |
| `stage` | `stage`, `label`, `index`, `total` | Comparison: start of each method |
| `pinn_instance` | `instance`, `rel_l2`, `seconds` | Comparison: after each PINN |
| `done` | `stopped`, `summary` | End of the run |
| `error` | `message`, `traceback` | Failure |

The `summary` of a PINN run contains `solutions` (method and time for `analytical`, `numerical`,
`learned`), `metrics` (per kind: `pde_residual`, `conditions`, `error_vs_analytical`,
`error_vs_numerical`), `config`, `params`, `n_parameters` and `final_losses`.

### Binary frames

```
bytes 0..3      uint32, little-endian: header length H (a multiple of 4)
bytes 4..4+H    UTF-8 JSON, space-padded: {"name", "shape", "dtype": "float32", "step", "run"}
bytes 4+H..     the array, little-endian float32, C order
```

Because the payload starts on a 4-byte boundary, a browser can wrap it without copying:

```ts
const h = new DataView(buf).getUint32(0, true);
const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, h)));
const data = new Float32Array(buf, 4 + h);
```

| Field name | Shape | Run kind |
| --- | --- | --- |
| `analytical`, `numerical` | grid | PINN, once |
| `prediction`, `abs_error` | grid | PINN and operator, every snapshot |
| `residual`, `points` | grid; (N, d) | PINN, at the end |
| `u0_test`, `u0_ood` | (n, nx) | operator |
| `u_test`, `u_ood`, `pred_*`, `abserr_*` | (n, nx, nt) | operator |
| `bench_u0`, `bench_reference`, `bench_<model>`, `bench_err_<model>` | (K, ...) | comparison |

## Python client example

```python
import asyncio, json, httpx, websockets

run = httpx.post("http://localhost:8000/api/runs",
                 json={"kind": "pinn", "problem": "poisson_2d", "config": {"epochs": 500}}).json()

async def follow():
    async with websockets.connect(f"ws://localhost:8000/ws/runs/{run['id']}", max_size=None) as ws:
        async for message in ws:
            if isinstance(message, bytes):
                continue                      # a field frame; decode as shown above
            event = json.loads(message)
            if event["type"] == "metrics":
                print(event["step"], event["losses"]["total"])
            if event["type"] in ("done", "error"):
                return event

print(asyncio.run(follow())["summary"]["metrics"]["learned"])
```

## Using the core directly

The service is a thin layer. The same functionality is available in Python:

```python
from neuralphysics.experiments import Experiment
from neuralphysics.operators import OperatorConfig, OperatorDataConfig, OperatorTrainer, generate_dataset
from neuralphysics.operators.benchmark import run_benchmark

result = Experiment("burgers_1d", params={"nu": 0.02}, config={"epochs": 2000}).run()
trainer = OperatorTrainer(generate_dataset(OperatorDataConfig()), OperatorConfig(model="fno"))
trainer.train(); metrics = trainer.evaluate()
table = run_benchmark()
```
