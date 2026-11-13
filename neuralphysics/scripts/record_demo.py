"""Record real training runs for the browser demo.

Every recording is the exact event stream the API would send for a live run:
the same worker functions are called, with wall-clock timestamps, and nothing
is synthesised. Arrays are stored as float16 to keep the demo small; that is a
display precision (about three significant digits), not a change to the science.

    python scripts/record_demo.py                 # everything (about 20 minutes on a laptop CPU)
    python scripts/record_demo.py pinn:heat_1d    # one recording
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "core")]

from app.schemas import BenchmarkRunRequest, OperatorData, OperatorSettings, PINNSettings  # noqa: E402
from app.worker import _jsonable, run_benchmark_job, run_operator, run_pinn  # noqa: E402
from neuralphysics.experiments import preset_config  # noqa: E402
from neuralphysics.operators import evaluate_new_input  # noqa: E402
from neuralphysics.operators.benchmark import DEFAULT_EPOCHS  # noqa: E402
from neuralphysics.problems import REGISTRY  # noqa: E402

OUT = ROOT / "frontend" / "public" / "demo"
MAX_SAMPLES = 24  # stacked per-sample fields are truncated to this many samples
STACKED = ("u0_", "u_", "pred_", "abserr_")
NEW_INPUT_SCALES = (0.05, 0.08, 0.12, 0.2)


def encode(array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array, dtype="<f2")
    return {"shape": list(a.shape), "dtype": "float16", "b64": base64.b64encode(a.tobytes()).decode()}


class Recorder:
    def __init__(self):
        self.messages, self.start = [], time.perf_counter()

    def __call__(self, event: dict) -> None:
        t = round(time.perf_counter() - self.start, 3)
        if event["type"] == "field":
            arr = event["array"]
            if event["name"].startswith(STACKED) and not event["name"].startswith("bench"):
                arr = arr[:MAX_SAMPLES]
            self.messages.append({"t": t, "f": {"name": event["name"], "step": event.get("step", 0), **encode(arr)}})
        else:
            self.messages.append({"t": t, "e": _jsonable(event)})


def save(key: str, kind: str, request: dict, rec: Recorder, out: dict, extra: dict | None = None) -> None:
    rec({"type": "done", "stopped": False, **out})
    doc = {"key": key, "kind": kind, "request": request, "duration": rec.messages[-1]["t"],
           "recorded": time.strftime("%Y-%m-%d"), "messages": rec.messages, **(extra or {})}
    path = OUT / f"{key.replace(':', '_')}.json"
    path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{key}: {doc['duration']:.1f} s, {len(rec.messages)} messages, {path.stat().st_size / 1e6:.2f} MB", flush=True)


def pinn_settings(name: str) -> dict:
    cfg = preset_config(name)
    total = cfg.epochs + cfg.lbfgs_iterations
    base = {k: v for k, v in cfg.to_dict().items() if k in PINNSettings.model_fields}
    return PINNSettings(**{**base, "snapshot_every": max(10, total // 18)}).model_dump()


def record_pinn(name: str) -> None:
    request = {"problem": name, "params": dict(REGISTRY[name]().p), "config": pinn_settings(name)}
    rec = Recorder()
    save(f"pinn:{name}", "pinn", request, rec, run_pinn(request, rec, lambda: True))


def record_operator(model: str) -> None:
    epochs = DEFAULT_EPOCHS[model]
    request = {"data": OperatorData().model_dump(),
               "config": OperatorSettings(model=model, epochs=epochs, snapshot_every=max(1, epochs // 20)).model_dump()}
    rec = Recorder()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_operator(request, rec, lambda: True, tmp)
        new_inputs = [evaluate_new_input(f"{tmp}/model.pt", s, request["data"]["amplitude"], seed=100 + i)
                      for i, s in enumerate(NEW_INPUT_SCALES)]
    for sample in new_inputs:  # three significant digits are enough for display
        for k in ("u0", "reference", "prediction", "abs_error"):
            sample[k] = np.round(np.asarray(sample[k]), 4).tolist()
    save(f"operator:{model}", "operator", request, rec, out, {"new_inputs": new_inputs})


def record_benchmark() -> None:
    request = {k: v for k, v in BenchmarkRunRequest().model_dump().items() if k != "kind"}
    rec = Recorder()
    save("benchmark", "benchmark", request, rec, run_benchmark_job(request, rec, lambda: True))


def write_index() -> None:
    """Catalogue the demo needs in place of the API: problems, defaults and what was recorded."""
    problems = [{**cls().describe(), "preset": pinn_settings(name)} for name, cls in REGISTRY.items()]
    index = {
        "problems": problems,
        "operators": {"data": OperatorData().model_dump(), "config": OperatorSettings().model_dump(),
                      "benchmark": BenchmarkRunRequest().model_dump(),
                      "configs": {m: OperatorSettings(model=m, epochs=DEFAULT_EPOCHS[m]).model_dump()
                                  for m in ("deeponet", "fno", "pino")}},
        "recordings": sorted(p.stem for p in OUT.glob("*.json") if p.stem != "index"),
    }
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sys.argv[1:] or [*(f"pinn:{n}" for n in REGISTRY), "operator:deeponet", "operator:fno",
                            "operator:pino", "benchmark"]
    for job in jobs:
        kind, _, name = job.partition(":")
        {"pinn": record_pinn, "operator": record_operator}.get(kind, lambda _n: record_benchmark())(name)
    write_index()
