"""FastAPI application: REST for runs, one WebSocket per run for live output."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Union

from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .jobs import ACTIVE, JobManager, Run
from .schemas import (BenchmarkRunRequest, ControlRequest, EvaluateRequest, OperatorData, OperatorRunRequest,
                      OperatorSettings, PINNRunRequest, PINNSettings)

manager = JobManager(max_active=int(os.environ.get("NEURALPHYSICS_MAX_RUNS", "2")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    manager.shutdown()


app = FastAPI(title="NeuralPhysics API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def get_run(run_id: str) -> Run:
    run = manager.runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"No run with id {run_id}")
    return run


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "active_runs": len(manager.active())}


@app.get("/api/problems")
def problems() -> list[dict[str, Any]]:
    """Catalogue: equation, parameter schema and training preset for every experiment."""
    from neuralphysics.experiments import preset_config
    from neuralphysics.problems import REGISTRY

    out = []
    for name, cls in REGISTRY.items():
        preset = PINNSettings(**{k: v for k, v in preset_config(name).to_dict().items()
                                 if k in PINNSettings.model_fields}).model_dump()
        out.append({**cls().describe(), "preset": preset})
    return out


@app.get("/api/operators/defaults")
def operator_defaults() -> dict[str, Any]:
    return {"data": OperatorData().model_dump(), "config": OperatorSettings().model_dump(),
            "benchmark": BenchmarkRunRequest().model_dump()}


@app.post("/api/runs", status_code=201)
async def create_run(
    request: Union[PINNRunRequest, OperatorRunRequest, BenchmarkRunRequest] = Body(discriminator="kind"),
) -> dict[str, Any]:
    payload = request.model_dump()
    kind = payload.pop("kind")
    if kind == "pinn":
        from neuralphysics.problems import make_problem
        try:
            make_problem(payload["problem"], **payload["params"])  # validate before spawning a process
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
    try:
        run = manager.start(kind, payload)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc
    return run.info()


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return [{k: v for k, v in r.info().items() if k != "summary"} for r in manager.runs.values()]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    return {**run.info(), "events": run.events}


@app.post("/api/runs/{run_id}/control")
async def control_run(run_id: str, body: ControlRequest) -> dict[str, Any]:
    run = get_run(run_id)
    manager.control(run, body.action)
    return {"id": run.id, "state": run.state}


@app.get("/api/runs/{run_id}/fields/{name}")
def run_field(run_id: str, name: str) -> Response:
    run = get_run(run_id)
    if name not in run.fields:
        raise HTTPException(404, f"Run {run_id} has no field {name!r}")
    return Response(run.fields[name], media_type="application/octet-stream")


@app.post("/api/runs/{run_id}/evaluate")
def evaluate_operator(run_id: str, body: EvaluateRequest) -> dict[str, Any]:
    """Apply a trained operator to a new random input and compare with the reference solver."""
    run = get_run(run_id)
    path = run.dir / "model.pt"
    if run.kind != "operator" or not path.exists():
        raise HTTPException(409, "Only a finished operator run can be evaluated on new inputs")
    from neuralphysics.operators import evaluate_new_input

    return evaluate_new_input(str(path), body.length_scale, body.amplitude, body.seed)


@app.delete("/api/runs/{run_id}", status_code=204)
async def delete_run(run_id: str) -> Response:
    manager.delete(get_run(run_id))
    return Response(status_code=204)


@app.websocket("/ws/runs/{run_id}")
async def run_stream(ws: WebSocket, run_id: str) -> None:
    """Replay everything recorded so far, then stream live events until the run ends."""
    await ws.accept()
    run = manager.runs.get(run_id)
    if run is None:
        await ws.close(code=4404)
        return
    q: asyncio.Queue = asyncio.Queue()
    # Snapshot and subscribe without an await in between, so no event is lost or duplicated.
    backlog, frames = list(run.events), list(run.fields.values())
    finished = run.state not in ACTIVE
    run.subscribers.add(q)
    try:
        await ws.send_json({"type": "hello", "run": {k: v for k, v in run.info().items() if k != "summary"}})
        for frame in frames:
            await ws.send_bytes(frame)
        for event in backlog:
            await ws.send_json(event)
        while not finished:
            item = await q.get()
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_json(item)
                finished = item["type"] in ("done", "error")
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        run.subscribers.discard(q)


# Serve the built front end, if present, so one process runs the whole application.
_dist = Path(os.environ.get("NEURALPHYSICS_FRONTEND", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
