"""API contract: codec round trip, catalogue, validation and one real streamed training run."""
import numpy as np
from fastapi.testclient import TestClient

from app.codec import decode_field, encode_field
from app.main import app


def test_codec_round_trip_is_aligned():
    arr = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)
    frame = encode_field("prediction", arr, step=12)
    header_len = int.from_bytes(frame[:4], "little")
    assert (4 + header_len) % 4 == 0
    header, out = decode_field(frame)
    assert header["name"] == "prediction" and header["step"] == 12
    np.testing.assert_array_equal(out, arr)


def test_catalogue_and_validation():
    with TestClient(app) as client:
        problems = client.get("/api/problems").json()
        assert [p["name"] for p in problems] == ["heat_1d", "heat_2d", "wave_1d", "burgers_1d", "poisson_2d"]
        assert all({"pde", "boundary"} & {s["group"] for s in p["params"]} for p in problems)
        bad = client.post("/api/runs", json={"kind": "pinn", "problem": "heat_1d", "params": {"alpha": 99}})
        assert bad.status_code == 422
        too_big = client.post("/api/runs", json={"kind": "pinn", "problem": "heat_1d", "config": {"neurons": 100000}})
        assert too_big.status_code == 422


def test_streamed_pinn_run_with_pause_and_resume():
    config = dict(hidden_layers=2, neurons=16, epochs=120, lbfgs_iterations=0, n_collocation=300,
                  log_every=20, snapshot_every=20)
    with TestClient(app) as client:
        run = client.post("/api/runs", json={"kind": "pinn", "problem": "heat_1d", "config": config}).json()
        assert run["state"] == "running"
        events, frames, paused = [], {}, False
        with client.websocket_connect(f"/ws/runs/{run['id']}") as ws:
            while True:
                msg = ws.receive()
                if msg.get("bytes"):
                    header, data = decode_field(msg["bytes"])
                    frames[header["name"]] = data
                    continue
                import json
                event = json.loads(msg["text"])
                events.append(event)
                if event["type"] == "metrics" and not paused:
                    paused = True
                    assert client.post(f"/api/runs/{run['id']}/control", json={"action": "pause"}).json()["state"] == "paused"
                    assert client.post(f"/api/runs/{run['id']}/control", json={"action": "resume"}).json()["state"] == "running"
                if event["type"] in ("done", "error"):
                    break
        assert events[-1]["type"] == "done", events[-1]
        assert {"analytical", "numerical", "prediction", "abs_error", "residual"} <= set(frames)
        assert frames["prediction"].shape == (101, 101)
        states = [e["state"] for e in events if e["type"] == "status"]
        assert states[:2] == ["paused", "running"]
        summary = events[-1]["summary"]
        assert set(summary["metrics"]) == {"analytical", "numerical", "learned"}
        detail = client.get(f"/api/runs/{run['id']}").json()
        assert detail["state"] == "completed"
        assert client.get(f"/api/runs/{run['id']}/fields/prediction").status_code == 200
        assert client.delete(f"/api/runs/{run['id']}").status_code == 204
