"""Drive the running service from Python: start a run, pause it, follow the stream.

    pip install httpx websockets
    (cd api && uvicorn app.main:app --port 8000) &
    python examples/05_api_client.py
"""
import asyncio
import json
import struct

import httpx
import numpy as np
import websockets

BASE = "localhost:8000"


def decode(frame: bytes):
    (n,) = struct.unpack_from("<I", frame, 0)
    header = json.loads(frame[4:4 + n])
    return header, np.frombuffer(frame, dtype="<f4", offset=4 + n).reshape(header["shape"])


async def main():
    async with httpx.AsyncClient(base_url=f"http://{BASE}") as http:
        run = (await http.post("/api/runs", json={
            "kind": "pinn", "problem": "wave_1d", "params": {"c": 1.5},
            "config": {"epochs": 600, "n_collocation": 2000, "neurons": 32}})).json()
        print("run", run["id"], run["state"])
        async with websockets.connect(f"ws://{BASE}/ws/runs/{run['id']}", max_size=None) as ws:
            paused = False
            async for message in ws:
                if isinstance(message, bytes):
                    header, field = decode(message)
                    if header["name"] == "prediction":
                        print(f"  prediction at step {header['step']}: {field.shape}, max {field.max():.3f}")
                    continue
                event = json.loads(message)
                if event["type"] == "metrics" and event["step"] >= 200 and not paused:
                    paused = True
                    print(await http.post(f"/api/runs/{run['id']}/control", json={"action": "pause"}))
                    await asyncio.sleep(1)
                    print(await http.post(f"/api/runs/{run['id']}/control", json={"action": "resume"}))
                if event["type"] in ("done", "error"):
                    print(event["type"], json.dumps(event.get("summary", {}).get("metrics", {}).get("learned"), indent=1))
                    break


asyncio.run(main())
