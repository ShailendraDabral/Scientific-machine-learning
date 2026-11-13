"""Job manager: one worker process per run, events relayed to subscribers."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue as queue_mod
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .codec import encode_field
from .worker import worker_main

ACTIVE = ("queued", "running", "paused")


class Run:
    """State of one run inside the API process."""

    def __init__(self, kind: str, request: dict[str, Any], root: Path):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.request = kind, request
        self.state = "queued"
        self.created = time.time()
        self.dir = root / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []  # every JSON event, for replay after a reconnect
        self.fields: dict[str, bytes] = {}  # newest binary frame per field name
        self.summary: dict[str, Any] | None = None
        self.error: str | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.process: mp.Process | None = None
        self.controls: dict[str, Any] = {}

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "state": self.state, "created": self.created,
                "request": self.request, "error": self.error, "fields": sorted(self.fields),
                "summary": self.summary}

    # Called on the event loop thread only.
    def publish(self, event: dict[str, Any]) -> None:
        if event["type"] == "field":
            frame = encode_field(event["name"], event["array"], step=event.get("step", 0), run=self.id)
            self.fields[event["name"]] = frame
            for q in self.subscribers:
                if q.qsize() < 200:  # a slow client falls behind on pictures, never on metrics
                    q.put_nowait(frame)
            return
        if event["type"] == "done":
            self.summary = event.get("summary")
            self.state = "cancelled" if event.get("stopped") else "completed"
        elif event["type"] == "error":
            self.error = event.get("message")
            self.state = "failed"
        self.events.append(event)
        for q in self.subscribers:
            q.put_nowait(event)

    def set_state(self, state: str) -> None:
        self.state = state
        self.publish({"type": "status", "state": state})


class JobManager:
    def __init__(self, max_active: int = 2, threads_per_run: int | None = None):
        self.runs: dict[str, Run] = {}
        self.max_active = max_active
        self.threads_per_run = threads_per_run
        self.root = Path(tempfile.mkdtemp(prefix="neuralphysics-runs-"))
        self.ctx = mp.get_context("spawn")

    def active(self) -> list[Run]:
        return [r for r in self.runs.values() if r.state in ACTIVE]

    def start(self, kind: str, request: dict[str, Any]) -> Run:
        if len(self.active()) >= self.max_active:
            raise RuntimeError(f"At most {self.max_active} runs can be active at once; stop one first")
        run = Run(kind, request, self.root)
        self.runs[run.id] = run
        out_q = self.ctx.Queue()
        run.controls = {"pause": self.ctx.Event(), "stop": self.ctx.Event()}
        run.process = self.ctx.Process(target=worker_main, daemon=True,
                                       args=(kind, request, out_q, run.controls, str(run.dir), self.threads_per_run))
        run.process.start()
        run.state = "running"
        loop = asyncio.get_running_loop()
        threading.Thread(target=self._drain, args=(run, out_q, loop), daemon=True).start()
        return run

    @staticmethod
    def _drain(run: Run, out_q, loop: asyncio.AbstractEventLoop) -> None:
        """Move worker events onto the event loop until the worker finishes."""
        finished = False
        while not finished:
            try:
                event = out_q.get(timeout=0.5)
            except queue_mod.Empty:
                if run.process is not None and not run.process.is_alive():
                    if run.state in ACTIVE:
                        event = {"type": "error", "message": "The training process exited unexpectedly"}
                    else:
                        break
                else:
                    continue
            finished = event["type"] in ("done", "error")
            loop.call_soon_threadsafe(run.publish, event)

    def control(self, run: Run, action: str) -> None:
        if run.state not in ACTIVE:
            return
        if action == "pause":
            run.controls["pause"].set()
            run.set_state("paused")
        elif action == "resume":
            run.controls["pause"].clear()
            run.set_state("running")
        elif action == "stop":
            run.controls["stop"].set()
            run.controls["pause"].clear()

    def delete(self, run: Run) -> None:
        if run.process is not None and run.process.is_alive():
            run.controls["stop"].set()
            run.process.join(timeout=3)
            if run.process.is_alive():
                run.process.terminate()
        shutil.rmtree(run.dir, ignore_errors=True)
        self.runs.pop(run.id, None)

    def shutdown(self) -> None:
        for run in list(self.runs.values()):
            self.delete(run)
        shutil.rmtree(self.root, ignore_errors=True)
