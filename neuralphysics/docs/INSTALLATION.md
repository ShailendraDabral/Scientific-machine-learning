# Installation

There are four ways to run NeuralPhysics. Pick by what you want to do.

| You want to | Use | Needs |
| --- | --- | --- |
| Look at real results in a browser, nothing to install | [Demo mode](#1-demo-mode-no-python) | A web browser |
| Train with your own settings | [Local install](#2-local-install-full-functionality) | Python 3.10+ |
| Run the full app in a container | [Docker](#3-docker) | Docker |
| Change the front end | [Development setup](#4-development-setup) | Python 3.10+, Node 20+ |

## 1. Demo mode (no Python)

Open `demo/index.html` in a browser. That is the whole installation.

The file is self-contained: the application, its fonts and the recorded runs are embedded in it, and it
makes no network requests. It replays recordings of real training runs for all five PINN experiments, the
three neural operators and the operator-versus-PINN comparison. Because the runs are recorded, the
settings are fixed; play, pause, stop, the training timeline and sample browsing all work.

To host the demo, upload that one file to any static host (GitHub Pages, S3, an intranet share).

## 2. Local install (full functionality)

```bash
git clone <your fork or release URL> neuralphysics && cd neuralphysics
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate

# CPU-only PyTorch is a much smaller download. For a GPU, use the selector at pytorch.org instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e core -e api

cd api
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. FastAPI serves the API and the pre-built front end (`frontend/dist`) from
one process, so Node is not needed.

Check the installation:

```bash
pip install -r requirements-dev.txt
(cd core && pytest -m "not slow")            # about 15 seconds
(cd api && pytest)                           # runs one real streamed training job
```

To use only the scientific core (no web service):

```bash
pip install -e core
python -m neuralphysics.cli run heat_1d
```

### Settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `NEURALPHYSICS_MAX_RUNS` | `2` | Training runs allowed at the same time |
| `NEURALPHYSICS_FRONTEND` | `frontend/dist` | Directory of the built front end to serve |

## 3. Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>. The image builds the front end in a Node stage and runs the API with
CPU-only PyTorch as a non-root user. To use a GPU, change the PyTorch index URL in the `Dockerfile` and
add the usual `deploy.resources.reservations.devices` block to `docker-compose.yml`.

## 4. Development setup

Run the API and the Vite dev server side by side; Vite proxies `/api` and `/ws` to port 8000.

```bash
# terminal 1
cd api && uvicorn app.main:app --port 8000 --reload

# terminal 2
cd frontend && npm install && npm run dev    # http://localhost:5173
```

| Command (in `frontend/`) | Result |
| --- | --- |
| `npm run build` | Type-check and build `dist/`, the app served by FastAPI |
| `npm run build:demo` | Build `dist-demo/index.html`, the single-file demo with recordings embedded |
| `npm run typecheck` | TypeScript only |

Rebuilding the demo data is described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Troubleshooting

- **"Cannot reach the NeuralPhysics API"** in the browser: the back end is not running on port 8000, or
  you opened the Vite dev server without it.
- **"At most 2 runs can be active at once"**: stop a run, or raise `NEURALPHYSICS_MAX_RUNS`.
- **Training is slow**: lower the collocation points or epochs. The presets are sized for accuracy, and
  the wave and Burgers presets take several minutes on a laptop CPU.
- **Windows**: the service starts each training job with the `spawn` start method, which is the Windows
  default, so no change is needed.
