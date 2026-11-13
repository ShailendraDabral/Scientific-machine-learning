"""Error and residual statistics. All inputs are NumPy arrays."""
from __future__ import annotations

import numpy as np


def stats(values: np.ndarray) -> dict[str, float]:
    """Root-mean-square and maximum magnitude of a field (e.g. a PDE residual)."""
    v = np.asarray(values, dtype=np.float64)
    return {"rms": float(np.sqrt(np.mean(v**2))), "max": float(np.max(np.abs(v)))}


def error_stats(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Relative L2, RMS and maximum absolute error of ``prediction`` against ``reference``."""
    p = np.asarray(prediction, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    diff = p - r
    denom = float(np.linalg.norm(r.ravel()))
    return {
        "rel_l2": float(np.linalg.norm(diff.ravel()) / denom) if denom > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
    }
