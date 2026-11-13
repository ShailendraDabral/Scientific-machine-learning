"""NeuralPhysics API service. Orchestrates jobs; contains no numerical code."""
import sys
from pathlib import Path

try:  # prefer the installed package; fall back to the sibling source tree
    import neuralphysics  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
