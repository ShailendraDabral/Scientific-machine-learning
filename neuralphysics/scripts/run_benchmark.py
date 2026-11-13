"""Run the operator-versus-PINN comparison and save its table, settings and arrays.

    python scripts/run_benchmark.py [output directory, default: results]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from neuralphysics.operators.benchmark import run_benchmark  # noqa: E402


def progress(event):
    if event["type"] == "stage":
        print(f"[{event['index'] + 1}/{event['total']}] {event['label']}", flush=True)
    elif event["type"] == "metrics" and event["step"] % 10 == 0:
        print(f"    epoch {event['step']:4d}  test relative L2 {event['test_rel_l2']:.4f}", flush=True)
    elif event["type"] == "pinn_instance":
        print(f"    instance {event['instance']}: relative L2 {event['rel_l2']:.4f} in {event['seconds']:.1f} s", flush=True)


out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
out.mkdir(parents=True, exist_ok=True)
result = run_benchmark(on_event=progress)
np.savez_compressed(out / "operator_benchmark.npz", x=result.pop("x"), t=result.pop("t"), **result.pop("arrays"))
(out / "operator_benchmark.json").write_text(json.dumps(result, indent=2))

print(f"\nResidual floor of the reference data: {result['reference_residual_rms']:.2e}")
print(f"{'method':<22}{'same K inputs':>14}{'full test':>11}{'out of dist.':>13}{'s per input':>13}")
for row in result["rows"]:
    cell = lambda v: "      n/a" if v is None else f"{v:9.4f}"  # noqa: E731
    print(f"{row['model']:<22}{cell(row['same_instances_rel_l2']):>14}{cell(row['test_rel_l2_mean']):>11}"
          f"{cell(row['ood_rel_l2_mean']):>13}{row['seconds_per_new_input']:>13.4f}")
