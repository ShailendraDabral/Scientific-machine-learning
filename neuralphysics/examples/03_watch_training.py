"""Follow a training run through the event hooks the web service uses, and stop it early.

    python examples/03_watch_training.py
"""
from neuralphysics.experiments import Experiment

experiment = Experiment("poisson_2d", config={"epochs": 1500, "lbfgs_iterations": 0, "snapshot_every": 100})
state = {"error": 1.0}


def on_event(event):
    if event["type"] == "snapshot":  # a prediction on the evaluation grid plus its error
        state["error"] = event["rel_l2"]
        print(f"step {event['step']:5d}  relative L2 error vs {event['reference_kind']}: {event['rel_l2']:.3e}")


def keep_going():  # called between iterations; return False to stop, or block to pause
    return state["error"] > 0.15


result = experiment.run(on_event, keep_going)
print("stopped early:", result.stopped_early, "after", result.history[-1]["step"], "steps")
