"""Command-line entry point: run experiments without the web service.

    python -m neuralphysics.cli list
    python -m neuralphysics.cli run heat_1d --out results
    python -m neuralphysics.cli run all --out results
    python -m neuralphysics.cli run burgers_1d --set nu=0.02 --epochs 2000 --lbfgs 500
"""
from __future__ import annotations

import argparse
import json

from .experiments import Experiment
from .problems import REGISTRY


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neuralphysics", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments and their parameters")
    run = sub.add_parser("run", help="run one experiment, or 'all'")
    run.add_argument("name", choices=[*REGISTRY, "all"])
    run.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="problem parameters")
    run.add_argument("--epochs", type=int)
    run.add_argument("--lbfgs", type=int, dest="lbfgs_iterations")
    run.add_argument("--layers", type=int, dest="hidden_layers")
    run.add_argument("--neurons", type=int)
    run.add_argument("--activation")
    run.add_argument("--lr", type=float, dest="learning_rate")
    run.add_argument("--collocation", type=int, dest="n_collocation")
    run.add_argument("--seed", type=int)
    run.add_argument("--out", help="directory for .npz arrays and .json summaries")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, cls in REGISTRY.items():
            print(f"{name}: {cls.title}")
            for spec in cls.PARAMS:
                print(f"    {spec.name:<20} default={spec.default!r:<10} [{spec.group}] {spec.label}")
        return

    params = dict(item.split("=", 1) for item in args.set)
    params = {k: _parse_value(v) for k, v in params.items()}
    config = {k: getattr(args, k) for k in ("epochs", "lbfgs_iterations", "hidden_layers", "neurons",
                                             "activation", "learning_rate", "n_collocation", "seed")
              if getattr(args, k) is not None}
    for name in (REGISTRY if args.name == "all" else [args.name]):
        result = Experiment(name, params if args.name != "all" else None, config).run()
        print(result.report())
        print(f"  learned: PDE residual max={result.metrics['learned']['pde_residual']['max']:.2e}, "
              f"max abs error vs numerical={result.metrics['learned']['error_vs_numerical']['max_abs']:.2e}, "
              f"training {result.learned.seconds:.1f} s, {result.n_parameters} parameters\n")
        if args.out:
            result.save(args.out)
    if args.out:
        print(f"Saved arrays and summaries to {args.out}/")


if __name__ == "__main__":
    main()
