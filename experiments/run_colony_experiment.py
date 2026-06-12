"""Run a *colony* experiment (shared-ocean ecology) and pickle the full results.

The ecological counterpart to ``experiments.run_experiment``. That one scores
each genome alone in its own ocean; this one runs :func:`src.colony.run_colony`,
where the whole colony competes for ONE shared, depleting fish pool, breeds by
foraging success, ages, and is culled by overcrowding -- the same lifecycle the
Watch GUI shows, but headless and recorded.

Run from the repo root::

    python -m experiments.run_colony_experiment --cycles 1000

Each run gets its own subfolder ``experiments/colony_<pop>sharks_<cycles>cyc/``
holding ``results.pkl`` and ``report.pdf``.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from dataclasses import replace

from experiments.colony_report import write_colony_report
from src.colony import run_colony
from src.environment.config import DEFAULT_CONFIG

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run + pickle a shared-ocean colony experiment")
    parser.add_argument("--cycles", type=int, default=1000, help="birth cycles to simulate")
    parser.add_argument("--population", type=int, default=None,
                        help="founding pod size (default: config watch_population_size)")
    parser.add_argument("--carrying-capacity", type=int, default=None,
                        help="overcrowding cull threshold (default: config value)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="output pickle path (default: experiments/<auto-named>/results.pkl)")
    parser.add_argument("--no-report", action="store_true",
                        help="skip the PDF report (only write the pickle)")
    args = parser.parse_args()

    # Build the config, overriding only the knobs the user asked for.
    overrides = {}
    if args.population is not None:
        overrides["watch_population_size"] = args.population
    if args.carrying_capacity is not None:
        overrides["watch_carrying_capacity"] = args.carrying_capacity
    config = replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG

    print(f"Simulating a colony for {args.cycles} cycles "
          f"(founding pod {config.watch_population_size}, "
          f"capacity {config.watch_carrying_capacity}, seed {args.seed})...\n")
    results = run_colony(cycles=args.cycles, config=config, seed=args.seed)

    if args.output:
        out_path = Path(args.output)
    else:
        exp_dir = (EXPERIMENTS_DIR /
                   f"colony_{config.watch_population_size}sharks_{args.cycles}cyc")
        out_path = exp_dir / "results.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(results, fh)

    hist = results["history"]
    peak = max((h["population"] for h in hist), default=0)
    final_pop = hist[-1]["population"] if hist else 0
    print(f"Finished {results['params']['cycles_completed']} cycles "
          f"in {results['duration_seconds']:.1f}s.")
    print(f"  population: founding {config.watch_population_size} -> "
          f"peak {peak} -> final {final_pop}   (extinct: {results['extinct']})")
    print("Best genome (highest single-cycle reward under competition):")
    for name, value in results["best_genome_values"].items():
        print(f"  {name:<22}{value:.3f}")
    print(f"  brain: {results['brain']['states_learned']} states learned, "
          f"epsilon {results['brain']['epsilon']:.3f}")
    print(f"\nSaved experiment -> {out_path}")

    if not args.no_report:
        report_path = write_colony_report(results, out_path.with_name("report.pdf"))
        print(f"Saved report     -> {report_path}")


if __name__ == "__main__":
    main()
