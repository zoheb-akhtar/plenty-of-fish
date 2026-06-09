"""Run a GA+RL shark-evolution experiment and pickle the full results.

Unlike ``src.main`` (which only writes the winning genome to JSON), this captures
the *whole* run for later analysis: every generation's fitness spread and best
genome, the run parameters, and the trained shared brain's Q-table.

Run from the repo root:

    python -m experiments.run_experiment --population 30 --generations 1000

Each run gets its own subfolder ``experiments/<pop>sharks_<gens>gen/`` holding
``results.pkl`` (a single dict, see ``results`` below) and ``report.pdf``. The
Q-table is stored as a plain dict so the pickle does not depend on the RL
package's ``defaultdict`` factory to load.
"""
from __future__ import annotations

import argparse
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

from experiments.report import write_report
from src.algorithms.genetic_algo import fitness_function, genetic_algorithm
from src.main import _annealing_brain
from src.shark import SharkGenome
from src.simulation import population_fitness

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def run_experiment(
    population: int,
    generations: int,
    max_steps: int,
    lives: int,
    mutation_rate: float,
    seed: int,
) -> dict:
    """Evolve a shark family and return a dict capturing the whole run."""
    rng = random.Random(seed)
    brain = _annealing_brain(generations)

    # The GA only returns the overall best genome, so we record per-generation
    # history by wrapping the scorer: it sees the whole population each gen.
    history: list[dict] = []

    def score(genomes: list[SharkGenome]) -> list[float]:
        fitnesses = population_fitness(
            genomes, brain, rng, max_steps=max_steps, lives_per_genome=lives
        )
        best_i = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        history.append({
            "generation": len(history) + 1,
            "fitnesses": list(fitnesses),
            "best_fitness": fitnesses[best_i],
            "best_genome_values": dict(genomes[best_i].values),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
        })
        return fitnesses

    print(f"Evolving {population} sharks over {generations} generations "
          f"(max_steps={max_steps}, lives={lives}, mutation_rate={mutation_rate}, "
          f"seed={seed})...\n")
    start = time.time()
    best = genetic_algorithm(
        population_size=population,
        generations=generations,
        mutation_rate=mutation_rate,
        seed=seed,
        show_plots=False,
        population_fitness=score,
    )
    duration = time.time() - start

    return {
        "params": {
            "population": population,
            "generations": generations,
            "max_steps": max_steps,
            "lives": lives,
            "mutation_rate": mutation_rate,
            "seed": seed,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration,
        "best_genome": best,                     # SharkGenome (needs src to unpickle)
        "best_genome_values": dict(best.values),  # plain dict (self-contained)
        "analytical_fitness": fitness_function(best),
        "history": history,
        "brain": {
            "alpha": brain.alpha,
            "gamma": brain.gamma,
            "epsilon": brain.epsilon,
            "epsilon_min": brain.epsilon_min,
            "epsilon_decay": brain.epsilon_decay,
            "states_learned": len(brain.q),
            # Convert the defaultdict(lambda) to a plain dict so it pickles.
            "q_table": dict(brain.q._table),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run + pickle a GA+RL shark experiment")
    parser.add_argument("--population", type=int, default=30, help="sharks per generation (even)")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200, help="steps in one shark's life")
    parser.add_argument("--lives", type=int, default=3, help="lives averaged per genome per gen")
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="output pickle path (default: experiments/<auto-named>.pkl)")
    parser.add_argument("--no-report", action="store_true",
                        help="skip the PDF report (only write the pickle)")
    args = parser.parse_args()

    results = run_experiment(
        population=args.population,
        generations=args.generations,
        max_steps=args.max_steps,
        lives=args.lives,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        # One subfolder per experiment config; re-running the same config refreshes it.
        exp_dir = EXPERIMENTS_DIR / f"{args.population}sharks_{args.generations}gen"
        out_path = exp_dir / "results.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(results, fh)

    print(f"\nFinished in {results['duration_seconds']:.1f}s.")
    print("Best genome (judged by reward earned in the Ocean env):")
    for name, value in results["best_genome_values"].items():
        print(f"  {name:<22}{value:.3f}")
    print(f"  analytical fitness    {results['analytical_fitness']:.4f}")
    print(f"  brain: {results['brain']['states_learned']} states learned, "
          f"epsilon {results['brain']['epsilon']:.3f}")
    print(f"\nSaved experiment -> {out_path}")

    if not args.no_report:
        report_path = write_report(results, out_path.with_name("report.pdf"))
        print(f"Saved report     -> {report_path}")


if __name__ == "__main__":
    main()
