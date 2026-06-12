"""Plenty of Fish -- entry point fitting the GA and RL together.

The GA evolves shark genomes; one shared RL brain learns to behave in the Ocean;
each genome is scored by the reward its temperament earns (see ``src.simulation``).
Run from the repo root:

    python -m src.main                 # headless: evolve + learn
    python -m src.main --watch         # GUI: watch the family evolve
    python -m src.main --play          # GUI: control one shark
"""
from __future__ import annotations

import argparse
import random
from dataclasses import replace

from src.algorithms.genetic_algo import fitness_function, genetic_algorithm
from src.algorithms.rl_algo import FamilyRL
from src.environment.config import DEFAULT_CONFIG
from src.shark import SHARK_TRAITS, SharkGenome
from src.simulation import population_fitness

# --population is shared by both modes, so it defaults to None and each mode
# falls back to its own sensible size.
DEFAULT_POPULATION = 30


def _annealing_brain(generations: int) -> FamilyRL:
    """A family brain whose epsilon decays from ~1.0 to its floor over the run.

    Sized to the generation budget (decay steps once per generation): explore
    early, exploit later.
    """
    brain = FamilyRL()
    if generations > 1:
        ratio = brain.epsilon_min / brain.epsilon
        brain.epsilon_decay = ratio ** (1.0 / generations)
    return brain


def _print_genome(genome: SharkGenome) -> None:
    for name in SHARK_TRAITS:
        marker = " (evolved)" if SHARK_TRAITS[name].evolvable else ""
        print(f"  {name:<22}{genome[name]:.3f}{marker}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.watch:
        from src.gui.watch_gui import run
        pop = args.population if args.population is not None else DEFAULT_CONFIG.watch_population_size
        run(replace(DEFAULT_CONFIG,
                    watch_population_size=pop,
                    watch_mutation_rate=args.mutation_rate))
        return
    if args.play:
        from src.gui.gui import run
        run()
        return

    population = args.population if args.population is not None else DEFAULT_POPULATION
    rng = random.Random(args.seed)
    brain = _annealing_brain(args.generations)

    # Capture the shared brain and rng so learning carries across generations.
    def score(genomes: list[SharkGenome]) -> list[float]:
        return population_fitness(
            genomes, brain, rng,
            max_steps=args.max_steps, lives_per_genome=args.lives,
        )

    print(f"Evolving {population} sharks over {args.generations} generations, "
          f"scored by life in the Ocean env...\n")
    best = genetic_algorithm(
        population_size=population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        show_plots=not args.no_plots,
        population_fitness=score,
    )

    print("\nBest shark found (judged by reward earned in the Ocean env):")
    _print_genome(best)
    print(f"  analytical fitness    {fitness_function(best):.4f}")
    print(f"  brain: {len(brain.q)} states learned, epsilon {brain.epsilon:.3f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plenty of Fish - GA + RL shark evolution")
    parser.add_argument("--population", type=int, default=None,
                        help="sharks per generation (even); default 30 headless and for --watch")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=200, help="steps in one shark's life")
    parser.add_argument("--lives", type=int, default=3, help="lives averaged per genome per gen")
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib output")
    parser.add_argument("--watch", action="store_true", help="open the Watch GUI (family evolving)")
    parser.add_argument("--play", action="store_true", help="open the Play GUI (control one shark)")
    return parser


if __name__ == "__main__":
    main()
