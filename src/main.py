"""Plenty of Fish -- entry point that ties the whole MVP together.

Three things live in this repo and this file connects them:

    * ``src.shark``            -- shark genomes + traits   (the *bodies*)
    * ``src.algorithms``       -- a genetic algorithm + a Q-learning brain
    * ``src.environment``      -- a real ocean to live and be drawn in

The loop is: the GA evolves shark bodies, a single shared RL brain learns how to
behave, and both are judged by how a whole family actually fares in the ocean
(``src.simulation``). Bodies and behaviour improve together.

Usage (run from the repo root):

    python -m src.main train      # evolve + learn, plot the run, save the best genome
    python -m src.main demo       # pygame: watch an evolved family hunt
    python -m src.main ocean      # just the scrollable ocean background demo
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.algorithms.genetic_algo import fitness_function, genetic_algorithm
from src.algorithms.rl_algo import FamilyRL, discretize
from src.shark import SHARK_TRAITS, SharkGenome
from src.simulation import population_fitness, run_episode
from src.environment.world import World

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
BEST_GENOME_PATH = OUTPUT_DIR / "best_genome.json"


# ---------------------------------------------------------------------------
# Brain helpers
# ---------------------------------------------------------------------------
def _annealing_brain(episodes: int) -> FamilyRL:
    """A family brain whose epsilon decays from ~1.0 to its floor over the run.

    The shared default decay (0.995) barely moves across a few dozen episodes,
    so we size the decay to the actual episode budget: the family explores early
    and exploits what it learned by the end.
    """
    brain = FamilyRL()
    if episodes > 1:
        ratio = brain.epsilon_min / brain.epsilon
        brain.epsilon_decay = ratio ** (1.0 / episodes)
    return brain


def _save_genome(genome: SharkGenome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genome.values, indent=2))


def _print_genome(genome: SharkGenome) -> None:
    for name in SHARK_TRAITS:
        marker = " (evolved)" if SHARK_TRAITS[name].evolvable else ""
        print(f"  {name:<22}{genome[name]:.3f}{marker}")


# ---------------------------------------------------------------------------
# train: evolve bodies + learn behaviour, headless
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    brain = _annealing_brain(args.generations)

    # The GA scores each generation by simulating it; the brain (and rng) are
    # captured here so learning carries over from one generation to the next.
    def score(genomes: list[SharkGenome]) -> list[float]:
        return population_fitness(
            genomes, brain, rng, width=args.width, height=args.height, steps=args.steps
        )

    print(f"Evolving {args.population} sharks over {args.generations} generations "
          f"in a {args.width}x{args.height} ocean...\n")
    best = genetic_algorithm(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        show_plots=not args.no_plots,
        population_fitness=score,
    )

    print("\nBest shark found (judged by life in the ocean):")
    _print_genome(best)
    print(f"  analytical fitness    {fitness_function(best):.4f}")
    print(f"  brain: {len(brain.q)} states learned, epsilon {brain.epsilon:.3f}")

    _save_genome(best, BEST_GENOME_PATH)
    print(f"\nSaved best genome -> {BEST_GENOME_PATH}")


# ---------------------------------------------------------------------------
# demo: watch an evolved family live in the ocean
# ---------------------------------------------------------------------------
def _demo_family(rng: random.Random, size: int) -> list[SharkGenome]:
    """A family for the demo: the saved best shark (if any) plus mutated kin,
    otherwise a fresh random family."""
    if BEST_GENOME_PATH.exists():
        base = SharkGenome(json.loads(BEST_GENOME_PATH.read_text()))
        return [base] + [base.mutate(0.5, rng) for _ in range(size - 1)]
    return [SharkGenome.random_evolvable(rng) for _ in range(size)]


def cmd_demo(args: argparse.Namespace) -> None:
    import pygame  # local import: training never needs a display

    from src.environment.ocean import Ocean
    from src.environment.render import draw_world

    rng = random.Random(args.seed)
    genomes = _demo_family(rng, args.family)

    # Train the shared brain briefly so the family acts on real experience.
    brain = _annealing_brain(args.warmup)
    warmup_world = World(width=args.width, height=args.height, rng=rng)
    print(f"Warming up the family brain for {args.warmup} episodes...")
    for _ in range(args.warmup):
        run_episode(warmup_world, brain, genomes, steps=args.steps, learn=True)
    brain.epsilon = brain.epsilon_min  # demo shows learned behaviour, not exploring

    pygame.init()
    ts = args.tile_size
    screen = pygame.display.set_mode((args.width * ts, args.height * ts))
    pygame.display.set_caption("Plenty of Fish - evolved family")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)
    ocean = Ocean(args.width, args.height, tile_size=ts, draw_grid=True)

    world = World(width=args.width, height=args.height, rng=rng)
    world.reset(genomes)
    obs = {sid: world.get_observation(sid) for sid in world.living_shark_ids()}
    generation = 1

    step_interval = 1.0 / args.speed  # sim steps per second
    accumulated = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Advance the simulation on its own (slower) clock so it's watchable.
        accumulated += dt
        while accumulated >= step_interval:
            accumulated -= step_interval
            living = world.living_shark_ids()
            if not living:
                # Whole family wiped out: start the next generation of kin.
                generation += 1
                genomes = _demo_family(rng, args.family)
                world.reset(genomes)
                obs = {sid: world.get_observation(sid) for sid in world.living_shark_ids()}
                break
            for sid in living:
                action = brain.select_action(discretize(obs[sid]))
                next_obs, _reward, _done = world.step(sid, action)
                obs[sid] = next_obs

        alive = len(world.living_shark_ids())
        total_eaten = sum(s.food_eaten for s in world.sharks)
        hud = [
            f"Generation {generation}",
            f"Alive: {alive}/{len(world.sharks)}",
            f"Fish eaten: {total_eaten}",
            "Green=safe  Purple=poison   ESC to quit",
        ]
        draw_world(screen, world, ocean, font, hud_lines=hud)
        pygame.display.flip()

    pygame.quit()


# ---------------------------------------------------------------------------
# ocean: the original scrollable background demo
# ---------------------------------------------------------------------------
def cmd_ocean(_args: argparse.Namespace) -> None:
    from src.environment.ocean import run_demo

    run_demo()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plenty of Fish - shark evolution simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="evolve + learn, plot the run, save the best genome")
    train.add_argument("--population", type=int, default=30, help="sharks per generation (even)")
    train.add_argument("--generations", type=int, default=30)
    train.add_argument("--steps", type=int, default=150, help="simulation steps per generation")
    train.add_argument("--width", type=int, default=20)
    train.add_argument("--height", type=int, default=20)
    train.add_argument("--mutation-rate", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--no-plots", action="store_true", help="skip matplotlib output")
    train.set_defaults(func=cmd_train)

    demo = sub.add_parser("demo", help="watch an evolved family in pygame")
    demo.add_argument("--family", type=int, default=12, help="sharks shown at once")
    demo.add_argument("--warmup", type=int, default=40, help="brain training episodes first")
    demo.add_argument("--steps", type=int, default=150, help="steps per warmup episode")
    demo.add_argument("--width", type=int, default=20)
    demo.add_argument("--height", type=int, default=20)
    demo.add_argument("--tile-size", type=int, default=32)
    demo.add_argument("--speed", type=float, default=8.0, help="simulation steps per second")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    ocean = sub.add_parser("ocean", help="the scrollable ocean background demo")
    ocean.set_defaults(func=cmd_ocean)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
