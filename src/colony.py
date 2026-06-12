"""Headless colony simulation: the shared-ocean ecology that Watch mode shows.

Runs the exact Watch-mode lifecycle without pygame so experiments can measure it:
a colony forages ONE shared, depleting ocean, breeds by foraging success, ages,
and culls overcrowding. The loop mirrors ``WatchGUI._start_cycle``/``_tick``/
``_end_cycle`` with rendering removed. Smoke test: ``python -m src.colony``.
"""
from __future__ import annotations

import random
import time
from datetime import datetime

from src.algorithms.genetic_algo import create_initial_population, fitness_function
from src.algorithms.rl_algo import FamilyRL, discretize
from src.environment.config import DEFAULT_CONFIG, EnvConfig
from src.environment.ocean import Ocean, advance_fish_pool, spawn_fish
from src.shark import MAX_AGE_YEARS, Shark, SharkGenome, evolvable_traits, reproduce
from src.simulation import biased_action, run_life


def _warmup_brain(brain: FamilyRL, config: EnvConfig, rng: random.Random) -> None:
    """Pre-train the shared brain on a default shark so the founding pod survives.

    A cold brain dies foraging and the colony goes extinct; mirrors ``WatchGUI._warmup_brain``.
    """
    lives = config.watch_brain_warmup_lives
    if lives <= 0:
        return
    warm = SharkGenome.default()
    env = Ocean(config, genome=warm, rng=rng)
    for _ in range(lives):
        run_life(env, brain, warm, rng, config.watch_max_steps)
    brain.epsilon = max(brain.epsilon_min, 0.1)


def _forage_one_cycle(
    population: list[Shark],
    brain: FamilyRL,
    config: EnvConfig,
    rng: random.Random,
) -> tuple[list[Ocean], list[float]]:
    """Run one cycle of interleaved foraging in a shared pool; return envs + rewards.

    Every shark has its own ``Ocean`` but (with ``watch_shared_fish_pool``) shares
    one fish list. Each step every living shark acts, then the pool drifts/respawns once.
    """
    envs = [
        Ocean(config, genome=s.genome, rng=rng, body_size=s.size)
        for s in population
    ]

    shared_fishes = None
    shared_target = 0
    if config.watch_shared_fish_pool:
        free_tiles = config.size * config.size - 1
        target = min(
            free_tiles,
            config.watch_fish_pool_cap,
            max(1, round(config.watch_fish_per_shark * len(population))),
        )
        shared_fishes = spawn_fish(
            config, rng, target_total=target, exclude={config.shark_start}
        )
        shared_target = len(shared_fishes)
        for e in envs:
            e.reset(shared_fishes=shared_fishes)
    else:
        for e in envs:
            e.reset()

    states = [discretize(e.observe()) for e in envs]
    totals = [0.0] * len(envs)

    for _ in range(config.watch_max_steps):
        all_done = True
        for i, env in enumerate(envs):
            if env.done:
                continue
            all_done = False
            action = biased_action(brain, states[i], population[i].genome, rng)
            obs, reward, done = env.step(action)
            next_state = discretize(obs)
            brain.update(states[i], action, reward, next_state, done)
            states[i] = next_state
            totals[i] += reward
            if not env.fishes:  # ate everything -> that life is over (survived)
                env.done = True
        if all_done:
            break
        if shared_fishes is not None:
            living = {envs[i].shark for i, e in enumerate(envs) if not e.done}
            advance_fish_pool(
                shared_fishes, config, rng, config.size,
                target=shared_target, blocked=living,
            )
    return envs, totals


def _advance_population(
    population: list[Shark],
    envs: list[Ocean],
    totals: list[float],
    config: EnvConfig,
    rng: random.Random,
    cull_rng: random.Random,
) -> tuple[list[Shark], dict]:
    """Resolve a finished cycle: deaths, breeding, ageing, culling. Mirror of ``_end_cycle``.

    Returns the next population and a tally dict (births / forage / old-age / cull).
    """
    # Record each shark's foraging outcome; foraging deaths drop from the gene pool.
    for shark, env, total in zip(population, envs, totals):
        shark.record_life(total, env.eaten)
        if not env.alive:
            shark.alive = False

    survived_forage = [s for s in population if s.alive]

    # Breeding favours well-fed sharks and is paced by the gestation trait.
    pups = reproduce(
        population, rng, config.watch_mutation_rate,
        require_food=config.watch_require_food_to_breed,
        fitness_weighted=config.watch_fitness_weighted_breeding,
        gestation_divisor=config.watch_gestation_divisor,
        brood_size=config.watch_brood_size,
    )
    for shark in population:
        shark.grow_older()
    survivors = [s for s in population if s.alive]
    next_pop = survivors + pups

    # Overcrowding culls the weakest foragers first; newborn pups are spared.
    cap = config.watch_carrying_capacity
    culled = max(0, len(next_pop) - cap)
    if culled:
        if config.watch_cull_weakest:
            weakest_first = sorted(survivors, key=lambda s: s.last_reward)
            cull_ids = {id(s) for s in weakest_first[:culled]}
            next_pop = [s for s in next_pop if id(s) not in cull_ids]
            if len(next_pop) > cap:  # too many pups to fit -> trim the remainder
                next_pop = cull_rng.sample(next_pop, cap)
        else:
            next_pop = cull_rng.sample(next_pop, cap)

    tally = {
        "births": len(pups),
        "forage_deaths": len(population) - len(survived_forage),
        "oldage_deaths": len(survived_forage) - len(survivors),
        "culled": culled,
    }
    return next_pop, tally


def run_colony(
    cycles: int,
    config: EnvConfig = DEFAULT_CONFIG,
    seed: int = 0,
    *,
    brain: FamilyRL | None = None,
) -> dict:
    """Evolve a shark colony in a shared, competitive ocean for ``cycles`` cycles.

    Returns a results dict (per-cycle population/fitness/trait history plus the
    trained brain) ready for pickling and reporting.
    """
    rng = random.Random(seed)
    cull_rng = random.Random(seed + 1)  # kept off the sim stream, like the GUI's view_rng
    if brain is None:
        brain = FamilyRL()
    _warmup_brain(brain, config, rng)

    genes = evolvable_traits()

    # Founding pod: random ages 0..14 so old-age deaths appear from the start.
    genomes = create_initial_population(config.watch_population_size, rng)
    population = [Shark(g, age=rng.randrange(MAX_AGE_YEARS)) for g in genomes]

    history: list[dict] = []
    best_genome: SharkGenome | None = None
    best_fitness_overall = float("-inf")
    extinct_at: int | None = None

    start = time.time()
    for cycle in range(1, cycles + 1):
        if not population:
            extinct_at = cycle - 1
            break

        envs, totals = _forage_one_cycle(population, brain, config, rng)

        best_i = max(range(len(totals)), key=lambda i: totals[i])
        best_shark = population[best_i]
        pop_n = len(population)
        history.append({
            "cycle": cycle,
            "population": pop_n,
            "best_fitness": totals[best_i],
            "mean_fitness": sum(totals) / pop_n,
            "min_fitness": min(totals),
            "max_fitness": max(totals),
            "best_genome_values": dict(best_shark.genome.values),
            "avg_trait_values": {
                g: sum(s.genome[g] for s in population) / pop_n for g in genes
            },
            "fitnesses": list(totals),
        })
        if totals[best_i] > best_fitness_overall:
            best_fitness_overall = totals[best_i]
            best_genome = best_shark.genome

        population, tally = _advance_population(
            population, envs, totals, config, rng, cull_rng
        )
        history[-1].update(tally)

        brain.decay_epsilon()
        if not population:
            extinct_at = cycle
            break
    duration = time.time() - start

    if best_genome is None:  # extinct before any cycle completed (shouldn't happen)
        best_genome = SharkGenome.default()

    return {
        "mode": "colony",
        "params": {
            "population": config.watch_population_size,
            "cycles": cycles,
            "cycles_completed": len(history),
            "max_steps": config.watch_max_steps,
            "mutation_rate": config.watch_mutation_rate,
            "carrying_capacity": config.watch_carrying_capacity,
            "warmup_lives": config.watch_brain_warmup_lives,
            "seed": seed,
            "shared_fish_pool": config.watch_shared_fish_pool,
            "fish_per_shark": config.watch_fish_per_shark,
            "fish_pool_cap": config.watch_fish_pool_cap,
            "require_food_to_breed": config.watch_require_food_to_breed,
            "fitness_weighted_breeding": config.watch_fitness_weighted_breeding,
            "cull_weakest": config.watch_cull_weakest,
            "gestation_divisor": config.watch_gestation_divisor,
            "brood_size": config.watch_brood_size,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration,
        "extinct": extinct_at is not None,
        "extinct_at_cycle": extinct_at,
        "best_genome": best_genome,
        "best_genome_values": dict(best_genome.values),
        "analytical_fitness": fitness_function(best_genome),
        "history": history,
        "brain": {
            "alpha": brain.alpha,
            "gamma": brain.gamma,
            "epsilon": brain.epsilon,
            "epsilon_min": brain.epsilon_min,
            "epsilon_decay": brain.epsilon_decay,
            "states_learned": len(brain.q),
            "q_table": dict(brain.q._table),
        },
    }


if __name__ == "__main__":
    # Smoke test: a short colony run, printing the population/fitness arc.
    results = run_colony(cycles=40, seed=0)
    print(f"Ran {results['params']['cycles_completed']} cycles "
          f"in {results['duration_seconds']:.1f}s "
          f"(extinct: {results['extinct']}).")
    print(f"{'cycle':>5} {'pop':>4} {'best':>8} {'mean':>8} "
          f"{'born':>4} {'forage':>6} {'old':>4} {'cull':>4}")
    hist = results["history"]
    for h in hist[:: max(1, len(hist) // 20)]:
        print(f"{h['cycle']:5d} {h['population']:4d} {h['best_fitness']:8.1f} "
              f"{h['mean_fitness']:8.1f} {h['births']:4d} {h['forage_deaths']:6d} "
              f"{h['oldage_deaths']:4d} {h['culled']:4d}")
    print("\nbest genome:")
    for name, val in results["best_genome_values"].items():
        print(f"  {name:<22}{val:.3f}")
