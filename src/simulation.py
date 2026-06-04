"""Glue layer: a family of sharks *living* in the ocean, scored for the GA.

This is where the two algorithms meet. One generation is one **episode**:

    1. Build a ``World`` and drop the whole population in as a family.
    2. Step every living shark each tick, choosing actions with the *shared*
       ``FamilyRL`` brain and learning from the rewards (tabular Q-learning).
    3. Score each shark by how well it actually did -- food eaten and time
       survived -- and hand those fitnesses back to the GA.

The brain is created once and **persists across generations**: as bodies evolve,
the family's collective behaviour keeps improving too. ``population_fitness``
packages this up as the population-level fitness the GA calls each generation.
"""
from __future__ import annotations

import random

from src.algorithms.rl_algo import Action, FamilyRL, discretize
from src.environment.world import World
from src.shark import SharkGenome

# Fitness weights: a meal is worth far more than a quiet step, but surviving
# longer is still rewarded so a starving shark beats a poisoned one.
FOOD_WEIGHT = 10.0
SURVIVAL_WEIGHT = 0.1


def run_episode(
    world: World,
    brain: FamilyRL,
    genomes: list[SharkGenome],
    steps: int,
    learn: bool = True,
) -> list[float]:
    """Run one episode in ``world`` and return each shark's fitness.

    With ``learn=True`` the shared Q-table is updated every step (training).
    With ``learn=False`` the family just acts on what it already knows -- used
    for the visual demo so the ocean shows learned behaviour, not exploration.
    """
    world.reset(genomes)

    # Cache the current observation per shark so each tick is one transition.
    obs = {sid: world.get_observation(sid) for sid in world.living_shark_ids()}

    for _ in range(steps):
        living = world.living_shark_ids()
        if not living:
            break
        for sid in living:
            state = discretize(obs[sid])
            action = brain.select_action(state)
            next_obs, reward, done = world.step(sid, action)
            if learn:
                brain.update(state, action, reward, discretize(next_obs), done)
            obs[sid] = next_obs

    if learn:
        brain.decay_epsilon()  # one episode = one step down the exploration ramp

    return [
        FOOD_WEIGHT * s.food_eaten + SURVIVAL_WEIGHT * s.steps_survived
        for s in world.sharks
    ]


def population_fitness(
    genomes: list[SharkGenome],
    brain: FamilyRL,
    rng: random.Random,
    width: int = 20,
    height: int = 20,
    steps: int = 150,
) -> list[float]:
    """Fitness function for the GA: score a whole population by simulation.

    A fresh ``World`` is built each generation (so geography doesn't bias the
    run) but the ``brain`` and ``rng`` are shared, keeping learning and the
    random stream continuous across generations.
    """
    world = World(width=width, height=height, rng=rng)
    return run_episode(world, brain, genomes, steps=steps, learn=True)


if __name__ == "__main__":
    # Watch a single family's average fitness climb as the shared brain learns,
    # holding the genomes fixed so the gain is purely behavioural.
    rng = random.Random(0)
    genomes = [SharkGenome.random_evolvable(rng) for _ in range(20)]
    brain = FamilyRL()
    world = World(width=20, height=20, rng=rng)

    for episode in range(1, 21):
        fitnesses = run_episode(world, brain, genomes, steps=150)
        avg = sum(fitnesses) / len(fitnesses)
        print(
            f"episode {episode:2d} | avg_fitness {avg:6.1f} | "
            f"epsilon {brain.epsilon:.3f} | states {len(brain.q)}"
        )
