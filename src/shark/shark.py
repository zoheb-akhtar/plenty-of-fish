"""A living shark: a genome plus the lifecycle state a genome doesn't carry.

A :class:`~src.shark.genome.SharkGenome` is only trait values. A ``Shark`` is an
individual living through the simulation: it counts its own age one birth cycle
at a time, breeds while it's alive, and dies once it reaches ``MAX_AGE_YEARS``.

Reproduction follows the rule *one pup per two living sharks*: dead sharks can't
breed, the living are paired off, and each pair yields a single age-0 pup. Since
a shark breeds once per cycle for up to 15 cycles, a pair that keeps surviving
more than replaces itself -- so the population grows or shrinks on its own merits
rather than being clamped to a fixed size.

Run as a demo from the repo root: ``python -m src.shark.shark``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .genome import SharkGenome

# Sharks live 15 birth cycles (read as "years"); on reaching this age they die.
MAX_AGE_YEARS = 15


@dataclass
class Shark:
    """An individual shark: its genome plus age and alive/dead state."""

    genome: SharkGenome
    age: int = 0          # completed birth cycles ("years"); newborns start at 0
    alive: bool = True

    @property
    def can_reproduce(self) -> bool:
        """Only living sharks reproduce -- a dead shark can't breed."""
        return self.alive

    def grow_older(self) -> None:
        """Advance one birth cycle; die on reaching ``MAX_AGE_YEARS``.

        A shark that's already dead stays dead and doesn't keep counting.
        """
        if not self.alive:
            return
        self.age += 1
        if self.age >= MAX_AGE_YEARS:
            self.alive = False


def reproduce(
    population: list[Shark],
    rng: random.Random = random,
    mutation_rate: float = 0.0,
) -> list[Shark]:
    """Breed one age-0 pup per two living sharks.

    Dead sharks are filtered out first (they can't reproduce); the survivors are
    shuffled and paired, and each pair produces a single pup via genome crossover
    (plus optional mutation). An odd shark out sits this cycle out.
    """
    breeders = [s for s in population if s.can_reproduce]
    rng.shuffle(breeders)
    pups: list[Shark] = []
    for i in range(0, len(breeders) - 1, 2):
        pup_genome, _ = SharkGenome.crossover(
            breeders[i].genome, breeders[i + 1].genome, rng
        )
        if mutation_rate:
            pup_genome = pup_genome.mutate(mutation_rate, rng)
        pups.append(Shark(genome=pup_genome))
    return pups


def advance_cycle(
    population: list[Shark],
    rng: random.Random = random,
    mutation_rate: float = 0.0,
) -> list[Shark]:
    """Run one birth cycle and return the next population.

    Order matters: the living breed *first* (so a shark gets a final breeding
    season in the cycle it turns 15), then every shark ages a year and any that
    reach ``MAX_AGE_YEARS`` die. Survivors plus this cycle's pups carry forward.
    """
    pups = reproduce(population, rng, mutation_rate)
    for shark in population:
        shark.grow_older()
    survivors = [s for s in population if s.alive]
    return survivors + pups


if __name__ == "__main__":
    # Watch a starter pod live out several cycles: births, ageing, and deaths.
    # Seeding ages randomly means deaths show up from the very first cycles.
    rng = random.Random(0)
    pop = [
        Shark(SharkGenome.random_evolvable(rng), age=rng.randrange(MAX_AGE_YEARS))
        for _ in range(10)
    ]

    print(f"start: {len(pop)} sharks")
    print(f"{'cycle':>5} {'living':>6} {'born':>4} {'died':>4} {'oldest':>6}")
    for cycle in range(1, 21):
        before = len(pop)
        pups = reproduce(pop, rng, mutation_rate=0.1)
        for s in pop:
            s.grow_older()
        survivors = [s for s in pop if s.alive]
        died = before - len(survivors)
        pop = survivors + pups
        oldest = max((s.age for s in pop), default=0)
        print(f"{cycle:5d} {len(pop):6d} {len(pups):4d} {died:4d} {oldest:6d}")
