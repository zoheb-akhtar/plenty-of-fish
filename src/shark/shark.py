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
from .traits import SHARK_TRAITS

# Sharks die on reaching this age (birth cycles, read as "years").
MAX_AGE_YEARS = 15

# Age of sexual maturity; also when a shark reaches full size (see ``Shark.size``).
MIN_BREEDING_AGE_YEARS = 5

# A newborn's length as a fraction of adult size; grows to full by MIN_BREEDING_AGE_YEARS.
BIRTH_SIZE_FRACTION = 0.25


@dataclass
class Shark:
    """An individual shark: its genome plus age and alive/dead state."""

    genome: SharkGenome
    age: int = 0          # completed birth cycles; newborns start at 0
    alive: bool = True
    # Most recent foraging outcome; the colony uses these to favour good foragers.
    last_reward: float = 0.0
    last_eaten: int = 0
    bred_at_age: int = -1  # age at last breeding; -1 = never bred

    # Only living sharks at or past MIN_BREEDING_AGE_YEARS can breed.
    @property
    def can_reproduce(self) -> bool:
        return self.alive and self.age >= MIN_BREEDING_AGE_YEARS

    # Current body length: adult size scaled from BIRTH_SIZE_FRACTION at age 0
    # to full by MIN_BREEDING_AGE_YEARS, clamped to the size trait's range.
    @property
    def size(self) -> float:
        adult = self.genome.size
        grown = min(1.0, self.age / max(1, MIN_BREEDING_AGE_YEARS))
        frac = BIRTH_SIZE_FRACTION + (1.0 - BIRTH_SIZE_FRACTION) * grown
        return SHARK_TRAITS["size"].clamp(adult * frac)

    # Store the outcome of the cycle this shark just foraged.
    def record_life(self, reward: float, eaten: int) -> None:
        self.last_reward = reward
        self.last_eaten = eaten

    # Has enough time passed since last breeding?
    # gestation_period / gestation_divisor cycles must elapse between broods.
    def ready_to_breed(self, gestation_divisor: float) -> bool:
        interval = max(1, round(self.genome.gestation_period / gestation_divisor))
        return self.bred_at_age < 0 or (self.age - self.bred_at_age) >= interval

    # Record that this shark bred at its current age (paces future broods).
    def mark_bred(self) -> None:
        self.bred_at_age = self.age

    # Advance one cycle and die at MAX_AGE_YEARS (dead sharks stay dead).
    def grow_older(self) -> None:
        if not self.alive:
            return
        self.age += 1
        if self.age >= MAX_AGE_YEARS:
            self.alive = False


# Pick two distinct parents, each chosen with probability ~ its weight.
def _weighted_pair(
    breeders: list[Shark], weights: list[float], rng: random.Random
) -> tuple[Shark, Shark]:
    a = rng.choices(breeders, weights=weights, k=1)[0]
    for _ in range(8):  # resample until we get a different second parent
        b = rng.choices(breeders, weights=weights, k=1)[0]
        if b is not a:
            return a, b
    # Degenerate fallback (all weight on one shark): pair with any other.
    b = next((s for s in breeders if s is not a), a)
    return a, b


# Breed ``brood_size`` age-0 pups per two eligible sharks.
# Optional flags add selection pressure: ``gestation_divisor`` spaces broods,
# ``require_food`` keeps only fed sharks fertile, ``fitness_weighted`` draws
# parents in proportion to foraging success. Each pup is its own crossover+
# mutation, and bred sharks are marked so broods can be spaced out.
def reproduce(
    population: list[Shark],
    rng: random.Random = random,
    mutation_rate: float = 0.0,
    *,
    require_food: bool = False,
    fitness_weighted: bool = False,
    gestation_divisor: float | None = None,
    brood_size: int = 1,
) -> list[Shark]:
    brood_size = max(1, brood_size)
    breeders = [s for s in population if s.can_reproduce]
    if gestation_divisor is not None:
        breeders = [s for s in breeders if s.ready_to_breed(gestation_divisor)]
    if require_food:
        breeders = [s for s in breeders if s.last_eaten > 0]
    if len(breeders) < 2:
        return []

    pups: list[Shark] = []
    n_pairs = len(breeders) // 2

    if fitness_weighted:
        # Weight by fish eaten + positive reward; +1 keeps every fertile shark in the lottery.
        weights = [1.0 + 2.0 * s.last_eaten + max(0.0, s.last_reward) for s in breeders]
        for _ in range(n_pairs):
            a, b = _weighted_pair(breeders, weights, rng)
            for _ in range(brood_size):
                pup_genome, _ = SharkGenome.crossover(a.genome, b.genome, rng)
                if mutation_rate:
                    pup_genome = pup_genome.mutate(mutation_rate, rng)
                pups.append(Shark(genome=pup_genome))
            a.mark_bred()
            b.mark_bred()
    else:
        rng.shuffle(breeders)
        for i in range(0, len(breeders) - 1, 2):
            for _ in range(brood_size):
                pup_genome, _ = SharkGenome.crossover(
                    breeders[i].genome, breeders[i + 1].genome, rng
                )
                if mutation_rate:
                    pup_genome = pup_genome.mutate(mutation_rate, rng)
                pups.append(Shark(genome=pup_genome))
            breeders[i].mark_bred()
            breeders[i + 1].mark_bred()
    return pups


# Run one birth cycle: living breed first, then all age and old ones die.
# Returns survivors plus this cycle's pups.
def advance_cycle(
    population: list[Shark],
    rng: random.Random = random,
    mutation_rate: float = 0.0,
) -> list[Shark]:
    pups = reproduce(population, rng, mutation_rate)
    for shark in population:
        shark.grow_older()
    survivors = [s for s in population if s.alive]
    return survivors + pups


if __name__ == "__main__":
    # Watch a starter pod live several cycles; random start ages so deaths appear early.
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
