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

# Sharks live 15 birth cycles (read as "years"); on reaching this age they die.
MAX_AGE_YEARS = 15

# Sharks are sexually immature until this age -- no breeding before it. A shark
# is also full-grown at this age (see ``Shark.size``), so maturity is one event.
MIN_BREEDING_AGE_YEARS = 5

# A newborn's body length as a fraction of its genetic (adult) size; it grows to
# the full value by ``MIN_BREEDING_AGE_YEARS``. Juveniles are therefore smaller,
# so they can only eat smaller prey and burn less energy than their genome's adult.
BIRTH_SIZE_FRACTION = 0.25


@dataclass
class Shark:
    """An individual shark: its genome plus age and alive/dead state."""

    genome: SharkGenome
    age: int = 0          # completed birth cycles ("years"); newborns start at 0
    alive: bool = True
    # How this shark's most recent foraging life went. The colony records these
    # each cycle so reproduction and overcrowding can favour good foragers.
    last_reward: float = 0.0
    last_eaten: int = 0
    bred_at_age: int = -1  # age at last breeding; -1 = never bred

    @property
    def can_reproduce(self) -> bool:
        """Only living sharks of breeding age reproduce.

        A dead shark can't breed, and a shark younger than
        ``MIN_BREEDING_AGE_YEARS`` is not yet sexually mature.
        """
        return self.alive and self.age >= MIN_BREEDING_AGE_YEARS

    @property
    def size(self) -> float:
        """Effective body length right now: the genome's *adult* size scaled by age.

        The genome carries a shark's adult length. A pup starts at
        ``BIRTH_SIZE_FRACTION`` of it and grows linearly to the full value by
        ``MIN_BREEDING_AGE_YEARS``; from maturity on it stays adult-sized. The
        result is clamped to the size trait's legal range.
        """
        adult = self.genome.size
        grown = min(1.0, self.age / max(1, MIN_BREEDING_AGE_YEARS))
        frac = BIRTH_SIZE_FRACTION + (1.0 - BIRTH_SIZE_FRACTION) * grown
        return SHARK_TRAITS["size"].clamp(adult * frac)

    def record_life(self, reward: float, eaten: int) -> None:
        """Store the outcome of the cycle this shark just foraged."""
        self.last_reward = reward
        self.last_eaten = eaten

    def ready_to_breed(self, gestation_divisor: float) -> bool:
        """Has enough time passed since this shark last bred?

        The ``gestation_period`` trait (in "months") is converted to a whole
        number of birth cycles by ``gestation_divisor`` -- so a longer gestation
        means a shark breeds less often, a real evolutionary trade-off.
        """
        interval = max(1, round(self.genome.gestation_period / gestation_divisor))
        return self.bred_at_age < 0 or (self.age - self.bred_at_age) >= interval

    def mark_bred(self) -> None:
        self.bred_at_age = self.age

    def grow_older(self) -> None:
        """Advance one birth cycle; die on reaching ``MAX_AGE_YEARS``.

        A shark that's already dead stays dead and doesn't keep counting.
        """
        if not self.alive:
            return
        self.age += 1
        if self.age >= MAX_AGE_YEARS:
            self.alive = False


def _weighted_pair(
    breeders: list[Shark], weights: list[float], rng: random.Random
) -> tuple[Shark, Shark]:
    """Pick two distinct parents, each chosen with probability ~ its weight."""
    a = rng.choices(breeders, weights=weights, k=1)[0]
    for _ in range(8):  # resample until we get a different second parent
        b = rng.choices(breeders, weights=weights, k=1)[0]
        if b is not a:
            return a, b
    # Degenerate fallback (all weight on one shark): pair with any other.
    b = next((s for s in breeders if s is not a), a)
    return a, b


def reproduce(
    population: list[Shark],
    rng: random.Random = random,
    mutation_rate: float = 0.0,
    *,
    require_food: bool = False,
    fitness_weighted: bool = False,
    gestation_divisor: float | None = None,
) -> list[Shark]:
    """Breed one age-0 pup per two eligible sharks.

    By default (all flags off) this is the simple rule -- shuffle the living and
    pair them, one pup per pair. The keyword flags add selection pressure:

    * ``gestation_divisor`` -- a shark only breeds if enough cycles have passed
      since its last brood (see :meth:`Shark.ready_to_breed`).
    * ``require_food`` -- only sharks that ate at least one fish last cycle are
      fertile, so foraging success gates reproduction.
    * ``fitness_weighted`` -- parents are drawn with probability proportional to
      how well they foraged, so good hunters leave more offspring.

    Sharks that breed are marked so ``ready_to_breed`` can space out their broods.
    """
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
        # Fertility favours good foragers: a shark's weight grows with the fish
        # it ate and any positive reward it earned. The +1 keeps every fertile
        # shark in the lottery so weak-but-fed sharks can still occasionally breed.
        weights = [1.0 + 2.0 * s.last_eaten + max(0.0, s.last_reward) for s in breeders]
        for _ in range(n_pairs):
            a, b = _weighted_pair(breeders, weights, rng)
            pup_genome, _ = SharkGenome.crossover(a.genome, b.genome, rng)
            if mutation_rate:
                pup_genome = pup_genome.mutate(mutation_rate, rng)
            pups.append(Shark(genome=pup_genome))
            a.mark_bred()
            b.mark_bred()
    else:
        rng.shuffle(breeders)
        for i in range(0, len(breeders) - 1, 2):
            pup_genome, _ = SharkGenome.crossover(
                breeders[i].genome, breeders[i + 1].genome, rng
            )
            if mutation_rate:
                pup_genome = pup_genome.mutate(mutation_rate, rng)
            pups.append(Shark(genome=pup_genome))
            breeders[i].mark_bred()
            breeders[i + 1].mark_bred()
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
