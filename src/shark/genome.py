"""A concrete shark's genome.

A :class:`SharkGenome` is one individual: a mapping of trait name -> value,
backed by the trait registry in :mod:`shark.traits`. The genome owns its own
gene mechanics (random init, crossover, mutation, GA vector round-trip) so the
genetic algorithm can stay trait-agnostic -- it asks a genome for its evolvable
genes and never hard-codes a trait name or bound.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .traits import SHARK_TRAITS, color_rgb, evolvable_traits


@dataclass
class SharkGenome:
    values: dict[str, float]

    # --- constructors ------------------------------------------------------
    @classmethod
    def default(cls) -> "SharkGenome":
        """A genome where every trait sits at its default value."""
        return cls({name: spec.default for name, spec in SHARK_TRAITS.items()})

    @classmethod
    def random(cls, rng: random.Random = random) -> "SharkGenome":
        """A genome with every trait drawn uniformly at random from its range."""
        return cls({name: spec.random_value(rng) for name, spec in SHARK_TRAITS.items()})

    @classmethod
    def random_evolvable(cls, rng: random.Random = random) -> "SharkGenome":
        """Randomise only the *evolvable* genes; keep the rest at their defaults.

        Use this to seed the GA: it makes a controlled experiment where just the
        traits under search (size/speed/caution) vary across individuals, while
        non-evolvable traits hold their defaults until you switch them on.
        """
        values = {name: spec.default for name, spec in SHARK_TRAITS.items()}
        for name in evolvable_traits():
            values[name] = SHARK_TRAITS[name].random_value(rng)
        return cls(values)

    # --- access ------------------------------------------------------------
    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def __getattr__(self, name: str) -> float:
        # Only consulted when normal attribute lookup fails, so this exposes
        # trait values (genome.size) without shadowing real attributes.
        try:
            return self.__dict__["values"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def color(self) -> tuple[int, int, int]:
        """This shark's RGB colour, derived from its ``color_hue`` gene."""
        return color_rgb(self.values["color_hue"])

    # --- GA bridge ---------------------------------------------------------
    def as_vector(self) -> list[float]:
        """Flatten the *evolvable* genes into an ordered list for the GA."""
        return [self.values[name] for name in evolvable_traits()]

    @classmethod
    def from_vector(
        cls, vector: Sequence[float], base: "SharkGenome | None" = None
    ) -> "SharkGenome":
        """Rebuild a genome from a GA vector of evolvable genes.

        Non-evolvable traits are taken from ``base`` (or their defaults), so the
        round-trip ``as_vector`` -> ``from_vector`` only ever rewrites the genes
        the GA is allowed to change. Incoming values are clamped to range.
        """
        base = base or cls.default()
        values = dict(base.values)
        for name, raw in zip(evolvable_traits(), vector):
            values[name] = SHARK_TRAITS[name].clamp(raw)
        return cls(values)

    # --- evolution ---------------------------------------------------------
    def mutate(self, mutation_rate: float, rng: random.Random = random) -> "SharkGenome":
        """Return a mutated copy. Only evolvable genes can change."""
        values = dict(self.values)
        for name in evolvable_traits():
            if rng.random() < mutation_rate:
                values[name] = SHARK_TRAITS[name].mutate(values[name], rng)
        return SharkGenome(values)

    @classmethod
    def crossover(
        cls, parent_a: "SharkGenome", parent_b: "SharkGenome", rng: random.Random = random
    ) -> tuple["SharkGenome", "SharkGenome"]:
        """Blend two parents into two children (whole-genome arithmetic crossover).

        Evolvable genes are blended with a random weight; non-evolvable genes are
        inherited unchanged from the matching parent so they never drift via the
        GA. All children stay in range.
        """
        alpha = rng.random()
        child_a, child_b = dict(parent_a.values), dict(parent_b.values)
        for name in evolvable_traits():
            spec = SHARK_TRAITS[name]
            va, vb = parent_a.values[name], parent_b.values[name]
            child_a[name] = spec.clamp(alpha * va + (1 - alpha) * vb)
            child_b[name] = spec.clamp(alpha * vb + (1 - alpha) * va)
        return cls(child_a), cls(child_b)


if __name__ == "__main__":
    rng = random.Random(0)
    a, b = SharkGenome.random(rng), SharkGenome.random(rng)
    kid1, kid2 = SharkGenome.crossover(a, b, rng)
    kid1 = kid1.mutate(mutation_rate=0.5, rng=rng)
    print("\nEvolvable genes (what the GA touches):", evolvable_traits())
    print("parent A vector:", [round(x, 2) for x in a.as_vector()])
    print("parent B vector:", [round(x, 2) for x in b.as_vector()])
    print("child  1 vector:", [round(x, 2) for x in kid1.as_vector()])
    print("child1 color RGB:", kid1.color(), "(non-evolvable, inherited)")
