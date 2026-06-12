from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .traits import SHARK_TRAITS, color_rgb, evolvable_traits


@dataclass
class SharkGenome:
    values: dict[str, float]

    # --- constructors ------------------------------------------------------
    # Build a genome with every gene at its default value.
    @classmethod
    def default(cls) -> "SharkGenome":
        return cls({name: spec.default for name, spec in SHARK_TRAITS.items()})

    # Build a genome with every gene at a fresh uniform-random value.
    @classmethod
    def random(cls, rng: random.Random = random) -> "SharkGenome":
        return cls({name: spec.random_value(rng) for name, spec in SHARK_TRAITS.items()})

    # Randomise only the evolvable genes; keep the rest at defaults (GA seed).
    @classmethod
    def random_evolvable(cls, rng: random.Random = random) -> "SharkGenome":
        values = {name: spec.default for name, spec in SHARK_TRAITS.items()}
        for name in evolvable_traits():
            values[name] = SHARK_TRAITS[name].random_value(rng)
        return cls(values)

    # --- access ------------------------------------------------------------
    # Trait value by name (genome["size"]).
    def __getitem__(self, name: str) -> float:
        return self.values[name]

    # Fallback lookup: exposes trait values (genome.size) without shadowing real attrs.
    def __getattr__(self, name: str) -> float:
        try:
            return self.__dict__["values"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    # This shark's RGB colour, derived from its ``color_hue`` gene.
    def color(self) -> tuple[int, int, int]:
        return color_rgb(self.values["color_hue"])

    # --- GA bridge ---------------------------------------------------------
    # Flatten the *evolvable* genes into an ordered list for the GA.
    def as_vector(self) -> list[float]:
        return [self.values[name] for name in evolvable_traits()]

    # Rebuild a genome from a GA vector of evolvable genes (clamped to range).
    # Non-evolvable traits come from ``base`` (or defaults).
    @classmethod
    def from_vector(
        cls, vector: Sequence[float], base: "SharkGenome | None" = None
    ) -> "SharkGenome":
        base = base or cls.default()
        values = dict(base.values)
        for name, raw in zip(evolvable_traits(), vector):
            values[name] = SHARK_TRAITS[name].clamp(raw)
        return cls(values)

    # --- evolution ---------------------------------------------------------
    # Return a mutated copy. Only evolvable genes can change.
    def mutate(self, mutation_rate: float, rng: random.Random = random) -> "SharkGenome":
        values = dict(self.values)
        for name in evolvable_traits():
            if rng.random() < mutation_rate:
                values[name] = SHARK_TRAITS[name].mutate(values[name], rng)
        return SharkGenome(values)

    # Blend two parents into two children (arithmetic crossover, clamped).
    # Evolvable genes blend with a random weight; non-evolvable genes are
    # inherited unchanged from the matching parent.
    @classmethod
    def crossover(
        cls, parent_a: "SharkGenome", parent_b: "SharkGenome", rng: random.Random = random
    ) -> tuple["SharkGenome", "SharkGenome"]:
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
