"""Shark domain model: traits and (later) behaviour."""

from .genome import SharkGenome
from .traits import (
    SHARK_TRAITS,
    TraitCategory,
    TraitSpec,
    color_rgb,
    evolvable_bounds,
    evolvable_traits,
    set_evolvable,
    trait_names,
)

__all__ = [
    "SHARK_TRAITS",
    "SharkGenome",
    "TraitCategory",
    "TraitSpec",
    "color_rgb",
    "evolvable_bounds",
    "evolvable_traits",
    "set_evolvable",
    "trait_names",
]
