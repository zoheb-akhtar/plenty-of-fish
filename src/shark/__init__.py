"""Shark domain model: traits and (later) behaviour."""

from .genome import SharkGenome
from .traits import (
    SHARK_TRAITS,
    TraitCategory,
    TraitSpec,
    color_rgb,
    evolvable_traits,
)

__all__ = [
    "SHARK_TRAITS",
    "SharkGenome",
    "TraitCategory",
    "TraitSpec",
    "color_rgb",
    "evolvable_traits",
]
