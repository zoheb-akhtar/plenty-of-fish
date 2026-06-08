"""Shark domain model: traits, genome, and the living-shark lifecycle."""

from .genome import SharkGenome
from .shark import MAX_AGE_YEARS, Shark, advance_cycle, reproduce
from .traits import (
    SHARK_TRAITS,
    TraitCategory,
    TraitSpec,
    color_rgb,
    evolvable_traits,
)

__all__ = [
    "MAX_AGE_YEARS",
    "SHARK_TRAITS",
    "Shark",
    "SharkGenome",
    "TraitCategory",
    "TraitSpec",
    "advance_cycle",
    "color_rgb",
    "evolvable_traits",
    "reproduce",
]
