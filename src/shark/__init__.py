"""Shark domain model: traits, genome, and the living-shark lifecycle."""

from .genome import SharkGenome
from .shark import (
    BIRTH_SIZE_FRACTION,
    MAX_AGE_YEARS,
    MIN_BREEDING_AGE_YEARS,
    Shark,
    advance_cycle,
    reproduce,
)
from .traits import (
    SHARK_TRAITS,
    TraitCategory,
    TraitSpec,
    color_rgb,
    evolvable_traits,
)

__all__ = [
    "BIRTH_SIZE_FRACTION",
    "MAX_AGE_YEARS",
    "MIN_BREEDING_AGE_YEARS",
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
