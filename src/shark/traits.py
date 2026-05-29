"""Shark trait definitions for the evolution simulation.

    Attributes per trait 
    ----------
    name: identifier used as the genome key and GA gene label.
    default: value used when a shark is built without evolution.
    min_value / max_value: inclusive legal range; values are clamped to it.
    evolvable: if True, the GA mutates/crosses this gene right now.
    mutation_sigma: std-dev of the Gaussian step used when mutating.
    unit / description: human-readable notes.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from enum import Enum


class TraitCategory(str, Enum):
    PRIMARY = "primary"        # core survival levers (size, speed, caution)
    SECONDARY = "secondary"    # flavour / situational (gestation, aggression, color)
    STATIONARY = "stationary"  # expected to stay put unless deliberately enabled


@dataclass(frozen=True)
class TraitSpec:
    name: str
    default: float
    min_value: float
    max_value: float
    category: TraitCategory
    evolvable: bool
    mutation_sigma: float
    unit: str = ""
    description: str = ""

    @property
    def span(self) -> float:
        return self.max_value - self.min_value

    def clamp(self, value: float) -> float:
        """Force ``value`` into the legal range."""
        return max(self.min_value, min(self.max_value, value))

    def random_value(self, rng: random.Random = random) -> float:
        """Draw a fresh uniform-random value inside the range."""
        return rng.uniform(self.min_value, self.max_value)

    def mutate(self, value: float, rng: random.Random = random) -> float:
        """Nudge ``value`` by a Gaussian step scaled to this trait, then clamp."""
        return self.clamp(value + rng.gauss(0.0, self.mutation_sigma))


# ---------------------------------------------------------------------------
# The shark trait registry.
#
# Order matters: it fixes the order of genes in the GA vector. To add a trait,
# append a TraitSpec here -- nothing else needs to change. Defaults and bounds
# are starting points tuned for a tile-based ocean; tweak freely.
# ---------------------------------------------------------------------------
_TRAIT_LIST: list[TraitSpec] = [
    TraitSpec(
        name="size",
        default=4.0,
        min_value=0.5,
        max_value=9.0,
        category=TraitCategory.PRIMARY,
        evolvable=True,
        mutation_sigma=0.4,
        unit="m",
        description="Body length. Bigger = stronger/more prey options but slower and hungrier.",
    ),
    TraitSpec(
        name="speed",
        default=5.0,
        min_value=1.0,
        max_value=12.0,
        category=TraitCategory.PRIMARY,
        evolvable=True,
        mutation_sigma=0.5,
        unit="tiles/step",
        description="Max travel per step. Aids hunting and fleeing but costs energy.",
    ),
    TraitSpec(
        name="caution",
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        category=TraitCategory.PRIMARY,
        evolvable=True,
        mutation_sigma=0.08,
        unit="0-1",
        description="Risk aversion. High = avoid threats/retreat early; low = take chances.",
    ),
    TraitSpec(
        name="gestation_period",
        default=12.0,
        min_value=4.0,
        max_value=24.0,
        category=TraitCategory.SECONDARY,
        evolvable=False,
        mutation_sigma=1.0,
        unit="months",
        description="Time to produce offspring. Shorter = faster breeding, costlier per cycle.",
    ),
    TraitSpec(
        name="aggression",
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        category=TraitCategory.SECONDARY,
        evolvable=False,
        mutation_sigma=0.08,
        unit="0-1",
        description="Tendency to attack vs. ignore. High = more hunts and more fights.",
    ),
    TraitSpec(
        name="field_of_perception",
        default=5.0,
        min_value=1.0,
        max_value=15.0,
        category=TraitCategory.STATIONARY,
        evolvable=False,
        mutation_sigma=0.6,
        unit="tiles (radius)",
        description="How far the shark senses prey/threats. Wider = more info, but assumed fixed for now.",
    ),
    TraitSpec(
        name="color_hue",
        default=205.0,
        min_value=0.0,
        max_value=360.0,
        category=TraitCategory.SECONDARY,
        evolvable=False,
        mutation_sigma=12.0,
        unit="degrees (HSV hue)",
        description="Camouflage/display colour as a hue 0-360. Use color_rgb() to render.",
    ),
]

#: Trait name -> spec. Insertion order = gene order in the GA vector.
SHARK_TRAITS: dict[str, TraitSpec] = {spec.name: spec for spec in _TRAIT_LIST}

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def evolvable_traits() -> list[str]:
    return [name for name, spec in SHARK_TRAITS.items() if spec.evolvable]


def color_rgb(hue: float) -> tuple[int, int, int]:
    """Convert a ``color_hue`` value (0-360) to an 0-255 RGB tuple for pygame."""
    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, 0.55, 0.75)
    return (int(r * 255), int(g * 255), int(b * 255))
