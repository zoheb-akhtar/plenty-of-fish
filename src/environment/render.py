"""Pygame drawing for a live ``World``.

Kept separate from ``world.py`` (pure simulation) and ``ocean.py`` (tile
background) so that headless training never imports pygame. The background tiles
come from ``Ocean``; this module just paints the fish and sharks on top, and a
small HUD.

Sharks are drawn in their genome colour and sized by their ``size`` gene, so the
GA's effect is visible at a glance: a generation of small, drab sharks looks
different from one of big, vivid ones.
"""
from __future__ import annotations

import pygame

from src.environment.ocean import Ocean
from src.environment.world import World
from src.shark import SHARK_TRAITS

SAFE_FISH_COLOR = (90, 220, 120)      # green = food
POISON_FISH_COLOR = (200, 80, 220)    # purple = danger
DEAD_SHARK_COLOR = (70, 70, 70)
HUD_COLOR = (255, 255, 255)


def _shark_radius(world_size_px_per_tile: int, size_value: float) -> int:
    """Map the ``size`` gene to a pixel radius that fits inside a tile."""
    spec = SHARK_TRAITS["size"]
    frac = (size_value - spec.min_value) / spec.span  # 0..1
    return max(3, int(world_size_px_per_tile * (0.25 + 0.2 * frac)))


def draw_world(
    surface: pygame.Surface,
    world: World,
    ocean: Ocean,
    font: pygame.font.Font,
    hud_lines: list[str] | None = None,
) -> None:
    """Render one frame: ocean tiles, fish, sharks, then the HUD text."""
    ts = ocean.tile_size

    ocean.draw(surface, cam_x=0, cam_y=0)

    # Fish: small filled circles centred in their tile.
    for fish in world.fishes:
        color = SAFE_FISH_COLOR if fish.kind == "safe" else POISON_FISH_COLOR
        center = (fish.x * ts + ts // 2, fish.y * ts + ts // 2)
        pygame.draw.circle(surface, color, center, max(3, ts // 5))

    # Sharks: bigger circles in genome colour; dead ones greyed out.
    for shark in world.sharks:
        center = (shark.x * ts + ts // 2, shark.y * ts + ts // 2)
        radius = _shark_radius(ts, shark.genome.size)
        color = shark.genome.color() if shark.alive else DEAD_SHARK_COLOR
        pygame.draw.circle(surface, color, center, radius)
        pygame.draw.circle(surface, (10, 10, 30), center, radius, width=1)
        if shark.alive:
            # A thin energy ring: brighter/fuller bodies are well-fed.
            energy_w = max(1, int(radius * shark.energy))
            pygame.draw.circle(surface, (255, 255, 255), center, energy_w, width=1)

    # HUD: a few status lines in the top-left, with a drop shadow for contrast.
    for i, line in enumerate(hud_lines or []):
        y = 6 + i * 18
        shadow = font.render(line, True, (0, 0, 0))
        text = font.render(line, True, HUD_COLOR)
        surface.blit(shadow, (7, y + 1))
        surface.blit(text, (6, y))
