"""Shark sprites for the ocean GUIs.

Builds a right-facing shark surface once; the GUI rotates it to match the
shark's current facing before blitting.
"""

from __future__ import annotations

import pygame

BODY = (96, 108, 124)
BELLY = (150, 162, 178)


def _lighten(color: tuple[int, int, int], amt: float = 0.4) -> tuple[int, int, int]:
    """Mix ``color`` toward white by ``amt`` (0..1) for a belly highlight."""
    return tuple(int(c + (255 - c) * amt) for c in color)


def _draw_shark(surf: pygame.Surface, h: float, body_color, belly_color) -> None:
    """Paint a right-facing shark of half-extent ``h`` centred on ``surf``."""
    cx = cy = surf.get_width() / 2
    body = [
        (cx + h, cy),                       # nose
        (cx - h * 0.55, cy - h * 0.55),     # back top
        (cx - h, cy - h * 0.45),            # tail top
        (cx - h * 0.6, cy),                 # tail notch
        (cx - h, cy + h * 0.45),            # tail bottom
        (cx - h * 0.55, cy + h * 0.55),     # back bottom
    ]
    pygame.draw.polygon(surf, body_color, body)
    pygame.draw.polygon(  # dorsal fin
        surf,
        body_color,
        [(cx - h * 0.1, cy - h * 0.5), (cx - h * 0.5, cy - h * 1.0), (cx - h * 0.5, cy - h * 0.5)],
    )
    pygame.draw.polygon(  # belly accent
        surf,
        belly_color,
        [(cx + h, cy), (cx - h * 0.55, cy + h * 0.55), (cx - h * 0.6, cy)],
    )
    pygame.draw.circle(surf, (245, 245, 245), (int(cx + h * 0.45), int(cy - h * 0.16)), max(2, int(h * 0.12)))
    pygame.draw.circle(surf, (20, 20, 25), (int(cx + h * 0.5), int(cy - h * 0.16)), max(1, int(h * 0.06)))


def make_genome_shark_sprite(
    tile: int, color: tuple[int, int, int], size_norm: float = 0.5
) -> pygame.Surface:
    """A right-facing shark tinted by ``color`` (genome.color()) and scaled by ``size_norm`` (0..1)."""
    surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    h = tile * (0.26 + 0.13 * max(0.0, min(1.0, size_norm)))  # bigger sharks look bigger
    _draw_shark(surf, h, color, _lighten(color))
    return surf


def make_shark_sprite(tile: int) -> pygame.Surface:
    """A right-facing shark drawn into a ``tile``x``tile`` transparent surface."""
    surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    cx = cy = tile / 2
    h = tile * 0.36
    body = [
        (cx + h, cy),                       # nose
        (cx - h * 0.55, cy - h * 0.55),     # back top
        (cx - h, cy - h * 0.45),            # tail top
        (cx - h * 0.6, cy),                 # tail notch
        (cx - h, cy + h * 0.45),            # tail bottom
        (cx - h * 0.55, cy + h * 0.55),     # back bottom
    ]
    pygame.draw.polygon(surf, BODY, body)
    # dorsal fin
    pygame.draw.polygon(
        surf,
        BODY,
        [(cx - h * 0.1, cy - h * 0.5), (cx - h * 0.5, cy - h * 1.0), (cx - h * 0.5, cy - h * 0.5)],
    )
    # belly accent
    pygame.draw.polygon(
        surf,
        BELLY,
        [(cx + h, cy), (cx - h * 0.55, cy + h * 0.55), (cx - h * 0.6, cy)],
    )
    # eye
    pygame.draw.circle(surf, (245, 245, 245), (int(cx + h * 0.45), int(cy - h * 0.16)), max(2, int(h * 0.12)))
    pygame.draw.circle(surf, (20, 20, 25), (int(cx + h * 0.5), int(cy - h * 0.16)), max(1, int(h * 0.06)))
    return surf
