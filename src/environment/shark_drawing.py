"""Shark sprite for the GridOcean GUI.

Builds a right-facing shark surface once; the GUI rotates it to match the
shark's current facing before blitting.
"""

from __future__ import annotations

import pygame

BODY = (96, 108, 124)
BELLY = (150, 162, 178)


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
