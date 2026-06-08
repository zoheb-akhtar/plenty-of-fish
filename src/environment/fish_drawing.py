"""Fish sprites for the ocean GUIs.

Safe fish are green; poisonous fish are purple and carry a ``!`` marker.
"""

from __future__ import annotations

import pygame

SAFE = (70, 205, 130)
POISON = (185, 80, 210)

# Body radius as a fraction of the tile, per fish size.
SIZE_SCALE = {"small": 0.18, "medium": 0.26, "large": 0.34}

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _marker_font(size: int) -> pygame.font.Font:
    """Cache the poison-marker font by size (avoid rebuilding it every frame)."""
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont(None, size, bold=True)
    return _FONT_CACHE[size]


def draw_fish(
    surface: pygame.Surface,
    kind: str,
    center: tuple[int, int],
    tile: int,
    size: str = "medium",
) -> None:
    """Draw one fish centred at ``center`` (pixels) on ``surface``, scaled by ``size``."""
    cx, cy = center
    color = SAFE if kind == "safe" else POISON
    r = max(3, int(tile * SIZE_SCALE.get(size, 0.26)))

    # body + tail (tail points left, "behind" the body)
    body = pygame.Rect(cx - r, cy - int(r * 0.62), 2 * r, int(r * 1.24))
    pygame.draw.ellipse(surface, color, body)
    pygame.draw.polygon(
        surface,
        color,
        [(cx - r, cy), (cx - r - int(r * 0.7), cy - int(r * 0.5)), (cx - r - int(r * 0.7), cy + int(r * 0.5))],
    )

    # eye
    pygame.draw.circle(surface, (250, 250, 250), (cx + int(r * 0.45), cy - int(r * 0.2)), max(2, int(r * 0.18)))
    pygame.draw.circle(surface, (20, 20, 25), (cx + int(r * 0.5), cy - int(r * 0.2)), max(1, int(r * 0.09)))

    # poisonous marker
    if kind != "safe":
        mark = _marker_font(int(tile * 0.34)).render("!", True, (255, 255, 255))
        surface.blit(mark, mark.get_rect(center=(cx, cy + int(r * 0.05))))
