"""Seabed tile renderer: a blue vertical gradient with per-tile jitter and grid lines.

Pure drawing with a camera offset; the world/simulation lives in ``ocean.py``.
"""

from __future__ import annotations

import random
from typing import Tuple

import pygame


class Seabed:
    # Build a seabed of tiles_x by tiles_y tiles of the given pixel size.
    def __init__(self, tiles_x: int, tiles_y: int, tile_size: int = 64, draw_grid: bool = True):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.tile_size = tile_size
        self.draw_grid = draw_grid

    # Total seabed width in pixels.
    @property
    def width(self) -> int:
        return self.tiles_x * self.tile_size

    # Total seabed height in pixels.
    @property
    def height(self) -> int:
        return self.tiles_y * self.tile_size

    # A blue shade for tile (tx, ty): vertical gradient + small per-tile jitter.
    def _tile_color(self, tx: int, ty: int) -> Tuple[int, int, int]:
        t = ty / max(1, self.tiles_y - 1)
        top_r, top_g, top_b = 20, 140, 200
        bot_r, bot_g, bot_b = 0, 40, 120
        r = int(top_r * (1 - t) + bot_r * t)
        g = int(top_g * (1 - t) + bot_g * t)
        b = int(top_b * (1 - t) + bot_b * t)

        rnd = random.Random((tx << 16) ^ ty)
        jitter = rnd.randint(-12, 12)
        g = max(0, min(255, g + jitter))
        b = max(0, min(255, b + jitter // 2))
        return (r, g, b)

    # Paint the visible portion of the seabed onto ``surface`` (cam = world px of top-left).
    def draw(self, surface: pygame.Surface, cam_x: int = 0, cam_y: int = 0) -> None:
        surf_w, surf_h = surface.get_size()

        start_tx = max(0, cam_x // self.tile_size)
        start_ty = max(0, cam_y // self.tile_size)
        end_tx = min(self.tiles_x - 1, (cam_x + surf_w) // self.tile_size)
        end_ty = min(self.tiles_y - 1, (cam_y + surf_h) // self.tile_size)

        for ty in range(start_ty, end_ty + 1):
            for tx in range(start_tx, end_tx + 1):
                rect = pygame.Rect(
                    tx * self.tile_size - cam_x,
                    ty * self.tile_size - cam_y,
                    self.tile_size,
                    self.tile_size,
                )
                surface.fill(self._tile_color(tx, ty), rect)

        if self.draw_grid:
            for tx in range(start_tx, end_tx + 2):
                x = tx * self.tile_size - cam_x
                pygame.draw.line(surface, (10, 10, 30), (x, 0), (x, surf_h), 1)
            for ty in range(start_ty, end_ty + 2):
                y = ty * self.tile_size - cam_y
                pygame.draw.line(surface, (10, 10, 30), (0, y), (surf_w, y), 1)
