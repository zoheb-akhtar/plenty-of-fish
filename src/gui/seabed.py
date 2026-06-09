"""Seabed tile renderer for the ocean GUIs.

Pure drawing: a blue vertical gradient with a little deterministic per-tile
jitter, optional grid lines, and a camera offset. The world/simulation lives in
``ocean.py``; this module only knows how to paint the floor.
"""

from __future__ import annotations

import random
from typing import Tuple

import pygame


class Seabed:
    def __init__(self, tiles_x: int, tiles_y: int, tile_size: int = 64, draw_grid: bool = True):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.tile_size = tile_size
        self.draw_grid = draw_grid

    @property
    def width(self) -> int:
        return self.tiles_x * self.tile_size

    @property
    def height(self) -> int:
        return self.tiles_y * self.tile_size

    def _tile_color(self, tx: int, ty: int) -> Tuple[int, int, int]:
        """A blue shade for tile (tx, ty): vertical gradient + small per-tile jitter."""
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

    def draw(self, surface: pygame.Surface, cam_x: int = 0, cam_y: int = 0) -> None:
        """Paint the visible portion of the seabed onto ``surface`` (cam = world px of top-left)."""
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
