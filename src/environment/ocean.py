"""Ocean background demo using pygame.

Features:
- Adjustable grid tile size via `tile_size`.
- Different shades of blue per tile (gradient + per-tile variation).
- Basic visible viewport with camera movement (arrow keys / WASD).
- Optional grid lines.

Run as a demo: `python -m src.environment.ocean` or `python src/environment/ocean.py`
"""
from __future__ import annotations

import pygame
import math
import random
import sys
from typing import Tuple

<<<<<<< HEAD
# used chat to build this quickly - for AI comment, will refine further later... 
=======

>>>>>>> 8cc10e695705d1379c239911c1d436df1cdec994

class Ocean:
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
        """Return a blue shade for tile at tile coords (tx, ty).

        Uses a vertical gradient (deeper at larger ty) plus a small deterministic
        per-tile variation so neighboring tiles have slightly different blues.
        """
        # Base gradient from light to deep blue
        t = ty / max(1, self.tiles_y - 1)
        # gradient components
        top_r, top_g, top_b = 20, 140, 200
        bot_r, bot_g, bot_b = 0, 40, 120
        r = int(top_r * (1 - t) + bot_r * t)
        g = int(top_g * (1 - t) + bot_g * t)
        b = int(top_b * (1 - t) + bot_b * t)

        # deterministic per-tile jitter
        rnd = random.Random((tx << 16) ^ ty)
        jitter = rnd.randint(-12, 12)
        g = max(0, min(255, g + jitter))
        b = max(0, min(255, b + jitter // 2))
        return (r, g, b)

    def draw(self, surface: pygame.Surface, cam_x: int, cam_y: int) -> None:
        """Draw the visible portion of the ocean onto `surface` using camera offset.

        cam_x/cam_y are world pixel coordinates of the top-left of the viewport.
        """
        surf_w, surf_h = surface.get_size()

        start_tx = max(0, cam_x // self.tile_size)
        start_ty = max(0, cam_y // self.tile_size)
        end_tx = min(self.tiles_x - 1, (cam_x + surf_w) // self.tile_size)
        end_ty = min(self.tiles_y - 1, (cam_y + surf_h) // self.tile_size)

        # draw tiles
        for ty in range(start_ty, end_ty + 1):
            for tx in range(start_tx, end_tx + 1):
                color = self._tile_color(tx, ty)
                rect = pygame.Rect(
                    tx * self.tile_size - cam_x,
                    ty * self.tile_size - cam_y,
                    self.tile_size,
                    self.tile_size,
                )
                surface.fill(color, rect)

        # optional grid lines
        if self.draw_grid:
            # vertical lines
            for tx in range(start_tx, end_tx + 2):
                x = tx * self.tile_size - cam_x
                pygame.draw.line(surface, (10, 10, 30), (x, 0), (x, surf_h), 1)
            # horizontal lines
            for ty in range(start_ty, end_ty + 2):
                y = ty * self.tile_size - cam_y
                pygame.draw.line(surface, (10, 10, 30), (0, y), (surf_w, y), 1)


def run_demo(
    viewport_size: Tuple[int, int] = (800, 600),
    tiles_x: int = 120,
    tiles_y: int = 80,
    tile_size: int = 48,
):
    pygame.init()
    pygame.display.set_caption("Ocean background demo")
    screen = pygame.display.set_mode(viewport_size)
    clock = pygame.time.Clock()

    ocean = Ocean(tiles_x, tiles_y, tile_size=tile_size, draw_grid=True)

    # camera is top-left pixel of viewport in world coords
    cam_x = max(0, (ocean.width - viewport_size[0]) // 2)
    cam_y = max(0, (ocean.height - viewport_size[1]) // 2)

    speed = 800  # pixels per second

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1

        # normalize diagonal movement
        if dx != 0 and dy != 0:
            nd = math.sqrt(2) / 2
            dx *= nd
            dy *= nd

        cam_x += int(dx * speed * dt)
        cam_y += int(dy * speed * dt)

        # clamp camera to world bounds
        cam_x = max(0, min(ocean.width - viewport_size[0], cam_x))
        cam_y = max(0, min(ocean.height - viewport_size[1], cam_y))

        # draw
        ocean.draw(screen, cam_x, cam_y)

        # small HUD
        font = pygame.font.SysFont(None, 20)
        text = font.render(
            f"Cam: ({cam_x}, {cam_y})  Tile: {tile_size}px  World: {ocean.tiles_x}x{ocean.tiles_y}",
            True,
            (255, 255, 255),
        )
        # subtle drop shadow
        screen.blit(text, (6, 6))
        screen.blit(text, (5, 5))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # Default demo parameters — change these to control the grid size and world size.
    run_demo(viewport_size=(1024, 768), tiles_x=140, tiles_y=100, tile_size=48)