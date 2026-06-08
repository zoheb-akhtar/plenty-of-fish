"""Play mode: control one shark in its own ocean (see ocean.py).

The shark's traits are randomised each run and shape how it plays:
  * size sets which fish you can eat (the biggest give the most energy),
  * speed sets how many tiles you cover per move,
  * perception sets how far you sense fish (the ring around you).

Run:  python -m src.environment.gui   (or python src/environment/gui.py)

Controls: Arrows/WASD move, Space attack, E rest, R new shark, Q/Esc quit.
"""

from __future__ import annotations

import os
import random
import sys

import pygame

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algorithms.rl_algo.actions import Action
from src.environment.config import DEFAULT_CONFIG, EnvConfig
from src.environment.fish_drawing import draw_fish
from src.environment.ocean import Ocean
from src.environment.seabed import Seabed
from src.environment.shark_drawing import make_genome_shark_sprite
from src.shark import SharkGenome

HUD_H = 112

HUD_BG = (12, 16, 28)
TEXT = (230, 236, 245)
TEXT_DIM = (150, 160, 175)
HIGHLIGHT = (255, 220, 90)
PERCEPTION_RING = (255, 255, 255, 28)

KEY_ACTIONS = {
    pygame.K_UP: Action.UP, pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN, pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT, pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT, pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.ATTACK,
    pygame.K_e: Action.REST,
}
FACING_ANGLE = {Action.RIGHT: 0, Action.UP: 90, Action.LEFT: 180, Action.DOWN: 270}
TIER_LABEL = ("small", "medium", "large")


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class PlayGUI:
    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config
        self.tile = config.tile_size
        self.size = config.size
        self.grid_px = self.size * self.tile

        pygame.init()
        pygame.display.set_caption("Plenty of Fish — Play")
        self.screen = pygame.display.set_mode((self.grid_px, HUD_H + self.grid_px))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.small = pygame.font.SysFont(None, 15)
        self.big = pygame.font.SysFont(None, 40, bold=True)
        self.seabed = Seabed(self.size, self.size, tile_size=self.tile, draw_grid=True)

        self.rng = random.Random(config.seed)
        self._new_shark()

    def _new_shark(self):
        self.genome = SharkGenome.random(self.rng)
        self.env = Ocean(self.config, genome=self.genome, rng=self.rng)
        self.env.reset()
        self.sprite = make_genome_shark_sprite(self.tile, self.genome.color(), self.env.size_norm)
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.result = ""

    def _over(self) -> bool:
        return self.env.done or not self.env.fishes

    def _apply(self, action: Action):
        if self._over():
            return
        _, reward, done = self.env.step(action)
        self.steps += 1
        self.last_reward = reward
        self.total_reward += reward
        if done:
            self.result = "You bit a poisonous fish!" if self.env.energy > 0 else "You starved"
        elif not self.env.fishes:
            self.result = "All fish eaten!"

    # --- loop ----------------------------------------------------------------
    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self._new_shark()
                    elif not self._over() and event.key in KEY_ACTIONS:
                        self._apply(KEY_ACTIONS[event.key])
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # --- rendering -----------------------------------------------------------
    def _center(self, tx, ty):
        return (tx * self.tile + self.tile // 2, ty * self.tile + self.tile // 2)

    def _draw(self):
        self.screen.fill(HUD_BG)
        play = self.screen.subsurface((0, HUD_H, self.grid_px, self.grid_px))
        self.seabed.draw(play, 0, 0)

        sx, sy = self.env.shark
        center = self._center(sx, sy)

        # perception ring (shows the sight trait)
        ring = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
        pygame.draw.circle(ring, PERCEPTION_RING, center, self.env.perception * self.tile, 0)
        play.blit(ring, (0, 0))

        # attack-reach highlight
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = sx + dx, sy + dy
            if 0 <= ax < self.size and 0 <= ay < self.size:
                pygame.draw.rect(play, (255, 255, 255, 40),
                                 pygame.Rect(ax * self.tile, ay * self.tile, self.tile, self.tile), 1)

        for f in self.env.fishes:
            fc = self._center(*f.pos)
            if self.env.in_reach(f.pos):
                pygame.draw.circle(play, HIGHLIGHT, fc, int(self.tile * 0.42), 2)
            draw_fish(play, f.kind, fc, self.tile, size=f.size)

        sprite = pygame.transform.rotate(self.sprite, FACING_ANGLE[self.env.facing])
        play.blit(sprite, sprite.get_rect(center=center))

        self._draw_hud()
        if self._over():
            self._draw_overlay()

    def _draw_hud(self):
        g, env = self.genome, self.env
        energy = max(0.0, env.energy)
        bar_x, bar_y, bar_w, bar_h = 12, 12, 220, 16
        col = _lerp((210, 60, 50), (70, 205, 130), energy)
        pygame.draw.rect(self.screen, (40, 46, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, col, (bar_x, bar_y, int(bar_w * energy), bar_h), border_radius=4)
        pygame.draw.rect(self.screen, (90, 100, 120), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)
        self.screen.blit(self.small.render(f"Energy {energy*100:3.0f}%", True, TEXT), (bar_x + bar_w + 10, bar_y + 1))

        traits = (
            f"Size {g.size:.1f} ({TIER_LABEL[env.size_tier]})   "
            f"Speed {g.speed:.1f} (->{env.move_tiles} tiles/move)   "
            f"Caution {g.caution:.2f}   Sight {env.perception}"
        )
        self.screen.blit(self.font.render(traits, True, TEXT), (12, 38))
        stats = f"Fish eaten {env.eaten}   Score {self.total_reward:6.1f}   Last {self.last_reward:+5.1f}"
        self.screen.blit(self.font.render(stats, True, TEXT), (12, 62))
        controls = "Arrows/WASD move  ·  Space attack  ·  E rest  ·  R new shark  ·  Q quit"
        self.screen.blit(self.small.render(controls, True, TEXT_DIM), (12, 88))

    def _draw_overlay(self):
        veil = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 150))
        self.screen.blit(veil, (0, HUD_H))
        cx, cy = self.grid_px // 2, HUD_H + self.grid_px // 2
        title = self.big.render(self.result or "Episode over", True, (255, 235, 235))
        hint = self.font.render("Press R for a new shark", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(cx, cy - 16)))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 28)))


def run():
    PlayGUI().run()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
