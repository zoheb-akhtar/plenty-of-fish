"""Interactive pygame GUI for :class:`GridOcean`.

Draws the seabed with the :class:`Ocean` tile renderer (``ocean.py``) and
overlays the shark and fish. Play manually, or hit Tab to watch a random agent.

Run:  ``python -m src.environment.gui``   (from the repo root)
  or  ``python src/environment/gui.py``

Controls:
  Arrows / WASD : move the shark one tile
  Space         : attack an adjacent fish
  E             : rest (regain energy)
  Tab           : toggle auto mode (random actions)
  R             : reset the episode
  Esc / Q       : quit
"""

from __future__ import annotations

import random
import sys

import pygame

# Work both as a package module (-m src.environment.gui) and as a script.
try:
    from .config import DEFAULT_CONFIG, EnvConfig
    from .fish_drawing import draw_fish
    from .grid_ocean import Action, GridOcean
    from .ocean import Ocean
    from .shark_drawing import make_shark_sprite
except ImportError:  # run directly: src/environment is on sys.path
    from config import DEFAULT_CONFIG, EnvConfig
    from environment.fish_drawing import draw_fish
    from grid_ocean import Action, GridOcean
    from ocean import Ocean
    from environment.shark_drawing import make_shark_sprite


# --- layout -----------------------------------------------------------------
HUD_H = 96
FONT_NAME = None  # default system font

# --- colours ----------------------------------------------------------------
HUD_BG = (12, 16, 28)
TEXT = (230, 236, 245)
TEXT_DIM = (150, 160, 175)
HIGHLIGHT = (255, 220, 90)

# Action bound to each key. Movement keys also set the shark's facing.
KEY_ACTIONS = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.ATTACK,
    pygame.K_e: Action.REST,
}

# Degrees to rotate the right-facing shark sprite for each facing (CCW).
FACING_ANGLE = {
    Action.RIGHT: 0,
    Action.UP: 90,
    Action.LEFT: 180,
    Action.DOWN: 270,
}

AUTO_STEP_INTERVAL = 0.30  # seconds between random actions in auto mode


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class GridOceanGUI:
    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config
        self.env = GridOcean(config)
        self.size = config.size
        self.tile = config.tile_size
        self.grid_px = self.size * self.tile

        pygame.init()
        pygame.display.set_caption("Plenty of Fish — GridOcean")
        self.screen = pygame.display.set_mode((self.grid_px, HUD_H + self.grid_px))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 20)
        self.small = pygame.font.SysFont(FONT_NAME, 15)
        self.big = pygame.font.SysFont(FONT_NAME, 40, bold=True)

        self.ocean = Ocean(self.size, self.size, tile_size=self.tile, draw_grid=True)
        self.shark_sprite = make_shark_sprite(self.tile)

        self.total_safe = config.num_safe_fish
        self._new_episode()

    def _new_episode(self):
        self.env.reset()
        self.facing = Action.RIGHT
        self.steps = 0
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.auto = False
        self.auto_timer = 0.0
        self.result = ""

    def _apply(self, action: Action):
        if self.env.done:
            return
        if action in FACING_ANGLE:
            self.facing = action
        _, reward, done = self.env.step(action)
        self.steps += 1
        self.last_reward = reward
        self.total_reward += reward
        if done:
            if self.env.energy <= 0:
                self.result = "Starved"
            elif reward <= -100:
                self.result = "Ate a poisonous fish!"
            else:
                self.result = "Episode over"

    # --- main loop ----------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self._new_episode()
                    elif event.key == pygame.K_TAB:
                        self.auto = not self.auto
                        self.auto_timer = 0.0
                    elif not self.auto and event.key in KEY_ACTIONS:
                        self._apply(KEY_ACTIONS[event.key])

            if self.auto and not self.env.done:
                self.auto_timer += dt
                if self.auto_timer >= AUTO_STEP_INTERVAL:
                    self.auto_timer = 0.0
                    self._apply(Action(random.randint(0, len(Action) - 1)))

            self._draw()
            pygame.display.flip()

        pygame.quit()

    # --- rendering ----------------------------------------------------------
    def _center(self, tx: int, ty: int) -> tuple[int, int]:
        return (tx * self.tile + self.tile // 2, ty * self.tile + self.tile // 2)

    def _draw(self):
        self.screen.fill(HUD_BG)
        play = self.screen.subsurface((0, HUD_H, self.grid_px, self.grid_px))
        self.ocean.draw(play, 0, 0)

        # highlight tiles adjacent to the shark (what ATTACK can reach)
        sx, sy = self.env.shark
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = sx + dx, sy + dy
            if 0 <= ax < self.size and 0 <= ay < self.size:
                rect = pygame.Rect(ax * self.tile, ay * self.tile, self.tile, self.tile)
                pygame.draw.rect(play, (255, 255, 255, 40), rect, 1)

        for kind, (fx, fy) in self.env.fishes:
            center = self._center(fx, fy)
            if self.env._adjacent((fx, fy)):
                pygame.draw.circle(play, HIGHLIGHT, center, int(self.tile * 0.42), 2)
            draw_fish(play, kind, center, self.tile)

        sprite = pygame.transform.rotate(self.shark_sprite, FACING_ANGLE[self.facing])
        play.blit(sprite, sprite.get_rect(center=self._center(sx, sy)))

        self._draw_hud()
        if self.env.done:
            self._draw_overlay()

    def _draw_hud(self):
        # energy bar
        bar_x, bar_y, bar_w, bar_h = 12, 14, 220, 18
        energy = max(0.0, self.env.energy)
        col = _lerp((210, 60, 50), (70, 205, 130), energy)
        pygame.draw.rect(self.screen, (40, 46, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, col, (bar_x, bar_y, int(bar_w * energy), bar_h), border_radius=4)
        pygame.draw.rect(self.screen, (90, 100, 120), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)
        self.screen.blit(self.small.render(f"Energy {energy*100:3.0f}%", True, TEXT), (bar_x + bar_w + 10, bar_y + 1))

        safe_left = sum(1 for kind, _ in self.env.fishes if kind == "safe")
        mode = "AUTO (random)" if self.auto else "Manual"
        line = (
            f"Steps {self.steps}   Score {self.total_reward:6.1f}   "
            f"Last {self.last_reward:+5.1f}   Safe fish {safe_left}/{self.total_safe}   Mode {mode}"
        )
        self.screen.blit(self.font.render(line, True, TEXT), (12, 42))

        controls = "Arrows/WASD move  ·  Space attack  ·  E rest  ·  Tab auto  ·  R reset  ·  Q quit"
        self.screen.blit(self.small.render(controls, True, TEXT_DIM), (12, 72))

    def _draw_overlay(self):
        veil = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 150))
        self.screen.blit(veil, (0, HUD_H))
        cx = self.grid_px // 2
        cy = HUD_H + self.grid_px // 2
        title = self.big.render(self.result or "Episode over", True, (255, 235, 235))
        hint = self.font.render("Press R to play again", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(cx, cy - 16)))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 28)))


def run():
    GridOceanGUI().run()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
