"""Watch mode: spectate a shark family evolve.

A small population of sharks each forage in their OWN ocean (a grid of
mini-oceans), all sharing one ``FamilyRL`` brain. As you watch:

  * the brain learns every step (tabular Q-learning), and
  * between generations the genetic algorithm breeds the next family
    (tournament selection + crossover + mutation, with elitism).

So both the bodies (traits) and the shared behaviour improve over time. This is
the animated version of ``genetic_algorithm(..., population_fitness=...)``.

Run:  python -m src.environment.watch_gui   (or python src/environment/watch_gui.py)
Controls: Space pause/resume · Q/Esc quit.
"""

from __future__ import annotations

import math
import os
import random
import sys

import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algorithms.genetic_algo import create_initial_population, selection
from src.algorithms.rl_algo import FamilyRL, discretize
from src.environment.config import DEFAULT_CONFIG, EnvConfig
from src.environment.fish_drawing import draw_fish
from src.environment.gui import FACING_ANGLE  # reuse the rotation map
from src.environment.ocean import Ocean
from src.environment.shark_drawing import make_genome_shark_sprite
from src.shark import SHARK_TRAITS, SharkGenome, evolvable_traits
from src.simulation import biased_action

HUD_H = 64
HUD_BG = (12, 16, 28)
TEXT = (230, 236, 245)
TEXT_DIM = (150, 160, 175)
CELL_BG = (10, 52, 96)
CELL_BORDER = (40, 46, 60)

# Buttons (Metrics / Back).
BTN_BG = (32, 54, 92)
BTN_HOT = (46, 84, 140)
BTN_BORDER = (90, 130, 190)

# Matplotlib styling for the metrics panel (hex strings for matplotlib).
TEXT_HEX = "#e6ecf5"    # = TEXT, as a hex string matplotlib understands
PLOT_FACE = "#0c1019"   # figure background, matches HUD_BG
AX_FACE = "#10243f"     # axes background, matches CELL_BG-ish
AX_GRID = "#26304a"
AX_SPINE = "#39435a"
AX_TEXT = "#aab2c5"
PLOT_BEST = "#ffd24a"
PLOT_AVG = "#7fd1a0"
PLOT_RANGE = "#4a6fa5"
PLOT_TRAIT = "#ff9d5c"


class WatchGUI:
    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config
        pop = config.watch_population_size
        self.pop_size = pop if pop % 2 == 0 else pop + 1  # breed in pairs
        self.cols = config.watch_grid_cols
        self.rows = math.ceil(self.pop_size / self.cols)

        # Window: a cols x rows grid of square-ish mini-oceans below the HUD.
        cell = 230
        self.cell_w = cell
        self.cell_h = cell
        self.cell_tile = max(4, (cell - 16) // config.size)
        self.width = self.cols * self.cell_w
        self.height = HUD_H + self.rows * self.cell_h

        pygame.init()
        pygame.display.set_caption("Plenty of Fish — Watch the family evolve")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.small = pygame.font.SysFont(None, 14)

        self.rng = random.Random(config.seed)
        self.brain = FamilyRL()
        self.generation = 0
        self.best_fitness = float("nan")
        self.avg_fitness = float("nan")
        self.paused = False

        # View state: "watch" (the grid) or "metrics" (the live graphs).
        self.view = "watch"
        self.metrics_btn = pygame.Rect(self.width - 132, 14, 118, 32)
        self.back_btn = pygame.Rect(14, 14, 96, 32)

        # Per-generation history that the metrics graphs plot.
        self.genes = evolvable_traits()
        self.best_hist: list[float] = []
        self.avg_hist: list[float] = []
        self.min_hist: list[float] = []
        self.max_hist: list[float] = []
        self.trait_hist: dict[str, list[float]] = {g: [] for g in self.genes}
        # Cache the rendered figure; re-render only when a new generation lands.
        self._metrics_surf: pygame.Surface | None = None
        self._metrics_cached_gen = -1

        self._start_generation(create_initial_population(self.pop_size, self.rng))

    # --- generation lifecycle ------------------------------------------------
    def _start_generation(self, population: list[SharkGenome]):
        self.population = population
        self.envs = [Ocean(self.config, genome=g, rng=self.rng) for g in population]
        for e in self.envs:
            e.reset()
        self.states = [discretize(e.observe()) for e in self.envs]
        self.totals = [0.0] * self.pop_size
        self.step_in_life = 0
        self.sprites = [
            make_genome_shark_sprite(self.cell_tile, g.color(), e.size_norm)
            for g, e in zip(population, self.envs)
        ]

    def _tick(self):
        """Advance every living shark one step (and learn); roll the generation when all are done."""
        all_done = True
        for i, env in enumerate(self.envs):
            if env.done:
                continue
            all_done = False
            action = biased_action(self.brain, self.states[i], self.population[i], self.rng)
            obs, reward, done = env.step(action)
            next_state = discretize(obs)
            self.brain.update(self.states[i], action, reward, next_state, done)
            self.states[i] = next_state
            self.totals[i] += reward
            if not env.fishes:  # ate everything -> that life is over
                env.done = True
        self.step_in_life += 1
        if all_done or self.step_in_life >= self.config.watch_max_steps:
            self._end_generation()

    def _end_generation(self):
        fitnesses = list(self.totals)
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_genome = self.population[best_idx]

        # Record this generation's stats so the metrics graphs can plot them.
        self.best_hist.append(self.best_fitness)
        self.avg_hist.append(self.avg_fitness)
        self.min_hist.append(min(fitnesses))
        self.max_hist.append(max(fitnesses))
        for gene in self.genes:
            self.trait_hist[gene].append(best_genome[gene])

        breeders = selection(self.population, fitnesses, self.rng)
        nxt: list[SharkGenome] = []
        for i in range(0, len(breeders), 2):
            c1, c2 = SharkGenome.crossover(breeders[i], breeders[i + 1], self.rng)
            nxt.append(c1.mutate(self.config.watch_mutation_rate, self.rng))
            nxt.append(c2.mutate(self.config.watch_mutation_rate, self.rng))
        nxt[0] = best_genome  # elitism: carry the best body over unchanged

        self.brain.decay_epsilon()
        self.generation += 1
        self._start_generation(nxt)

    # --- loop ----------------------------------------------------------------
    def run(self):
        running = True
        acc = 0.0
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        # Esc backs out of metrics first, then quits.
                        if self.view == "metrics":
                            self.view = "watch"
                        else:
                            running = False
                    elif event.key == pygame.K_m:
                        self.view = "watch" if self.view == "metrics" else "metrics"
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            # The simulation keeps running in either view, so the graphs update live.
            if not self.paused:
                acc += dt
                if acc >= self.config.watch_tick_interval:
                    acc = 0.0
                    self._tick()
            if self.view == "metrics":
                self._draw_metrics()
            else:
                self._draw()
            pygame.display.flip()
        pygame.quit()

    def _handle_click(self, pos):
        if self.view == "watch" and self.metrics_btn.collidepoint(pos):
            self.view = "metrics"
        elif self.view == "metrics" and self.back_btn.collidepoint(pos):
            self.view = "watch"

    # --- rendering -----------------------------------------------------------
    def _draw(self):
        self.screen.fill(HUD_BG)
        for i in range(self.pop_size):
            cx = (i % self.cols) * self.cell_w
            cy = HUD_H + (i // self.cols) * self.cell_h
            self._draw_cell(i, cx, cy)
        self._draw_hud()

    def _draw_cell(self, i: int, x0: int, y0: int):
        env = self.envs[i]
        cell = pygame.Rect(x0 + 1, y0 + 1, self.cell_w - 2, self.cell_h - 2)
        pygame.draw.rect(self.screen, CELL_BG, cell)

        t = self.cell_tile
        grid_px = env.size * t
        ox = x0 + (self.cell_w - grid_px) // 2
        oy = y0 + (self.cell_h - grid_px) // 2

        for f in env.fishes:
            fc = (ox + f.pos[0] * t + t // 2, oy + f.pos[1] * t + t // 2)
            draw_fish(self.screen, f.kind, fc, t, size=f.size)

        sx, sy = env.shark
        sprite = pygame.transform.rotate(self.sprites[i], FACING_ANGLE[env.facing])
        self.screen.blit(sprite, sprite.get_rect(center=(ox + sx * t + t // 2, oy + sy * t + t // 2)))

        if env.done:  # dim finished lives
            veil = pygame.Surface(cell.size, pygame.SRCALPHA)
            veil.fill((0, 0, 0, 120))
            self.screen.blit(veil, cell.topleft)

        tag = f"#{i}  r={self.totals[i]:.0f}  ate={env.eaten}" + ("  x" if env.done else "")
        self.screen.blit(self.small.render(tag, True, TEXT), (x0 + 5, y0 + 4))
        pygame.draw.rect(self.screen, CELL_BORDER, cell, 1)

    def _draw_hud(self):
        best = "—" if math.isnan(self.best_fitness) else f"{self.best_fitness:6.1f}"
        avg = "—" if math.isnan(self.avg_fitness) else f"{self.avg_fitness:6.1f}"
        line1 = (
            f"Generation {self.generation}   Population {self.pop_size}   "
            f"Best {best}   Avg {avg}   "
            f"epsilon {self.brain.epsilon:.3f}   Q-states {len(self.brain.q)}"
        )
        self.screen.blit(self.font.render(line1, True, TEXT), (12, 10))
        line2 = "Each shark forages its own ocean; all share one evolving brain.  Space pause · M metrics · Q quit"
        if self.paused:
            line2 += "   [PAUSED]"
        self.screen.blit(self.small.render(line2, True, TEXT_DIM), (12, 38))
        self._draw_button(self.metrics_btn, "Metrics >")

    # --- buttons / metrics view ----------------------------------------------
    def _draw_button(self, rect: pygame.Rect, label: str):
        hot = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, BTN_HOT if hot else BTN_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BTN_BORDER, rect, 1, border_radius=6)
        txt = self.font.render(label, True, TEXT)
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_metrics(self):
        self.screen.fill(HUD_BG)
        self._draw_button(self.back_btn, "< Back")
        title = self.font.render(
            f"Metrics · generation {self.generation} · population {self.pop_size}", True, TEXT
        )
        self.screen.blit(title, (self.back_btn.right + 18, 20))

        top = HUD_H
        area_w, area_h = self.width, self.height - top
        if not self.best_hist:
            msg = self.small.render(
                "Collecting data — the first generation is still running.", True, TEXT_DIM
            )
            self.screen.blit(msg, msg.get_rect(center=(self.width // 2, top + area_h // 2)))
            return

        # Re-render the figure only when a new generation has been recorded.
        if self._metrics_surf is None or self._metrics_cached_gen != len(self.best_hist):
            self._metrics_surf = self._render_metrics_figure(area_w, area_h)
            self._metrics_cached_gen = len(self.best_hist)
        self.screen.blit(self._metrics_surf, (0, top))

    def _render_metrics_figure(self, px_w: int, px_h: int) -> pygame.Surface:
        """Draw the per-generation graphs to a pygame surface via matplotlib (Agg)."""
        dpi = 100
        fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor=PLOT_FACE)

        n_plots = 1 + len(self.genes)  # fitness + one per evolvable trait
        cols = 2
        rows = math.ceil(n_plots / cols)
        axes = fig.subplots(rows, cols, squeeze=False)
        flat = [axes[r][c] for r in range(rows) for c in range(cols)]
        for ax in flat[n_plots:]:  # hide any unused grid cells
            ax.set_visible(False)

        gens = range(1, len(self.best_hist) + 1)

        def _style(ax, title: str, ylabel: str):
            ax.set_title(title, color=TEXT_HEX, fontsize=10)
            ax.set_xlabel("generation", color=AX_TEXT, fontsize=8)
            ax.set_ylabel(ylabel, color=AX_TEXT, fontsize=8)
            ax.set_facecolor(AX_FACE)
            ax.tick_params(colors=AX_TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(AX_SPINE)
            ax.grid(True, color=AX_GRID, linewidth=0.5)

        # Fitness: best line, average line, and the population min..max band.
        ax = flat[0]
        ax.fill_between(gens, self.min_hist, self.max_hist,
                        color=PLOT_RANGE, alpha=0.35, label="pop range")
        ax.plot(gens, self.best_hist, color=PLOT_BEST, linewidth=1.8, label="best")
        ax.plot(gens, self.avg_hist, color=PLOT_AVG, linewidth=1.2, label="avg")
        _style(ax, "Fitness (reward) over generations", "reward")
        leg = ax.legend(fontsize=6, facecolor=AX_FACE, edgecolor=AX_SPINE)
        for text in leg.get_texts():
            text.set_color(TEXT_HEX)

        # One subplot per evolvable trait: the best shark's value over time.
        for ax, gene in zip(flat[1:], self.genes):
            spec = SHARK_TRAITS[gene]
            ax.plot(gens, self.trait_hist[gene], color=PLOT_TRAIT,
                    marker="o", markersize=2, linewidth=1.4)
            ax.set_ylim(spec.min_value, spec.max_value)
            _style(ax, f"Best shark's {gene}", spec.unit)

        fig.tight_layout(pad=1.4)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        return pygame.image.frombuffer(bytes(canvas.buffer_rgba()), (w, h), "RGBA")


def run():
    WatchGUI().run()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
