"""Watch mode: spectate a shark colony live, compete, breed, age, and die.

A colony forages ONE shared, depleting ocean (a fish eaten by one is gone for all)
sharing one ``FamilyRL`` brain. Each cycle is one shark lifetime of foraging plus
the :mod:`src.shark.shark` lifecycle: foraging deaths leave the gene pool, only
fed sharks breed (better foragers leave more), all age and die at MAX_AGE_YEARS,
and overcrowding above the carrying capacity culls the weakest foragers.

The whole colony is simulated; the grid shows up to ``watch_display_slots`` of
them. The brain is warmed headlessly at startup so the founding pod survives.

Run:  python -m src.gui.watch_gui
Controls: Space pause/resume · M metrics · Q/Esc quit.
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algorithms.genetic_algo import create_initial_population
from src.algorithms.rl_algo import NUM_ACTIONS, Action, FamilyRL, discretize
from src.environment.config import DEFAULT_CONFIG, EnvConfig
from src.environment.ocean import Ocean, advance_fish_pool, spawn_fish
from src.gui.fish_drawing import draw_fish
from src.gui.gui import FACING_ANGLE  # reuse the rotation map
from src.gui.shark_drawing import make_genome_shark_sprite
from src.shark import MAX_AGE_YEARS, SHARK_TRAITS, Shark, SharkGenome, evolvable_traits, reproduce
from src.simulation import biased_action, run_life

HUD_H = 84
HUD_BG = (12, 16, 28)
TEXT = (230, 236, 245)
TEXT_DIM = (150, 160, 175)
TEXT_WARN = (235, 170, 90)
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
PLOT_TRAIT = "#ff9d5c"        # best shark's trait value
PLOT_TRAIT_AVG = "#ffd24a"    # colony-average trait value (yellow)
PLOT_POP = "#6fb6ff"        # population line
PLOT_BIRTHS = "#7fd1a0"     # births (green, good)
PLOT_FORAGE = "#ff6b6b"     # foraging deaths (red, poison/starve)
PLOT_OLDAGE = "#c9a0ff"     # old-age deaths (purple)
PLOT_CULL = "#ff9d5c"       # overcrowding culls (orange)
HEATMAP_CMAP = "RdYlGn"     # Q-values: red = bad, green = good


class WatchGUI:
    # Set up the window, shared brain, history buffers, and founding pod.
    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config

        # Whole colony is simulated; the grid shows up to ``slots`` of them.
        self.slots = config.watch_display_slots
        self.cols = config.watch_grid_cols
        self.rows = math.ceil(self.slots / self.cols)
        self.view_rng = random.Random(config.seed)  # picks the shown sample; kept off the sim rng

        # Window: a cols x rows grid of square-ish mini-oceans below the HUD.
        cell = 230
        self.cell_w = cell
        self.cell_h = cell
        self.cell_tile = max(4, (cell - 16) // config.size)
        self.width = self.cols * self.cell_w
        self.height = HUD_H + self.rows * self.cell_h

        pygame.init()
        pygame.display.set_caption("Plenty of Fish — Watch the colony live & breed")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.small = pygame.font.SysFont(None, 14)

        self.rng = random.Random(config.seed)
        self.brain = FamilyRL()
        self.cycle = 0
        self.paused = False
        self.extinct = False
        self.best_fitness = float("nan")
        self.avg_fitness = float("nan")
        # Per-cycle lifecycle tallies for the HUD.
        self.pop_total = 0
        self.births = self.forage_deaths = self.oldage_deaths = self.culled = 0

        # View state: "watch" (the grid) or "metrics" (the live graphs).
        self.view = "watch"
        # Vertical scroll (px) for the metrics page; clamped when drawn.
        self.metrics_scroll = 0
        # Which page of the colony the grid is showing (←/→ to page through all).
        self.page = 0
        self.metrics_btn = pygame.Rect(self.width - 132, 12, 118, 30)
        self.back_btn = pygame.Rect(14, 12, 96, 30)

        # Per-cycle history that the metrics graphs plot.
        self.genes = evolvable_traits()
        self.best_hist: list[float] = []
        self.avg_hist: list[float] = []
        self.min_hist: list[float] = []
        self.max_hist: list[float] = []
        self.trait_hist: dict[str, list[float]] = {g: [] for g in self.genes}
        self.avg_trait_hist: dict[str, list[float]] = {g: [] for g in self.genes}
        # Population size and per-cycle lifecycle event counts, for their charts.
        self.pop_hist: list[int] = []
        self.births_hist: list[int] = []
        self.forage_death_hist: list[int] = []
        self.oldage_death_hist: list[int] = []
        self.cull_hist: list[int] = []
        # Cache the rendered figure; re-render only when a new cycle lands.
        self._metrics_surf: pygame.Surface | None = None
        self._metrics_cached_gen = -1

        self._warmup_brain(config.watch_brain_warmup_lives)

        # Founding pod: random ages so old-age deaths appear from the start.
        genomes = create_initial_population(config.watch_population_size, self.rng)
        founding = [Shark(g, age=self.rng.randrange(MAX_AGE_YEARS)) for g in genomes]
        self._start_cycle(founding)

    # Pre-train the shared brain on a default shark so the founding pod survives.
    # A cold brain dies foraging; after warming we drop exploration low, not to zero.
    def _warmup_brain(self, lives: int) -> None:
        if lives <= 0:
            return
        warm = SharkGenome.default()
        env = Ocean(self.config, genome=warm, rng=self.rng)
        for _ in range(lives):
            run_life(env, self.brain, warm, self.rng, self.config.watch_max_steps)
        self.brain.epsilon = max(self.brain.epsilon_min, 0.1)

    # --- cycle lifecycle -----------------------------------------------------
    # Begin a birth cycle: give every shark an ocean and pick the on-screen window.
    def _start_cycle(self, population: list[Shark]):
        self.population = population
        self.pop_total = len(population)
        self.envs = [
            Ocean(self.config, genome=s.genome, rng=self.rng, body_size=s.size)
            for s in population
        ]

        # The colony forages ONE shared, depleting pool sized to the population
        # (capped), so good hunters take food from the rest.
        if self.config.watch_shared_fish_pool:
            free_tiles = self.config.size * self.config.size - 1
            target = min(
                free_tiles,
                self.config.watch_fish_pool_cap,
                max(1, round(self.config.watch_fish_per_shark * len(population))),
            )
            self.shared_fishes = spawn_fish(
                self.config, self.rng, target_total=target,
                exclude={self.config.shark_start},
            )
            self.shared_fish_target = len(self.shared_fishes)
            for e in self.envs:
                e.reset(shared_fishes=self.shared_fishes)
        else:
            self.shared_fishes = None
            for e in self.envs:
                e.reset()
        self.states = [discretize(e.observe()) for e in self.envs]
        self.totals = [0.0] * len(population)
        self.step_in_life = 0
        self.sprites = [
            make_genome_shark_sprite(self.cell_tile, s.genome.color(), e.size_norm)
            for s, e in zip(population, self.envs)
        ]
        # Fill the grid from the current page so you can ←/→ through the whole pod.
        self._recompute_shown()

    # Pick which sharks the grid shows, based on the current page.
    def _recompute_shown(self) -> None:
        pop = len(self.population)
        max_page = max(0, (pop - 1) // self.slots)
        self.page = max(0, min(self.page, max_page))  # clamp (population shrinks/grows)
        start = self.page * self.slots
        self.shown = list(range(start, min(start + self.slots, pop)))

    # Advance every living shark one foraging step (and learn); roll the cycle when all finish.
    def _tick(self):
        if self.extinct:
            return
        all_done = True
        for i, env in enumerate(self.envs):
            if env.done:
                continue
            all_done = False
            action = biased_action(self.brain, self.states[i], self.population[i].genome, self.rng)
            obs, reward, done = env.step(action)
            next_state = discretize(obs)
            self.brain.update(self.states[i], action, reward, next_state, done)
            self.states[i] = next_state
            self.totals[i] += reward
            if not env.fishes:  # ate everything -> that life is over (survived)
                env.done = True
        # Drift/respawn the colony-owned shared pool once per tick.
        if self.shared_fishes is not None and not all_done:
            living = {self.envs[i].shark for i, e in enumerate(self.envs) if not e.done}
            advance_fish_pool(
                self.shared_fishes, self.config, self.rng, self.config.size,
                target=self.shared_fish_target, blocked=living,
            )
        self.step_in_life += 1
        if all_done or self.step_in_life >= self.config.watch_max_steps:
            self._end_cycle()

    # Record stats, resolve foraging deaths, breed survivors, age the colony, cull overcrowding.
    def _end_cycle(self):
        fitnesses = list(self.totals)
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_shark = self.population[best_idx]

        # Record this cycle's stats so the metrics graphs can plot them.
        self.best_hist.append(self.best_fitness)
        self.avg_hist.append(self.avg_fitness)
        self.min_hist.append(min(fitnesses))
        self.max_hist.append(max(fitnesses))
        self.pop_hist.append(len(self.population))
        pop_n = len(self.population)
        for gene in self.genes:
            self.trait_hist[gene].append(best_shark.genome[gene])
            self.avg_trait_hist[gene].append(
                sum(s.genome[gene] for s in self.population) / pop_n
            )

        # Record each shark's outcome; foraging deaths leave the gene pool.
        for i, (shark, env) in enumerate(zip(self.population, self.envs)):
            shark.record_life(self.totals[i], env.eaten)
            if not env.alive:
                shark.alive = False

        # Living breed (favouring well-fed sharks, paced by gestation), then all age.
        survived_forage = [s for s in self.population if s.alive]
        pups = reproduce(
            self.population, self.rng, self.config.watch_mutation_rate,
            require_food=self.config.watch_require_food_to_breed,
            fitness_weighted=self.config.watch_fitness_weighted_breeding,
            gestation_divisor=self.config.watch_gestation_divisor,
            brood_size=self.config.watch_brood_size,
        )
        for shark in self.population:
            shark.grow_older()
        survivors = [s for s in self.population if s.alive]
        next_pop = survivors + pups

        # Overcrowding culls the weakest foragers first; newborn pups are spared.
        cap = self.config.watch_carrying_capacity
        self.culled = max(0, len(next_pop) - cap)
        if self.culled:
            if self.config.watch_cull_weakest:
                weakest_first = sorted(survivors, key=lambda s: s.last_reward)
                cull_ids = {id(s) for s in weakest_first[: self.culled]}
                next_pop = [s for s in next_pop if id(s) not in cull_ids]
                if len(next_pop) > cap:  # too many pups to fit -> trim the remainder
                    next_pop = self.view_rng.sample(next_pop, cap)
            else:
                next_pop = self.view_rng.sample(next_pop, cap)

        # Tally for the HUD.
        self.births = len(pups)
        self.forage_deaths = len(self.population) - len(survived_forage)
        self.oldage_deaths = len(survived_forage) - len(survivors)

        # Record lifecycle events so the death-types chart can plot them.
        self.births_hist.append(self.births)
        self.forage_death_hist.append(self.forage_deaths)
        self.oldage_death_hist.append(self.oldage_deaths)
        self.cull_hist.append(self.culled)

        self.brain.decay_epsilon()
        self.cycle += 1
        if not next_pop:
            self.extinct = True
            self.pop_total = 0
            return
        self._start_cycle(next_pop)

    # --- loop ----------------------------------------------------------------
    # Main loop: handle input, tick the colony, and redraw the active view.
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
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.view == "watch":
                        self.page += -1 if event.key == pygame.K_LEFT else 1
                        self._recompute_shown()
                    elif event.key in (pygame.K_UP, pygame.K_DOWN) and self.view == "metrics":
                        self.metrics_scroll += -60 if event.key == pygame.K_UP else 60
                    elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN) and self.view == "metrics":
                        self.metrics_scroll += -300 if event.key == pygame.K_PAGEUP else 300
                elif event.type == pygame.MOUSEWHEEL and self.view == "metrics":
                    self.metrics_scroll -= event.y * 80
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            # The simulation keeps running in either view, so the graphs update live.
            if not self.paused and not self.extinct:
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

    # Route mouse clicks to the Metrics / Back buttons.
    def _handle_click(self, pos):
        if self.view == "watch" and self.metrics_btn.collidepoint(pos):
            self.view = "metrics"
        elif self.view == "metrics" and self.back_btn.collidepoint(pos):
            self.view = "watch"

    # --- rendering -----------------------------------------------------------
    # Render the watch grid of mini-oceans plus the HUD.
    def _draw(self):
        self.screen.fill(HUD_BG)
        for cell in range(self.slots):
            cx = (cell % self.cols) * self.cell_w
            cy = HUD_H + (cell // self.cols) * self.cell_h
            if cell < len(self.shown):
                self._draw_cell(self.shown[cell], cx, cy)
            else:
                self._draw_empty_cell(cx, cy)
        self._draw_hud()

    # A vacant slot when the population is smaller than the grid: dark blue.
    def _draw_empty_cell(self, x0: int, y0: int):
        cell = pygame.Rect(x0 + 1, y0 + 1, self.cell_w - 2, self.cell_h - 2)
        pygame.draw.rect(self.screen, CELL_BG, cell)
        pygame.draw.rect(self.screen, CELL_BORDER, cell, 1)

    # Draw one shark's mini-ocean: fish, shark sprite, dim-if-done, and label.
    def _draw_cell(self, i: int, x0: int, y0: int):
        shark, env = self.population[i], self.envs[i]
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

        dead = not env.alive
        tag = f"#{i}  age {shark.age}/{MAX_AGE_YEARS}  r={self.totals[i]:.0f}  ate={env.eaten}"
        if dead:
            tag += "  DIED"
        self.screen.blit(self.small.render(tag, True, TEXT_WARN if dead else TEXT), (x0 + 5, y0 + 4))
        pygame.draw.rect(self.screen, CELL_BORDER, cell, 1)

    # Draw the top HUD: cycle stats, last-cycle births/deaths, and paging info.
    def _draw_hud(self):
        best = "—" if math.isnan(self.best_fitness) else f"{self.best_fitness:6.1f}"
        avg = "—" if math.isnan(self.avg_fitness) else f"{self.avg_fitness:6.1f}"
        line1 = (
            f"Cycle {self.cycle}   Population {self.pop_total}   Best {best}   Avg {avg}   "
            f"epsilon {self.brain.epsilon:.3f}   Q-states {len(self.brain.q)}"
        )
        self.screen.blit(self.font.render(line1, True, TEXT), (12, 8))

        deaths = f"-{self.forage_deaths} foraging   -{self.oldage_deaths} old age"
        if self.culled:
            deaths += f"   -{self.culled} overcrowding"
        line2 = f"last cycle:  +{self.births} born   {deaths}"
        self.screen.blit(self.small.render(line2, True, TEXT_DIM), (12, 32))

        n_pages = max(1, (self.pop_total + self.slots - 1) // self.slots)
        lo = self.page * self.slots + 1 if self.shown else 0
        hi = self.page * self.slots + len(self.shown)
        forage_note = (
            "all compete for one shared ocean & brain"
            if self.config.watch_shared_fish_pool
            else "each forages its own ocean; all share one brain"
        )
        line3 = (
            f"Sharks {lo}-{hi} of {self.pop_total}  (page {self.page + 1}/{n_pages}, ←/→).  "
            f"{forage_note}.  Space pause · M metrics · Q quit"
        )
        if self.extinct:
            line3 = "COLONY EXTINCT — every shark died.  Q quit"
        elif self.paused:
            line3 += "   [PAUSED]"
        self.screen.blit(self.small.render(line3, True, TEXT_DIM), (12, 54))
        self._draw_button(self.metrics_btn, "Metrics >")

    # --- buttons / metrics view ----------------------------------------------
    # Draw a clickable button, highlighted when hovered.
    def _draw_button(self, rect: pygame.Rect, label: str):
        hot = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, BTN_HOT if hot else BTN_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BTN_BORDER, rect, 1, border_radius=6)
        txt = self.font.render(label, True, TEXT)
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    # Render the scrollable metrics page (cached figure blitted at the scroll offset).
    def _draw_metrics(self):
        self.screen.fill(HUD_BG)
        self._draw_button(self.back_btn, "< Back")
        title = self.font.render(
            f"Metrics · cycle {self.cycle} · population {self.pop_total}"
            "    (↑/↓ or mouse wheel to scroll)",
            True, TEXT,
        )
        self.screen.blit(title, (self.back_btn.right + 18, 18))

        top = HUD_H
        area_w, viewport_h = self.width, self.height - top
        if not self.best_hist:
            msg = self.small.render(
                "Collecting data — the first cycle is still running.", True, TEXT_DIM
            )
            self.screen.blit(msg, msg.get_rect(center=(self.width // 2, top + viewport_h // 2)))
            return

        # Re-render only on a new cycle; blit at a scroll offset and clip.
        if self._metrics_surf is None or self._metrics_cached_gen != len(self.best_hist):
            self._metrics_surf = self._render_metrics_figure(area_w, viewport_h)
            self._metrics_cached_gen = len(self.best_hist)

        surf = self._metrics_surf
        max_scroll = max(0, surf.get_height() - viewport_h)
        self.metrics_scroll = max(0, min(self.metrics_scroll, max_scroll))

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(0, top, self.width, viewport_h))
        self.screen.blit(surf, (0, top - self.metrics_scroll))
        self.screen.set_clip(prev_clip)

        if max_scroll > 0:
            self._draw_scrollbar(top, viewport_h, surf.get_height(), max_scroll)

    # A thin position indicator on the right edge of the metrics viewport.
    def _draw_scrollbar(self, top: int, viewport_h: int, content_h: int, max_scroll: int):
        x = self.width - 9
        pygame.draw.rect(self.screen, (28, 34, 50),
                         pygame.Rect(x, top, 6, viewport_h), border_radius=3)
        thumb_h = max(28, int(viewport_h * viewport_h / content_h))
        thumb_y = top + int((viewport_h - thumb_h) * (self.metrics_scroll / max_scroll))
        pygame.draw.rect(self.screen, BTN_BORDER,
                         pygame.Rect(x, thumb_y, 6, thumb_h), border_radius=3)

    # Draw the per-cycle graphs to a tall pygame surface via matplotlib (Agg).
    # Small line charts sit two per row at the top, fitness gets a full-width row,
    # and the Q-table heatmap a full-width band at the bottom. Taller than the view.
    def _render_metrics_figure(self, px_w: int, min_h: int) -> pygame.Surface:
        dpi = 100

        n_small = 2 + len(self.genes)       # population, deaths, + one per trait
        small_rows = math.ceil(n_small / 2)
        fit_rows = 1                        # fitness spans a full-width row of its own
        heat_rows = 3                       # grid rows the full-width heatmap spans
        total_rows = small_rows + fit_rows + heat_rows
        px_h = max(min_h, total_rows * 240)  # ~240 px per grid row → taller than the view

        fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor=PLOT_FACE)
        gs = fig.add_gridspec(total_rows, 2)
        flat = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_small)]
        ax_fit = fig.add_subplot(gs[small_rows, :])              # full-width fitness row
        ax_heat = fig.add_subplot(gs[small_rows + fit_rows:, :])  # full-width heatmap band

        gens = range(1, len(self.best_hist) + 1)

        # Apply the shared dark theme (title, labels, ticks, grid) to one axes.
        def _style(ax, title: str, ylabel: str):
            ax.set_title(title, color=TEXT_HEX, fontsize=10)
            ax.set_xlabel("cycle", color=AX_TEXT, fontsize=8)
            ax.set_ylabel(ylabel, color=AX_TEXT, fontsize=8)
            ax.set_facecolor(AX_FACE)
            ax.tick_params(colors=AX_TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(AX_SPINE)
            ax.grid(True, color=AX_GRID, linewidth=0.5)

        # Add a legend styled for the dark theme.
        def _legend(ax):
            leg = ax.legend(fontsize=6, facecolor=AX_FACE, edgecolor=AX_SPINE)
            for text in leg.get_texts():
                text.set_color(TEXT_HEX)

        # Population: colony size each cycle, with the carrying-capacity line.
        ax = flat[0]
        ax.plot(gens, self.pop_hist, color=PLOT_POP, linewidth=1.8,
                marker="o", markersize=2, label="population")
        ax.axhline(self.config.watch_carrying_capacity, color=PLOT_CULL,
                   linewidth=0.9, linestyle="--", alpha=0.8, label="capacity")
        ax.set_ylim(bottom=0)
        _style(ax, "Population over cycles", "sharks")
        _legend(ax)

        # Death types (and births) per cycle.
        ax = flat[1]
        ax.plot(gens, self.births_hist, color=PLOT_BIRTHS, linewidth=1.4, label="births")
        ax.plot(gens, self.forage_death_hist, color=PLOT_FORAGE, linewidth=1.4,
                label="foraging (poison/starve)")
        ax.plot(gens, self.oldage_death_hist, color=PLOT_OLDAGE, linewidth=1.4, label="old age")
        ax.plot(gens, self.cull_hist, color=PLOT_CULL, linewidth=1.4, label="overcrowding")
        ax.set_ylim(bottom=0)
        _style(ax, "Births & deaths per cycle", "count")
        _legend(ax)

        # One subplot per evolvable trait: best shark vs colony average.
        for ax, gene in zip(flat[2:], self.genes):
            spec = SHARK_TRAITS[gene]
            ax.plot(gens, self.trait_hist[gene], color=PLOT_TRAIT,
                    marker="o", markersize=2, linewidth=1.4, label="best")
            ax.plot(gens, self.avg_trait_hist[gene], color=PLOT_TRAIT_AVG,
                    linewidth=1.4, label="avg")
            ax.set_ylim(spec.min_value, spec.max_value)
            _style(ax, gene, spec.unit)
            _legend(ax)

        # Fitness full-width row: best, average, and the population min..max band.
        ax = ax_fit
        ax.fill_between(gens, self.min_hist, self.max_hist,
                        color=PLOT_RANGE, alpha=0.35, label="pop range")
        ax.plot(gens, self.best_hist, color=PLOT_BEST, linewidth=1.8, label="best")
        ax.plot(gens, self.avg_hist, color=PLOT_AVG, linewidth=1.2, label="avg")
        _style(ax, "Fitness (reward) over cycles", "reward")
        _legend(ax)

        # Q-table heatmap in its own full-width band so tiles are readable.
        self._draw_qtable_heatmap(fig, ax_heat)

        fig.tight_layout(pad=1.4)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        return pygame.image.frombuffer(bytes(canvas.buffer_rgba()), (w, h), "RGBA")

    # Render the shared brain's Q-table as a state x action heatmap.
    # Colour is the learned Q-value (red = avoid, green = good); row labels show
    # only when the table is small enough to read.
    def _draw_qtable_heatmap(self, fig, ax) -> None:
        table = self.brain.q._table
        ax.set_facecolor(AX_FACE)
        if not table:
            ax.set_title("Q-table heatmap", color=TEXT_HEX, fontsize=10)
            ax.text(0.5, 0.5, "no states learned yet", color=AX_TEXT,
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        states = sorted(table.keys())
        matrix = np.array([table[s] for s in states], dtype=float)
        vmax = float(np.abs(matrix).max()) or 1.0

        im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP,
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Q-table heatmap ({len(states)} states)", color=TEXT_HEX, fontsize=10)
        ax.set_xticks(range(NUM_ACTIONS))
        ax.set_xticklabels([Action(i).name for i in range(NUM_ACTIONS)],
                           rotation=45, ha="right", fontsize=6, color=AX_TEXT)
        # Outline tiles with gridlines when sparse enough to stay readable.
        if len(states) <= 60:
            ax.set_xticks(np.arange(-0.5, NUM_ACTIONS, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
            ax.grid(which="minor", color=AX_SPINE, linewidth=0.5)
            ax.tick_params(which="minor", length=0)
        # Row labels are unreadable past ~24 states, so only show them when sparse.
        if len(states) <= 24:
            ax.set_yticks(range(len(states)))
            ax.set_yticklabels(["·".join(s) for s in states], fontsize=5, color=AX_TEXT)
        else:
            ax.set_yticks([])
            ax.set_ylabel("state", color=AX_TEXT, fontsize=8)
        ax.tick_params(colors=AX_TEXT)
        for spine in ax.spines.values():
            spine.set_color(AX_SPINE)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=AX_TEXT, labelsize=6)
        cbar.outline.set_edgecolor(AX_SPINE)


# Launch the watch GUI.
def run(config: EnvConfig = DEFAULT_CONFIG):
    WatchGUI(config).run()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
