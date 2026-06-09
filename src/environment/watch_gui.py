"""Watch mode: spectate a shark colony live, compete, breed, age, and die.

A colony of sharks forages ONE shared, depleting ocean (a fish eaten by one is
gone for all of them), all sharing one ``FamilyRL`` brain. Each cycle ("year")
is one shark lifetime of foraging followed by the :mod:`src.shark.shark`
lifecycle, with selection pressure layered on top:

  * the brain learns every step (tabular Q-learning);
  * a shark that dies foraging (starves or bites poison) is out of the gene pool;
  * only sharks that *ate* breed, better foragers leave more offspring, and each
    shark's ``gestation_period`` trait spaces out its broods;
  * everyone ages a year, dying of old age at ``MAX_AGE_YEARS`` (15 cycles);
  * when the pod exceeds the carrying capacity, overcrowding culls the *weakest*
    foragers first (newborn pups are spared that cycle).

So the colony grows on its own merits toward a carrying capacity where food
competition keeps it in check. The whole colony is simulated, but the grid is a
fixed viewport that only ever shows up to ``watch_display_slots`` of them (a
random sample when the pod is larger); the HUD reports the true population.

A cold brain would kill the founding pod before it learns anything, so the brain
is warmed headlessly at startup (``watch_brain_warmup_lives``) -- you then watch
a *competent* colony play out its lifecycle.

Run:  python -m src.environment.watch_gui   (or python src/environment/watch_gui.py)
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
from src.environment.fish_drawing import draw_fish
from src.environment.gui import FACING_ANGLE  # reuse the rotation map
from src.environment.ocean import Ocean, advance_fish_pool, spawn_fish
from src.environment.shark_drawing import make_genome_shark_sprite
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
PLOT_TRAIT = "#ff9d5c"
PLOT_POP = "#6fb6ff"        # population line
PLOT_BIRTHS = "#7fd1a0"     # births (green, good)
PLOT_FORAGE = "#ff6b6b"     # foraging deaths (red, poison/starve)
PLOT_OLDAGE = "#c9a0ff"     # old-age deaths (purple)
PLOT_CULL = "#ff9d5c"       # overcrowding culls (orange)
HEATMAP_CMAP = "RdYlGn"     # Q-values: red = bad, green = good


class WatchGUI:
    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config

        # The whole colony is simulated, but the grid only ever shows up to
        # ``slots`` of them: a random sample when the population is larger, and
        # empty (dark blue) cells for the remainder when it is smaller.
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

        # Founding pod: random ages 0..14 so old-age deaths show up from the start
        # (not only after the first 15 cycles), alongside foraging deaths.
        genomes = create_initial_population(config.watch_population_size, self.rng)
        founding = [Shark(g, age=self.rng.randrange(MAX_AGE_YEARS)) for g in genomes]
        self._start_cycle(founding)

    def _warmup_brain(self, lives: int) -> None:
        """Teach the shared brain on a default shark so the founding pod can survive.

        Without this the cold (fully exploring) brain dies foraging almost every
        time and the colony goes extinct in a cycle or two. After warming we drop
        exploration low so behaviour is competent but not frozen.
        """
        if lives <= 0:
            return
        warm = SharkGenome.default()
        env = Ocean(self.config, genome=warm, rng=self.rng)
        for _ in range(lives):
            run_life(env, self.brain, warm, self.rng, self.config.watch_max_steps)
        self.brain.epsilon = max(self.brain.epsilon_min, 0.1)

    # --- cycle lifecycle -----------------------------------------------------
    def _start_cycle(self, population: list[Shark]):
        """Begin a birth cycle: give every shark an ocean and pick the on-screen window."""
        self.population = population
        self.pop_total = len(population)
        self.envs = [Ocean(self.config, genome=s.genome, rng=self.rng) for s in population]

        # Competition: the whole colony forages ONE shared, depleting fish pool
        # sized to the population (capped by the grid). A fish eaten by one shark
        # is gone for all of them, so good hunters take food from the rest.
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

    def _recompute_shown(self) -> None:
        """Pick which sharks the grid shows, based on the current page."""
        pop = len(self.population)
        max_page = max(0, (pop - 1) // self.slots)
        self.page = max(0, min(self.page, max_page))  # clamp (population shrinks/grows)
        start = self.page * self.slots
        self.shown = list(range(start, min(start + self.slots, pop)))

    def _tick(self):
        """Advance every living shark one foraging step (and learn); roll the cycle when all finish."""
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
        # The shared pool is colony-owned, so drift/respawn it once per tick here
        # (each Ocean skips its own fish upkeep when foraging a shared pool).
        if self.shared_fishes is not None and not all_done:
            living = {self.envs[i].shark for i, e in enumerate(self.envs) if not e.done}
            advance_fish_pool(
                self.shared_fishes, self.config, self.rng, self.config.size,
                target=self.shared_fish_target, blocked=living,
            )
        self.step_in_life += 1
        if all_done or self.step_in_life >= self.config.watch_max_steps:
            self._end_cycle()

    def _end_cycle(self):
        """Record stats, resolve foraging deaths, breed survivors, age the colony, cull overcrowding."""
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
        for gene in self.genes:
            self.trait_hist[gene].append(best_shark.genome[gene])

        # Record each shark's foraging outcome, and mark foraging deaths (env.alive
        # False) out of the gene pool. The recorded reward/food drives who breeds.
        for i, (shark, env) in enumerate(zip(self.population, self.envs)):
            shark.record_life(self.totals[i], env.eaten)
            if not env.alive:
                shark.alive = False

        # Lifecycle: living breed, then everyone ages a year. Breeding now favours
        # well-fed sharks (require_food + fitness_weighted) and is paced by each
        # shark's gestation trait, so reproduction reflects foraging success.
        survived_forage = [s for s in self.population if s.alive]
        pups = reproduce(
            self.population, self.rng, self.config.watch_mutation_rate,
            require_food=self.config.watch_require_food_to_breed,
            fitness_weighted=self.config.watch_fitness_weighted_breeding,
            gestation_divisor=self.config.watch_gestation_divisor,
        )
        for shark in self.population:
            shark.grow_older()
        survivors = [s for s in self.population if s.alive]
        next_pop = survivors + pups

        # Carrying capacity: overcrowding culls the weakest foragers first (the
        # newborn pups, with no foraging record yet, are spared this cycle).
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

    def _handle_click(self, pos):
        if self.view == "watch" and self.metrics_btn.collidepoint(pos):
            self.view = "metrics"
        elif self.view == "metrics" and self.back_btn.collidepoint(pos):
            self.view = "watch"

    # --- rendering -----------------------------------------------------------
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

    def _draw_empty_cell(self, x0: int, y0: int):
        """A vacant slot when the population is smaller than the grid: dark blue."""
        cell = pygame.Rect(x0 + 1, y0 + 1, self.cell_w - 2, self.cell_h - 2)
        pygame.draw.rect(self.screen, CELL_BG, cell)
        pygame.draw.rect(self.screen, CELL_BORDER, cell, 1)

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
            f"Metrics · cycle {self.cycle} · population {self.pop_total}", True, TEXT
        )
        self.screen.blit(title, (self.back_btn.right + 18, 18))

        top = HUD_H
        area_w, area_h = self.width, self.height - top
        if not self.best_hist:
            msg = self.small.render(
                "Collecting data — the first cycle is still running.", True, TEXT_DIM
            )
            self.screen.blit(msg, msg.get_rect(center=(self.width // 2, top + area_h // 2)))
            return

        # Re-render the figure only when a new cycle has been recorded.
        if self._metrics_surf is None or self._metrics_cached_gen != len(self.best_hist):
            self._metrics_surf = self._render_metrics_figure(area_w, area_h)
            self._metrics_cached_gen = len(self.best_hist)
        self.screen.blit(self._metrics_surf, (0, top))

    def _render_metrics_figure(self, px_w: int, px_h: int) -> pygame.Surface:
        """Draw the per-cycle graphs to a pygame surface via matplotlib (Agg)."""
        dpi = 100
        fig = Figure(figsize=(px_w / dpi, px_h / dpi), dpi=dpi, facecolor=PLOT_FACE)

        # Core panels (fitness, population, deaths, Q-table) + one per trait.
        n_plots = 4 + len(self.genes)
        cols = 2
        rows = math.ceil(n_plots / cols)
        axes = fig.subplots(rows, cols, squeeze=False)
        flat = [axes[r][c] for r in range(rows) for c in range(cols)]
        for ax in flat[n_plots:]:  # hide any unused grid cells
            ax.set_visible(False)

        gens = range(1, len(self.best_hist) + 1)

        def _style(ax, title: str, ylabel: str):
            ax.set_title(title, color=TEXT_HEX, fontsize=10)
            ax.set_xlabel("cycle", color=AX_TEXT, fontsize=8)
            ax.set_ylabel(ylabel, color=AX_TEXT, fontsize=8)
            ax.set_facecolor(AX_FACE)
            ax.tick_params(colors=AX_TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(AX_SPINE)
            ax.grid(True, color=AX_GRID, linewidth=0.5)

        def _legend(ax):
            leg = ax.legend(fontsize=6, facecolor=AX_FACE, edgecolor=AX_SPINE)
            for text in leg.get_texts():
                text.set_color(TEXT_HEX)

        # Fitness: best line, average line, and the population min..max band.
        ax = flat[0]
        ax.fill_between(gens, self.min_hist, self.max_hist,
                        color=PLOT_RANGE, alpha=0.35, label="pop range")
        ax.plot(gens, self.best_hist, color=PLOT_BEST, linewidth=1.8, label="best")
        ax.plot(gens, self.avg_hist, color=PLOT_AVG, linewidth=1.2, label="avg")
        _style(ax, "Fitness (reward) over cycles", "reward")
        _legend(ax)

        # Population: colony size each cycle, with the carrying-capacity line.
        ax = flat[1]
        ax.plot(gens, self.pop_hist, color=PLOT_POP, linewidth=1.8,
                marker="o", markersize=2, label="population")
        ax.axhline(self.config.watch_carrying_capacity, color=PLOT_CULL,
                   linewidth=0.9, linestyle="--", alpha=0.8, label="capacity")
        ax.set_ylim(bottom=0)
        _style(ax, "Population over cycles", "sharks")
        _legend(ax)

        # Death types (and births) per cycle.
        ax = flat[2]
        ax.plot(gens, self.births_hist, color=PLOT_BIRTHS, linewidth=1.4, label="births")
        ax.plot(gens, self.forage_death_hist, color=PLOT_FORAGE, linewidth=1.4,
                label="foraging (poison/starve)")
        ax.plot(gens, self.oldage_death_hist, color=PLOT_OLDAGE, linewidth=1.4, label="old age")
        ax.plot(gens, self.cull_hist, color=PLOT_CULL, linewidth=1.4, label="overcrowding")
        ax.set_ylim(bottom=0)
        _style(ax, "Births & deaths per cycle", "count")
        _legend(ax)

        # Q-table heatmap: every learned state (row) x action (column).
        self._draw_qtable_heatmap(fig, flat[3])

        # One subplot per evolvable trait: the best shark's value over time.
        for ax, gene in zip(flat[4:], self.genes):
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

    def _draw_qtable_heatmap(self, fig, ax) -> None:
        """Render the shared brain's Q-table as a state x action heatmap.

        Rows are the discrete states the brain has visited, columns are the six
        actions, and colour is the learned Q-value (red = avoid, green = good).
        Row labels are only drawn when the table is small enough to read.
        """
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


def run(config: EnvConfig = DEFAULT_CONFIG):
    WatchGUI(config).run()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
