"""The ocean world: one shark foraging among randomly-placed, sized fish.

Replaces the placeholder ``ToyOcean`` (``src/algorithms/rl_algo/toy_env.py``):
fish spawn at random tiles every ``reset()`` and are deleted when eaten, and the
shark's *traits* (from a :class:`~src.shark.SharkGenome`) drive the mechanics:

  * size  -> which fish it can eat (bigger fish give more) + metabolic cost
  * speed -> how many tiles a MOVE covers + metabolic cost
  * field_of_perception -> how far it senses the nearest fish (its observation)

Each shark in a family gets its OWN ``Ocean`` instance; they share only the
brain (see ``src/simulation.py``). The observation keeps the exact shape
``discretize()`` expects, so the RL code is unchanged.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

# Allow both ``-m src.environment.ocean`` and bare ``python src/environment/ocean.py``.
if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algorithms.rl_algo.actions import Action  # leaf import -> no package cycle
from src.environment.config import DEFAULT_CONFIG, EnvConfig
from src.shark import SHARK_TRAITS, SharkGenome

SIZE_NAMES = ("small", "medium", "large")
_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
_DELTA = {Action.UP: (0, -1), Action.DOWN: (0, 1), Action.LEFT: (-1, 0), Action.RIGHT: (1, 0)}


@dataclass
class Fish:
    kind: str   # "safe" | "poisonous"
    size: str   # "small" | "medium" | "large"
    pos: tuple[int, int]


def _norm(name: str, value: float) -> float:
    """Scale a trait to 0..1 using its declared range (same idea as genetic_algo._norm)."""
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


def _size_tier(size_value: float, thresholds: tuple[float, float]) -> int:
    lo, hi = thresholds
    if size_value < lo:
        return 0
    if size_value < hi:
        return 1
    return 2


class Ocean:
    """Single-shark grid world. ``(x, y)`` with x rightward, y downward."""

    def __init__(
        self,
        config: EnvConfig = DEFAULT_CONFIG,
        genome: SharkGenome | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.genome = genome if genome is not None else SharkGenome.default()
        self.rng = rng if rng is not None else random.Random(config.seed)
        self.size = config.size
        self._derive_traits()

        self.shark = config.shark_start
        self.energy = config.start_energy
        self.facing = Action.RIGHT
        self.alive = True
        self.eaten = 0
        self.fishes: list[Fish] = []
        self.done = False

    # --- trait-derived parameters (computed once from the genome) -------------
    def _derive_traits(self) -> None:
        cfg, g = self.config, self.genome
        self.size_norm = _norm("size", g.size)
        self.speed_norm = _norm("speed", g.speed)
        self.size_tier = _size_tier(g.size, cfg.size_tier_thresholds)
        self.move_tiles = 1 + round(self.speed_norm * cfg.max_extra_move_tiles)
        self.perception = (
            max(1, round(g.field_of_perception)) if cfg.perception_uses_genome
            else cfg.fixed_perception_radius
        )
        # Metabolism mirrors genetic_algo's fitness term: 0.45*size^2 + 0.35*speed^2 (normalized).
        metab = 0.45 * self.size_norm**2 + 0.35 * self.speed_norm**2
        self.step_cost = cfg.base_move_cost + cfg.metab_scale * metab

    # --- episode lifecycle ----------------------------------------------------
    def reset(self) -> dict:
        if self.config.seed is not None:
            self.rng.seed(self.config.seed)
        self.shark = self.config.shark_start
        self.energy = self.config.start_energy
        self.facing = Action.RIGHT
        self.alive = True
        self.eaten = 0
        self.done = False
        self.fishes = self._spawn_fish()
        return self.observe()

    def _spawn_fish(self) -> list[Fish]:
        cfg = self.config
        spec = [
            ("safe", "small", cfg.num_safe_small),
            ("safe", "medium", cfg.num_safe_medium),
            ("safe", "large", cfg.num_safe_large),
            ("poisonous", "small", cfg.num_poison_small),
            ("poisonous", "medium", cfg.num_poison_medium),
            ("poisonous", "large", cfg.num_poison_large),
        ]
        total = sum(n for _, _, n in spec)
        free = [
            (x, y)
            for x in range(cfg.size)
            for y in range(cfg.size)
            if (x, y) != cfg.shark_start
        ]
        if total > len(free):
            raise ValueError(
                f"Cannot place {total} fish on {len(free)} free tiles (grid {cfg.size}x{cfg.size})."
            )
        positions = self.rng.sample(free, total)
        fishes, i = [], 0
        for kind, size, n in spec:
            for _ in range(n):
                fishes.append(Fish(kind, size, positions[i]))
                i += 1
        return fishes

    def step(self, action: Action) -> tuple[dict, float, bool]:
        """Advance one timestep. Returns ``(obs, reward, done)`` (done = shark died)."""
        if self.done:
            return self.observe(), 0.0, True

        cfg = self.config
        reward = cfg.step_penalty

        if action in _MOVES:
            self.facing = action
            self._move(action)
        elif action == Action.ATTACK:
            reward = self._attack()  # may set self.done (poison)

        if not self.done:  # poison death short-circuits the energy economy
            if action == Action.REST:
                self.energy = min(cfg.max_energy, self.energy + cfg.rest_energy_gain)
            else:
                self.energy -= self.step_cost
            if self.energy <= 0:
                self.energy = 0.0
                reward = cfg.starve_penalty
                self.alive = False
                self.done = True

        return self.observe(), reward, self.done

    def _move(self, action: Action) -> None:
        dx, dy = _DELTA[action]
        x, y = self.shark
        for _ in range(self.move_tiles):  # speed -> multi-tile, clamped at edges
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                x, y = nx, ny
            else:
                break
        self.shark = (x, y)

    def _attack(self) -> float:
        cfg = self.config
        reachable = [f for f in self.fishes if self.in_reach(f.pos)]
        poison = next((f for f in reachable if f.kind == "poisonous"), None)
        if poison is not None:
            # Eating poison halves the shark's health. It survives the bite unless
            # it was already tired (low energy) or has gorged on 2+ fish, in which
            # case the poison finishes it off.
            fatal = self.energy <= cfg.tired_energy or self.eaten >= 2
            self.energy *= 0.5
            self.fishes.remove(poison)
            self.eaten += 1
            if fatal:
                self.alive = False
                self.done = True
            return cfg.poison_penalty
        edible = [
            f for f in reachable
            if f.kind == "safe" and SIZE_NAMES.index(f.size) <= self.size_tier
        ]
        if edible:
            f = edible[0]
            tier = SIZE_NAMES.index(f.size)
            self.energy = min(cfg.max_energy, self.energy + cfg.eat_energy_by_size[tier])
            self.fishes.remove(f)
            self.eaten += 1
            return cfg.eat_reward_by_size[tier]
        return cfg.step_penalty  # nothing edible in reach (or fish too big): wasted attack

    def in_reach(self, pos: tuple[int, int]) -> bool:
        """Can the shark attack a fish here? On its own tile or one step away."""
        sx, sy = self.shark
        return abs(sx - pos[0]) + abs(sy - pos[1]) <= 1

    # --- observation (discretize-compatible) ----------------------------------
    def observe(self) -> dict:
        return {"energy": max(0.0, self.energy), "nearest_fish": self._nearest_fish()}

    def _nearest_fish(self) -> dict:
        sx, sy = self.shark
        best, best_d = None, 10**9
        for f in self.fishes:
            d = abs(sx - f.pos[0]) + abs(sy - f.pos[1])
            if d <= self.perception and d < best_d:  # only fish within sight
                best, best_d = f, d
        if best is None:
            return {"exists": False, "type": None, "distance": 0, "size": None, "dx": 0, "dy": 0}
        return {
            "exists": True,
            "type": best.kind,
            "distance": best_d,
            "size": best.size,
            "dx": best.pos[0] - sx,  # signed offset to the fish -> direction bucket
            "dy": best.pos[1] - sy,
        }


if __name__ == "__main__":
    env = Ocean()
    env.reset()
    print("layout A:", [(f.kind, f.size, f.pos) for f in env.fishes])
    env.reset()
    print("layout B:", [(f.kind, f.size, f.pos) for f in env.fishes], "(random each reset)")
    print(
        f"default shark: size_tier={env.size_tier} move_tiles={env.move_tiles} "
        f"perception={env.perception} step_cost={env.step_cost:.4f}"
    )
    for action in [Action.RIGHT, Action.DOWN, Action.ATTACK]:
        obs, reward, done = env.step(action)
        print(f"{action.name:7s} shark {env.shark} reward {reward:6.1f} done {done} obs {obs['nearest_fish']}")
        if done:
            break
