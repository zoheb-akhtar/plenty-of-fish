"""The ocean world: one shark foraging among randomly-placed, sized fish.

Replaces the placeholder ``ToyOcean`` (``src/algorithms/rl_algo/toy_env.py``):
fish spawn at random tiles every ``reset()`` and are deleted when eaten, and the
shark's *traits* (from a :class:`~src.shark.SharkGenome`) drive the mechanics:

  * size  -> which fish it can eat (bigger fish give more) + metabolic cost
  * speed -> how many tiles a MOVE covers + metabolic cost
  * field_of_perception -> how far it senses the nearest fish (its observation)

Each shark in a family gets its OWN ``Ocean`` instance; they share only the
brain (see ``src/simulation.py``). In Watch mode the colony can also forage a
single shared fish pool (``reset(shared_fishes=...)``) so sharks compete for the
same food. The observation keeps the exact shape ``discretize()`` expects, so
the RL code is unchanged.
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


# Scale a trait to 0..1 using its declared range (same idea as genetic_algo._norm).
def _norm(name: str, value: float) -> float:
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


# Bucket a size value into 0=small / 1=medium / 2=large by the thresholds.
def _size_tier(size_value: float, thresholds: tuple[float, float]) -> int:
    lo, hi = thresholds
    if size_value < lo:
        return 0
    if size_value < hi:
        return 1
    return 2


# The configured (kind, size, count) make-up of a fresh fish set.
def _fish_spec(cfg: EnvConfig) -> list[tuple[str, str, int]]:
    return [
        ("safe", "small", cfg.num_safe_small),
        ("safe", "medium", cfg.num_safe_medium),
        ("safe", "large", cfg.num_safe_large),
        ("poisonous", "small", cfg.num_poison_small),
        ("poisonous", "medium", cfg.num_poison_medium),
        ("poisonous", "large", cfg.num_poison_large),
    ]


# Scale the configured counts to roughly ``target_total`` fish, same mix.
def _scaled_counts(spec: list[tuple[str, str, int]], target_total: int) -> list[tuple[str, str, int]]:
    base_total = sum(n for *_, n in spec)
    if base_total == 0 or target_total <= 0:
        return spec
    scale = target_total / base_total
    scaled = [(k, s, max(0, round(n * scale))) for k, s, n in spec]
    if sum(n for *_, n in scaled) == 0:
        k, s, _ = scaled[0]
        scaled[0] = (k, s, 1)
    return scaled


# A fresh fish whose (kind, size) follows the configured spawn mix.
def _random_fish_at(cfg: EnvConfig, rng: random.Random, pos: tuple[int, int]) -> "Fish":
    pool = [(k, s) for k, s, n in _fish_spec(cfg) for _ in range(n)]
    kind, size = rng.choice(pool) if pool else ("safe", "small")
    return Fish(kind, size, pos)


# Place a fish set on random free tiles.
# ``target_total`` scales the configured mix (for the colony's shared pool);
# ``None`` uses the exact configured counts. Total is clamped to the free tiles.
def spawn_fish(
    cfg: EnvConfig,
    rng: random.Random,
    target_total: int | None = None,
    exclude: tuple[tuple[int, int], ...] | set = (),
) -> list["Fish"]:
    spec = _fish_spec(cfg)
    if target_total is not None:
        spec = _scaled_counts(spec, target_total)
    excl = set(exclude)
    free = [
        (x, y)
        for x in range(cfg.size)
        for y in range(cfg.size)
        if (x, y) not in excl
    ]
    total = sum(n for *_, n in spec)
    if total > len(free):  # clamp to fit, keeping the mix
        spec = _scaled_counts(_fish_spec(cfg), len(free))
        total = min(sum(n for *_, n in spec), len(free))
    positions = rng.sample(free, total)
    fishes: list[Fish] = []
    i = 0
    for kind, size, n in spec:
        for _ in range(n):
            fishes.append(Fish(kind, size, positions[i]))
            i += 1
    return fishes


# Drift some fish one tile and occasionally respawn toward target.
# Mutates ``fishes`` in place; ``blocked`` tiles are avoided on respawn. Call once per tick.
def advance_fish_pool(
    fishes: list["Fish"],
    cfg: EnvConfig,
    rng: random.Random,
    size: int,
    target: int | None = None,
    blocked: tuple[tuple[int, int], ...] | set = (),
) -> None:
    occupied = {f.pos for f in fishes}
    if cfg.fish_move_prob > 0:
        for f in fishes:
            if rng.random() < cfg.fish_move_prob:
                dx, dy = rng.choice(((0, -1), (0, 1), (-1, 0), (1, 0)))
                nxt = (f.pos[0] + dx, f.pos[1] + dy)
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in occupied:
                    occupied.discard(f.pos)
                    occupied.add(nxt)
                    f.pos = nxt
    if target is not None and len(fishes) < target and cfg.fish_respawn_prob > 0:
        if rng.random() < cfg.fish_respawn_prob:
            taken = occupied | set(blocked)
            free = [
                (x, y)
                for x in range(size)
                for y in range(size)
                if (x, y) not in taken
            ]
            if free:
                fishes.append(_random_fish_at(cfg, rng, rng.choice(free)))


class Ocean:
    """Single-shark grid world. ``(x, y)`` with x rightward, y downward."""

    # Set up the world from the config + genome (or a colony-supplied body size).
    def __init__(
        self,
        config: EnvConfig = DEFAULT_CONFIG,
        genome: SharkGenome | None = None,
        rng: random.Random | None = None,
        *,
        body_size: float | None = None,
    ):
        self.config = config
        self.genome = genome if genome is not None else SharkGenome.default()
        self.rng = rng if rng is not None else random.Random(config.seed)
        self.size = config.size
        # Length this shark forages at: the genome's adult size, or an age-scaled
        # size the colony passes so juveniles forage smaller.
        self.body_size = self.genome.size if body_size is None else body_size
        self._derive_traits()

        self.shark = config.shark_start
        self.energy = config.start_energy
        self.facing = Action.RIGHT
        self.alive = True
        self.eaten = 0
        self.fishes: list[Fish] = []
        self.done = False
        # A solo Ocean owns and manages its fish; when foraging a shared colony
        # pool it only reads/eats (the colony drifts and respawns it).
        self.owns_fish = True
        self._fish_target = 0

    # --- trait-derived parameters (computed once from the genome) -------------
    # Precompute the trait-driven parameters (tiers, move distance, sight, metabolism).
    def _derive_traits(self) -> None:
        cfg, g = self.config, self.genome
        self.size_norm = _norm("size", self.body_size)
        self.speed_norm = _norm("speed", g.speed)
        self.size_tier = _size_tier(self.body_size, cfg.size_tier_thresholds)
        self.move_tiles = 1 + round(self.speed_norm * cfg.max_extra_move_tiles)
        self.perception = (
            max(1, round(g.field_of_perception)) if cfg.perception_uses_genome
            else cfg.fixed_perception_radius
        )
        # Metabolism mirrors genetic_algo's fitness term: 0.45*size^2 + 0.35*speed^2 (normalized).
        metab = 0.45 * self.size_norm**2 + 0.35 * self.speed_norm**2
        self.step_cost = cfg.base_move_cost + cfg.metab_scale * metab

    # --- episode lifecycle ----------------------------------------------------
    # Start a new life: reset shark state and either spawn or adopt a shared fish set.
    def reset(self, shared_fishes: list[Fish] | None = None) -> dict:
        if self.config.seed is not None:
            self.rng.seed(self.config.seed)
        self.shark = self.config.shark_start
        self.energy = self.config.start_energy
        self.facing = Action.RIGHT
        self.alive = True
        self.eaten = 0
        self.done = False
        if shared_fishes is not None:
            # Forage a colony-owned pool: don't spawn or drift it ourselves.
            self.fishes = shared_fishes
            self.owns_fish = False
        else:
            self.fishes = self._spawn_fish()
            self.owns_fish = True
            self._fish_target = len(self.fishes)
        return self.observe()

    # Spawn this ocean's own fish set at random free tiles.
    def _spawn_fish(self) -> list[Fish]:
        return spawn_fish(self.config, self.rng, exclude={self.config.shark_start})

    # Drift/respawn this ocean's own fish one tick (no-op for shared pools).
    def advance_fish(self) -> None:
        if not self.owns_fish:
            return
        advance_fish_pool(
            self.fishes, self.config, self.rng, self.size,
            target=self._fish_target, blocked={self.shark},
        )

    # Advance one timestep. Returns ``(obs, reward, done)`` (done = shark died).
    def step(self, action: Action) -> tuple[dict, float, bool]:
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
                # Resting recovers energy but still burns some, so idling can't sustain a shark.
                self.energy = min(
                    cfg.max_energy,
                    self.energy + cfg.rest_energy_gain - cfg.rest_passive_cost,
                )
            else:
                self.energy -= self.step_cost
            if self.energy <= 0:
                self.energy = 0.0
                reward = cfg.starve_penalty
                self.alive = False
                self.done = True

        if not self.done:  # prey drift/respawn over the shark's life
            self.advance_fish()

        return self.observe(), reward, self.done

    # Step the shark up to move_tiles in the action's direction, clamped to the grid.
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

    # Resolve an attack on a reachable tile: take poison or eat edible safe prey.
    def _attack(self) -> float:
        cfg = self.config
        reachable = [f for f in self.fishes if self.in_reach(f.pos)]
        poison = next((f for f in reachable if f.kind == "poisonous"), None)
        if poison is not None:
            # Poison halves health; fatal only if already tired or gorged on 2+ fish.
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

    # Can the shark attack a fish here? On its own tile or one step away.
    def in_reach(self, pos: tuple[int, int]) -> bool:
        sx, sy = self.shark
        return abs(sx - pos[0]) + abs(sy - pos[1]) <= 1

    # --- observation (discretize-compatible) ----------------------------------
    # Current observation: energy plus the nearest sensed fish.
    def observe(self) -> dict:
        return {"energy": max(0.0, self.energy), "nearest_fish": self._nearest_fish()}

    # Find the closest fish within perception as a type/distance/direction dict.
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
