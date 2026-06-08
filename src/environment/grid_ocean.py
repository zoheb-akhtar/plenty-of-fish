"""Grid-world ocean environment: one shark foraging among safe/poisonous fish.

A self-contained port of the headless RL toy env
(``src/algorithms/rl-algo/toy_env.py``) so the ``environment`` package can be
imported on its own (the ``rl-algo`` folder is not importable: its name is
hyphenated and its modules use flat imports). The reward structure is kept
identical so a policy trained headless behaves the same here.

Use :class:`GridOcean` for logic and ``gui.py`` for the pygame front-end.
"""

from __future__ import annotations

import random
from enum import IntEnum

try:
    from .config import DEFAULT_CONFIG, EnvConfig
except ImportError:  # run directly: src/environment is on sys.path
    from config import DEFAULT_CONFIG, EnvConfig


class Action(IntEnum):
    """What the shark can do in one timestep."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    REST = 5


class GridOcean:
    """Discrete grid world. ``(x, y)`` with x rightward, y downward.

    The shark moves one tile per step, loses energy by moving, regains it by
    resting, and eats by attacking an adjacent fish. Attacking a poisonous fish
    ends the episode; so does running out of energy. All tunables live in
    :class:`~config.EnvConfig`.
    """

    def __init__(self, config: EnvConfig = DEFAULT_CONFIG):
        self.config = config
        self.size = config.size
        self._rng = random.Random(config.seed)
        self.shark = config.shark_start
        self.energy = config.start_energy
        self.fishes: list[tuple[str, tuple[int, int]]] = []
        self.done = False

    def reset(self):
        # Reproducible layout when a seed is set; fresh one each episode otherwise.
        if self.config.seed is not None:
            self._rng.seed(self.config.seed)
        self.shark = self.config.shark_start
        self.energy = self.config.start_energy
        self.fishes = self._place_fish()
        self.done = False
        return self._get_obs()

    def _place_fish(self) -> list[tuple[str, tuple[int, int]]]:
        """Scatter the configured fish on distinct random tiles (never the shark's)."""
        cfg = self.config
        total = cfg.num_safe_fish + cfg.num_poisonous_fish
        free = [
            (x, y)
            for x in range(cfg.size)
            for y in range(cfg.size)
            if (x, y) != cfg.shark_start
        ]
        if total > len(free):
            raise ValueError(
                f"Cannot place {total} fish on {len(free)} free tiles "
                f"(grid {cfg.size}x{cfg.size})."
            )
        positions = self._rng.sample(free, total)
        kinds = ["safe"] * cfg.num_safe_fish + ["poisonous"] * cfg.num_poisonous_fish
        return list(zip(kinds, positions))

    def step(self, action: Action):
        """Advance one timestep. Returns ``(obs, reward, done)``."""
        if self.done:
            return self._get_obs(), 0.0, True

        cfg = self.config
        reward = cfg.step_penalty
        x, y = self.shark

        if action == Action.UP:
            y = max(0, y - 1)
        elif action == Action.DOWN:
            y = min(self.size - 1, y + 1)
        elif action == Action.LEFT:
            x = max(0, x - 1)
        elif action == Action.RIGHT:
            x = min(self.size - 1, x + 1)
        elif action == Action.REST:
            pass
        elif action == Action.ATTACK:
            # If adjacent to poison and safe at once, the poisonous outcome wins.
            poison_adj = [
                i
                for i, (kind, pos) in enumerate(self.fishes)
                if kind == "poisonous" and self._adjacent(pos)
            ]
            safe_adj = [
                i
                for i, (kind, pos) in enumerate(self.fishes)
                if kind == "safe" and self._adjacent(pos)
            ]
            if poison_adj:
                reward = cfg.poison_penalty
                self.done = True
            elif safe_adj:
                reward = cfg.eat_reward
                self.energy = min(cfg.max_energy, self.energy + cfg.eat_energy_gain)
                self.fishes.pop(safe_adj[0])

        self.shark = (x, y)

        if action != Action.REST:
            self.energy -= cfg.move_energy_cost
        else:
            self.energy = min(cfg.max_energy, self.energy + cfg.rest_energy_gain)

        if self.energy <= 0:
            reward = cfg.starve_penalty
            self.done = True

        return self._get_obs(), reward, self.done

    def _adjacent(self, pos) -> bool:
        sx, sy = self.shark
        px, py = pos
        return abs(sx - px) + abs(sy - py) == 1

    def _nearest_fish(self) -> dict:
        """Info about the closest fish by Manhattan distance."""
        if not self.fishes:
            return {"exists": False, "type": None, "distance": 0, "size": None}

        best_kind = ""
        best_d = 999
        for kind, pos in self.fishes:
            d = abs(self.shark[0] - pos[0]) + abs(self.shark[1] - pos[1])
            if d < best_d:
                best_d = d
                best_kind = kind

        return {
            "exists": True,
            "type": best_kind,
            "distance": best_d,
            "size": "small",
        }

    def _get_obs(self) -> dict:
        return {
            "energy": self.energy,
            "nearest_fish": self._nearest_fish(),
        }


if __name__ == "__main__":
    env = GridOcean()
    env.reset()
    print("layout A:", env.fishes)
    env.reset()
    print("layout B:", env.fishes, "(differs each reset; set config.seed to fix it)")

    for action in [Action.RIGHT, Action.DOWN, Action.ATTACK]:
        obs, reward, done = env.step(action)
        print(
            f"{action.name:7s} shark {env.shark} reward {reward:6.1f} "
            f"done {done} fish_left {len(env.fishes)}"
        )
        if done:
            break
