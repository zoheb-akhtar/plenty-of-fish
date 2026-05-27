"""Tiny grid world for headless RL: one shark and multiple fish."""

from __future__ import annotations

from actions import Action

# (type, (x, y)) — spread on default 10x10 grid, non-overlapping (shark starts at 0,0)
INITIAL_FISHES: list[tuple[str, tuple[int, int]]] = [
    ("safe", (5, 5)),
    ("safe", (1, 8)),
    ("safe", (8, 2)),
    ("poisonous", (9, 3)),
    ("poisonous", (7, 1)),
    ("poisonous", (4, 6)),
    ("poisonous", (2, 3)),
]


class ToyOcean:
    def __init__(self, size: int = 10, initial_fishes: list[tuple[str, tuple[int, int]]] | None = None):
        self.size = size
        self._fish_template = (
            initial_fishes if initial_fishes is not None else list(INITIAL_FISHES)
        )
        self.shark = (0, 0)
        self.energy = 1.0
        self.fishes: list[tuple[str, tuple[int, int]]] = []
        self.done = False

    def reset(self):
        self.shark = (0, 0)
        self.energy = 1.0
        self.fishes = [tuple(f) for f in self._fish_template]
        self.done = False
        return self._get_obs()

    def step(self, action: Action):
        """One timestep. Returns (obs, reward, done)."""
        if self.done:
            return self._get_obs(), 0.0, True

        reward = -0.1

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
            # If adjacent to poison + safe at once, poisonous outcome wins first
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
                reward = -100.0
                self.done = True
            elif safe_adj:
                reward = 25.0
                self.energy = min(1.0, self.energy + 0.3)
                eaten_idx = safe_adj[0]
                self.fishes.pop(eaten_idx)

        self.shark = (x, y)

        if action != Action.REST:
            self.energy -= 0.02
        else:
            self.energy = min(1.0, self.energy + 0.05)

        if self.energy <= 0:
            reward = -50.0
            self.done = True

        return self._get_obs(), reward, self.done

    def _adjacent(self, pos):
        sx, sy = self.shark
        px, py = pos
        return abs(sx - px) + abs(sy - py) == 1

    def _nearest_fish(self):
        """Return info about closest fish (nearest by Manhattan distance)."""
        if not self.fishes:
            return {"exists": False, "type": None, "distance": 0, "size": None}

        best_kind = ""
        best_d = 999
        best_pos = (0, 0)

        for kind, pos in self.fishes:
            d = abs(self.shark[0] - pos[0]) + abs(self.shark[1] - pos[1])
            if d < best_d:
                best_d = d
                best_kind = kind
                best_pos = pos

        return {
            "exists": True,
            "type": best_kind,
            "distance": best_d,
            "size": "small",
        }

    def _get_obs(self):
        return {
            "energy": self.energy,
            "nearest_fish": self._nearest_fish(),
        }


if __name__ == "__main__":
    from state import discretize

    env = ToyOcean()
    obs = env.reset()
    print("fish count", len(env.fishes), "reset", discretize(obs), obs)

    for action in [Action.RIGHT, Action.RIGHT, Action.ATTACK]:
        obs, reward, done = env.step(action)
        print(
            action.name,
            "shark",
            env.shark,
            "reward",
            reward,
            "done",
            done,
            "state",
            discretize(obs),
            "remaining fish",
            len(env.fishes),
        )
        if done:
            break
