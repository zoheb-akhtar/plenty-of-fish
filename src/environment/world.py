"""The real ocean: a multi-shark grid world the GA + RL plug into.

This is the environment the RL README calls for. It keeps the exact
observation/step contract ``ToyOcean`` already established, so ``FamilyRL`` and
``state.discretize`` work against it with **no changes**:

    obs                  = world.get_observation(shark_id)
    obs, reward, done    = world.step(shark_id, action)

What makes it "real" instead of a toy:
  * many sharks live in it at once (one family, one shared brain);
  * each shark's *genome* (``src.shark``) drives its dynamics --
      - speed   -> how far a move action travels (forage faster),
      - caution -> moves shorter (forage less) but lowers predation risk,
      - size    -> lowers predation risk but raises metabolism,
      - field_of_perception -> how far it can sense fish;
  * fish are consumed and respawn, so the ocean keeps feeding the population.

The class is deliberately pygame-free; rendering lives in
``src.environment.render`` so headless training never imports a display library.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.algorithms.rl_algo import Action
from src.shark import SHARK_TRAITS, SharkGenome

# ---------------------------------------------------------------------------
# Tunables. These mirror ToyOcean's reward scale so a shark trained on the toy
# grid behaves sensibly here; tweak freely.
# ---------------------------------------------------------------------------
STEP_PENALTY = -0.1          # small per-step cost: discourages dithering
EAT_REWARD = 25.0            # eating a safe fish
EAT_ENERGY = 0.3             # energy refunded by a meal
POISON_REWARD = -100.0       # attacking a poisonous fish (fatal)
STARVE_PENALTY = -50.0       # energy hit 0 (fatal)
PREDATION_REWARD = -75.0     # eaten by an off-screen predator (fatal, random)
REST_GAIN = 0.05             # energy recovered by resting

BASE_MOVE_COST = 0.02        # energy burned per tile travelled, before genome
MAX_MOVE = 4                 # tiles a maximally fast shark covers in one move
BASE_PREDATION = 0.01        # baseline per-step chance of predation


def _norm(name: str, value: float) -> float:
    """Scale a trait to 0..1 using its declared range (same idea as the GA)."""
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


@dataclass
class Fish:
    """A single fish. ``kind`` is "safe" or "poisonous"; size is fixed for now."""

    kind: str
    x: int
    y: int
    size: str = "small"


@dataclass
class Shark:
    """One shark living in the ocean. Carries its genome and live game state."""

    id: int
    genome: SharkGenome
    x: int
    y: int
    energy: float = 1.0
    alive: bool = True
    food_eaten: int = 0
    steps_survived: int = 0
    cause_of_death: str | None = None

    # Genome-derived constants, cached once so step() stays cheap. ----------
    move_range: int = field(init=False)
    metabolism: float = field(init=False)
    perception: int = field(init=False)
    predation_chance: float = field(init=False)

    def __post_init__(self) -> None:
        size = _norm("size", self.genome.size)
        speed = _norm("speed", self.genome.speed)
        caution = _norm("caution", self.genome.caution)

        # Fast sharks cover more ground; cautious sharks hold back (forage less).
        reach = 1 + round(speed * (MAX_MOVE - 1))
        self.move_range = max(1, round(reach * (1.0 - 0.5 * caution)))

        # Big, fast bodies are expensive to run.
        self.metabolism = BASE_MOVE_COST * (1.0 + 0.5 * size + 0.5 * speed)

        # Sensing radius is the (currently fixed) perception trait, in tiles.
        self.perception = max(1, round(self.genome.field_of_perception))

        # Caution and bulk both make a shark a harder target for predators.
        self.predation_chance = BASE_PREDATION * (1.0 - caution) * (1.0 - 0.5 * size)


class World:
    """A grid ocean holding one family of sharks and a pool of fish."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        genomes: list[SharkGenome] | None = None,
        num_safe: int = 18,
        num_poison: int = 12,
        rng: random.Random | None = None,
    ):
        self.width = width
        self.height = height
        self.num_safe = num_safe
        self.num_poison = num_poison
        self.rng = rng or random.Random()

        self.sharks: list[Shark] = []
        self.fishes: list[Fish] = []
        if genomes is not None:
            self.reset(genomes)

    # --- setup -------------------------------------------------------------
    def reset(self, genomes: list[SharkGenome]) -> None:
        """Repopulate the ocean with a fresh family and a full pool of fish."""
        self.fishes = []
        self.sharks = []

        for kind, count in (("safe", self.num_safe), ("poisonous", self.num_poison)):
            for _ in range(count):
                x, y = self._free_tile()
                self.fishes.append(Fish(kind, x, y))

        for i, genome in enumerate(genomes):
            x, y = self.rng.randrange(self.width), self.rng.randrange(self.height)
            self.sharks.append(Shark(id=i, genome=genome, x=x, y=y))

    def _free_tile(self) -> tuple[int, int]:
        """A random tile with no fish on it (sharks may overlap fish/each other)."""
        occupied = {(f.x, f.y) for f in self.fishes}
        while True:
            x, y = self.rng.randrange(self.width), self.rng.randrange(self.height)
            if (x, y) not in occupied:
                return x, y

    # --- queries -----------------------------------------------------------
    def living_shark_ids(self) -> list[int]:
        return [s.id for s in self.sharks if s.alive]

    def get_observation(self, shark_id: int) -> dict:
        """Same dict shape as ``ToyOcean``: energy + nearest perceptible fish."""
        shark = self.sharks[shark_id]
        return {
            "energy": shark.energy,
            "nearest_fish": self._nearest_fish(shark),
        }

    def _nearest_fish(self, shark: Shark) -> dict:
        """Closest fish within the shark's perception radius (Manhattan)."""
        best: Fish | None = None
        best_d = shark.perception + 1
        for fish in self.fishes:
            d = abs(shark.x - fish.x) + abs(shark.y - fish.y)
            if d <= shark.perception and d < best_d:
                best, best_d = fish, d

        if best is None:
            return {"exists": False, "type": None, "distance": 0, "size": None}
        return {
            "exists": True,
            "type": best.kind,
            "distance": best_d,
            "size": best.size,
        }

    # --- dynamics ----------------------------------------------------------
    def step(self, shark_id: int, action: Action) -> tuple[dict, float, bool]:
        """Advance one shark by one action. Returns (obs, reward, done)."""
        shark = self.sharks[shark_id]
        if not shark.alive:
            return self.get_observation(shark_id), 0.0, True

        reward = STEP_PENALTY

        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            tiles = self._move(shark, action)
            shark.energy -= shark.metabolism * tiles
        elif action == Action.REST:
            shark.energy = min(1.0, shark.energy + REST_GAIN)
        elif action == Action.ATTACK:
            reward += self._attack(shark)

        # Random predation: caution and size buy safety, nothing else.
        if shark.alive and self.rng.random() < shark.predation_chance:
            return self._kill(shark, "predation", PREDATION_REWARD)

        if shark.alive and shark.energy <= 0.0:
            return self._kill(shark, "starvation", STARVE_PENALTY)

        if shark.alive:
            shark.steps_survived += 1
        return self.get_observation(shark_id), reward, not shark.alive

    def _move(self, shark: Shark, action: Action) -> int:
        """Slide up to ``move_range`` tiles in one direction. Returns tiles moved."""
        dx, dy = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }[action]

        moved = 0
        for _ in range(shark.move_range):
            nx, ny = shark.x + dx, shark.y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                break  # don't pay energy for walking into a wall
            shark.x, shark.y = nx, ny
            moved += 1
        return moved

    def _attack(self, shark: Shark) -> float:
        """Bite an adjacent fish. Poison wins ties and is fatal; safe fish feed."""
        poison = self._adjacent_fish(shark, "poisonous")
        if poison is not None:
            shark.alive = False
            shark.cause_of_death = "poison"
            return POISON_REWARD

        safe = self._adjacent_fish(shark, "safe")
        if safe is not None:
            self.fishes.remove(safe)
            shark.food_eaten += 1
            shark.energy = min(1.0, shark.energy + EAT_ENERGY)
            self._respawn_fish("safe")  # keep the ocean's food supply stocked
            return EAT_REWARD
        return 0.0  # bit at empty water

    def _adjacent_fish(self, shark: Shark, kind: str) -> Fish | None:
        for fish in self.fishes:
            if fish.kind == kind and abs(shark.x - fish.x) + abs(shark.y - fish.y) == 1:
                return fish
        return None

    def _respawn_fish(self, kind: str) -> None:
        x, y = self._free_tile()
        self.fishes.append(Fish(kind, x, y))

    def _kill(self, shark: Shark, cause: str, reward: float) -> tuple[dict, float, bool]:
        shark.alive = False
        shark.cause_of_death = cause
        return self.get_observation(shark.id), reward, True


if __name__ == "__main__":
    # Quick smoke test: drop a small family in and let them wander randomly.
    rng = random.Random(0)
    genomes = [SharkGenome.random_evolvable(rng) for _ in range(5)]
    world = World(width=12, height=12, genomes=genomes, rng=rng)

    print(f"start: {len(world.living_shark_ids())} sharks, {len(world.fishes)} fish")
    for _ in range(50):
        for sid in world.living_shark_ids():
            world.step(sid, Action(rng.randrange(len(Action))))
    survivors = world.sharks
    alive = world.living_shark_ids()
    print(f"after 50 steps: {len(alive)} alive")
    for s in survivors:
        print(
            f"  shark {s.id}: alive={s.alive} eaten={s.food_eaten} "
            f"steps={s.steps_survived} cause={s.cause_of_death}"
        )
