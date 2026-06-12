"""Discretize environment observations into Q-learning states.

The state tuple is ``(energy, fish_type, distance, size, direction)``. ``direction``
points at the nearest fish so the brain can learn to *swim toward food* and
attack once it is in reach (``distance == "reach"``); without it the shark could
only sit and avoid poison. ``direction`` is derived from the signed ``dx``/``dy``
offsets the env reports in ``nearest_fish``.
"""

from typing import Any, Tuple

State = Tuple[str, str, str, str, str]


def _bucket_energy(energy: float) -> str:
    if energy < 0.33:
        return "low"
    if energy < 0.66:
        return "medium"
    return "high"


def _bucket_distance(distance: int, exists: bool) -> str:
    if not exists:
        return "none"
    if distance <= 1:
        return "reach"   # on the shark's tile or one step away -> attackable
    if distance <= 3:
        return "close"
    if distance <= 6:
        return "medium"
    return "far"


def _bucket_direction(dx: int, dy: int, exists: bool) -> str:
    """Dominant cardinal direction to the nearest fish (y grows downward)."""
    if not exists:
        return "none"
    if dx == 0 and dy == 0:
        return "here"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _bucket_fish_type(fish_type: str | None, exists: bool) -> str:
    if not exists or fish_type is None:
        return "none"
    return fish_type  # "safe" or "poisonous"


def _bucket_size(size: str | None, exists: bool) -> str:
    if not exists or size is None:
        return "none"
    return size  # "small", "medium", "large"


def discretize(obs: dict[str, Any]) -> State:
    """Convert env observation dict to a discrete state tuple."""
    energy = _bucket_energy(float(obs["energy"]))
    nf = obs["nearest_fish"]
    exists = bool(nf.get("exists", False))
    fish_type = _bucket_fish_type(nf.get("type"), exists)
    distance = _bucket_distance(int(nf.get("distance", 0)), exists)
    size = _bucket_size(nf.get("size"), exists)
    direction = _bucket_direction(int(nf.get("dx", 0)), int(nf.get("dy", 0)), exists)
    return (energy, fish_type, distance, size, direction)


if __name__ == "__main__":
    from .actions import Action
    from .toy_env import ToyOcean

    env = ToyOcean()
    obs = env.reset()
    print("reset:", discretize(obs))

    obs, reward, done = env.step(Action.RIGHT)
    print("after RIGHT:", discretize(obs), "reward=", reward, "done=", done)