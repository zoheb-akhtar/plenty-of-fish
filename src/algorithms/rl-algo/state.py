"""Discretize environment observations into Q-learning states."""

from typing import Any, Tuple

State = Tuple[str, str, str, str]


def _bucket_energy(energy: float) -> str:
    if energy < 0.33:
        return "low"
    if energy < 0.66:
        return "medium"
    return "high"


def _bucket_distance(distance: int, exists: bool) -> str:
    if not exists:
        return "none"
    if distance <= 2:
        return "close"
    if distance <= 5:
        return "medium"
    return "far"


def _bucket_fish_type(fish_type: str | None, exists: bool) -> str:
    if not exists or fish_type is None:
        return "none"
    return fish_type  # "safe", "poisonous", "strong" later


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
    return (energy, fish_type, distance, size)


if __name__ == "__main__":
    from actions import Action
    from toy_env import ToyOcean

    env = ToyOcean()
    obs = env.reset()
    print("reset:", discretize(obs))

    obs, reward, done = env.step(Action.RIGHT)
    print("after RIGHT:", discretize(obs), "reward=", reward, "done=", done)