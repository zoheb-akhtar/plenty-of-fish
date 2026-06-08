"""Single place to set every GridOcean parameter.

Edit the defaults below, or build your own ``EnvConfig(...)`` and pass it to
:class:`~grid_ocean.GridOcean` / the GUI. Nothing else hard-codes these values.

Fish spawn at random free tiles on every ``reset()``. Leave ``seed = None`` for
a fresh layout each episode; set an int for a reproducible layout (useful for
RL training / comparing runs).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # --- grid & spawning ----------------------------------------------------
    size: int = 10                      # grid is size x size tiles
    num_safe_fish: int = 3              # green fish (good to eat)
    num_poisonous_fish: int = 4         # purple fish (eating ends the episode)
    shark_start: tuple[int, int] = (0, 0)
    seed: int | None = None             # None = random fish each episode; int = reproducible

    # --- energy -------------------------------------------------------------
    start_energy: float = 1.0
    max_energy: float = 1.0             # energy is capped here when regained
    move_energy_cost: float = 0.02      # energy lost per non-rest action
    rest_energy_gain: float = 0.05      # energy regained by resting
    eat_energy_gain: float = 0.30       # energy regained by eating a safe fish

    # --- rewards ------------------------------------------------------------
    step_penalty: float = -0.1          # base reward applied every step
    eat_reward: float = 25.0            # eating a safe fish
    poison_penalty: float = -100.0      # attacking a poisonous fish (ends episode)
    starve_penalty: float = -50.0       # energy reaching zero (ends episode)

    # --- rendering (GUI) ----------------------------------------------------
    tile_size: int = 60                 # pixels per tile in the pygame window


#: The config used when none is supplied.
DEFAULT_CONFIG = EnvConfig()
