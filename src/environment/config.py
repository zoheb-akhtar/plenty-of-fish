"""Single place to set every Ocean / GUI parameter.

Edit the defaults here, or build your own ``EnvConfig(...)`` and pass it to
:class:`~ocean.Ocean` or the GUIs. Nothing else hard-codes these values.

Fish spawn at random free tiles on every ``reset()`` (leave ``seed = None`` for
a fresh layout each life; set an int for reproducible layouts). Shark traits
turn several of these knobs into per-shark behaviour (see ``ocean.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # --- grid -----------------------------------------------------------------
    size: int = 10
    shark_start: tuple[int, int] = (0, 0)
    seed: int | None = None

    # --- fish: how many of each (type, size) spawn at random tiles each reset --
    num_safe_small: int = 4
    num_safe_medium: int = 2
    num_safe_large: int = 1
    num_poison_small: int = 2
    num_poison_medium: int = 2
    num_poison_large: int = 1

    # --- size tiers (one shared scale for the shark's size trait AND fish size) -
    # Shark ``size`` trait ranges 0.5..9.0: < thresholds[0] -> small(0),
    # < thresholds[1] -> medium(1), else large(2). Fish sizes map by the same idx.
    size_tier_thresholds: tuple[float, float] = (3.5, 6.5)

    # --- eating (indexed by the fish's size tier 0=small / 1=medium / 2=large) --
    eat_reward_by_size: tuple[float, float, float] = (15.0, 25.0, 40.0)
    eat_energy_by_size: tuple[float, float, float] = (0.20, 0.30, 0.45)

    # --- energy economy -------------------------------------------------------
    start_energy: float = 1.0
    max_energy: float = 1.0
    base_move_cost: float = 0.015   # baseline energy burned on any non-REST action
    metab_scale: float = 0.03       # scales the size/speed metabolism term (see ocean.py)
    rest_energy_gain: float = 0.05

    # --- rewards / penalties --------------------------------------------------
    step_penalty: float = -0.1
    poison_penalty: float = -100.0
    starve_penalty: float = -50.0

    # --- movement (speed trait -> tiles travelled per MOVE action) ------------
    max_extra_move_tiles: int = 2   # speed_norm 1.0 -> 1 + 2 = 3 tiles/step

    # --- perception (field_of_perception trait -> how far the shark senses fish) -
    perception_uses_genome: bool = True
    fixed_perception_radius: int = 5

    # --- rendering / GUI ------------------------------------------------------
    tile_size: int = 60
    tick_interval: float = 0.30     # seconds between world ticks (Watch mode + auto)
    max_steps: int = 300            # per-life step cap

    # --- watch mode (the evolving family) -------------------------------------
    watch_population_size: int = 12   # kept even (sharks breed in pairs)
    watch_grid_cols: int = 4
    watch_mutation_rate: float = 0.2
    watch_tick_interval: float = 0.05  # seconds between world steps (fast, to watch evolution)
    watch_max_steps: int = 80          # per-life cap inside Watch (short, so generations turn over)


#: The config used when none is supplied.
DEFAULT_CONFIG = EnvConfig()
