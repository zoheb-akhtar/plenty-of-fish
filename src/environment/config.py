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

    # --- size tiers: shared scale for the shark size trait AND fish size ------
    # < thresholds[0] -> small(0), < thresholds[1] -> medium(1), else large(2).
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
    rest_passive_cost: float = 0.03  # hunger paid even while resting; below rest_energy_gain
    tired_energy: float = 0.33      # at/below this the shark is "tired" (the "low" state bucket)

    # --- prey behaviour (fish drift and replenish over a life) ----------------
    fish_move_prob: float = 0.05    # chance each fish wanders one tile per world tick
    fish_respawn_prob: float = 0.70  # chance per tick to respawn toward target. High = fast refill,
                                     # feeding a bigger colony without a denser standing pool.

    # --- rewards / penalties --------------------------------------------------
    step_penalty: float = -0.1
    poison_penalty: float = -50.0
    starve_penalty: float = -50.0

    # --- movement (speed trait -> tiles travelled per MOVE action) ------------
    max_extra_move_tiles: int = 2   # speed_norm 1.0 -> 1 + 2 = 3 tiles/step

    # --- fish movement --------------------------------------------------------
    fish_move: bool = True          # if True, fish random-walk; False = static prey

    # --- perception (field_of_perception trait -> how far the shark senses fish) -
    perception_uses_genome: bool = True
    fixed_perception_radius: int = 5

    # --- rendering / GUI ------------------------------------------------------
    tile_size: int = 60
    tick_interval: float = 0.30     # seconds between world ticks (Watch mode + auto)
    max_steps: int = 300            # per-life step cap

    # --- watch mode (the evolving family) -------------------------------------
    watch_population_size: int = 30   # founding pod size (bigger survives the competition better)
    watch_grid_cols: int = 4
    watch_mutation_rate: float = 0.2
    watch_tick_interval: float = 0.05  # seconds between world steps (fast, to watch evolution)
    watch_max_steps: int = 80          # per-life cap inside Watch (short, so generations turn over)

    # --- watch mode: shark lifecycle (age, reproduction, death) ---------------
    # The pod grows freely and is culled only above the carrying capacity; the grid
    # only animates ``display_slots`` of them, the rest forage headlessly.
    watch_display_slots: int = 12      # mini-oceans shown on screen (4 cols x 3 rows)
    watch_carrying_capacity: int = 100  # cull overcrowding above this many sharks
    watch_brain_warmup_lives: int = 400  # headless lives to teach the brain first (else extinction)

    # --- watch mode: realism levers (selection + competition) -----------------
    # The colony forages ONE shared, depleting pool (competition); only fed sharks
    # breed, broods are spaced by gestation, and overcrowding culls weakest foragers.
    watch_shared_fish_pool: bool = True   # one shared, depleting fish set for the colony
    watch_fish_per_shark: float = 1.8     # shared-pool size = this * population (capped below)
    watch_fish_pool_cap: int = 30         # cap on standing shared-pool fish; with the high respawn
                                          # rate this sustains a stable demo colony (small long-run risk)
    watch_require_food_to_breed: bool = True   # a shark must have eaten to reproduce
    watch_fitness_weighted_breeding: bool = True  # better foragers leave more offspring
    watch_cull_weakest: bool = True       # overcrowding removes the weakest foragers first
    watch_gestation_divisor: float = 6.0  # gestation_period / this = cycles between broods
    watch_brood_size: int = 2             # pups per breeding pair (2 = "2 sharks per 2 sharks")


#: The config used when none is supplied.
DEFAULT_CONFIG = EnvConfig()
