"""Bridge between the genetic algorithm and reinforcement learning.

RL (``FamilyRL``) handles within-life decisions; the GA evolves genomes. They
meet through behaviour: a genome's temperament biases *which action* the shared
brain takes (not the Q-update itself), so a bold shark attacks more and a cautious
one rests more. A genome's fitness is the reward its biased behaviour earns. The
brain persists across generations, so bodies and behaviour improve together.
"""
from __future__ import annotations

import random

from src.algorithms.rl_algo import NUM_ACTIONS, Action, FamilyRL, discretize
from src.algorithms.rl_algo.state import State
from src.environment.ocean import Ocean
from src.shark import SHARK_TRAITS, Shark, SharkGenome, advance_cycle

# How strongly temperament tilts choices, in Q-value units. Rewards run +25 for a
# safe meal and -100 for poison, so these only sway genuinely close calls.
ATTACK_BIAS = 20.0   # boldness (low caution) -> bite more
REST_BIAS = 6.0      # caution -> hang back and recover


def _norm(name: str, value: float) -> float:
    """Scale a trait to 0..1 using its declared range."""
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


def biased_action(
    brain: FamilyRL, state: State, genome: SharkGenome, rng: random.Random
) -> Action:
    """Pick an action: explore like the brain, else exploit Q nudged by genome.

    The nudge is only in action selection; the Q-update still learns true values.
    """
    if rng.random() < brain.epsilon:
        return Action(rng.randrange(NUM_ACTIONS))

    q = [brain.q.get(state, Action(i)) for i in range(NUM_ACTIONS)]

    caution = _norm("caution", genome.caution)
    boldness = _norm("aggression", genome.aggression) - caution
    q[Action.ATTACK] += ATTACK_BIAS * boldness
    q[Action.REST] += REST_BIAS * caution

    return Action(max(range(NUM_ACTIONS), key=lambda i: q[i]))


def run_life(
    env: Ocean,
    brain: FamilyRL,
    genome: SharkGenome,
    rng: random.Random,
    max_steps: int = 200,
    learn: bool = True,
) -> float:
    """Live one shark for up to ``max_steps`` and return the reward it earned.

    With ``learn=True`` the shared Q-table is updated each step (training).
    """
    obs = env.reset()
    state = discretize(obs)
    total = 0.0

    for _ in range(max_steps):
        action = biased_action(brain, state, genome, rng)
        obs, reward, done = env.step(action)
        next_state = discretize(obs)
        if learn:
            brain.update(state, action, reward, next_state, done)
        state = next_state
        total += reward
        if done:
            break

    return total


def population_fitness(
    genomes: list[SharkGenome],
    brain: FamilyRL,
    rng: random.Random,
    max_steps: int = 200,
    lives_per_genome: int = 3,
) -> list[float]:
    """GA fitness: score each genome by averaging a few lives in its own Ocean.

    All lives feed the one shared brain, which decays exploration once per generation.
    """
    fitnesses: list[float] = []
    for genome in genomes:
        env = Ocean(genome=genome, rng=rng)  # each shark forages its own ocean
        score = (
            sum(run_life(env, brain, genome, rng, max_steps) for _ in range(lives_per_genome))
            / lives_per_genome
        )
        fitnesses.append(score)
    brain.decay_epsilon()  # one generation = one step down the exploration ramp
    return fitnesses


def live_one_cycle(
    population: list[Shark],
    brain: FamilyRL,
    rng: random.Random,
    max_steps: int = 200,
    mutation_rate: float = 0.0,
) -> list[Shark]:
    """Advance the living population by one birth cycle.

    Each living shark forages its own ocean (the shared brain learns); a shark
    that dies there can't breed. Survivors then reproduce and age. Returns the
    next population (survivors + this cycle's pups).
    """
    for shark in population:
        if not shark.alive:
            continue
        env = Ocean(genome=shark.genome, rng=rng, body_size=shark.size)
        run_life(env, brain, shark.genome, rng, max_steps)
        if not env.alive:
            shark.alive = False  # died foraging -> out of the gene pool this cycle
    brain.decay_epsilon()  # one cycle = one step down the exploration ramp
    return advance_cycle(population, rng, mutation_rate)


if __name__ == "__main__":
    # A bold genome vs. a cautious one sharing a warmed brain; boldness should
    # out-eat caution here (few poison traps).
    rng = random.Random(0)
    brain = FamilyRL()

    bold = SharkGenome.default()
    bold.values["caution"] = 0.05
    cautious = SharkGenome.default()
    cautious.values["caution"] = 0.95
    bold_env = Ocean(genome=bold, rng=rng)
    cautious_env = Ocean(genome=cautious, rng=rng)

    # Warm the shared brain up so the exploit branch has something to bias.
    for _ in range(300):
        run_life(bold_env, brain, bold, rng)
    brain.epsilon = brain.epsilon_min

    for name, g, env in (("bold", bold, bold_env), ("cautious", cautious, cautious_env)):
        avg = sum(run_life(env, brain, g, rng, learn=False) for _ in range(50)) / 50
        print(f"{name:9s} caution={g.caution:.2f} -> avg reward {avg:6.1f}")
