"""The bridge between the genetic algorithm and reinforcement learning.

The two algorithms work on different timescales and this module joins them:

    * RL (``FamilyRL``) handles **within-life** decisions -- a shark learns, by
      trial and error in ``ToyOcean``, which actions pay off in which states.
    * The GA handles **who reproduces** -- it evolves shark genomes.

They meet through behaviour. A genome doesn't change the Q-table; instead its
temperament *biases which action the shark takes* (the rl-algo README's plan:
"bias action choice with genome ... not inside the Q-update itself"). A cautious
shark attacks less and rests more; a bold one attacks more -- which is great when
the fish is safe and fatal when it's poisonous. A genome's fitness is simply the
reward its biased behaviour earns, so the GA selects for temperaments that do
well given what the shared brain has learned.

The brain is created once and **persists across generations**, so behaviour and
bodies improve together.
"""
from __future__ import annotations

import random

from src.algorithms.rl_algo import NUM_ACTIONS, Action, FamilyRL, discretize
from src.algorithms.rl_algo.state import State
from src.environment.ocean import Ocean
from src.shark import SHARK_TRAITS, SharkGenome

# How strongly temperament tilts the shark's choices, in Q-value units. ToyOcean
# pays +25 for a safe meal and -100 for biting poison, so these nudges sway
# genuinely close calls without overriding a hard-learned "don't eat poison".
ATTACK_BIAS = 20.0   # boldness (low caution) -> bite more
REST_BIAS = 6.0      # caution -> hang back and recover


def _norm(name: str, value: float) -> float:
    """Scale a trait to 0..1 using its declared range."""
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


def biased_action(
    brain: FamilyRL, state: State, genome: SharkGenome, rng: random.Random
) -> Action:
    """Pick an action: explore like the brain would, else exploit Q nudged by genome.

    The nudge lives entirely in *action selection*. The Q-update afterwards is
    the plain Bellman update on whatever action was actually taken, so the brain
    still learns the true value of moves -- the genome only colours the choice.
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
    """Fitness function for the GA: score every genome by living it in its own Ocean.

    Each genome lives a few times (exploration makes a single life noisy) in its
    OWN ocean and is scored by its average reward. All lives feed the one shared
    brain, which decays its exploration once per generation.
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


if __name__ == "__main__":
    # Show the link in miniature: a bold genome vs. a cautious one, sharing a
    # brain that has already learned ToyOcean. Boldness should out-eat caution
    # here (few poison traps), but pay off less as poison density rises.
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
