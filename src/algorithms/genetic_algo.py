# Run as a demo from the repo root: ``python -m src.algorithms.genetic_algo``.

from __future__ import annotations

import random
import matplotlib.pyplot as plt

from src.shark import SHARK_TRAITS, SharkGenome, evolvable_traits

# Structure follows https://www.datacamp.com/tutorial/genetic-algorithm-python
# Used claude to integrate traits 

# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
def _norm(name: str, value: float) -> float:
    """Scale a trait value to 0..1 using its spec range, so each gene's
    contribution to fitness is comparable regardless of its native units."""
    spec = SHARK_TRAITS[name]
    return (value - spec.min_value) / spec.span


def fitness_function(genome: SharkGenome) -> float:
    """Score a shark's survival/foraging trade-off (higher is better).

    coefficent explaination: 
    * size & speed help catch food, but high caution makes a shark forage less;
    * being big and fast is metabolically expensive (quadratic cost);
    * being reckless (low caution) and small is dangerous (predation risk),
      while caution and size reduce that risk.
    """
    s = _norm("size", genome.size)
    v = _norm("speed", genome.speed)
    c = _norm("caution", genome.caution)

    food = (0.5 * v + 0.5 * s) * (1.0 - 0.6 * c)
    metabolism = 0.45 * s**2 + 0.35 * v**2
    predation_risk = max(0.0, (1.0 - c) * (0.6 - 0.4 * s))

    return food - metabolism - predation_risk

# if we add in secondary traits make sure we include them here 


# ---------------------------------------------------------------------------
# GA operators (delegate gene mechanics to the genome / trait specs)
# ---------------------------------------------------------------------------
# A population of sharks with random evolvable genes (others at default).
def create_initial_population(size: int, rng: random.Random) -> list[SharkGenome]:
    return [SharkGenome.random_evolvable(rng) for _ in range(size)]


def selection(
    population: list[SharkGenome],
    fitnesses: list[float],
    rng: random.Random,
    tournament_size: int = 3,
) -> list[SharkGenome]:
    #Tournament selection: each slot is won by the fittest of k random picks.
    selected = []
    paired = list(zip(population, fitnesses))
    for _ in range(len(population)):
        tournament = rng.sample(paired, tournament_size)
        winner = max(tournament, key=lambda pair: pair[1])[0]
        selected.append(winner)
    return selected


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def genetic_algorithm(
    population_size: int = 100,
    generations: int = 40,
    mutation_rate: float = 0.2,
    seed: int | None = None,
    show_plots: bool = True,
) -> SharkGenome:
    """Evolve a population of sharks and return the fittest found."""
    if population_size % 2 != 0:
        raise ValueError("population_size must be even (sharks breed in pairs).")

    rng = random.Random(seed)
    population = create_initial_population(population_size, rng)
    genes = evolvable_traits()  # gene order, e.g. ["size", "speed", "caution"]

    # History for plotting/reporting.
    best_per_gen: list[tuple[SharkGenome, float]] = []
    fitness_range: list[tuple[float, float]] = []  # (min, max) per generation

    headers = ["Gen", *[g.capitalize() for g in genes], "Fitness"]
    rows: list[list[str]] = []

    for _ in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]

        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_per_gen.append((best_individual, best_fitness))
        fitness_range.append((min(fitnesses), max(fitnesses)))
        rows.append(
            [str(len(best_per_gen))]
            + [f"{best_individual[g]:.3f}" for g in genes]
            + [f"{best_fitness:.4f}"]
        )

        # Select breeders, then produce the next generation pair by pair.
        breeders = selection(population, fitnesses, rng)
        next_population: list[SharkGenome] = []
        for i in range(0, len(breeders), 2):
            child1, child2 = SharkGenome.crossover(breeders[i], breeders[i + 1], rng)
            next_population.append(child1.mutate(mutation_rate, rng))
            next_population.append(child2.mutate(mutation_rate, rng))

        # Elitism: carry the best shark over unchanged.
        next_population[0] = best_individual
        population = next_population

    _print_table(headers, rows)

    overall_best = max(best_per_gen, key=lambda pair: pair[1])[0]
    if show_plots:
        _plot_history(genes, best_per_gen, fitness_range)
        plt.show()

    return overall_best


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned text table (no external table dependency)."""
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    line = "-+-".join("-" * w for w in widths)
    fmt = lambda cells: " | ".join(c.rjust(widths[i]) for i, c in enumerate(cells))
    print(fmt(headers))
    print(line)
    for row in rows:
        print(fmt(row))


def _plot_history(
    genes: list[str],
    best_per_gen: list[tuple[SharkGenome, float]],
    fitness_range: list[tuple[float, float]],
) -> None:
    gens = range(1, len(best_per_gen) + 1)

    # One subplot per evolvable trait: the best shark's value over time.
    fig, axs = plt.subplots(len(genes), 1, figsize=(10, 3 * len(genes)), squeeze=False)
    for ax, gene in zip(axs[:, 0], genes):
        values = [genome[gene] for genome, _ in best_per_gen]
        ax.plot(gens, values, marker="o", markersize=3)
        ax.set_ylabel(f"{gene}\n({SHARK_TRAITS[gene].unit})")
        ax.set_ylim(SHARK_TRAITS[gene].min_value, SHARK_TRAITS[gene].max_value)
    axs[-1, 0].set_xlabel("Generation")
    axs[0, 0].set_title("Best shark's evolvable traits over generations")
    fig.tight_layout()

    # Fitness: best line plus the population's min..max band.
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    best_fitness = [f for _, f in best_per_gen]
    mins = [lo for lo, _ in fitness_range]
    maxs = [hi for _, hi in fitness_range]
    ax2.plot(gens, best_fitness, color="black", label="Best fitness")
    ax2.fill_between(gens, mins, maxs, color="gray", alpha=0.4, label="Population range")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness over generations")
    ax2.legend()
    fig2.tight_layout()


if __name__ == "__main__":
    best = genetic_algorithm(
        population_size=100,
        generations=40,
        mutation_rate=0.2,
        seed=0,
    )
    print("\nBest shark found:")
    for name in SHARK_TRAITS:
        marker = " (evolved)" if SHARK_TRAITS[name].evolvable else ""
        print(f"  {name:<22}{best[name]:.3f}{marker}")
    print(f"  fitness               {fitness_function(best):.4f}")
