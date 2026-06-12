"""Render a self-contained PDF report from a *colony* run's results.

The colony analogue of :mod:`experiments.report`. Where that reports the isolated
GA+RL run (each shark scored alone), this reports a :func:`src.colony.run_colony`
result -- a shared-ocean ecology -- so the pages centre on population dynamics:
colony size vs. carrying capacity, births/deaths per cycle, fitness under
competition, and how the evolvable traits drift when food is contested.

Used automatically by ``run_colony_experiment.py``, or standalone::

    python -m experiments.colony_report experiments/colony_30sharks/results.pkl
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")  # headless: render straight to the PDF, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.shark import SHARK_TRAITS, evolvable_traits  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Palette, kept consistent with experiments/report.py.
_BEST = "#08519c"
_MEAN = "#e6550d"
_BAND = "#9ecae1"
_BAR = "#3182bd"
_POP = "#2171b5"
_CAP = "#d94801"
_BIRTHS = "#31a354"
_FORAGE = "#de2d26"
_OLDAGE = "#756bb1"
_CULL = "#fd8d3c"
_AVG = "#e6550d"
_HEADER_BG = "#eef4fa"
_LINE = "#d6e0ea"
_MUTED = "#5a6b7b"
_PAGE = (8.5, 11)  # US Letter portrait, inches


def _draw_table(ax, headers, rows, title=None, cell_loc="center") -> None:
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold",
                     color=_BEST, pad=8)
    table = ax.table(
        cellText=rows, colLabels=headers, cellLoc=cell_loc,
        loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor(_LINE)
        if r == 0:
            cell.set_facecolor(_HEADER_BG)
            cell.set_text_props(fontweight="bold")


def _ratios(*counts: int) -> list[int]:
    return [c + 1 for c in counts]


def _cover_page(results: dict) -> plt.Figure:
    p = results["params"]
    history = results["history"]
    brain = results["brain"]
    pops = [h["population"] for h in history]
    peak_pop = max(pops, default=0)
    final_pop = pops[-1] if pops else 0
    best_overall = max((h["best_fitness"] for h in history), default=float("nan"))
    final_mean = history[-1]["mean_fitness"] if history else float("nan")
    extinct = "yes — at cycle %s" % p["cycles_completed"] if results["extinct"] else "no"

    summary = [
        ["Cycles requested / completed", f"{p['cycles']} / {p['cycles_completed']}"],
        ["Founding pod", str(p["population"])],
        ["Carrying capacity", str(p["carrying_capacity"])],
        ["Peak population", str(peak_pop)],
        ["Final population", str(final_pop)],
        ["Colony went extinct?", extinct],
        ["Best fitness (any cycle)", f"{best_overall:.3f}"],
        ["Final mean fitness", f"{final_mean:.3f}"],
        ["Runtime", f"{results['duration_seconds']:.1f} s"],
    ]
    eco = [
        ["Shared, depleting ocean", str(p["shared_fish_pool"])],
        ["Fish per shark / pool cap", f"{p['fish_per_shark']} / {p['fish_pool_cap']}"],
        ["Must eat to breed", str(p["require_food_to_breed"])],
        ["Fitness-weighted breeding", str(p["fitness_weighted_breeding"])],
        ["Cull weakest on overcrowding", str(p["cull_weakest"])],
        ["Gestation divisor / brood", f"{p['gestation_divisor']} / {p['brood_size']}"],
        ["Steps per life / brain warmup", f"{p['max_steps']} / {p['warmup_lives']}"],
        ["Mutation rate / seed", f"{p['mutation_rate']} / {p['seed']}"],
        ["Run timestamp", results["timestamp"]],
    ]
    brain_rows = [
        ["States learned", str(brain["states_learned"])],
        ["Final epsilon (explore rate)", f"{brain['epsilon']:.3f}"],
        ["Epsilon min / decay", f"{brain['epsilon_min']:.3f} / {brain['epsilon_decay']:.4f}"],
        ["Alpha (learning rate)", f"{brain['alpha']:.3f}"],
        ["Gamma (discount)", f"{brain['gamma']:.3f}"],
    ]

    fig = plt.figure(figsize=_PAGE)
    fig.text(0.08, 0.95, "Shark Colony Experiment", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.925, f"Shared-ocean ecology (GA + RL) — {results['timestamp']}",
             fontsize=11, color=_MUTED)

    gs = fig.add_gridspec(
        3, 1, left=0.08, right=0.92, top=0.88, bottom=0.05, hspace=0.4,
        height_ratios=_ratios(len(summary), len(eco), len(brain_rows)),
    )
    _draw_table(fig.add_subplot(gs[0]), ["Summary", ""], summary,
                title="Summary", cell_loc="left")
    _draw_table(fig.add_subplot(gs[1]), ["Ecology setting", "Value"], eco,
                title="Ecology & competition", cell_loc="left")
    _draw_table(fig.add_subplot(gs[2]), ["Shared brain (Q-learning)", "Value"], brain_rows,
                title="Shared brain (Q-learning)", cell_loc="left")
    return fig


def _population_page(results: dict) -> plt.Figure:
    history = results["history"]
    cycles = [h["cycle"] for h in history]
    cap = results["params"]["carrying_capacity"]

    fig, (ax_pop, ax_evt) = plt.subplots(2, 1, figsize=_PAGE)
    fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.28)

    ax_pop.plot(cycles, [h["population"] for h in history], color=_POP, lw=1.8,
                marker="o", markersize=2, label="population")
    ax_pop.axhline(cap, color=_CAP, lw=1.0, ls="--", alpha=0.8, label="carrying capacity")
    ax_pop.set_ylim(bottom=0)
    ax_pop.set_xlabel("Cycle")
    ax_pop.set_ylabel("Colony size (sharks)")
    ax_pop.set_title("Population over cycles", fontsize=14, fontweight="bold", color=_BEST)
    ax_pop.legend()
    ax_pop.grid(alpha=0.3)

    ax_evt.plot(cycles, [h["births"] for h in history], color=_BIRTHS, lw=1.4, label="births")
    ax_evt.plot(cycles, [h["forage_deaths"] for h in history], color=_FORAGE, lw=1.4,
                label="foraging deaths (poison/starve)")
    ax_evt.plot(cycles, [h["oldage_deaths"] for h in history], color=_OLDAGE, lw=1.4,
                label="old-age deaths")
    ax_evt.plot(cycles, [h["culled"] for h in history], color=_CULL, lw=1.4,
                label="overcrowding culls")
    ax_evt.set_ylim(bottom=0)
    ax_evt.set_xlabel("Cycle")
    ax_evt.set_ylabel("Count per cycle")
    ax_evt.set_title("Births & deaths per cycle", fontsize=14, fontweight="bold", color=_BEST)
    ax_evt.legend()
    ax_evt.grid(alpha=0.3)
    return fig


def _fitness_page(history: list[dict]) -> plt.Figure:
    cycles = [h["cycle"] for h in history]
    fig, ax = plt.subplots(figsize=_PAGE)
    fig.subplots_adjust(top=0.9, bottom=0.1)
    ax.fill_between(
        cycles,
        [h["min_fitness"] for h in history],
        [h["max_fitness"] for h in history],
        color=_BAND, alpha=0.4, label="population range",
    )
    ax.plot(cycles, [h["best_fitness"] for h in history], color=_BEST, lw=1.8, label="best")
    ax.plot(cycles, [h["mean_fitness"] for h in history], color=_MEAN, lw=1.2, label="mean")
    ax.axhline(0, color=_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fitness (reward earned competing in the shared Ocean)")
    ax.set_title("Fitness under competition over cycles",
                 fontsize=14, fontweight="bold", color=_BEST)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def _traits_page(history: list[dict]) -> plt.Figure:
    genes = evolvable_traits()
    cycles = [h["cycle"] for h in history]
    fig, axs = plt.subplots(len(genes), 1, figsize=_PAGE, squeeze=False)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35)
    for ax, gene in zip(axs[:, 0], genes):
        spec = SHARK_TRAITS[gene]
        ax.plot(cycles, [h["best_genome_values"][gene] for h in history],
                color=_BEST, lw=1.4, label="best shark")
        ax.plot(cycles, [h["avg_trait_values"][gene] for h in history],
                color=_AVG, lw=1.2, label="colony average")
        ax.set_ylabel(f"{gene}\n({spec.unit})")
        ax.set_ylim(spec.min_value, spec.max_value)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[-1, 0].set_xlabel("Cycle")
    axs[0, 0].set_title("Evolvable traits over cycles (best shark vs. colony average)",
                        fontsize=14, fontweight="bold", color=_BEST)
    return fig


def _distribution_and_genome_page(results: dict) -> plt.Figure:
    history = results["history"]
    final = history[-1]["fitnesses"] if history else [0.0]
    genome_rows = [
        [name, f"{results['best_genome_values'][name]:.3f}", spec.unit,
         "yes" if spec.evolvable else "no (fixed)"]
        for name, spec in SHARK_TRAITS.items()
    ]

    fig = plt.figure(figsize=_PAGE)
    gs = fig.add_gridspec(2, 1, left=0.1, right=0.9, top=0.92, bottom=0.07, hspace=0.35,
                          height_ratios=[5, len(genome_rows) + 1])

    ax_hist = fig.add_subplot(gs[0])
    ax_hist.hist(final, bins=min(20, max(5, len(final) // 2)), color=_BAR, edgecolor="white")
    ax_hist.set_xlabel("Fitness")
    ax_hist.set_ylabel("Number of sharks")
    ax_hist.set_title(
        f"Final-cycle fitness distribution (cycle {history[-1]['cycle'] if history else 0})",
        fontsize=14, fontweight="bold", color=_BEST,
    )
    ax_hist.grid(alpha=0.3)

    _draw_table(
        fig.add_subplot(gs[1]),
        ["Trait", "Value", "Unit", "Evolved?"], genome_rows,
        title="Best genome found (highest single-cycle reward under competition)",
        cell_loc="left",
    )
    return fig


def _cycle_page(history: list[dict], max_rows: int = 30) -> plt.Figure:
    genes = evolvable_traits()
    headers = ["Cycle", "Pop", *[g.capitalize() for g in genes],
               "Best", "Mean", "Born", "Died", "Cull"]
    step = max(1, math.ceil(len(history) / max_rows))
    idxs = sorted({0, len(history) - 1, *range(0, len(history), step)}) if history else []
    rows = [
        [str(history[i]["cycle"]), str(history[i]["population"])]
        + [f"{history[i]['best_genome_values'][g]:.2f}" for g in genes]
        + [f"{history[i]['best_fitness']:.1f}", f"{history[i]['mean_fitness']:.1f}",
           str(history[i]["births"]),
           str(history[i]["forage_deaths"] + history[i]["oldage_deaths"]),
           str(history[i]["culled"])]
        for i in idxs
    ]

    fig = plt.figure(figsize=_PAGE)
    title = "Per-cycle summary"
    if step > 1:
        title += f"  (showing {len(rows)} of {len(history)} cycles, every {step})"
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
    _draw_table(ax, headers, rows, title=title)
    return fig


def write_colony_report(results: dict, out_path: str | Path) -> Path:
    """Write the multi-page PDF report for a colony ``results`` dict; return its path."""
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = [
        _cover_page(results),
        _population_page(results),
        _fitness_page(results["history"]),
        _traits_page(results["history"]),
        _distribution_and_genome_page(results),
        _cycle_page(results["history"]),
    ]
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = "Shark Colony Experiment Report"
        meta["Subject"] = (
            f"{results['params']['population']} founding sharks, "
            f"{results['params']['cycles_completed']} cycles, shared ocean"
        )
        meta["Creator"] = "experiments/colony_report.py"
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PDF report from a colony pickle")
    parser.add_argument("pickle", type=str, nargs="?", default=None,
                        help="colony .pkl file (default: most recent colony_* in experiments/)")
    parser.add_argument("--output", type=str, default=None,
                        help="output .pdf path (default: alongside the pickle)")
    args = parser.parse_args()

    if args.pickle:
        pkl_path = Path(args.pickle)
    else:
        candidates = sorted(EXPERIMENTS_DIR.glob("colony_*/**/*.pkl"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            parser.error(f"no pickle given and no colony .pkl files found under {EXPERIMENTS_DIR}")
        pkl_path = candidates[-1]
        print(f"No pickle given; using most recent: {pkl_path.relative_to(EXPERIMENTS_DIR)}")

    with pkl_path.open("rb") as fh:
        results = pickle.load(fh)

    out_path = Path(args.output) if args.output else pkl_path.with_name("report.pdf")
    write_colony_report(results, out_path)
    print(f"Wrote report -> {out_path}")


if __name__ == "__main__":
    main()
