"""Render a single self-contained PDF report from an experiment's results.

Takes the ``results`` dict produced by :func:`experiments.run_experiment.run_experiment`
(or an experiment pickle on disk) and writes one multi-page ``.pdf`` with the run
parameters, summary stats, the fitness/trait graphs, and a per-generation table --
all in one portable file (built with matplotlib, no extra dependencies).

Used by ``run_experiment.py`` automatically (the report is refreshed every run), or
standalone to (re)generate a report from a saved pickle::

    python -m experiments.report                      # most recent pickle in experiments/
    python -m experiments.report experiments/run.pkl  # a specific pickle
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

# Allow both ``-m experiments.report`` and bare ``python experiments/report.py``.
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

# Ocean-ish palette, kept consistent with src/algorithms/genetic_algo._plot_history.
_BEST = "#08519c"
_MEAN = "#e6550d"
_BAND = "#9ecae1"
_BAR = "#3182bd"
_HEADER_BG = "#eef4fa"
_LINE = "#d6e0ea"
_MUTED = "#5a6b7b"
_PAGE = (8.5, 11)  # US Letter portrait, inches


# ---------------------------------------------------------------------------
# Tables (rendered onto a matplotlib axis so they live inside the PDF)
# ---------------------------------------------------------------------------
def _draw_table(ax, headers, rows, title=None, cell_loc="center") -> None:
    """Fill ``ax`` with a styled table; optionally title it above."""
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
    """Gridspec height ratios so each table's height tracks its row count."""
    return [c + 1 for c in counts]  # +1 for the header row


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def _cover_page(results: dict) -> plt.Figure:
    p = results["params"]
    history = results["history"]
    brain = results["brain"]
    best_overall = max((h["best_fitness"] for h in history), default=float("nan"))
    final_mean = history[-1]["mean_fitness"] if history else float("nan")

    summary = [
        ["Generations", str(p["generations"])],
        ["Population (sharks/gen)", str(p["population"])],
        ["Best fitness", f"{best_overall:.3f}"],
        ["Final mean fitness", f"{final_mean:.3f}"],
        ["Analytical fitness (best genome)", f"{results['analytical_fitness']:.4f}"],
        ["Runtime", f"{results['duration_seconds']:.1f} s"],
    ]
    params = [
        ["Max steps per life", str(p["max_steps"])],
        ["Lives averaged per genome", str(p["lives"])],
        ["Mutation rate", f"{p['mutation_rate']}"],
        ["Random seed", str(p["seed"])],
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
    fig.text(0.08, 0.95, "Shark Evolution Experiment", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.925, f"GA + RL run — {results['timestamp']}",
             fontsize=11, color=_MUTED)

    gs = fig.add_gridspec(
        3, 1, left=0.08, right=0.92, top=0.88, bottom=0.06, hspace=0.4,
        height_ratios=_ratios(len(summary), len(params), len(brain_rows)),
    )
    _draw_table(fig.add_subplot(gs[0]), ["Summary", ""], summary,
                title="Summary", cell_loc="left")
    _draw_table(fig.add_subplot(gs[1]), ["Run parameter", "Value"], params,
                title="Run parameters", cell_loc="left")
    _draw_table(fig.add_subplot(gs[2]), ["Shared brain (Q-learning)", "Value"], brain_rows,
                title="Shared brain (Q-learning)", cell_loc="left")
    return fig


def _fitness_page(history: list[dict]) -> plt.Figure:
    gens = [h["generation"] for h in history]
    fig, ax = plt.subplots(figsize=_PAGE)
    fig.subplots_adjust(top=0.9, bottom=0.1)
    ax.fill_between(
        gens,
        [h["min_fitness"] for h in history],
        [h["max_fitness"] for h in history],
        color=_BAND, alpha=0.4, label="population range",
    )
    ax.plot(gens, [h["best_fitness"] for h in history], color=_BEST, lw=1.8, label="best")
    ax.plot(gens, [h["mean_fitness"] for h in history], color=_MEAN, lw=1.2, label="mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (reward earned in Ocean)")
    ax.set_title("Fitness over generations", fontsize=14, fontweight="bold", color=_BEST)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def _traits_page(history: list[dict]) -> plt.Figure:
    genes = evolvable_traits()
    gens = [h["generation"] for h in history]
    fig, axs = plt.subplots(len(genes), 1, figsize=_PAGE, squeeze=False)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)
    for ax, gene in zip(axs[:, 0], genes):
        spec = SHARK_TRAITS[gene]
        ax.plot(gens, [h["best_genome_values"][gene] for h in history], color=_BEST)
        ax.set_ylabel(f"{gene}\n({spec.unit})")
        ax.set_ylim(spec.min_value, spec.max_value)
        ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("Generation")
    axs[0, 0].set_title("Best shark's evolvable traits over generations",
                        fontsize=14, fontweight="bold", color=_BEST)
    return fig


def _distribution_and_genome_page(results: dict) -> plt.Figure:
    history = results["history"]
    final = history[-1]["fitnesses"]
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
        f"Final-generation fitness distribution (gen {history[-1]['generation']})",
        fontsize=14, fontweight="bold", color=_BEST,
    )
    ax_hist.grid(alpha=0.3)

    _draw_table(
        fig.add_subplot(gs[1]),
        ["Trait", "Value", "Unit", "Evolved?"], genome_rows,
        title="Best genome found (judged by reward earned in the Ocean)",
        cell_loc="left",
    )
    return fig


def _generation_page(history: list[dict], max_rows: int = 30) -> plt.Figure:
    """Per-generation summary, downsampled to keep the table readable."""
    genes = evolvable_traits()
    headers = ["Gen", *[g.capitalize() for g in genes], "Best", "Mean", "Min", "Max"]
    step = max(1, math.ceil(len(history) / max_rows))
    idxs = sorted({0, len(history) - 1, *range(0, len(history), step)})
    rows = [
        [str(history[i]["generation"])]
        + [f"{history[i]['best_genome_values'][g]:.3f}" for g in genes]
        + [f"{history[i]['best_fitness']:.3f}", f"{history[i]['mean_fitness']:.3f}",
           f"{history[i]['min_fitness']:.3f}", f"{history[i]['max_fitness']:.3f}"]
        for i in idxs
    ]

    fig = plt.figure(figsize=_PAGE)
    title = "Per-generation summary"
    if step > 1:
        title += f"  (showing {len(rows)} of {len(history)} generations, every {step})"
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.88])
    _draw_table(ax, headers, rows, title=title)
    return fig


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def write_report(results: dict, out_path: str | Path) -> Path:
    """Write the multi-page PDF report for ``results`` and return its path.

    Any existing report at ``out_path`` is overwritten, so re-running an
    experiment refreshes the same report file.
    """
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = [
        _cover_page(results),
        _fitness_page(results["history"]),
        _traits_page(results["history"]),
        _distribution_and_genome_page(results),
        _generation_page(results["history"]),
    ]
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = "Shark Evolution Experiment Report"
        meta["Subject"] = (
            f"{results['params']['population']} sharks, "
            f"{results['params']['generations']} generations"
        )
        meta["Creator"] = "experiments/report.py"
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PDF report from an experiment pickle")
    parser.add_argument("pickle", type=str, nargs="?", default=None,
                        help="experiment .pkl file (default: most recent in experiments/)")
    parser.add_argument("--output", type=str, default=None,
                        help="output .pdf path (default: alongside the pickle)")
    args = parser.parse_args()

    if args.pickle:
        pkl_path = Path(args.pickle)
    else:
        # Search experiment subfolders (and the top level) for the newest pickle.
        candidates = sorted(EXPERIMENTS_DIR.glob("**/*.pkl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            parser.error(f"no pickle given and no .pkl files found under {EXPERIMENTS_DIR}")
        pkl_path = candidates[-1]
        print(f"No pickle given; using most recent: {pkl_path.relative_to(EXPERIMENTS_DIR)}")

    with pkl_path.open("rb") as fh:
        results = pickle.load(fh)

    out_path = Path(args.output) if args.output else pkl_path.with_name("report.pdf")
    write_report(results, out_path)
    print(f"Wrote report -> {out_path}")


if __name__ == "__main__":
    main()
