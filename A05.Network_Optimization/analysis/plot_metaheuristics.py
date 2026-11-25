# -*- coding: utf-8 -*-
"""
plot_metaheuristics.py
======================
Plot-only companion of optimize_metaheuristics.py (Figure 4 of A05).

Reads the saved search results, so the figure can be redrawn without
rerunning Simulated Annealing and the Genetic Algorithms:
    outputs/comparison_metaheuristics.csv   (greedy vs best SA/GA per N)
    outputs/unconstrained_ga.csv            (unconstrained GA best network)

Writes a two-panel figure drawn at its printed size (12 cm wide), 600 dpi:
    (a) average system cost per tonne against the number of active plants;
    (b) gap of the greedy curve above the best metaheuristic, in percent.
Output: results/figures/comparison_metaheuristics.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_PATHS = [
    os.path.join(ROOT, "results", "figures", "comparison_metaheuristics.png"),
]
FIG_WIDTH_IN = 12.0 / 2.54


def plot_comparison(df_comp, n_unc, cost_unc, paths=FIG_PATHS):
    """Draw the cost frontier and the greedy gap; save to every path."""
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, (ax, axg) = plt.subplots(
        2, 1, sharex=True, figsize=(FIG_WIDTH_IN, 3.7),
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.22})
    n = df_comp["n_plants"]

    ax.plot(n, df_comp["greedy_cost_eur_t"], color="black", lw=1.2, marker="o",
            ms=2.6, label="Greedy pruning")
    ax.plot(n, df_comp["best_meta_cost_eur_t"], color="#c0392b", lw=1.0,
            ls=(0, (4, 2)), marker="s", ms=3.2, mfc="none", mew=0.7,
            label="Best of SA and constrained GA")
    ax.plot(n_unc, cost_unc, ls="None", marker="*", ms=9, color="#1f7a3a",
            mec="black", mew=0.4,
            label=f"Unconstrained GA: N = {n_unc}, {cost_unc:.2f} €/t")
    ax.set_ylabel("Average system cost (€/t)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.6)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=7.5, pad=2)

    gap = df_comp["gap_percentage"].clip(lower=0.0)
    axg.bar(n, gap, width=0.7, color="0.35")
    axg.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    axg.set_ylim(0, 8)
    axg.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1]))
    axg.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axg.yaxis.set_minor_locator(FixedLocator([]))
    axg.set_ylabel("Greedy gap (%)")
    axg.set_xlabel("Number of active plants, N")
    axg.grid(axis="y", alpha=0.3, lw=0.5)
    axg.set_title("(b)", loc="left", fontweight="bold", fontsize=7.5, pad=2)
    axg.invert_xaxis()  # shared x: consolidation reads left to right
    axg.set_xticks([32, 28, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3])

    for a in (ax, axg):
        a.tick_params(length=2.5, pad=2)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.11)
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=600)
        print(f"Saved comparison plot to {path}")
    plt.close(fig)


def main():
    df_comp = pd.read_csv(os.path.join(OUT_DIR, "comparison_metaheuristics.csv"))
    df_unc = pd.read_csv(os.path.join(OUT_DIR, "unconstrained_ga.csv"))
    row = df_unc.iloc[0]
    plot_comparison(df_comp, int(row["n_plants"]), float(row["cost_eur_t"]))


if __name__ == "__main__":
    main()
