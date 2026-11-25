# -*- coding: utf-8 -*-
"""
plot_cost_component_kdes.py
===========================
Kernel-density estimates of the per-municipality unit cost, split into
transport / treatment / total, for S1 (proximity baseline), S2 (observed use
with conservative closure) and S3 (cost-based fixed assignment). Reads
muni_costs_{s1,s2,s3}.csv exported by compute_three_scenarios.py.

Figure 3 of the WM&R submission. Written at twice the printed size (260 mm wide,
300 dpi) so that 15 pt text prints at 7.5 pt at 130 mm. Scenarios differ by line
style and marker as well as by colour, so the figure reads in greyscale. Costs
are non-negative, so each KDE is reflected at 0 and drawn only for x >= 0.
Markers above each panel give the production-weighted means (one row per scenario).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
OUT_JPG = os.path.join(OUT_DIR, "cost_component_kdes.jpg")
FIG_WIDTH_IN = 260 / 25.4

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "savefig.dpi": 300})

SCEN = {
    "s1": {"label": "S1 proximity\nbaseline", "color": "#000000", "ls": "-", "marker": "o",
           "mk_start": 0},
    "s2": {"label": "S2 observed use\n(conservative closure)", "color": "#c05a00",
           "ls": (0, (6, 3)), "marker": "s", "mk_start": 40},
    "s3": {"label": "S3 cost-based\nfixed assignment", "color": "#3b6fb6",
           "ls": (0, (1.5, 2)), "marker": "^", "mk_start": 20},
}
COMPONENTS = [
    ("transport", "(a) Transport cost"),
    ("treatment", "(b) Treatment cost"),
    ("total", "(c) Total unit cost"),
]


def kde_reflected(values: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Gaussian KDE with reflection at 0 (support is x >= 0)."""
    kde = gaussian_kde(values)
    return kde(xs) + kde(-xs)


def main() -> None:
    data = {k: pd.read_csv(os.path.join(OUT_DIR, f"muni_costs_{k}.csv")) for k in SCEN}
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 4.6))
    for ax, (col, title) in zip(axes, COMPONENTS):
        allv = np.concatenate([data[k][col].values for k in SCEN])
        allv = allv[np.isfinite(allv)]
        hi = allv.max() * 1.04
        xs = np.linspace(0.0, hi, 800)
        ymax = max(kde_reflected(data[k][col].dropna().values, xs).max() for k in SCEN)
        for j, (k, meta) in enumerate(SCEN.items()):
            v = data[k][col].dropna().values
            y = kde_reflected(v, xs)
            ax.plot(xs, y, color=meta["color"], ls=meta["ls"], lw=2.4, label=meta["label"],
                    marker=meta["marker"], ms=7, zorder=3,
                    markevery=[i for i in range(meta["mk_start"], len(xs), 80)
                               if y[i] > 0.08 * ymax])
            wmean = np.average(data[k][col].values, weights=data[k]["prod"].values)
            ax.plot([wmean], [(1.14 + 0.09 * j) * ymax], marker=meta["marker"], color=meta["color"],
                    ms=10, ls="None", clip_on=False, zorder=4)
        ax.axhline(1.08 * ymax, color="0.75", lw=0.8, zorder=0)
        ax.text(hi, 1.36 * ymax, "means", ha="right", va="top", fontsize=13, color="0.35")
        rug_h = 0.06 * ymax
        for j, (k, meta) in enumerate(SCEN.items()):
            v = data[k][col].dropna().values
            ax.plot(v, np.full_like(v, -(j + 1) * rug_h), "|", color=meta["color"],
                    ms=7, alpha=0.6, mew=1.0, zorder=1)
        ax.set_title(title, fontsize=15, loc="left", pad=8)
        ax.set_xlabel("€ t$^{-1}$")
        ax.set_xlim(0, hi)
        ax.set_yticks([])
        ax.set_ylim(-(len(SCEN) + 0.6) * rug_h, 1.36 * ymax)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=14)
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=14,
               handlelength=3.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT_JPG, dpi=300, pil_kwargs={"quality": 95})
    plt.close(fig)
    print("Saved:", OUT_JPG)


if __name__ == "__main__":
    main()
