#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figure: Algorithmic complexity comparison
====================================================

Single-column figure showing theoretical complexity curves (log-log)
normalized to n=10, with Extremadura (n=383) marked and complexity
multipliers annotated.

Three curves:
  - O(n)          — Voronoi / Selective verification (same class)
  - O(n log n)    — k-nearest
  - O(n^2 log n)  — Network Voronoi
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- style ----------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = Path(__file__).resolve().parent          # figuras_clean/
FIGDIR = OUTDIR.parent.parent / "figures"         # paper figures/

N_EXT = 383          # Extremadura municipalities
N_BASE = 10          # normalization reference


def main():
    n = np.logspace(1, 4, 500)  # 10 to 10,000

    # Complexity functions
    voronoi   = n                          # O(n)
    k_nearest = n * np.log2(n)             # O(n log n)
    network   = n**2 * np.log2(n)          # O(n^2 log n)

    # Normalize: each curve = 1.0 at n=10
    def norm(f):
        f0 = np.interp(N_BASE, n, f)
        return f / f0

    vor_n = norm(voronoi)
    knn_n = norm(k_nearest)
    net_n = norm(network)

    # Values at n = 383
    v383 = np.interp(N_EXT, n, vor_n)
    k383 = np.interp(N_EXT, n, knn_n)
    w383 = np.interp(N_EXT, n, net_n)

    mult_net_vor = w383 / v383   # ~99x
    mult_net_knn = w383 / k383   # ~38x

    # ---- figure (single column) -------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 4.0))

    # Curves
    ax.loglog(n, vor_n, color="#2c3e50", ls="-",  lw=1.8,
              label=r"$O(n)$ — Voronoi / Selective verif.")
    ax.loglog(n, knn_n, color="#3498db", ls="--", lw=1.5,
              label=r"$O(n \log n)$ — $k$-nearest")
    ax.loglog(n, net_n, color="#95a5a6", ls=":",  lw=2.2,
              label=r"$O(n^2 \log n)$ — Network Voronoi")

    # Extremadura vertical
    ax.axvline(N_EXT, color="#27ae60", ls="--", lw=0.9, alpha=0.5, zorder=1)

    # Dots at n = 383
    for val, col, ms in [(v383, "#2c3e50", 5),
                          (k383, "#3498db", 4),
                          (w383, "#95a5a6", 5)]:
        ax.plot(N_EXT, val, "o", color=col, ms=ms, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)

    # --- annotations at n=383 ---

    # "Extremadura" label
    ax.annotate(f"Extremadura\n($n = {N_EXT}$)",
                xy=(N_EXT, v383), xytext=(55, 8),
                textcoords="offset points",
                fontsize=7, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.7))

    # Vertical connector line (Voronoi → Network gap)
    x_bar = N_EXT * 1.12
    ax.plot([x_bar, x_bar], [v383, w383],
            color="#e74c3c", lw=1.2, ls="-", alpha=0.6, zorder=2)
    # small caps
    cap = 0.08  # in log-space fraction
    for y in [v383, w383]:
        ax.plot([x_bar / (1 + cap), x_bar * (1 + cap)], [y, y],
                color="#e74c3c", lw=1.0, alpha=0.6, zorder=2)

    # Multiplier label at geometric midpoint
    mid_y = np.sqrt(v383 * w383)
    ax.annotate(f"$\\times\\,{mult_net_vor:.0f}$",
                xy=(x_bar, mid_y),
                xytext=(18, 0), textcoords="offset points",
                fontsize=9, color="#e74c3c", fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#e74c3c", alpha=0.9, lw=0.6))

    ax.set_xlabel("Number of spatial units ($n$)")
    ax.set_ylabel("Relative computational cost\n(normalized to $n = 10$)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.15, ls=":")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(0.5, 1e7)

    # ---- save -------------------------------------------------------------
    for fmt in ("pdf", "png"):
        out = OUTDIR / f"algorithmic_complexity_analysis.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    out = FIGDIR / "algorithmic_complexity_analysis.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")

    plt.close()

    # ---- summary ----------------------------------------------------------
    print(f"\nMultipliers at n = {N_EXT}:")
    print(f"  Network / Voronoi (or Selective):  {mult_net_vor:,.0f}x")
    print(f"  Network / k-nearest:               {mult_net_knn:,.0f}x")


if __name__ == "__main__":
    main()
