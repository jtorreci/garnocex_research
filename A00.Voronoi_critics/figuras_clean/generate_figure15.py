#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure: Euclidean vs Network Distance Scatter
=======================================================

Gray cloud  : all municipality-plant pairs (~17 k points).
Blue dots   : Voronoi assignments that match the network-optimal (correct).
Red dots    : Voronoi assignments that differ from network-optimal (misallocated).

Each municipality contributes exactly one coloured dot (its Voronoi-assigned
plant).  ~322 blue + ~61 red = 383 total.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- paths ---------------------------------------------------------------
BASE    = Path(__file__).resolve().parent.parent          # support_material/
TABLES  = BASE / "tables"
FIGDIR  = BASE.parent / "figures"
OUTDIR  = Path(__file__).resolve().parent                 # figuras_clean/

EUCLIDEAN_FILE = TABLES / "D_euclidea_plantas_clean.csv"
NETWORK_FILE   = BASE / "codigo" / "tablas" / "D_real_plantas_clean_corrected.csv"

# ---- style ---------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data():
    df_euc = pd.read_csv(EUCLIDEAN_FILE)
    df_net = pd.read_csv(NETWORK_FILE)

    df_euc.columns = ["municipality", "plant_name", "d_euclidean"]
    df_net.columns = ["municipality", "plant_name",
                      "entry_cost", "network_cost", "exit_cost", "d_network"]

    # Add plant index to handle duplicate plant names
    df_euc["plant_idx"] = df_euc.groupby("municipality").cumcount()
    df_net["plant_idx"] = df_net.groupby("municipality").cumcount()

    df = df_euc.merge(df_net[["municipality", "plant_idx", "d_network"]],
                      on=["municipality", "plant_idx"])
    return df


def compute_assignments(df):
    """For each municipality: Voronoi assignment, network-optimal, correct/incorrect."""
    records = []
    for muni, sub in df.groupby("municipality"):
        # Voronoi: nearest Euclidean
        vor_row = sub.loc[sub["d_euclidean"].idxmin()]
        # Network-optimal: nearest by network
        net_row = sub.loc[sub["d_network"].idxmin()]

        correct = (vor_row["plant_idx"] == net_row["plant_idx"])

        records.append({
            "municipality": muni,
            "d_euclidean": vor_row["d_euclidean"],
            "d_network":   vor_row["d_network"],
            "correct":     correct,
        })
    return pd.DataFrame(records)


def main():
    print("Loading data...")
    df = load_data()
    n_muni = df["municipality"].nunique()
    print(f"  {n_muni} municipalities, {len(df)} total pairs")

    print("Computing assignments...")
    assign = compute_assignments(df)
    n_ok  = assign["correct"].sum()
    n_bad = len(assign) - n_ok
    print(f"  Correct: {n_ok}  |  Misallocated: {n_bad}")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    # 1) Gray cloud: all pairs
    ax.scatter(df["d_euclidean"], df["d_network"],
               s=4, color="#cccccc", alpha=0.35, edgecolors="none",
               zorder=1, label=f"All pairs ($n = {len(df):,}$)")

    # 2) Blue: correct assignments
    ok = assign[assign["correct"]]
    ax.scatter(ok["d_euclidean"], ok["d_network"],
               s=22, color="#2980b9", alpha=0.75, edgecolors="black",
               linewidths=0.3, zorder=3,
               label=f"Correct assignment ({n_ok})")

    # 3) Red: misallocated
    bad = assign[~assign["correct"]]
    ax.scatter(bad["d_euclidean"], bad["d_network"],
               s=28, color="#e74c3c", alpha=0.85, edgecolors="black",
               linewidths=0.3, zorder=4,
               label=f"Misallocated ({n_bad})")

    # Reference line beta = 1
    lims = [0, max(df["d_euclidean"].max(), df["d_network"].max()) * 1.03]
    ax.plot(lims, lims, color="#e74c3c", ls="--", lw=1.0, alpha=0.5, zorder=2)
    ax.text(lims[1] * 0.72, lims[1] * 0.67, r"$\beta = 1$",
            fontsize=8, color="#e74c3c", alpha=0.7, rotation=38)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Euclidean distance (m)")
    ax.set_ylabel("Network distance (m)")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls=":")

    # ---- save ------------------------------------------------------------
    for fmt in ("pdf", "png"):
        out = OUTDIR / f"euclidean_vs_real_scatterplot.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")
    fig.savefig(FIGDIR / "euclidean_vs_real_scatterplot.pdf",
                format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {FIGDIR / 'euclidean_vs_real_scatterplot.pdf'}")

    plt.close()


if __name__ == "__main__":
    main()
