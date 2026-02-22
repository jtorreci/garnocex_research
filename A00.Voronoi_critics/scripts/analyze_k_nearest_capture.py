#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Nearest Capture Rate Analysis
================================

For each municipality, checks whether the network-optimal plant
(the one closest by road) is among the k nearest Euclidean plants.

This validates the selective verification procedure: if k=3 already
captures ~100% of network-optimal assignments, then routing only to
the 3 nearest Euclidean facilities is sufficient.

Output: table showing capture rate for k=1,2,...,10
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent  # support_material/
TABLES = BASE / "tables"

EUCLIDEAN_FILE = TABLES / "D_euclidea_plantas_clean.csv"
NETWORK_FILE  = BASE / "codigo" / "tablas" / "D_real_plantas_clean_corrected.csv"

# ---------- load -----------------------------------------------------------
def load_and_index(euclidean_path, network_path):
    """
    Load both distance tables and add a consistent plant index
    within each municipality (to handle duplicate plant names).
    """
    df_euc = pd.read_csv(euclidean_path)
    df_net = pd.read_csv(network_path)

    # Rename for consistency
    df_euc.columns = ["municipality", "plant_name", "d_euclidean"]
    df_net.columns = ["municipality", "plant_name",
                      "entry_cost", "network_cost", "exit_cost", "d_network"]

    # Add sequential plant index within each municipality
    df_euc["plant_idx"] = df_euc.groupby("municipality").cumcount()
    df_net["plant_idx"] = df_net.groupby("municipality").cumcount()

    # Merge on (municipality, plant_idx)
    df = df_euc.merge(df_net[["municipality", "plant_idx", "d_network"]],
                      on=["municipality", "plant_idx"])

    return df


def compute_capture_rates(df, k_max=10):
    """
    For each municipality:
      - rank plants by Euclidean distance (1 = nearest)
      - identify the network-optimal plant (min d_network)
      - for k=1..k_max, check if network-optimal is among the k nearest Euclidean

    Returns DataFrame with columns: k, captured, total, capture_rate_pct
    """
    results = []
    municipalities = df["municipality"].unique()
    n_muni = len(municipalities)

    # Pre-compute per-municipality rankings
    misallocated_details = []  # for k=1 (Voronoi baseline)

    capture_counts = {k: 0 for k in range(1, k_max + 1)}

    for muni in municipalities:
        sub = df[df["municipality"] == muni].copy()

        # Rank by Euclidean distance
        sub = sub.sort_values("d_euclidean")
        sub["euc_rank"] = range(1, len(sub) + 1)

        # Network-optimal plant
        net_best_idx = sub["d_network"].idxmin()
        net_best_euc_rank = sub.loc[net_best_idx, "euc_rank"]

        # Voronoi assignment (k=1) details
        if net_best_euc_rank > 1:
            misallocated_details.append({
                "municipality": muni,
                "network_optimal_rank": net_best_euc_rank,
            })

        for k in range(1, k_max + 1):
            if net_best_euc_rank <= k:
                capture_counts[k] += 1

    rows = []
    for k in range(1, k_max + 1):
        captured = capture_counts[k]
        missed = n_muni - captured
        rows.append({
            "k": k,
            "captured": captured,
            "missed": missed,
            "total": n_muni,
            "capture_rate_pct": 100.0 * captured / n_muni,
            "misallocation_pct": 100.0 * missed / n_muni,
        })

    results_df = pd.DataFrame(rows)

    # Details about where the network-optimal sits in Euclidean ranking
    details_df = pd.DataFrame(misallocated_details)

    return results_df, details_df


def main():
    print("=" * 70)
    print("K-NEAREST CAPTURE RATE ANALYSIS")
    print("=" * 70)

    # Load
    print("\nLoading distance matrices...")
    df = load_and_index(EUCLIDEAN_FILE, NETWORK_FILE)
    municipalities = df["municipality"].unique()
    plants_per_muni = df.groupby("municipality").size()
    print(f"  Municipalities: {len(municipalities)}")
    print(f"  Plants per municipality: {plants_per_muni.iloc[0]}")
    print(f"  Total records: {len(df)}")

    # Compute
    print("\nComputing capture rates...")
    results_df, details_df = compute_capture_rates(df, k_max=10)

    # Display table
    print("\n" + "-" * 70)
    print("CAPTURE RATE TABLE")
    print("-" * 70)
    print(f"{'k':>3}  {'Captured':>9}  {'Missed':>7}  {'Capture %':>10}  {'Misalloc %':>11}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        k = int(row["k"])
        marker = " <-- Voronoi baseline" if k == 1 else ""
        print(f"{k:>3}  {int(row['captured']):>9}  {int(row['missed']):>7}"
              f"  {row['capture_rate_pct']:>9.1f}%  {row['misallocation_pct']:>10.1f}%{marker}")

    # Details about misallocated municipalities
    if len(details_df) > 0:
        print(f"\n\nMISALLOCATED MUNICIPALITIES (Voronoi vs network): {len(details_df)}")
        print("-" * 70)
        rank_dist = details_df["network_optimal_rank"].value_counts().sort_index()
        print("\nNetwork-optimal plant's Euclidean rank distribution:")
        for rank, count in rank_dist.items():
            print(f"  Rank {rank}: {count} municipalities")

        print(f"\n  Max Euclidean rank of network-optimal: "
              f"{details_df['network_optimal_rank'].max()}")
        print(f"  Mean Euclidean rank of network-optimal: "
              f"{details_df['network_optimal_rank'].mean():.1f}")

    # Save
    output_csv = BASE / "tables" / "k_nearest_capture_rates.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
