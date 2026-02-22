#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose the 59 vs 61 misallocation discrepancy.

Compare different distance table combinations to find which
one produces 59 and which produces 61.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # support_material/
TABLES = BASE / "tables"
CODIGO = BASE / "codigo" / "tablas"

# All candidate distance files
EUCLIDEAN_FILES = {
    "tables/D_euclidea_plantas_clean": TABLES / "D_euclidea_plantas_clean.csv",
    "codigo/D_euclidea_plantas_clean": CODIGO / "D_euclidea_plantas_clean.csv",
}

NETWORK_FILES = {
    "tables/D_real_plantas_clean":         TABLES / "D_real_plantas_clean.csv",
    "codigo/D_real_plantas_clean":         CODIGO / "D_real_plantas_clean.csv",
    "codigo/D_real_plantas_clean_corrected": CODIGO / "D_real_plantas_clean_corrected.csv",
}


def load_euc(path):
    df = pd.read_csv(path)
    df.columns = ["municipality", "plant_name", "d_euclidean"]
    df["plant_idx"] = df.groupby("municipality").cumcount()
    return df


def load_net(path):
    df = pd.read_csv(path)
    if len(df.columns) == 6:
        df.columns = ["municipality", "plant_name",
                       "entry_cost", "network_cost", "exit_cost", "d_network"]
    elif len(df.columns) == 3:
        df.columns = ["municipality", "plant_name", "d_network"]
    else:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    df["plant_idx"] = df.groupby("municipality").cumcount()
    return df


def compute_misalloc(df_euc, df_net):
    """Return set of misallocated municipalities."""
    df = df_euc.merge(df_net[["municipality", "plant_idx", "d_network"]],
                      on=["municipality", "plant_idx"])

    misallocated = set()
    for muni, sub in df.groupby("municipality"):
        vor_idx = sub.loc[sub["d_euclidean"].idxmin(), "plant_idx"]
        net_idx = sub.loc[sub["d_network"].idxmin(), "plant_idx"]
        if vor_idx != net_idx:
            misallocated.add(muni)
    return misallocated


def check_duplicates(df_euc, df_net):
    """Check for potential issues."""
    euc_counts = df_euc.groupby("municipality").size()
    net_counts = df_net.groupby("municipality").size()
    euc_munis = set(df_euc["municipality"].unique())
    net_munis = set(df_net["municipality"].unique())

    print(f"  Euclidean: {len(euc_munis)} municipalities, "
          f"{euc_counts.iloc[0]}-{euc_counts.iloc[-1]} plants/muni")
    print(f"  Network:   {len(net_munis)} municipalities, "
          f"{net_counts.iloc[0]}-{net_counts.iloc[-1]} plants/muni")

    # Check for mismatched plant counts
    merged_counts = pd.DataFrame({"euc": euc_counts, "net": net_counts})
    mismatch = merged_counts[merged_counts["euc"] != merged_counts["net"]]
    if len(mismatch) > 0:
        print(f"  WARNING: {len(mismatch)} municipalities with different plant counts!")
        print(mismatch.head())

    # Check for duplicate plant names within a municipality
    for name, grp in df_euc.groupby("municipality"):
        dupes = grp["plant_name"].duplicated().sum()
        if dupes > 0:
            dupe_names = grp[grp["plant_name"].duplicated(keep=False)]["plant_name"].unique()
            print(f"  Duplicate plants in {name}: {list(dupe_names)}")
            break  # just show first one


def main():
    print("=" * 70)
    print("MISALLOCATION COUNT DIAGNOSIS")
    print("=" * 70)

    # Try all combinations
    results = {}
    for euc_name, euc_path in EUCLIDEAN_FILES.items():
        for net_name, net_path in NETWORK_FILES.items():
            if not euc_path.exists() or not net_path.exists():
                continue
            print(f"\n--- {euc_name} + {net_name} ---")
            df_euc = load_euc(euc_path)
            df_net = load_net(net_path)
            check_duplicates(df_euc, df_net)
            misalloc = compute_misalloc(df_euc, df_net)
            key = f"{euc_name} | {net_name}"
            results[key] = misalloc
            print(f"  => Misallocated: {len(misalloc)}")

    # Compare the sets that give 59 vs 61
    print("\n" + "=" * 70)
    print("COMPARISON OF RESULTS")
    print("=" * 70)

    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s1, s2 = results[keys[i]], results[keys[j]]
            if s1 != s2:
                only_1 = s1 - s2
                only_2 = s2 - s1
                print(f"\n  {keys[i]} ({len(s1)})")
                print(f"  vs")
                print(f"  {keys[j]} ({len(s2)})")
                if only_1:
                    print(f"    Only in first:  {sorted(only_1)}")
                if only_2:
                    print(f"    Only in second: {sorted(only_2)}")


if __name__ == "__main__":
    main()
