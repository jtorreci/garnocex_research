# -*- coding: utf-8 -*-
"""
analisis_podado.py
==================
Plant pruning analysis for A05: progressively remove the least efficient
plants and reassign municipalities, computing the Pareto frontier of
(number of plants) vs (total system cost).

Uses the A01 cost model and real-distance assignment data.

Output:
  - pareto_podado.csv: number of plants, total cost, mean cost, max cost, Gini
  - pareto_podado.png: Pareto frontier plot
"""

import argparse
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

# ============================================================
# COST MODEL (same as A01)
# ============================================================
C0 = 40_000
T0 = 5_000
V_TREAT = 0.35
RHO_TRANS = 0.35
DOTACION = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="A05 greedy pruning analysis")
    parser.add_argument("--rho", type=float, default=RHO_TRANS, help="Transport cost parameter")
    parser.add_argument("--dotacion", type=float, default=DOTACION, help="Per-capita CDW generation")
    parser.add_argument("--scenario", type=str, default="", help="Scenario suffix for outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser.parse_args()


ARGS = parse_args()
RHO_TRANS = ARGS.rho
DOTACION = ARGS.dotacion
SCENARIO_SUFFIX = f"_{ARGS.scenario}" if ARGS.scenario else ""


def treatment_cost_per_tonne(T):
    if T <= 0:
        return 0.0
    if T < T0:
        cfix = C0
    else:
        cfix = C0 * (math.log2(T / T0) + 1)
    return cfix / T + V_TREAT


def gini_coefficient(values):
    """Compute Gini coefficient of a list of values."""
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


# ============================================================
# LOAD DATA
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
A01_DIR = os.path.join(SCRIPT_DIR, "..", "..", "A01.Voronoi", "scripts")

# Full distance matrix: plant_id -> municipality -> distance (road network)
# The real-distance matrix covers all 46x383 pairs
asig_real = pd.read_csv(os.path.join(A01_DIR, "asignacion_municipios_real.csv"))
df_dist = pd.read_csv(os.path.join(A01_DIR, "tablas_distancias", "distancias_reales_plantas_municipios.csv"))

print(f"Distance matrix (road): {len(df_dist)} rows")
# Standardize columns: origin_id=plant, destination_id=municipality, total_cost=distance_m
df_dist = df_dist.rename(columns={"origin_id": "plant_id", "destination_id": "municipio", "total_cost": "distance"})

# Population data
df_mun_data = pd.read_csv(os.path.join(A01_DIR, "datos_voronoi_municipios.csv"),
                           encoding="utf-8", on_bad_lines="warn")

# Build population lookup (reuse A01 normalization)
def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()

pop_lookup = {}
for _, row in df_mun_data.iterrows():
    key = norm(row.get("NOMBRE", ""))
    if key and not pd.isna(row.get("HBT")):
        pop_lookup[key] = int(row["HBT"])

# Build municipality list with populations
municipalities = []
for _, row in asig_real.iterrows():
    mun = str(row["municipio"]).strip()
    key = norm(mun)
    pop = pop_lookup.get(key, 0)
    if pop > 0:
        municipalities.append({"municipio": mun, "key": key, "poblacion": pop, "produccion": pop * DOTACION})

df_mun = pd.DataFrame(municipalities)
print(f"Municipalities with population: {len(df_mun)}")

# Build distance lookup: (plant_id, municipality_key) -> distance_km
dist_lookup = {}
for _, row in df_dist.iterrows():
    pid = row["plant_id"]
    mun_key = norm(str(row["municipio"]))
    dist_km = float(row["distance"]) / 1000.0
    if mun_key and dist_km > 0:
        dist_lookup[(pid, mun_key)] = dist_km

# Get all plant IDs
all_plants_raw = sorted(set(pid for pid, _ in dist_lookup.keys()))
print(f"Plants in distance matrix: {len(all_plants_raw)}")

# Plant consolidation (< 15 km) — consistent with A01
CONSOLIDATION_15KM = {
    2: 1, 12: 1,           # Ribera del Fresno
    17: 3,                  # Don Benito + Villanueva
    15: 6,                  # Quintana + Castuera
    40: 19,                 # Trujillo
    32: 20,                 # Moraleja + Coria
    23: 22, 26: 22, 28: 22, # Cabezuela cluster (ARAPLASA)
    34: 27, 35: 27, 37: 27, # Navalmoral cluster
    38: 36, 43: 36,         # Escurial + Miajadas
}

# Remap distances: for consolidated groups, use minimum distance
dist_lookup_consolidated = {}
for (pid, mkey), dist in dist_lookup.items():
    cpid = CONSOLIDATION_15KM.get(pid, pid)
    existing = dist_lookup_consolidated.get((cpid, mkey))
    if existing is None or dist < existing:
        dist_lookup_consolidated[(cpid, mkey)] = dist

all_plants = sorted(set(cpid for cpid, _ in dist_lookup_consolidated.keys()))
dist_lookup = dist_lookup_consolidated
print(f"After consolidation (< 15 km): {len(all_plants)} plant groups")


# ============================================================
# SIMULATION: PROGRESSIVE PRUNING
# ============================================================
def evaluate_network(active_plants, df_mun, dist_lookup, return_df=False):
    """
    Given a set of active plants, assign each municipality to the nearest
    active plant and compute system costs. When return_df=True, also return
    the per-municipality assignment DataFrame for downstream visualisation.
    """
    records = []
    for _, mun in df_mun.iterrows():
        key = mun["key"]
        prod = mun["produccion"]

        # Find nearest active plant
        best_dist = float("inf")
        best_plant = None
        for pid in active_plants:
            d = dist_lookup.get((pid, key))
            if d is not None and d < best_dist:
                best_dist = d
                best_plant = pid

        if best_plant is None:
            continue

        records.append({
            "municipio": mun["municipio"],
            "plant_id": best_plant,
            "distance_km": best_dist,
            "produccion": prod,
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        return None

    # Aggregate by plant
    plant_agg = df.groupby("plant_id").agg(
        produccion_total=("produccion", "sum"),
        tkm_total=("produccion", lambda x: (x * df.loc[x.index, "distance_km"]).sum()),
        n_municipios=("municipio", "count"),
    ).reset_index()

    # Compute costs per plant
    plant_agg["C_trans"] = (plant_agg["tkm_total"] * RHO_TRANS) / plant_agg["produccion_total"]
    plant_agg["C_trat"] = plant_agg["produccion_total"].apply(treatment_cost_per_tonne)
    plant_agg["C_tot"] = plant_agg["C_trans"] + plant_agg["C_trat"]

    # Municipality-level costs
    plant_costs = plant_agg.set_index("plant_id")["C_trat"].to_dict()
    df["C_trans"] = df["distance_km"] * RHO_TRANS
    df["C_trat"] = df["plant_id"].map(plant_costs)
    df["C_tot"] = df["C_trans"] + df["C_trat"]

    # System metrics
    total_cost = (df["C_tot"] * df["produccion"]).sum()
    mean_cost = df["C_tot"].mean()
    median_cost = df["C_tot"].median()
    max_cost = df["C_tot"].max()
    gini = gini_coefficient(df["C_tot"].values)
    prod_total = df["produccion"].sum()

    metrics = {
        "n_plants": len(active_plants),
        "total_cost_eur": total_cost,
        "mean_cost": mean_cost,
        "median_cost": median_cost,
        "max_cost": max_cost,
        "cost_per_tonne": total_cost / prod_total if prod_total > 0 else 0,
        "gini": gini,
        "n_municipalities": len(df),
    }
    if return_df:
        return metrics, df
    return metrics


# ============================================================
# RUN PRUNING
# ============================================================
print()
print("=" * 70)
print("PROGRESSIVE PLANT PRUNING ANALYSIS")
print("=" * 70)
print()

active = set(all_plants)
results = []

# Baseline: all plants
baseline = evaluate_network(active, df_mun, dist_lookup)
if baseline:
    results.append(baseline)
    print(f"Baseline: {baseline['n_plants']} plants, "
          f"cost={baseline['cost_per_tonne']:.2f} EUR/t, "
          f"max={baseline['max_cost']:.1f}, gini={baseline['gini']:.3f}")

# Progressive pruning: remove the plant whose removal causes minimum cost increase
while len(active) > 3:
    best_removal = None
    best_result = None
    best_cost_increase = float("inf")

    for pid in list(active):
        trial = active - {pid}
        result = evaluate_network(trial, df_mun, dist_lookup)
        if result is None:
            continue
        cost_increase = result["cost_per_tonne"] - results[-1]["cost_per_tonne"]
        if cost_increase < best_cost_increase:
            best_cost_increase = cost_increase
            best_removal = pid
            best_result = result

    if best_removal is None:
        break

    active.remove(best_removal)
    results.append(best_result)

    # Snapshot the active set at the 24-plant greedy pivot for downstream visualisation
    if best_result["n_plants"] == 24:
        greedy_24_active = set(active)

    if len(results) % 5 == 0 or len(active) <= 10:
        print(f"  {best_result['n_plants']:2d} plants (removed {best_removal}): "
              f"cost={best_result['cost_per_tonne']:.2f} EUR/t "
              f"(+{best_cost_increase:+.2f}), "
              f"max={best_result['max_cost']:.1f}, "
              f"gini={best_result['gini']:.3f}")

# ============================================================
# OUTPUT
# ============================================================
df_results = pd.DataFrame(results)
output_dir = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, f"pareto_podado{SCENARIO_SUFFIX}.csv")
df_results.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved: {csv_path}")

# Dump per-municipality assignment at the 24-plant greedy pivot (used by the
# three-panel map). Only emitted if the greedy curve actually passed through n=24.
if "greedy_24_active" in dir():
    _, df_g24 = evaluate_network(greedy_24_active, df_mun, dist_lookup, return_df=True)
    g24_path = os.path.join(output_dir, f"greedy_assignments_24{SCENARIO_SUFFIX}.csv")
    df_g24.to_csv(g24_path, index=False, float_format="%.4f")
    print(f"Saved: {g24_path} ({len(df_g24)} municipalities, {len(greedy_24_active)} active plants)")

# Plot
if not ARGS.no_plot:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Cost per tonne vs number of plants
        axes[0].plot(df_results["n_plants"], df_results["cost_per_tonne"], "k.-", lw=2)
        axes[0].set_xlabel("Number of plants")
        axes[0].set_ylabel("System cost (EUR/t)")
        axes[0].set_title("Cost vs Network Size")
        axes[0].invert_xaxis()
        axes[0].grid(alpha=0.3)

        # Max municipal cost vs number of plants
        axes[1].plot(df_results["n_plants"], df_results["max_cost"], "r.-", lw=2)
        axes[1].set_xlabel("Number of plants")
        axes[1].set_ylabel("Max municipal cost (EUR/t)")
        axes[1].set_title("Equity (worst case) vs Network Size")
        axes[1].invert_xaxis()
        axes[1].grid(alpha=0.3)

        # Gini vs number of plants
        axes[2].plot(df_results["n_plants"], df_results["gini"], "b.-", lw=2)
        axes[2].set_xlabel("Number of plants")
        axes[2].set_ylabel("Gini coefficient")
        axes[2].set_title("Inequality vs Network Size")
        axes[2].invert_xaxis()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"pareto_podado{SCENARIO_SUFFIX}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

print("\nDone.")
