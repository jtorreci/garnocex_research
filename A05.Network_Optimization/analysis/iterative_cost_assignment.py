# -*- coding: utf-8 -*-
"""
iterative_cost_assignment.py
============================
Iterative cost-optimal plant assignment for A05.

Algorithm:
  1. Consolidate co-located plants (< 15 km) into single facilities
  2. Assign each municipality to the plant with MINIMUM TOTAL COST
     (transport + treatment), not just minimum distance
  3. Recalculate throughputs → recalculate treatment costs → iterate
  4. At convergence, identify plants below viability threshold → eliminate
  5. Re-iterate until stable

This differs from A01's distance-based assignment because the cost-optimal
plant may NOT be the nearest one: a larger, more distant plant can be
cheaper due to economies of scale.

Uses A01 cost model: C₀=40k, T₀=5k, log₂, v=0.35, ρ=0.35
"""

import argparse
import math
import os
import re
import unicodedata

import numpy as np
import pandas as pd

# ============================================================
# COST MODEL (identical to A01)
# ============================================================
C0 = 40_000
T0 = 5_000
V_TREAT = 0.35
RHO_TRANS = 0.35
DOTACION = 0.5

VIABILITY_THRESHOLD = 2_000  # t/year: below this, plant is unviable


def parse_args():
    parser = argparse.ArgumentParser(description="A05 iterative cost assignment")
    parser.add_argument("--rho", type=float, default=RHO_TRANS, help="Transport cost parameter")
    parser.add_argument("--dotacion", type=float, default=DOTACION, help="Per-capita CDW generation")
    parser.add_argument("--scenario", type=str, default="", help="Scenario suffix for outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser.parse_args()


ARGS = parse_args()
RHO_TRANS = ARGS.rho
DOTACION = ARGS.dotacion
SCENARIO_SUFFIX = f"_{ARGS.scenario}" if ARGS.scenario else ""


def fixed_cost_total(T):
    if T <= 0: return 0.0
    if T < T0: return C0
    return C0 * (math.log2(T / T0) + 1)


def treatment_cost_per_tonne(T):
    if T <= 0: return 999.0
    return fixed_cost_total(T) / T + V_TREAT


def total_cost_for_municipality(dist_km, plant_throughput):
    """Total cost per tonne for a municipality at dist_km from a plant
    with the given throughput."""
    return dist_km * RHO_TRANS + treatment_cost_per_tonne(plant_throughput)


# ============================================================
# DATA LOADING
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
A01_DIR = os.path.join(SCRIPT_DIR, "..", "..", "A01.Voronoi", "scripts")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


print("Loading data...")

# Distance matrix: plant_id -> municipality -> distance (road network)
# The real-distance matrix covers all 46x383 pairs
df_dist_raw = pd.read_csv(os.path.join(A01_DIR, "tablas_distancias", "distancias_reales_plantas_municipios.csv"))

# Population
df_mun_raw = pd.read_csv(os.path.join(A01_DIR, "datos_voronoi_municipios.csv"),
                          encoding="utf-8", on_bad_lines="warn")
pop_lookup = {}
for _, row in df_mun_raw.iterrows():
    key = norm(row.get("NOMBRE", ""))
    if key and not pd.isna(row.get("HBT")):
        pop_lookup[key] = int(row["HBT"])

# Build municipality list
asig = pd.read_csv(os.path.join(A01_DIR, "asignacion_municipios_real.csv"))
municipalities = []
for _, row in asig.iterrows():
    mun = str(row["municipio"]).strip()
    key = norm(mun)
    pop = pop_lookup.get(key, 0)
    if pop > 0:
        municipalities.append({"name": mun, "key": key, "pop": pop, "prod": pop * DOTACION})
df_mun = pd.DataFrame(municipalities)

# Build distance lookup: (plant_id, mun_key) -> dist_km
dist_lookup = {}
for _, row in df_dist_raw.iterrows():
    pid = int(row["origin_id"])
    mkey = norm(str(row["destination_id"]))
    dist_km = float(row["total_cost"]) / 1000.0  # total_cost is in meters despite name
    if mkey and dist_km > 0:
        dist_lookup[(pid, mkey)] = dist_km

all_plants = sorted(set(pid for pid, _ in dist_lookup.keys()))
print(f"Municipalities: {len(df_mun)}, Plants: {len(all_plants)}")

# Plant consolidation (< 15 km) — consistent with A01
cod = pd.read_csv(os.path.join(A01_DIR, "tablas_distancias", "codigos_plantas.csv"))
id_to_mun = dict(zip(cod["Id"], cod["MUNICIPIO"].str.strip()))

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

# Apply consolidation
consolidated_plants = set()
for pid in all_plants:
    consolidated_plants.add(CONSOLIDATION_15KM.get(pid, pid))
consolidated_plants = sorted(consolidated_plants)

# Remap distances: for consolidated plants, use minimum distance
dist_consolidated = {}
for (pid, mkey), dist in dist_lookup.items():
    cpid = CONSOLIDATION_15KM.get(pid, pid)
    existing = dist_consolidated.get((cpid, mkey))
    if existing is None or dist < existing:
        dist_consolidated[(cpid, mkey)] = dist

print(f"After consolidation (< 15 km): {len(consolidated_plants)} plants")


# ============================================================
# ITERATIVE ALGORITHM
# ============================================================
def assign_by_cost(active_plants, plant_throughputs, df_mun, dist_lookup):
    """
    Assign each municipality to the plant with minimum TOTAL cost
    (transport + treatment), given current plant throughputs.
    """
    assignments = []
    for _, mun in df_mun.iterrows():
        key = mun["key"]
        best_cost = float("inf")
        best_plant = None
        best_dist = None

        for pid in active_plants:
            d = dist_lookup.get((pid, key))
            if d is None:
                continue
            T = plant_throughputs.get(pid, 0)
            c = total_cost_for_municipality(d, max(T, 1))
            if c < best_cost:
                best_cost = c
                best_plant = pid
                best_dist = d

        if best_plant is not None:
            assignments.append({
                "mun_key": key,
                "name": mun["name"],
                "prod": mun["prod"],
                "plant_id": best_plant,
                "dist_km": best_dist,
                "cost": best_cost,
            })
    return pd.DataFrame(assignments)


def compute_throughputs(df_assign):
    """Compute throughput per plant from assignments."""
    return df_assign.groupby("plant_id")["prod"].sum().to_dict()


def compute_system_metrics(df_assign, plant_throughputs):
    """Compute system-wide cost metrics."""
    costs = []
    for _, row in df_assign.iterrows():
        T = plant_throughputs.get(row["plant_id"], 0)
        c_trans = row["dist_km"] * RHO_TRANS
        c_treat = treatment_cost_per_tonne(T)
        costs.append(c_trans + c_treat)

    costs = np.array(costs)
    prods = df_assign["prod"].values
    total_cost = np.sum(costs * prods)
    mean_cost = total_cost / prods.sum()

    return {
        "n_plants": df_assign["plant_id"].nunique(),
        "cost_per_tonne": mean_cost,
        "max_cost": costs.max(),
        "min_cost": costs.min(),
        "total_cost": total_cost,
        "total_tkm": np.sum(prods * df_assign["dist_km"].values),
        "n_municipalities": len(df_assign),
    }


def iterate_assignment(active_plants, df_mun, dist_lookup, max_iter=50, tol=0.01):
    """
    Iterate cost-optimal assignment until convergence.
    Returns: final assignments, throughputs, convergence history.
    """
    # Initial throughputs: assign by distance first
    throughputs = {}
    for pid in active_plants:
        throughputs[pid] = 1000  # seed with small value

    # First pass: distance-based to get initial throughputs
    df_assign = assign_by_cost(active_plants, {p: 50000 for p in active_plants},
                                df_mun, dist_lookup)
    throughputs = compute_throughputs(df_assign)

    history = []
    for iteration in range(max_iter):
        # Assign by cost using current throughputs
        df_assign_new = assign_by_cost(active_plants, throughputs, df_mun, dist_lookup)
        throughputs_new = compute_throughputs(df_assign_new)

        metrics = compute_system_metrics(df_assign_new, throughputs_new)
        metrics["iteration"] = iteration
        history.append(metrics)

        # Check convergence
        max_change = 0
        for pid in active_plants:
            old = throughputs.get(pid, 0)
            new = throughputs_new.get(pid, 0)
            if old > 0:
                max_change = max(max_change, abs(new - old) / old)

        throughputs = throughputs_new
        df_assign = df_assign_new

        if max_change < tol:
            break

    return df_assign, throughputs, history


print()
print("=" * 70)
print("ITERATIVE COST-OPTIMAL ASSIGNMENT")
print("=" * 70)

# Phase 1: Iterate with all consolidated plants
active = set(consolidated_plants)
df_assign, throughputs, history = iterate_assignment(
    active, df_mun, dist_consolidated)

metrics = compute_system_metrics(df_assign, throughputs)
print(f"\nPhase 1 (all {metrics['n_plants']} plants):")
print(f"  Converged in {len(history)} iterations")
print(f"  Cost: {metrics['cost_per_tonne']:.2f} EUR/t")
print(f"  Max: {metrics['max_cost']:.1f}, Total t-km: {metrics['total_tkm']:,.0f}")

# Phase 2: Progressive elimination of unviable plants
print()
print("Phase 2: Progressive elimination")
print("-" * 70)

elimination_results = [metrics.copy()]
eliminated_plants = []

while True:
    # Find plants below viability threshold (including zero-throughput)
    unviable = [pid for pid in active
                if throughputs.get(pid, 0) < VIABILITY_THRESHOLD]

    if not unviable:
        print("  No unviable plants remaining. Converged.")
        break

    # Eliminate the plant with lowest throughput
    worst = min(unviable, key=lambda p: throughputs.get(p, 0))
    worst_T = throughputs.get(worst, 0)
    worst_mun = id_to_mun.get(worst, f"Plant_{worst}")
    active.remove(worst)
    eliminated_plants.append((worst, worst_mun, worst_T))

    # Re-iterate
    df_assign, throughputs, history = iterate_assignment(
        active, df_mun, dist_consolidated)
    metrics = compute_system_metrics(df_assign, throughputs)
    metrics["eliminated"] = f"{worst_mun} (T={worst_T:.0f})"
    elimination_results.append(metrics.copy())

    print(f"  Eliminated {worst_mun:25s} (T={worst_T:6.0f} t/y) -> "
          f"{metrics['n_plants']:2d} plants, "
          f"cost={metrics['cost_per_tonne']:.2f} EUR/t, "
          f"max={metrics['max_cost']:.1f}")

# Final summary
print()
print("=" * 70)
print("FINAL OPTIMIZED NETWORK")
print("=" * 70)
print(f"  Active plants: {len(active)}")
print(f"  Cost per tonne: {metrics['cost_per_tonne']:.2f} EUR/t")
print(f"  Max municipal cost: {metrics['max_cost']:.1f} EUR/t")
print(f"  Total t-km: {metrics['total_tkm']:,.0f}")
print()

# Eliminated plants
print(f"  Plants eliminated ({len(eliminated_plants)}):")
for pid, mun, T in eliminated_plants:
    print(f"    ID {pid:2d} {mun:25s} (was {T:6.0f} t/y)")

# Surviving plants
print(f"\n  Surviving plants ({len(active)}):")
for pid in sorted(active):
    T = throughputs.get(pid, 0)
    mun = id_to_mun.get(pid, f"Plant_{pid}")
    c_treat = treatment_cost_per_tonne(T)
    print(f"    ID {pid:2d} {mun:25s} T={T:8.0f} t/y  C_treat={c_treat:.2f} EUR/t")

# Comparison with A01 baseline
print()
print("=" * 70)
print("COMPARISON WITH A01 BASELINE")
print("=" * 70)
baseline = elimination_results[0]
final = elimination_results[-1]
print(f"  {'Metric':<25s} {'A01 baseline':>15s} {'Optimized':>15s} {'Change':>10s}")
print(f"  {'-'*65}")
for key, label in [("n_plants", "Plants"), ("cost_per_tonne", "Cost (EUR/t)"),
                    ("max_cost", "Max municipal"), ("total_tkm", "Total t-km")]:
    b = baseline[key]
    f = final[key]
    if isinstance(b, float):
        pct = f"{(f-b)/b*100:+.1f}%" if b != 0 else ""
        print(f"  {label:<25s} {b:15.2f} {f:15.2f} {pct:>10s}")
    else:
        print(f"  {label:<25s} {b:15d} {f:15d}")

# Save results
df_elim = pd.DataFrame(elimination_results)
df_elim.to_csv(os.path.join(OUTPUT_DIR, f"iterative_optimization{SCENARIO_SUFFIX}.csv"),
                index=False, float_format="%.4f")

# Save final assignments
df_assign.to_csv(os.path.join(OUTPUT_DIR, f"optimal_assignments{SCENARIO_SUFFIX}.csv"),
                  index=False, float_format="%.4f")

# Plot
if not ARGS.no_plot:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        n = df_elim["n_plants"].values
        axes[0].plot(n, df_elim["cost_per_tonne"], "ko-", lw=2, ms=5)
        axes[0].set_ylabel("System cost (EUR/t)")
        axes[0].set_title("Cost vs Network Size\n(iterative cost-optimal)")
        axes[0].invert_xaxis()
        axes[0].grid(alpha=0.3)

        axes[1].plot(n, df_elim["max_cost"], "ro-", lw=2, ms=5)
        axes[1].set_ylabel("Max municipal cost (EUR/t)")
        axes[1].set_title("Worst-case equity")
        axes[1].invert_xaxis()
        axes[1].grid(alpha=0.3)

        axes[2].plot(n, df_elim["total_tkm"] / 1e6, "bo-", lw=2, ms=5)
        axes[2].set_ylabel("Total transport (M t-km/y)")
        axes[2].set_title("Transport effort")
        axes[2].invert_xaxis()
        axes[2].grid(alpha=0.3)

        for ax in axes:
            ax.set_xlabel("Number of plants")

        plt.suptitle("A05: Iterative cost-optimal network pruning", fontweight="bold")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"iterative_optimization{SCENARIO_SUFFIX}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {plot_path}")
    except ImportError:
        pass

print("\nDone.")
