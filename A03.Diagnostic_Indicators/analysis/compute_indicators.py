#!/usr/bin/env python3
"""
compute_indicators.py
=====================
Compute the 13-indicator two-panel framework for the Extremadura CDW network.

Plant-level panel (7 indicators):
  1. C_i          : Unit total cost (EUR/t), decomposed into C_trans + C_treat
  2. U_i          : Capacity utilisation (skipped - no capacity data)
  3. D_i^(90)     : 90th-percentile weighted accessibility (km)
  4. Prec_i       : Voronoi precision
  5. Rec_i        : Voronoi recall
  6. Leak_dg_i    : Cost-penalised leakage (EUR/t)
  7. IET_i        : Transport efficiency (%)

Network-level panel (7 indicators):
  1. C_bar        : Production-weighted average unit cost (EUR/t)
  2. Gini(C_m)    : Gini coefficient of municipal costs
  3. MAD_C        : Mean absolute deviation of costs (EUR/t)
  4. CVaR_0.9(D)  : Conditional value at risk of distance (km)
  5. micro-F1     : System-wide baseline compliance
  6. IGET         : Global transport efficiency (%)
  7. CV(T_i)      : Coefficient of variation of throughput

Data sources (from A01):
  - datos_red_real_municipios.csv  : municipality-level data with real assignments
  - datos_red_real_plantas.csv     : plant-level summary
  - datos_red_real_municipios.csv        : municipality-level road-distance assignment
  - asignacion_municipios_real.csv       : Real network assignment
"""

import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Optional: matplotlib for figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available; skipping figure generation.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_A01 = PROJECT_ROOT / "A01.Voronoi" / "scripts"
BASE_A03 = PROJECT_ROOT / "A03. Indicadores"

DATA_DIR        = BASE_A01 / "datos_red_real"
REAL_ASSIGN     = BASE_A01 / "asignacion_municipios_real.csv"

OUTPUT_DIR = BASE_A03 / "scripts"

# ---------------------------------------------------------------------------
# Parameters (same as A01)
# ---------------------------------------------------------------------------
C0   = 40_000   # EUR, minimum annual fixed cost
T0   = 5_000    # t/year, reference throughput for log cost
RHO  = 0.35     # EUR/t-km, transport tariff
V    = 0.35     # EUR/t, variable treatment cost
DOT  = 0.5      # t/hab/year, CDW endowment rate

# Consolidation map: physical plant -> group head
CONSOLIDATION = {
    2: 1, 12: 1,
    17: 3,
    15: 6,
    40: 19,
    32: 20,
    23: 22, 26: 22, 28: 22,
    34: 27, 35: 27, 37: 27,
    38: 36, 43: 36,
}

# ---------------------------------------------------------------------------
# Name normalisation (same as A01)
# ---------------------------------------------------------------------------
def norm(s):
    """Normalise a municipality name for matching across CSVs."""
    if pd.isna(s):
        return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def consolidate_plant_id(pid):
    """Map physical plant id to consolidated group id."""
    pid = int(pid)
    return CONSOLIDATION.get(pid, pid)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def fixed_cost(T):
    """Annual fixed cost for a plant with throughput T (t/year).
    Matches A01 model: C0 below T0, C0*(log2(T/T0)+1) above T0."""
    if T <= 0:
        return C0
    if T < T0:
        return C0
    return C0 * (np.log2(T / T0) + 1)


def unit_treatment_cost(T):
    """Unit treatment cost (EUR/t) for a plant with throughput T."""
    if T <= 0:
        return np.inf
    return fixed_cost(T) / T + V


def unit_total_cost(d_km, T):
    """Unit total cost (EUR/t) for a municipality at distance d_km from a plant with throughput T."""
    return RHO * d_km + unit_treatment_cost(T)


# ---------------------------------------------------------------------------
# Weighted quantile
# ---------------------------------------------------------------------------
def weighted_quantile(values, weights, q):
    """Compute the q-th weighted quantile (0 <= q <= 1)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cum_w = np.cumsum(weights)
    cum_w_norm = (cum_w - 0.5 * weights) / cum_w[-1]
    return np.interp(q, cum_w_norm, values)


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------
def gini_coefficient(values):
    """Compute the Gini coefficient of an array of values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or np.sum(values) == 0:
        return np.nan
    n = len(values)
    values_sorted = np.sort(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values_sorted) - (n + 1) * np.sum(values_sorted)) / (n * np.sum(values_sorted))


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------
def cvar(values, weights, alpha=0.9):
    """
    Compute the Conditional Value at Risk at level alpha.
    CVaR_alpha = expected value of the worst (1-alpha) fraction (weighted by weights).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cum_w = np.cumsum(weights)
    total_w = cum_w[-1]
    threshold_w = alpha * total_w
    # Find VaR (the alpha-quantile)
    var_idx = np.searchsorted(cum_w, threshold_w)
    if var_idx >= len(values):
        var_idx = len(values) - 1
    var_val = values[var_idx]
    # CVaR = E[X | X >= VaR]
    tail_mask = values >= var_val
    tail_values = values[tail_mask]
    tail_weights = weights[tail_mask]
    if np.sum(tail_weights) == 0:
        return var_val
    return np.average(tail_values, weights=tail_weights)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("COMPUTING 13-INDICATOR TWO-PANEL FRAMEWORK")
    print("Extremadura CDW Network")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------
    print("\n[1] Loading data...")

    # Municipality data (A01 road-distance assignment)
    df_mun = pd.read_csv(DATA_DIR / "datos_red_real_municipios.csv")
    df_mun["norm_name"] = df_mun["municipio"].apply(norm)
    df_mun["planta_id_cons"] = df_mun["planta_id"].apply(consolidate_plant_id)

    # Real assignment (for cross-check)
    df_real = pd.read_csv(REAL_ASSIGN)
    df_real["norm_name"] = df_real["municipio"].apply(norm)
    df_real["planta_real_cons"] = df_real["planta_asignada"].apply(consolidate_plant_id)

    print(f"  Municipalities loaded: {len(df_mun)}")
    print(f"  Road-baseline assignments: {len(df_mun)}")

    # -------------------------------------------------------------------
    # 2. Merge theoretical baseline assignment into municipality data
    # -------------------------------------------------------------------
    print("\n[2] Building road-distance baseline...")

    df_mun["planta_voronoi"] = df_mun["planta_id_cons"]
    df_mun["euclidean_distance"] = df_mun["distancia_km"] * 1000.0
    df = df_mun.copy()

    # Check baseline quality
    n_matched = df["planta_voronoi"].notna().sum()
    print(f"  Matched road baseline: {n_matched}/{len(df)}")

    # Ensure types
    df["planta_voronoi"] = df["planta_voronoi"].astype(int)
    df["planta_id_cons"] = df["planta_id_cons"].astype(int)

    # -------------------------------------------------------------------
    # 3. Build baseline distances
    # -------------------------------------------------------------------
    print("\n[3] Building distance columns...")

    # Use the distance from the data file (network distance in km) as the primary distance
    # The 'distancia_km' column is the road-network distance
    df["d_real_km"] = df["distancia_km"]  # road-network distance to real plant

    # The theoretical baseline now uses the same nearest-plant assignment by road distance.
    df["d_voronoi_km"] = df["distancia_km"]

    print(f"  NaN in d_real_km: {df['d_real_km'].isna().sum()}")
    print(f"  NaN in d_voronoi_km: {df['d_voronoi_km'].isna().sum()}")

    # -------------------------------------------------------------------
    # 4. Compute plant throughput and treatment costs
    # -------------------------------------------------------------------
    print("\n[4] Computing plant throughput and costs...")

    # Production
    df["produccion_t"] = df["poblacion"] * DOT

    # Real plant throughput (consolidated)
    plant_throughput = df.groupby("planta_id_cons")["produccion_t"].sum().rename("T_i")

    # Number of physical plants per consolidated group (for split throughput, matching A01)
    from collections import Counter
    absorbed_into = Counter(CONSOLIDATION.values())
    n_plants_in_group = {}
    all_group_ids = set(range(1, 47)) - set(CONSOLIDATION.keys())  # heads
    for gid in all_group_ids:
        n_plants_in_group[gid] = 1 + absorbed_into.get(gid, 0)

    # Treatment cost per plant using split throughput (A01 methodology):
    # each physical plant in a group processes T_group / n_plants
    def treatment_cost_split(T_group, group_id):
        n = n_plants_in_group.get(group_id, 1)
        T_per_plant = T_group / n
        return unit_treatment_cost(T_per_plant)

    plant_treat = pd.Series(
        {gid: treatment_cost_split(T, gid) for gid, T in plant_throughput.items()},
        name="C_treat_i"
    )

    # Merge back
    df = df.merge(plant_throughput, left_on="planta_id_cons", right_index=True, how="left")
    df = df.merge(plant_treat, left_on="planta_id_cons", right_index=True, how="left")

    # Also compute throughput for Voronoi-assigned plants
    voronoi_throughput = df.groupby("planta_voronoi")["produccion_t"].sum().rename("T_voronoi")

    # Municipal unit cost under REAL assignment
    df["C_trans_m"] = RHO * df["d_real_km"]
    df["C_m"] = df["C_trans_m"] + df["C_treat_i"]

    # Municipal unit cost under VORONOI assignment (also using split throughput)
    df = df.merge(voronoi_throughput, left_on="planta_voronoi", right_index=True, how="left")
    df["C_treat_voronoi"] = df.apply(
        lambda r: treatment_cost_split(r["T_voronoi"], int(r["planta_voronoi"])), axis=1
    )
    df["C_m_voronoi"] = RHO * df["d_voronoi_km"] + df["C_treat_voronoi"]

    # Transport effort (t-km)
    df["tkm_real"] = df["produccion_t"] * df["d_real_km"]
    df["tkm_voronoi"] = df["produccion_t"] * df["d_voronoi_km"]

    total_production = df["produccion_t"].sum()
    print(f"  Total production: {total_production:,.0f} t/year")
    print(f"  Number of consolidated plants: {plant_throughput.shape[0]}")

    # -------------------------------------------------------------------
    # 5. TP / FP / FN classification
    # -------------------------------------------------------------------
    print("\n[5] Computing TP/FP/FN classification...")

    # For each municipality:
    #   - If planta_id_cons == planta_voronoi -> TP for that plant
    #   - If planta_id_cons != planta_voronoi -> FP for the real plant, FN for the Voronoi plant
    df["is_tp"] = (df["planta_id_cons"] == df["planta_voronoi"]).astype(int)

    # TP per plant: sum of production from municipalities that ARE in its Voronoi cell AND assigned to it
    tp_by_plant = df[df["is_tp"] == 1].groupby("planta_id_cons")["produccion_t"].sum().rename("TP")

    # FP per plant: sum of production from municipalities NOT in its Voronoi cell but assigned to it
    fp_by_plant = df[df["is_tp"] == 0].groupby("planta_id_cons")["produccion_t"].sum().rename("FP")

    # FN per plant (by Voronoi plant): sum of production from municipalities IN its Voronoi cell but NOT assigned to it
    fn_by_plant = df[df["is_tp"] == 0].groupby("planta_voronoi")["produccion_t"].sum().rename("FN")

    # Total theoretical intake per Voronoi cell
    voronoi_intake = df.groupby("planta_voronoi")["produccion_t"].sum().rename("T_voronoi_in")

    # Combine into plant-level DataFrame
    all_plants = sorted(plant_throughput.index)
    plant_df = pd.DataFrame(index=all_plants)
    plant_df.index.name = "plant_id"
    plant_df["T_i"] = plant_throughput.reindex(all_plants).fillna(0)
    plant_df["TP"] = tp_by_plant.reindex(all_plants).fillna(0)
    plant_df["FP"] = fp_by_plant.reindex(all_plants).fillna(0)
    plant_df["FN"] = fn_by_plant.reindex(all_plants).fillna(0)
    plant_df["T_voronoi_in"] = voronoi_intake.reindex(all_plants).fillna(0)

    # Precision, Recall, F1
    plant_df["Prec"] = plant_df["TP"] / (plant_df["TP"] + plant_df["FP"])
    plant_df["Rec"] = plant_df["TP"] / (plant_df["TP"] + plant_df["FN"])
    plant_df["F1"] = 2 * plant_df["Prec"] * plant_df["Rec"] / (plant_df["Prec"] + plant_df["Rec"])

    # Handle plants with zero TP+FP or TP+FN
    plant_df["Prec"] = plant_df["Prec"].fillna(0)
    plant_df["Rec"] = plant_df["Rec"].fillna(0)
    plant_df["F1"] = plant_df["F1"].fillna(0)

    n_voronoi_compliant = df["is_tp"].sum()
    n_total = len(df)
    print(f"  Voronoi-compliant municipalities: {n_voronoi_compliant}/{n_total} "
          f"({100*n_voronoi_compliant/n_total:.1f}%)")

    # -------------------------------------------------------------------
    # 6. Plant-level indicators
    # -------------------------------------------------------------------
    print("\n[6] Computing plant-level indicators...")

    # 6a. Unit total cost C_i (decomposed)
    plant_cost = df.groupby("planta_id_cons").agg(
        total_tkm=("tkm_real", "sum"),
        total_prod=("produccion_t", "sum"),
    )
    plant_df["C_trans_i"] = RHO * plant_cost["total_tkm"] / plant_cost["total_prod"]
    plant_df["C_treat_i"] = plant_df["T_i"].apply(unit_treatment_cost)
    plant_df["C_i"] = plant_df["C_trans_i"] + plant_df["C_treat_i"]

    # 6b. D_i^(90): 90th-percentile weighted accessibility
    d90_vals = {}
    for pid in all_plants:
        mun_mask = df["planta_id_cons"] == pid
        dists = df.loc[mun_mask, "d_real_km"].values
        weights = df.loc[mun_mask, "produccion_t"].values
        d90_vals[pid] = weighted_quantile(dists, weights, 0.9)
    plant_df["D90"] = pd.Series(d90_vals)

    # 6c. Cost-penalised leakage Leak_dg_i
    leak_vals = {}
    for pid in all_plants:
        # Municipalities in Voronoi cell of this plant
        in_voronoi = df[df["planta_voronoi"] == pid]
        if len(in_voronoi) == 0:
            leak_vals[pid] = 0.0
            continue

        total_voronoi_prod = in_voronoi["produccion_t"].sum()
        if total_voronoi_prod == 0:
            leak_vals[pid] = 0.0
            continue

        # Municipalities that leak (not assigned to this plant)
        leaked = in_voronoi[in_voronoi["planta_id_cons"] != pid]
        if len(leaked) == 0:
            leak_vals[pid] = 0.0
            continue

        # For each leaked municipality, compute cost penalty
        # g(m, voronoi_plant) = RHO * d_voronoi + C_treat_voronoi
        # g(m, real_plant) = RHO * d_real + C_treat_real = C_m
        # penalty = max(0, g_real - g_voronoi) * production
        # Need C_treat for the Voronoi plant (pid) with its throughput.
        # Use split (per-physical-plant) throughput to match the C_m cost model
        # (Section 3.4: T_tilde = T_i / n_i), so g_real and g_voronoi are consistent.
        c_treat_vor = treatment_cost_split(plant_df.loc[pid, "T_i"], pid)

        penalty_sum = 0.0
        for _, row in leaked.iterrows():
            g_voronoi = RHO * row["d_voronoi_km"] + c_treat_vor
            g_real = row["C_m"]  # already computed
            delta_g = max(0, g_real - g_voronoi)
            penalty_sum += row["produccion_t"] * delta_g

        leak_vals[pid] = penalty_sum / total_voronoi_prod

    plant_df["Leak_dg"] = pd.Series(leak_vals)

    # 6d. IET_i: Transport efficiency
    # IET_i = 100 * T_theo_i / T_real_i
    # T_real_i = sum of tkm for municipalities assigned to plant i (real)
    # T_theo_i = sum of tkm for municipalities assigned to plant i (Voronoi)
    # NOTE: IET compares the same PLANT's effort under two assignments.
    # Under Voronoi, plant i would receive T_voronoi_in tonnes from its Voronoi cell.
    # Under real, it receives T_i tonnes from various municipalities.

    real_tkm_by_plant = df.groupby("planta_id_cons")["tkm_real"].sum()
    voronoi_tkm_by_plant = df.groupby("planta_voronoi")["tkm_voronoi"].sum()

    plant_df["tkm_real"] = real_tkm_by_plant.reindex(all_plants).fillna(0)
    plant_df["tkm_voronoi"] = voronoi_tkm_by_plant.reindex(all_plants).fillna(0)

    plant_df["IET"] = np.where(
        plant_df["tkm_real"] > 0,
        100 * plant_df["tkm_voronoi"] / plant_df["tkm_real"],
        np.nan
    )

    # Plant name lookup
    plant_name_map = df_mun.drop_duplicates("planta_id").set_index("planta_id")["planta_municipio"]
    # Also map consolidated plant id to a representative name
    plant_names = {}
    for pid in all_plants:
        # Find any municipality assigned to this consolidated plant
        mask = df["planta_id_cons"] == pid
        if mask.any():
            plant_names[pid] = df.loc[mask, "planta_municipio"].iloc[0]
        else:
            plant_names[pid] = f"Plant {pid}"
    plant_df["plant_name"] = pd.Series(plant_names)

    # -------------------------------------------------------------------
    # 7. Network-level indicators
    # -------------------------------------------------------------------
    print("\n[7] Computing network-level indicators...")

    # 7a. C_bar: production-weighted average unit cost
    C_bar = np.average(df["C_m"], weights=df["produccion_t"])
    C_bar_trans = np.average(df["C_trans_m"], weights=df["produccion_t"])
    C_bar_treat = np.average(df["C_treat_i"], weights=df["produccion_t"])

    # 7b. Gini(C_m)
    gini_val = gini_coefficient(df["C_m"].values)

    # 7c. MAD_C (production-weighted)
    mad_c = np.average(np.abs(df["C_m"] - C_bar), weights=df["produccion_t"])

    # 7d. CVaR_0.9(D) on distances
    cvar_val = cvar(df["d_real_km"].values, df["produccion_t"].values, alpha=0.9)

    # 7e. micro-F1
    total_TP = plant_df["TP"].sum()
    total_FP = plant_df["FP"].sum()
    total_FN = plant_df["FN"].sum()
    micro_f1 = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) > 0 else 0

    # 7f. IGET: global transport efficiency
    total_tkm_real = df["tkm_real"].sum()
    total_tkm_voronoi = df["tkm_voronoi"].sum()
    IGET = 100 * total_tkm_voronoi / total_tkm_real if total_tkm_real > 0 else np.nan

    # 7g. CV(T_i): coefficient of variation of throughput
    cv_ti = plant_df["T_i"].std() / plant_df["T_i"].mean() if plant_df["T_i"].mean() > 0 else np.nan

    # Also compute legacy indices for redundancy verification
    IGCR = 10000.0 / IGET if IGET > 0 else np.nan
    IGER = IGCR  # same under constant kappa
    IGSD = 10000.0 / IGET - 100 if IGET > 0 else np.nan

    # -------------------------------------------------------------------
    # 8. Additional diagnostics
    # -------------------------------------------------------------------
    # How much of the leakage is irrational?
    total_leaked_production = df[df["is_tp"] == 0]["produccion_t"].sum()
    irrational_leaked = 0.0
    rational_leaked = 0.0
    irrational_penalty_total = 0.0

    for _, row in df[df["is_tp"] == 0].iterrows():
        vor_plant = int(row["planta_voronoi"])
        c_treat_vor = treatment_cost_split(plant_df.loc[vor_plant, "T_i"], vor_plant)
        g_voronoi = RHO * row["d_voronoi_km"] + c_treat_vor
        g_real = row["C_m"]
        if g_real > g_voronoi:
            irrational_leaked += row["produccion_t"]
            irrational_penalty_total += row["produccion_t"] * (g_real - g_voronoi)
        else:
            rational_leaked += row["produccion_t"]

    pct_irrational = 100 * irrational_leaked / total_leaked_production if total_leaked_production > 0 else 0
    pct_rational = 100 * rational_leaked / total_leaked_production if total_leaked_production > 0 else 0

    # Mean distance (weighted)
    mean_dist = np.average(df["d_real_km"], weights=df["produccion_t"])

    # -------------------------------------------------------------------
    # 9. Print summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- NETWORK-LEVEL PANEL ---")
    print(f"  C_bar (avg unit cost)        : {C_bar:.2f} EUR/t")
    print(f"    - Transport component      : {C_bar_trans:.2f} EUR/t")
    print(f"    - Treatment component      : {C_bar_treat:.2f} EUR/t")
    print(f"  Gini(C_m)                    : {gini_val:.4f}")
    print(f"  MAD_C                        : {mad_c:.2f} EUR/t")
    print(f"  CVaR_0.9(D)                  : {cvar_val:.1f} km")
    print(f"  micro-F1                     : {micro_f1:.4f}")
    print(f"  IGET                         : {IGET:.1f}%")
    print(f"  CV(T_i)                      : {cv_ti:.3f}")
    print(f"")
    print(f"  Mean weighted distance       : {mean_dist:.1f} km")
    print(f"  Total T*km (real)            : {total_tkm_real:,.0f}")
    print(f"  Total T*km (Voronoi)         : {total_tkm_voronoi:,.0f}")

    print("\n--- REDUNDANCY VERIFICATION ---")
    print(f"  IGET  = {IGET:.2f}%")
    print(f"  IGCR  = {IGCR:.2f}%  (should = 10000/IGET = {10000/IGET:.2f})")
    print(f"  IGER  = {IGER:.2f}%  (should = IGCR)")
    print(f"  IGSD  = {IGSD:.2f}%  (should = 10000/IGET - 100 = {10000/IGET - 100:.2f})")

    print("\n--- LEAKAGE ANALYSIS ---")
    print(f"  Total leaked production      : {total_leaked_production:,.0f} t/year")
    print(f"    - Rational (cheaper)       : {rational_leaked:,.0f} t/year ({pct_rational:.1f}%)")
    print(f"    - Irrational (more expen.) : {irrational_leaked:,.0f} t/year ({pct_irrational:.1f}%)")
    print(f"  Total irrational cost penalty: {irrational_penalty_total:,.0f} EUR/year")

    print("\n--- PLANT-LEVEL PANEL ---")
    cols_display = ["plant_name", "T_i", "C_i", "C_trans_i", "C_treat_i",
                    "D90", "Prec", "Rec", "F1", "Leak_dg", "IET"]
    print(plant_df[cols_display].to_string(float_format=lambda x: f"{x:.2f}"))

    # -------------------------------------------------------------------
    # 10. Export to CSV
    # -------------------------------------------------------------------
    print("\n[10] Exporting results...")

    # Plant-level CSV
    plant_out = plant_df[["plant_name", "T_i", "C_i", "C_trans_i", "C_treat_i",
                          "D90", "Prec", "Rec", "F1", "Leak_dg", "IET",
                          "TP", "FP", "FN", "T_voronoi_in"]].copy()
    plant_out.index.name = "plant_id"
    plant_csv = OUTPUT_DIR / "plant_level_indicators.csv"
    plant_out.to_csv(plant_csv)
    print(f"  Plant-level: {plant_csv}")

    # Network-level CSV
    net_data = {
        "Indicator": [
            "C_bar (EUR/t)", "C_bar_trans (EUR/t)", "C_bar_treat (EUR/t)",
            "Gini(C_m)", "MAD_C (EUR/t)", "CVaR_0.9(D) (km)",
            "micro-F1", "IGET (%)", "CV(T_i)",
            "IGCR (%)", "IGER (%)", "IGSD (%)",
            "Mean dist (km)", "Total tkm real", "Total tkm Voronoi",
            "Total production (t/yr)",
            "Leaked production (t/yr)", "Rational leaked (t/yr)",
            "Irrational leaked (t/yr)", "Irrational penalty (EUR/yr)",
        ],
        "Value": [
            f"{C_bar:.2f}", f"{C_bar_trans:.2f}", f"{C_bar_treat:.2f}",
            f"{gini_val:.4f}", f"{mad_c:.2f}", f"{cvar_val:.1f}",
            f"{micro_f1:.4f}", f"{IGET:.1f}", f"{cv_ti:.3f}",
            f"{IGCR:.2f}", f"{IGER:.2f}", f"{IGSD:.2f}",
            f"{mean_dist:.1f}", f"{total_tkm_real:.0f}", f"{total_tkm_voronoi:.0f}",
            f"{total_production:.0f}",
            f"{total_leaked_production:.0f}", f"{rational_leaked:.0f}",
            f"{irrational_leaked:.0f}", f"{irrational_penalty_total:.0f}",
        ]
    }
    net_csv = OUTPUT_DIR / "network_level_indicators.csv"
    pd.DataFrame(net_data).to_csv(net_csv, index=False)
    print(f"  Network-level: {net_csv}")

    # -------------------------------------------------------------------
    # 11. Generate figure
    # -------------------------------------------------------------------
    if HAS_MPL:
        print("\n[11] Generating figure...")
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

        # Sort plants by throughput for consistent ordering
        plant_sorted = plant_df.sort_values("T_i", ascending=True)
        plant_labels = [f"{pid} ({name})" for pid, name in
                        zip(plant_sorted.index, plant_sorted["plant_name"])]

        # --- Panel A: Unit cost decomposition (horizontal bar) ---
        ax1 = fig.add_subplot(gs[0, 0])
        y_pos = np.arange(len(plant_sorted))
        ax1.barh(y_pos, plant_sorted["C_trans_i"], color="#2196F3", label="Transport")
        ax1.barh(y_pos, plant_sorted["C_treat_i"], left=plant_sorted["C_trans_i"],
                 color="#FF9800", label="Treatment")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(plant_labels, fontsize=5)
        ax1.set_xlabel("Unit cost (EUR/t)")
        ax1.set_title("(a) Unit cost decomposition")
        ax1.legend(fontsize=7)

        # --- Panel B: Precision vs Recall scatter ---
        ax2 = fig.add_subplot(gs[0, 1])
        sizes = plant_sorted["T_i"] / plant_sorted["T_i"].max() * 300
        scatter = ax2.scatter(plant_sorted["Rec"], plant_sorted["Prec"],
                              s=sizes, c=plant_sorted["F1"], cmap="RdYlGn",
                              vmin=0, vmax=1, edgecolors="black", linewidth=0.5)
        ax2.set_xlabel("Recall (Voronoi)")
        ax2.set_ylabel("Precision (Voronoi)")
        ax2.set_title("(b) Precision vs Recall")
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="F1 score")

        # --- Panel C: IET distribution ---
        ax3 = fig.add_subplot(gs[0, 2])
        colors = ["#4CAF50" if v >= 100 else "#F44336" for v in plant_sorted["IET"]]
        ax3.barh(y_pos, plant_sorted["IET"], color=colors)
        ax3.axvline(100, color="black", linestyle="--", linewidth=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(plant_labels, fontsize=5)
        ax3.set_xlabel("IET (%)")
        ax3.set_title("(c) Transport Efficiency Index")

        # --- Panel D: D90 accessibility ---
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.barh(y_pos, plant_sorted["D90"], color="#9C27B0")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(plant_labels, fontsize=5)
        ax4.set_xlabel("D90 (km)")
        ax4.set_title("(d) 90th-percentile accessibility")

        # --- Panel E: Cost-penalised leakage ---
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.barh(y_pos, plant_sorted["Leak_dg"], color="#E91E63")
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(plant_labels, fontsize=5)
        ax5.set_xlabel("Leak_dg (EUR/t)")
        ax5.set_title("(e) Cost-penalised leakage")

        # --- Panel F: Network indicators summary ---
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        net_text = (
            f"NETWORK-LEVEL PANEL\n"
            f"{'='*35}\n"
            f"Avg unit cost (C_bar): {C_bar:.2f} EUR/t\n"
            f"  Transport: {C_bar_trans:.2f} EUR/t\n"
            f"  Treatment: {C_bar_treat:.2f} EUR/t\n"
            f"Gini(C_m): {gini_val:.4f}\n"
            f"MAD_C: {mad_c:.2f} EUR/t\n"
            f"CVaR_0.9(D): {cvar_val:.1f} km\n"
            f"micro-F1: {micro_f1:.4f}\n"
            f"IGET: {IGET:.1f}%\n"
            f"CV(T_i): {cv_ti:.3f}\n"
            f"{'='*35}\n"
            f"Mean distance: {mean_dist:.1f} km\n"
            f"Total prod: {total_production:,.0f} t/yr\n"
        )
        ax6.text(0.05, 0.95, net_text, transform=ax6.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax6.set_title("(f) Network-level summary")

        # --- Panel G: Distribution of municipal costs ---
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(df["C_m"], bins=30, weights=df["produccion_t"],
                 color="#607D8B", edgecolor="white")
        ax7.axvline(C_bar, color="red", linestyle="--", label=f"Mean = {C_bar:.1f}")
        ax7.set_xlabel("Municipal unit cost (EUR/t)")
        ax7.set_ylabel("Production (t/year)")
        ax7.set_title("(g) Distribution of municipal costs")
        ax7.legend()

        # --- Panel H: Redundancy verification ---
        ax8 = fig.add_subplot(gs[2, 1])
        plant_df_valid = plant_df[plant_df["IET"].notna() & (plant_df["IET"] > 0)]
        iet_vals = plant_df_valid["IET"]
        icr_computed = 10000.0 / iet_vals
        ax8.scatter(iet_vals, icr_computed, color="#3F51B5", s=40)
        x_range = np.linspace(iet_vals.min() * 0.8, iet_vals.max() * 1.2, 100)
        ax8.plot(x_range, 10000.0 / x_range, "r-", label="ICR = 10000/IET")
        ax8.set_xlabel("IET (%)")
        ax8.set_ylabel("ICR (%)")
        ax8.set_title("(h) Redundancy: IET vs ICR")
        ax8.legend()

        # --- Panel I: Leakage decomposition ---
        ax9 = fig.add_subplot(gs[2, 2])
        labels_leak = ["Voronoi-\ncompliant", "Rational\nleakage", "Irrational\nleakage"]
        compliant_prod = df[df["is_tp"] == 1]["produccion_t"].sum()
        sizes_leak = [compliant_prod, rational_leaked, irrational_leaked]
        colors_leak = ["#4CAF50", "#FFC107", "#F44336"]
        ax9.pie(sizes_leak, labels=labels_leak, colors=colors_leak, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 8})
        ax9.set_title("(i) Waste flow decomposition")

        fig_path = OUTPUT_DIR / "indicator_panels.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {fig_path}")
        plt.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
