# -*- coding: utf-8 -*-
"""
sensibilidad_dotacion.py
========================
Sensitivity sweep of the system-level CDW cost to the per-capita
generation rate (rho_CDW), addressing reviewer comment on Round 1
revision of A01.

Logic (per user reasoning):
    - Transport cost per tonne does NOT depend on rho_CDW: it is
      rho_trans * d_{m,i}, where d is distance. Total t*km scale
      with rho_CDW but the per-tonne component does not.
    - Treatment cost per tonne DOES depend on rho_CDW because plant
      throughput T_i scales linearly with rho_CDW, and the fixed-cost
      curve C0*(log2(T/T0)+1) is concave in T -> C_fix/T decreases
      with T. So system cost has a slight downward trend with rho.
    - Second-order effect (plants becoming more attractive and
      capturing demand from neighbours) is ignored: assignments are
      held fixed at the road-network optimum computed for rho = 0.5.

Range: CEDEX 2014 official range, 0.29 - 0.93 t/inhab/year, sampled
at fine resolution.

Outputs:
    - sensibilidad_dotacion.csv (per-rho aggregate statistics)
    - sensibilidad_dotacion.png (cost-vs-rho curve, for paper)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalcula_costes_red_real import (
    C0, T0, V_TREAT, RHO_TRANS, PLANT_CONSOLIDATION, normalize_name,
    ASIGNACION_REAL, DATOS_MUNICIPIOS, CODIGOS_PLANTAS,
    fixed_cost_total,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_red_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_base_dataframe():
    """Load assignments and population once. Returns a dataframe with
    one row per municipality: population, assigned plant, distance,
    plus the count of physical plants per consolidated group."""
    df_asig = pd.read_csv(ASIGNACION_REAL)
    df_asig["planta_asignada"] = df_asig["planta_asignada"].replace(PLANT_CONSOLIDATION)

    df_mun_raw = pd.read_csv(DATOS_MUNICIPIOS, encoding="utf-8", on_bad_lines="warn")
    pop_lookup = {}
    for _, row in df_mun_raw.iterrows():
        nombre = row.get("NOMBRE", "")
        key = normalize_name(nombre)
        if key and not pd.isna(row.get("HBT")):
            pop_lookup[key] = int(row["HBT"])

    records = []
    for _, row in df_asig.iterrows():
        mun_name = str(row["municipio"]).strip()
        plant_id = int(row["planta_asignada"])
        dist_km = float(row["real_distance"]) / 1000.0
        key = normalize_name(mun_name)
        pop = pop_lookup.get(key)
        if pop is None:
            continue
        records.append({
            "municipio": mun_name,
            "poblacion": pop,
            "planta_id": plant_id,
            "distancia_km": dist_km,
        })
    df = pd.DataFrame(records)

    # Count physical plants per consolidated group (1..46)
    n_phys = Counter()
    for orig_id in range(1, 47):
        canonical = PLANT_CONSOLIDATION.get(orig_id, orig_id)
        n_phys[canonical] += 1
    df["n_plantas_fisicas"] = df["planta_id"].map(lambda p: n_phys.get(p, 1))
    return df


def compute_for_rho(df_base, rho):
    """Recompute system-level costs for a given per-capita generation
    rate. Returns a dict of aggregate statistics."""
    df = df_base.copy()
    df["produccion_t"] = df["poblacion"] * rho

    # Aggregate to plant groups
    plant_agg = df.groupby("planta_id").agg(
        produccion_total=("produccion_t", "sum"),
        n_plantas_fisicas=("n_plantas_fisicas", "first"),
    ).reset_index()

    # Per-physical-plant throughput (split model, matches A01)
    plant_agg["prod_por_planta"] = (
        plant_agg["produccion_total"] / plant_agg["n_plantas_fisicas"]
    )
    plant_agg["C_fix_total"] = (
        plant_agg["prod_por_planta"].apply(fixed_cost_total)
        * plant_agg["n_plantas_fisicas"]
    )
    plant_agg["C_fix_unit"] = (
        plant_agg["C_fix_total"] / plant_agg["produccion_total"]
    )
    plant_agg["C_trat"] = plant_agg["C_fix_unit"] + V_TREAT

    # Map back to municipalities
    trat_by_plant = plant_agg.set_index("planta_id")["C_trat"].to_dict()
    df["C_trans"] = df["distancia_km"] * RHO_TRANS
    df["C_trat"] = df["planta_id"].map(trat_by_plant)
    df["C_tot"] = df["C_trans"] + df["C_trat"]

    # Aggregates (production-weighted at system level)
    prod_total = df["produccion_t"].sum()
    w_trans = (df["C_trans"] * df["produccion_t"]).sum() / prod_total
    w_trat = (df["C_trat"] * df["produccion_t"]).sum() / prod_total
    w_tot = w_trans + w_trat

    return {
        "rho_CDW": rho,
        "prod_total_t": prod_total,
        "C_trans_pond": w_trans,
        "C_trat_pond": w_trat,
        "C_total_pond": w_tot,
        "C_trans_mean_mun": df["C_trans"].mean(),
        "C_trat_mean_mun": df["C_trat"].mean(),
        "C_total_mean_mun": df["C_tot"].mean(),
        "C_total_max_mun": df["C_tot"].max(),
        "C_total_min_mun": df["C_tot"].min(),
        "n_plants_sub_T0": int((plant_agg["prod_por_planta"] < T0).sum()),
    }


def main():
    df_base = build_base_dataframe()
    print(f"Base loaded: {len(df_base)} municipalities, "
          f"{df_base['planta_id'].nunique()} plant groups")

    # CEDEX 2014 official range
    rho_values = np.round(np.arange(0.29, 0.94, 0.02), 3)
    # Ensure key anchor values are present (0.5 base, 0.78 INE 2021 national)
    for anchor in (0.29, 0.50, 0.78, 0.93):
        if anchor not in rho_values:
            rho_values = np.append(rho_values, anchor)
    rho_values = np.sort(np.unique(rho_values))

    results = [compute_for_rho(df_base, rho) for rho in rho_values]
    df_out = pd.DataFrame(results)

    out_csv = os.path.join(OUTPUT_DIR, "sensibilidad_dotacion.csv")
    df_out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved: {out_csv}")

    # Validation against published baseline (rho = 0.5 -> C_total_pond = 12.22)
    base_row = df_out[np.isclose(df_out["rho_CDW"], 0.50)].iloc[0]
    print(f"\nValidation at rho=0.50:")
    print(f"  C_trans_pond  = {base_row['C_trans_pond']:.2f} EUR/t  (expected: ~6.03)")
    print(f"  C_trat_pond   = {base_row['C_trat_pond']:.2f} EUR/t  (expected: ~6.19)")
    print(f"  C_total_pond  = {base_row['C_total_pond']:.2f} EUR/t  (expected: 12.22)")

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rho = df_out["rho_CDW"]
    ax1.plot(rho, df_out["C_total_pond"], color="#1f4e79",
             linewidth=2.5, marker="o", markersize=4, label="Total")
    ax1.plot(rho, df_out["C_trans_pond"], color="#7f7f7f",
             linewidth=1.5, linestyle="--", label="Transport")
    ax1.plot(rho, df_out["C_trat_pond"], color="#c0504d",
             linewidth=1.5, linestyle="-.", label="Treatment")
    ax1.axvline(0.50, color="black", linestyle=":", alpha=0.6)
    ax1.text(0.505, ax1.get_ylim()[1] * 0.95, r"$\rho_{base}=0.5$",
             fontsize=9, verticalalignment="top")
    ax1.set_xlabel(r"Per-capita CDW generation rate, $\rho_{CDW}$ (t/inhab$\cdot$year)")
    ax1.set_ylabel("Production-weighted cost (EUR/t)")
    ax1.set_title("System-level cost vs. generation rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    base_total = base_row["C_total_pond"]
    ax2.plot(rho, 100 * (df_out["C_total_pond"] - base_total) / base_total,
             color="#1f4e79", linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax2.axvline(0.50, color="black", linestyle=":", alpha=0.6)
    ax2.set_xlabel(r"Per-capita CDW generation rate, $\rho_{CDW}$ (t/inhab$\cdot$year)")
    ax2.set_ylabel(r"Relative variation of $\bar{C}$ vs. base (%)")
    ax2.set_title("Robustness of the production-weighted mean")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "Latex", "Imagenes", "sensibilidad_dotacion.png",
    )
    out_png = os.path.normpath(out_png)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # ---- Summary text ----
    extremes = df_out.iloc[[0, -1]]
    print("\nSummary across CEDEX range (0.29 to 0.93):")
    for _, r in extremes.iterrows():
        print(f"  rho={r['rho_CDW']:.2f}: "
              f"C_total = {r['C_total_pond']:.2f} EUR/t  "
              f"(C_trat = {r['C_trat_pond']:.2f}, sub-T0 plants = {r['n_plants_sub_T0']:.0f})")
    rel_range = 100 * (df_out["C_total_pond"].max() - df_out["C_total_pond"].min()) / base_total
    print(f"  Total spread across the range: {rel_range:.1f}% of base value")


if __name__ == "__main__":
    main()
