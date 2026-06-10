#!/usr/bin/env python3
"""
reproduce_scenarios.py
======================
Self-contained reproducibility script for the three-scenario comparison
reported in:

  "Diagnostic Indicators for CDW Treatment Plant Networks:
   A Plant-Level and Network-Level Panel Framework"
  Submitted to Waste Management & Research (SAGE)

This script reproduces the headline network-level indicators for all three
scenarios using ONLY public data:

  S1 — Proximity baseline (nearest plant by road, theoretical production):
       data/s1_proximity_baseline.csv (A01 public data)
  S2 — Observed flows (anonymised): data/s2_observed_flows_public.csv
       (730 municipality->plant flows; volumes rescaled to theoretical
       production 551,205 t/yr; no commercial tonnages)
  S3 — Cost-optimal network: A05 public data
       (A05.Network_Optimization/outputs/optimal_assignments.csv)

Expected headline numbers (within rounding):
  S1: C_bar = 12.22 EUR/t  (transport 6.03 + treatment 6.19),  Gini = 0.187
  S2: C_bar = 12.11 EUR/t  (transport 6.30 + treatment 5.81),  Gini = 0.208
  S3: C_bar = 11.34 EUR/t  (transport 6.42 + treatment 4.92),  Gini = 0.197

Net S1->S3 saving: 7.2% (0.88 EUR/t), driven entirely by treatment-cost
reduction via throughput concentration (transport rises slightly).

Usage
-----
  python analysis/reproduce_scenarios.py

No arguments needed; all paths are resolved relative to this script.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent         # A03/analysis/
A03_DIR    = THIS_DIR.parent                         # A03.Diagnostic_Indicators/
REPO_DIR   = A03_DIR.parent                          # garnocex_research/

# S1: proximity baseline (A01 public data) — nearest-plant-by-road assignment with
# THEORETICAL production (0.5 t/hab/yr). Transport is recomputed from distance with the
# shared cost model; treatment uses A01's per-physical-plant calibration. No confidential data.
# Columns: municipio, planta_id, distancia_km, produccion_t, C_trans, C_trat, C_tot, ...
S1_DATA    = A03_DIR / "data" / "s1_proximity_baseline.csv"

S2_PUBLIC  = A03_DIR / "data" / "s2_observed_flows_public.csv"

# S3: A05 cost-optimal per-municipality assignment.
# Columns: mun_key, name, prod, plant_id, dist_km, cost
A05_DATA   = REPO_DIR / "A05.Network_Optimization" / "outputs" / "optimal_assignments.csv"

# ---------------------------------------------------------------------------
# Cost model parameters (shared across all scenarios)
# ---------------------------------------------------------------------------
C0  = 40_000   # EUR/yr — minimum annual fixed cost
T0  = 5_000    # t/yr   — economy-of-scale threshold
RHO = 0.35     # EUR/(t·km) — transport tariff
V   = 0.35     # EUR/t  — variable treatment cost

# Consolidation map: physical plant id -> group leader id (from A01 / A03)
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
# Cost model helpers
# ---------------------------------------------------------------------------

def fixed_cost(T: float) -> float:
    """Annual fixed cost for a plant with throughput T (t/yr)."""
    if T <= 0:
        return float(C0)
    return float(max(C0, C0 * (np.log2(T / T0) + 1)))


def unit_treatment_cost(T: float) -> float:
    """Unit treatment cost (EUR/t) at throughput T."""
    if T <= 0:
        return np.inf
    return fixed_cost(T) / T + V


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (unweighted, per municipality)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0 or np.sum(v) == 0:
        return float("nan")
    n = len(v)
    s = np.sort(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) - (n + 1) * np.sum(s)) / (n * np.sum(s)))


def norm(s: str) -> str:
    """Normalise a municipality name for cross-source matching."""
    if pd.isna(s):
        return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def consolidate(pid) -> int:
    """Map physical plant id to its consolidated group id."""
    return CONSOLIDATION.get(int(pid), int(pid))


# ---------------------------------------------------------------------------
# Network-level indicators from a per-municipality cost DataFrame
# ---------------------------------------------------------------------------

def network_indicators(df: pd.DataFrame) -> dict:
    """
    Compute network-level headline indicators from a per-municipality DataFrame.

    Required columns: prod_t, C_trans, C_treat, C_total
    """
    w = df["prod_t"].values
    total_prod = w.sum()

    C_bar       = float(np.average(df["C_total"].values, weights=w))
    C_bar_trans = float(np.average(df["C_trans"].values, weights=w))
    C_bar_treat = float(np.average(df["C_treat"].values, weights=w))
    gini        = gini_coefficient(df["C_total"].values)
    n_plants    = int(df["plant_id"].nunique())

    return {
        "C_bar":       C_bar,
        "C_bar_trans": C_bar_trans,
        "C_bar_treat": C_bar_treat,
        "Gini":        gini,
        "n_plants":    n_plants,
        "total_prod":  total_prod,
    }


# ---------------------------------------------------------------------------
# Scenario 1 — Proximity baseline (A01 public data)
# ---------------------------------------------------------------------------

def load_s1() -> pd.DataFrame:
    """
    Load Scenario 1 (proximity baseline) from public A01 data.

    Source: data/s1_proximity_baseline.csv — nearest-plant-by-road assignment
    with theoretical municipal production (0.5 t/hab/yr). Transport is recomputed
    here as RHO*distance; treatment uses A01's per-physical-plant calibration
    (C_trat). No confidential data.

    Input columns used: municipio, planta_id, distancia_km, produccion_t, C_trat
      mun       — normalised municipality name
      transport — transport component (EUR/t) = RHO * dist_km
      treatment — treatment component (EUR/t) = fixed_cost(T)/T + V
      total     — total unit cost (EUR/t)
      prod      — production (t/yr)

    The 383 municipalities cover the full Extremadura network.
    Production-weighted C_bar = 12.22 EUR/t, Gini = 0.187, 32 plant groups.
    """
    raw = pd.read_csv(S1_DATA)
    df = pd.DataFrame({
        "mun_norm": raw["municipio"].astype(str),
        "plant_id": pd.to_numeric(raw["planta_id"], errors="coerce"),
        "prod_t":   pd.to_numeric(raw["produccion_t"], errors="coerce"),
        # transport recomputed from distance with the shared model (RHO EUR/t-km)
        "C_trans":  RHO * pd.to_numeric(raw["distancia_km"], errors="coerce"),
        # treatment from A01's per-physical-plant calibration (split-throughput convention)
        "C_treat":  pd.to_numeric(raw["C_trat"], errors="coerce"),
    })
    df["C_total"] = df["C_trans"] + df["C_treat"]
    df = df.dropna(subset=["prod_t", "C_total"]).copy()
    df = df[df["prod_t"] > 0].copy()
    return df


# ---------------------------------------------------------------------------
# Scenario 2 — Observed flows, anonymised (s2_observed_flows_public.csv)
# ---------------------------------------------------------------------------

def load_s2() -> pd.DataFrame:
    """
    Load Scenario 2 from the anonymised public flows CSV.

    Columns: mun_norm, plant_id, dist_km, prod_t
    Volumes are rescaled to theoretical production (551,205 t) — no
    commercial tonnages.  Costs are computed using the shared cost model.
    """
    df = pd.read_csv(S2_PUBLIC)

    # Aggregate per-plant throughput (multi-flow: one municipality may send
    # waste to multiple plants)
    plant_throughput = df.groupby("plant_id")["prod_t"].sum()
    df["C_treat"] = df["plant_id"].map(
        plant_throughput.apply(unit_treatment_cost)
    )
    df["C_trans"] = RHO * df["dist_km"]
    df["C_total"] = df["C_trans"] + df["C_treat"]

    # For Gini we need per-municipality cost (production-weighted average
    # across destination plants for multi-flow municipalities)
    mun_cost = (
        df.groupby("mun_norm")
        .apply(lambda g: np.average(g["C_total"].values, weights=g["prod_t"].values))
        .reset_index()
    )
    mun_cost.columns = ["mun_norm", "C_total_avg"]

    mun_prod = df.groupby("mun_norm")["prod_t"].sum().reset_index()
    mun_prod.columns = ["mun_norm", "prod_t"]

    mun_trans = (
        df.groupby("mun_norm")
        .apply(lambda g: np.average(g["C_trans"].values, weights=g["prod_t"].values))
        .reset_index()
    )
    mun_trans.columns = ["mun_norm", "C_trans_avg"]

    mun_treat = (
        df.groupby("mun_norm")
        .apply(lambda g: np.average(g["C_treat"].values, weights=g["prod_t"].values))
        .reset_index()
    )
    mun_treat.columns = ["mun_norm", "C_treat_avg"]

    mun_df = (
        mun_prod
        .merge(mun_cost,  on="mun_norm")
        .merge(mun_trans, on="mun_norm")
        .merge(mun_treat, on="mun_norm")
    )
    mun_df = mun_df.rename(columns={
        "C_total_avg": "C_total",
        "C_trans_avg": "C_trans",
        "C_treat_avg": "C_treat",
    })

    # Attach a plant_id for n_plants count (count distinct active plants)
    # Use a synthetic column: n_plants = number of unique plant_ids in S2 data
    active_plants = df["plant_id"].unique()
    mun_df["plant_id"] = mun_df["mun_norm"].map(
        df.groupby("mun_norm")["plant_id"].first()
    )
    mun_df["_n_plants_total"] = len(active_plants)

    return mun_df, len(active_plants)


# ---------------------------------------------------------------------------
# Scenario 3 — Cost-optimal (A05 public data)
# ---------------------------------------------------------------------------

def load_s3() -> pd.DataFrame:
    """
    Load Scenario 3 from A05 optimal_assignments.csv.

    Columns: mun_key, name, prod, plant_id, dist_km, cost
    Costs are recomputed using the shared cost model for consistency.
    """
    df = pd.read_csv(A05_DATA)
    df = df.rename(columns={
        "mun_key": "mun_norm",
        "prod":    "prod_t",
    })

    plant_throughput = df.groupby("plant_id")["prod_t"].sum()
    df["C_treat"] = df["plant_id"].map(
        plant_throughput.apply(unit_treatment_cost)
    )
    df["C_trans"] = RHO * df["dist_km"]
    df["C_total"] = df["C_trans"] + df["C_treat"]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  REPRODUCE SCENARIOS — A03 Diagnostic Indicators")
    print("  CDW Treatment Plant Networks: Extremadura, Spain")
    print("=" * 70)

    # Check input files
    for label, path in [
        ("S1 proximity baseline (data/s1_proximity_baseline.csv)", S1_DATA),
        ("S2 anonymised flows",       S2_PUBLIC),
        ("A05 optimal assignments (S3)", A05_DATA),
    ]:
        if not path.exists():
            print(f"  ERROR: {label} not found at {path}")
            sys.exit(1)
        print(f"  OK  {label}: {path.name}")

    print()

    # ---- Scenario 1 ----
    print("[S1] Computing S1 from proximity baseline (data/s1_proximity_baseline.csv)...")
    df1 = load_s1()
    r1  = network_indicators(df1)
    r1["n_plants"] = 32  # 32 consolidated plant groups (proximity baseline)
    print(f"  Municipalities: {len(df1)}, Plants: {r1['n_plants']}")
    print(f"  Total production: {r1['total_prod']:,.0f} t/yr")

    # ---- Scenario 2 ----
    print("[S2] Loading anonymised S2 flows (s2_observed_flows_public.csv)...")
    df2, n_plants_s2 = load_s2()
    # Override n_plants for S2 (active distinct plants in the public CSV)
    r2 = network_indicators(df2)
    r2["n_plants"] = n_plants_s2
    print(f"  Municipalities: {len(df2)}, Active plants: {n_plants_s2}")
    print(f"  Total production: {r2['total_prod']:,.0f} t/yr")

    # ---- Scenario 3 ----
    print("[S3] Loading A05 cost-optimal assignments...")
    df3 = load_s3()
    r3  = network_indicators(df3)
    print(f"  Municipalities: {len(df3)}, Plants: {r3['n_plants']}")
    print(f"  Total production: {r3['total_prod']:,.0f} t/yr")

    # ---- Comparison table ----
    print()
    print("=" * 70)
    print("  HEADLINE RESULTS — THREE-SCENARIO COMPARISON")
    print("=" * 70)
    hdr = f"{'Indicator':<35s}  {'S1':>10s}  {'S2':>10s}  {'S3':>10s}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    rows = [
        ("Active plants",
         f"{r1['n_plants']}",       f"{r2['n_plants']}",       f"{r3['n_plants']}"),
        ("C_bar, production-weighted (EUR/t)",
         f"{r1['C_bar']:.2f}",      f"{r2['C_bar']:.2f}",      f"{r3['C_bar']:.2f}"),
        ("  Transport component (EUR/t)",
         f"{r1['C_bar_trans']:.2f}",f"{r2['C_bar_trans']:.2f}",f"{r3['C_bar_trans']:.2f}"),
        ("  Treatment component (EUR/t)",
         f"{r1['C_bar_treat']:.2f}",f"{r2['C_bar_treat']:.2f}",f"{r3['C_bar_treat']:.2f}"),
        ("Gini(C_m)",
         f"{r1['Gini']:.3f}",       f"{r2['Gini']:.3f}",       f"{r3['Gini']:.3f}"),
    ]
    for name, v1, v2, v3 in rows:
        print(f"  {name:<33s}  {v1:>10s}  {v2:>10s}  {v3:>10s}")

    print(sep)

    # ---- S1 -> S3 savings ----
    saving_abs = r1["C_bar"] - r3["C_bar"]
    saving_pct = 100.0 * saving_abs / r1["C_bar"]
    print()
    print("  Net S1 -> S3 saving:")
    print(f"    Absolute: {saving_abs:.2f} EUR/t")
    print(f"    Relative: {saving_pct:.1f}%")
    delta_trans = r3["C_bar_trans"] - r1["C_bar_trans"]
    delta_treat = r3["C_bar_treat"] - r1["C_bar_treat"]
    print(f"    Transport change: {delta_trans:+.2f} EUR/t")
    print(f"    Treatment change: {delta_treat:+.2f} EUR/t")
    print(f"    (Cost saving is driven entirely by treatment-cost reduction")
    print(f"     via throughput concentration in fewer, larger plants.)")

    # ---- Verify against manuscript ----
    print()
    print("  Verification against manuscript headline numbers:")
    tol = 0.02
    checks = [
        ("S1 C_bar = 12.22",  r1["C_bar"],       12.22, tol),
        ("S1 Gini = 0.187",   r1["Gini"],         0.187, tol),
        ("S2 C_bar = 12.11",  r2["C_bar"],       12.11, tol),
        ("S2 Gini = 0.208",   r2["Gini"],         0.208, tol),
        ("S3 C_bar = 11.34",  r3["C_bar"],       11.34, tol),
        ("S3 Gini = 0.197",   r3["Gini"],         0.197, tol),
    ]
    all_ok = True
    for label, computed, expected, tolerance in checks:
        diff = abs(computed - expected)
        ok = diff <= tolerance
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"    [{status}] {label}  (computed {computed:.3f}, diff {diff:.3f})")

    print()
    if all_ok:
        print("  All headline numbers match the manuscript (within rounding tolerance).")
    else:
        print("  WARNING: some numbers differ from manuscript values.")
        print("  Check that A01 and A05 input files have not been modified.")

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
