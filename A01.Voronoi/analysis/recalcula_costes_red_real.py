# -*- coding: utf-8 -*-
"""
recalcula_costes_red_real.py
============================
Recalculates CDW management costs using real road-distance assignments
(validated in A00) and the correct cost model:

    C₀ = 40,000 €/year   (1 operator + basic machinery)
    T₀ = 5,000 t/year    (250 days × 20 t/day)
    log base 2            (doubling rule)
    v  = 0.35 €/t         (variable treatment: energy + consumables)
    ρ  = 0.35 €/t·km      (transport, MITMA 2025 + CDW factors)
    Generation = 0.5 t/inhab/year

Inputs:
    - asignacion_municipios_real.csv      (municipality → plant by road distance)
    - datos_voronoi_municipios.csv        (population, area, province per municipality)
    - tablas_distancias/codigos_plantas.csv (plant ID → name mapping)

Outputs:
    - datos_red_real_plantas.csv          (plant-level aggregated costs)
    - datos_red_real_municipios.csv       (municipality-level costs)
    - resumen_recalculo.txt               (summary statistics)
"""

import pandas as pd
import numpy as np
import math
import os
import unicodedata
import re

# ============================================================
# PARAMETERS
# ============================================================
C0 = 40_000       # Base fixed cost (€/year)
T0 = 5_000        # Scale threshold (t/year)
V_TREAT = 0.35    # Variable treatment cost (€/t)
RHO_TRANS = 0.35  # Transport cost (€/t·km)
DOTACION = 0.5    # CDW generation rate (t/inhab/year)

# ============================================================
# PLANT CONSOLIDATION (threshold 15 km)
# ============================================================
# Plants within 15 km of each other are treated as a single facility.
# This reflects that transport cost is practically indifferent between
# co-located or very proximate plants, and avoids spurious Voronoi
# boundary effects at plant cluster vertices.
# Groups (canonical_id <- absorbed_ids):
#   Ribera del Fresno: 1 <- {2, 12}
#   Don Benito:        3 <- {17}
#   Quintana:          6 <- {15}
#   Trujillo:         19 <- {40}
#   Moraleja:         20 <- {32}
#   Cabezuela:        22 <- {23, 26, 28}
#   Navalmoral:       27 <- {34, 35, 37}
#   Escurial:         36 <- {38, 43}
PLANT_CONSOLIDATION = {
    2: 1, 12: 1,           # Ribera del Fresno cluster
    17: 3,                  # Don Benito + Villanueva de la Serena
    15: 6,                  # Quintana de la Serena + Castuera
    40: 19,                 # Trujillo (2 plants, same town)
    32: 20,                 # Moraleja + Coria (ARAPLASA)
    23: 22, 26: 22, 28: 22, # Cabezuela + Aldeanueva + Jaraíz + Jarandilla (ARAPLASA)
    34: 27, 35: 27, 37: 27, # Navalmoral + Millanes + Casatejada + Almaraz
    38: 36, 43: 36,         # Escurial + Miajadas (2 plants)
}

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASIGNACION_REAL = os.path.join(SCRIPT_DIR, "asignacion_municipios_real.csv")
DATOS_MUNICIPIOS = os.path.join(SCRIPT_DIR, "datos_voronoi_municipios.csv")
CODIGOS_PLANTAS = os.path.join(SCRIPT_DIR, "tablas_distancias", "codigos_plantas.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datos_red_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COST MODEL FUNCTIONS
# ============================================================
def fixed_cost_total(T):
    """Annual fixed cost (€/year) using log₂ doubling rule."""
    if T <= 0:
        return 0.0
    if T < T0:
        return C0
    return C0 * (math.log2(T / T0) + 1)


def fixed_cost_per_tonne(T):
    """Fixed cost per tonne (€/t)."""
    if T <= 0:
        return 0.0
    return fixed_cost_total(T) / T


def treatment_cost_per_tonne(T):
    """Total treatment cost per tonne: fixed + variable."""
    return fixed_cost_per_tonne(T) + V_TREAT


def transport_cost_per_tonne(distance_km):
    """Transport cost per tonne (€/t) for a given distance."""
    return distance_km * RHO_TRANS


# ============================================================
# NAME NORMALIZATION (to handle encoding mismatches)
# ============================================================
def normalize_name(name):
    """Normalize municipality name for matching across CSVs with different encodings."""
    if pd.isna(name):
        return ""
    s = str(name).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    # Strip everything non-ASCII (handles corrupted encoding in QGIS exports)
    s = s.encode("ascii", errors="ignore").decode("ascii")
    # Normalize articles: ". La" / ", La" / ". Los" etc → standardize
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", r" \1", s, flags=re.IGNORECASE)
    # Lowercase, collapse whitespace, strip punctuation
    s = re.sub(r"[^a-z0-9\s]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("RECALCULO DE COSTES CON ASIGNACION POR DISTANCIA REAL")
print("=" * 70)
print(f"Parametros: C0={C0}, T0={T0}, log2, v={V_TREAT}, rho={RHO_TRANS}")
print()

# 1. Real-distance assignments (+ consolidation)
df_asig = pd.read_csv(ASIGNACION_REAL)
df_asig["planta_asignada"] = df_asig["planta_asignada"].replace(PLANT_CONSOLIDATION)
n_consolidated = sum(1 for v in PLANT_CONSOLIDATION.values() if v != v)  # just count keys
print(f"Asignaciones por distancia real: {len(df_asig)} municipios")
print(f"Plantas consolidadas (umbral 15 km): {len(PLANT_CONSOLIDATION)} absorbidas -> "
      f"{df_asig['planta_asignada'].nunique()} plantas efectivas")

# Also load euclidean assignments for misallocation comparison
df_asig_euc = pd.read_csv(os.path.join(SCRIPT_DIR, "asignacion_municipios_euclidiana.csv"))
df_asig_euc["planta_asignada"] = df_asig_euc["planta_asignada"].replace(PLANT_CONSOLIDATION)
_comp = df_asig.merge(df_asig_euc[["municipio", "planta_asignada"]], on="municipio", suffixes=("_real", "_euc"))
_misalloc = (_comp["planta_asignada_real"] != _comp["planta_asignada_euc"]).sum()
print(f"Misallocaciones (Voronoi vs red real, consolidado): {_misalloc}/{len(_comp)} "
      f"({100*_misalloc/len(_comp):.1f}%)")

# 2. Plant codes
df_plantas_cod = pd.read_csv(CODIGOS_PLANTAS)
# Build plant ID → info mapping
plantas_info = {}
for _, row in df_plantas_cod.iterrows():
    pid = int(row["Id"])
    plantas_info[pid] = {
        "denominacion": str(row["DENOMINACI"]).replace("\n", " ").strip(),
        "municipio_planta": str(row["MUNICIPIO"]).replace("\n", " ").strip(),
        "nombre": str(row["Nombre"]).replace("\n", " ").strip(),
    }
print(f"Plantas registradas: {len(plantas_info)}")

# 3. Municipality population data
df_mun_raw = pd.read_csv(DATOS_MUNICIPIOS, encoding="utf-8", on_bad_lines="warn")
print(f"Datos municipales cargados: {len(df_mun_raw)} filas")

# Build population lookup by normalized name
pop_lookup = {}
area_lookup = {}
density_lookup = {}
province_lookup = {}

for _, row in df_mun_raw.iterrows():
    nombre = row.get("NOMBRE", "")
    key = normalize_name(nombre)
    if key and not pd.isna(row.get("HBT")):
        pop_lookup[key] = int(row["HBT"])
        area_lookup[key] = float(row.get("KM2", 0))
        density_lookup[key] = float(row.get("Densidad", 0))
        province_lookup[key] = str(row.get("PROVIN", ""))

print(f"Poblacion disponible para {len(pop_lookup)} municipios")
print()

# ============================================================
# MERGE: ASSIGNMENT + POPULATION
# ============================================================
records = []
unmatched = []

for _, row in df_asig.iterrows():
    mun_name = str(row["municipio"]).strip()
    plant_id = int(row["planta_asignada"])
    real_dist_m = float(row["real_distance"])
    real_dist_km = real_dist_m / 1000.0

    key = normalize_name(mun_name)
    pop = pop_lookup.get(key)

    if pop is None:
        unmatched.append(mun_name)
        continue

    production = pop * DOTACION
    tkm = production * real_dist_km

    plant_info = plantas_info.get(plant_id, {})

    records.append({
        "municipio": mun_name,
        "poblacion": pop,
        "km2": area_lookup.get(key, 0),
        "densidad": density_lookup.get(key, 0),
        "provincia": province_lookup.get(key, ""),
        "planta_id": plant_id,
        "planta_nombre": plant_info.get("denominacion", f"Planta_{plant_id}"),
        "planta_municipio": plant_info.get("municipio_planta", ""),
        "distancia_km": real_dist_km,
        "produccion_t": production,
        "tkm": tkm,
    })

df = pd.DataFrame(records)
print(f"Municipios procesados: {len(df)}")
if unmatched:
    print(f"AVISO: {len(unmatched)} municipios sin datos de poblacion:")
    for m in unmatched[:10]:
        print(f"  - {m}")
    if len(unmatched) > 10:
        print(f"  ... y {len(unmatched) - 10} mas")
print()

# ============================================================
# PLANT-LEVEL AGGREGATION
# ============================================================
print("Calculando costes a nivel de planta...")

# Count physical plants per consolidated group
# (inverse of PLANT_CONSOLIDATION: how many original IDs map to each canonical)
from collections import Counter
_canonical_counts = Counter()
# All 46 original plant IDs
all_original_ids = set(range(1, 47))
for orig_id in all_original_ids:
    canonical = PLANT_CONSOLIDATION.get(orig_id, orig_id)
    _canonical_counts[canonical] += 1

plant_agg = df.groupby("planta_id").agg(
    planta_nombre=("planta_nombre", "first"),
    planta_municipio=("planta_municipio", "first"),
    n_municipios=("municipio", "count"),
    poblacion_total=("poblacion", "sum"),
    produccion_total=("produccion_t", "sum"),
    tkm_total=("tkm", "sum"),
    superficie_total=("km2", "sum"),
).reset_index()

# Number of physical plants in each consolidated group
plant_agg["n_plantas_fisicas"] = plant_agg["planta_id"].map(_canonical_counts).fillna(1).astype(int)

# Treatment cost: throughput is SPLIT among physical plants in the group
# Each physical plant processes produccion_total / n_plantas_fisicas
plant_agg["produccion_por_planta"] = plant_agg["produccion_total"] / plant_agg["n_plantas_fisicas"]

# Calculate costs
plant_agg["densidad_media"] = plant_agg["poblacion_total"] / plant_agg["superficie_total"].replace(0, np.nan)
plant_agg["C_trans"] = (plant_agg["tkm_total"] * RHO_TRANS) / plant_agg["produccion_total"].replace(0, np.nan)
# Treatment cost uses per-physical-plant throughput (reflects actual splitting)
plant_agg["C_fix_total"] = plant_agg["produccion_por_planta"].apply(fixed_cost_total) * plant_agg["n_plantas_fisicas"]
plant_agg["C_fix_unit"] = plant_agg["C_fix_total"] / plant_agg["produccion_total"]
plant_agg["C_trat"] = plant_agg["C_fix_unit"] + V_TREAT
plant_agg["C_tot"] = plant_agg["C_trans"] + plant_agg["C_trat"]

# SCENARIO: if each group were rationalized to ONE physical plant
plant_agg["C_trat_racional"] = plant_agg["produccion_total"].apply(treatment_cost_per_tonne)
plant_agg["C_tot_racional"] = plant_agg["C_trans"] + plant_agg["C_trat_racional"]
plant_agg["ahorro_racional"] = plant_agg["C_tot"] - plant_agg["C_tot_racional"]

plant_agg = plant_agg.sort_values("produccion_total", ascending=False)

# Save plant-level results
plant_output = os.path.join(OUTPUT_DIR, "datos_red_real_plantas.csv")
plant_agg.to_csv(plant_output, index=False, float_format="%.2f")
print(f"Guardado: {plant_output}")

# ============================================================
# MUNICIPALITY-LEVEL COSTS
# ============================================================
print("Calculando costes a nivel de municipio...")

# Each municipality gets:
# - Transport cost = its distance × ρ_trans
# - Treatment cost = the treatment cost of its assigned plant
plant_treatment = plant_agg.set_index("planta_id")["C_trat"].to_dict()
plant_fix_unit = plant_agg.set_index("planta_id")["C_fix_unit"].to_dict()

df["C_trans"] = df["distancia_km"] * RHO_TRANS
df["C_trat"] = df["planta_id"].map(plant_treatment)
df["C_fix_unit"] = df["planta_id"].map(plant_fix_unit)
df["C_tot"] = df["C_trans"] + df["C_trat"]

# Sort by total cost descending
df_mun_out = df.sort_values("C_tot", ascending=False)

mun_output = os.path.join(OUTPUT_DIR, "datos_red_real_municipios.csv")
df_mun_out.to_csv(mun_output, index=False, float_format="%.2f")
print(f"Guardado: {mun_output}")

# ============================================================
# COMPARISON: Voronoi vs Real assignment
# ============================================================
print()
print("=" * 70)
print("COMPARACION CON ASIGNACION VORONOI (datos anteriores)")
print("=" * 70)

try:
    df_voronoi = pd.read_csv(os.path.join(SCRIPT_DIR, "datos_voronoi_municipios.csv"),
                             encoding="utf-8", on_bad_lines="warn")
    # Normalize and match
    df_voronoi["_key"] = df_voronoi["NOMBRE"].apply(normalize_name)
    df_mun_out["_key"] = df_mun_out["municipio"].apply(normalize_name)

    merged = df_mun_out.merge(
        df_voronoi[["_key", "C_Trans", "C_Trat", "C_Tot", "Planta"]],
        on="_key", how="inner", suffixes=("_real", "_voronoi")
    )

    n_match = len(merged)
    print(f"Municipios comparados: {n_match}")

    if n_match > 0:
        # Count plant assignment changes
        merged["planta_voronoi_norm"] = merged["Planta"].apply(
            lambda x: normalize_name(str(x)) if pd.notna(x) else "")
        merged["planta_real_norm"] = merged["planta_nombre"].apply(
            lambda x: normalize_name(str(x)) if pd.notna(x) else "")
        cambios = (merged["planta_voronoi_norm"] != merged["planta_real_norm"]).sum()
        print(f"Municipios que cambian de planta: {cambios} ({100*cambios/n_match:.1f}%)")

        # Cost comparison
        print(f"\nCostes medios (€/t):")
        print(f"  {'':25s} {'Voronoi':>10s} {'Red real':>10s} {'Diferencia':>10s}")
        for label, col_v, col_r in [
            ("Transporte", "C_Trans", "C_trans"),
            ("Tratamiento", "C_Trat", "C_trat"),
            ("Total", "C_Tot", "C_tot"),
        ]:
            mean_v = merged[col_v].mean()
            mean_r = merged[col_r].mean()
            print(f"  {label:25s} {mean_v:10.2f} {mean_r:10.2f} {mean_r - mean_v:+10.2f}")

except Exception as e:
    print(f"No se pudo comparar con datos Voronoi: {e}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print()
print("=" * 70)
print("RESUMEN ESTADISTICO - DATOS RECALCULADOS (RED REAL)")
print("=" * 70)

print(f"\n--- NIVEL PLANTA ({len(plant_agg)} plantas con municipios asignados) ---")
print(f"  Produccion total sistema: {plant_agg['produccion_total'].sum():,.0f} t/año")
print(f"  t·km totales sistema:     {plant_agg['tkm_total'].sum():,.0f} t·km/año")
print(f"  Rango produccion/planta:  {plant_agg['produccion_total'].min():,.0f} - {plant_agg['produccion_total'].max():,.0f} t/año")
print(f"  Plantas < T0 (5000 t):    {(plant_agg['produccion_total'] < T0).sum()}")
print(f"  Plantas >= T0:            {(plant_agg['produccion_total'] >= T0).sum()}")
print(f"\n  Costes por planta (€/t):  {'Media':>8s} {'Mediana':>8s} {'Min':>8s} {'Max':>8s}")
for label, col in [("C_trans", "C_trans"), ("C_trat", "C_trat"), ("C_tot", "C_tot")]:
    s = plant_agg[col].dropna()
    print(f"    {label:20s}  {s.mean():8.2f} {s.median():8.2f} {s.min():8.2f} {s.max():8.2f}")

print(f"\n--- NIVEL MUNICIPIO ({len(df_mun_out)} municipios) ---")
print(f"  Costes por municipio (€/t): {'Media':>8s} {'Mediana':>8s} {'Min':>8s} {'Max':>8s}")
for label, col in [("C_trans", "C_trans"), ("C_trat", "C_trat"), ("C_tot", "C_tot")]:
    s = df_mun_out[col].dropna()
    print(f"    {label:20s}  {s.mean():8.2f} {s.median():8.2f} {s.min():8.2f} {s.max():8.2f}")

# ============================================================
# RATIONALIZATION ANALYSIS
# ============================================================
print()
print("=" * 70)
print("ANALISIS DE RACIONALIZACION: EFECTO DE CONSOLIDAR PLANTAS FISICAS")
print("=" * 70)
print()

multi_plant = plant_agg[plant_agg["n_plantas_fisicas"] > 1].copy()
single_plant = plant_agg[plant_agg["n_plantas_fisicas"] == 1].copy()

print(f"Grupos con multiples plantas fisicas: {len(multi_plant)}")
print(f"Plantas individuales: {len(single_plant)}")
print()

if len(multi_plant) > 0:
    print(f"{'Grupo':<28s} {'N':>3s} {'Prod':>8s} {'Prod/pl':>8s} "
          f"{'C_trat':>7s} {'C_trat*':>7s} {'Ahorro':>7s}")
    print("-" * 80)
    total_ahorro_ponderado = 0
    total_prod_multi = 0
    for _, row in multi_plant.sort_values("ahorro_racional", ascending=False).iterrows():
        prod_pp = row["produccion_por_planta"]
        print(f"  {row['planta_municipio']:<26s} {row['n_plantas_fisicas']:3.0f} "
              f"{row['produccion_total']:8,.0f} {prod_pp:8,.0f} "
              f"{row['C_trat']:7.2f} {row['C_trat_racional']:7.2f} "
              f"{row['ahorro_racional']:+7.2f}")
        total_ahorro_ponderado += row["ahorro_racional"] * row["produccion_total"]
        total_prod_multi += row["produccion_total"]

    print()
    prod_total = plant_agg["produccion_total"].sum()

    # System-wide costs: current vs rationalized
    coste_total_actual = (plant_agg["C_tot"] * plant_agg["produccion_total"]).sum()
    coste_total_racional = (plant_agg["C_tot_racional"] * plant_agg["produccion_total"]).sum()
    ahorro_total = coste_total_actual - coste_total_racional

    # Number of plants that could be closed
    plantas_eliminables = multi_plant["n_plantas_fisicas"].sum() - len(multi_plant)

    print(f"  C_trat = coste actual (produccion repartida entre plantas fisicas)")
    print(f"  C_trat* = coste si cada grupo operase como una unica planta")
    print()
    print(f"  Plantas fisicas eliminables: {plantas_eliminables:.0f} "
          f"(de {plant_agg['n_plantas_fisicas'].sum():.0f} actuales a {len(plant_agg)} consolidadas)")
    print(f"  Coste total actual del sistema:       {coste_total_actual:>12,.0f} EUR/anyo")
    print(f"  Coste total racionalizado:            {coste_total_racional:>12,.0f} EUR/anyo")
    print(f"  Ahorro potencial por racionalizacion: {ahorro_total:>12,.0f} EUR/anyo ({100*ahorro_total/coste_total_actual:.1f}%)")
    print(f"  Coste medio actual:        {coste_total_actual/prod_total:.2f} EUR/t")
    print(f"  Coste medio racionalizado: {coste_total_racional/prod_total:.2f} EUR/t")

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "resumen_recalculo.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("RECALCULO DE COSTES CDW - ASIGNACION POR RED REAL\n")
    f.write("=" * 60 + "\n")
    f.write(f"Parametros: C0={C0}, T0={T0}, log2, v={V_TREAT}, rho={RHO_TRANS}\n")
    f.write(f"Consolidacion: umbral 15 km, {len(PLANT_CONSOLIDATION)} plantas absorbidas\n")
    f.write(f"Dotacion: {DOTACION} t/hab/anyo\n\n")
    f.write(f"Municipios procesados: {len(df)}\n")
    f.write(f"Plantas consolidadas (grupos): {len(plant_agg)}\n")
    f.write(f"Plantas fisicas totales: {plant_agg['n_plantas_fisicas'].sum():.0f}\n")
    f.write(f"Misallocaciones (consolidado): {_misalloc}/{len(_comp)} ({100*_misalloc/len(_comp):.1f}%)\n")
    f.write(f"Produccion total: {plant_agg['produccion_total'].sum():,.0f} t/anyo\n\n")
    f.write("COSTES MEDIOS POR MUNICIPIO (EUR/t):\n")
    f.write(f"  Transporte:  {df_mun_out['C_trans'].mean():.2f}\n")
    f.write(f"  Tratamiento: {df_mun_out['C_trat'].mean():.2f}\n")
    f.write(f"  Total:       {df_mun_out['C_tot'].mean():.2f}\n")
    if len(multi_plant) > 0:
        f.write(f"\nRACIONALIZACION:\n")
        f.write(f"  Plantas eliminables: {plantas_eliminables:.0f}\n")
        f.write(f"  Ahorro potencial: {ahorro_total:,.0f} EUR/anyo ({100*ahorro_total/coste_total_actual:.1f}%)\n")
        f.write(f"  Coste medio actual:        {coste_total_actual/prod_total:.2f} EUR/t\n")
        f.write(f"  Coste medio racionalizado: {coste_total_racional/prod_total:.2f} EUR/t\n")
print(f"\nResumen guardado: {summary_path}")
print("\nRecalculo completado.")
