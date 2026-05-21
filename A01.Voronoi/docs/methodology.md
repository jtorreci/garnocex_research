# Methodology — A01 Spatial Cost Analysis

This document outlines the analytical pipeline implemented in `analysis/` and `replication/`. For the full derivation and discussion, see the manuscript.

## 1. Network-based allocation

Each municipality `m` is assigned to the treatment plant `p` minimising the road-network distance from the municipal centroid to the plant:

```
A(m) = argmin_{p ∈ P} d_network(m, p)
```

The 383x46 road-network distance matrix (`data/Matriz_Municipios.csv`) is computed externally with QGIS 3.28 + the ORS Tools plugin (OpenRouteService) on the OpenStreetMap network restricted to Extremadura. Plants located within 15 km of each other are consolidated into plant groups for allocation only; treatment costs are computed at the physical-plant level.

See `analysis/script_asignacion_planta_distancia_real.py` (network) and `analysis/script_asignacion_planta_distancia_euclidea.py` (Euclidean reference).

## 2. CDW generation

Per-municipality annual CDW generation is

```
G_m = P_m · ρ_CDW
```

with `ρ_CDW = 0.5 t/inhab·yr` (CEDEX 2014 lower bound; sensitivity scan 0.29–0.93 in `analysis/sensibilidad_dotacion.py`). Per-plant throughput is the sum of `G_m` over the municipalities assigned to the plant group, divided by the number of physical plants in the group (split-throughput convention).

See `analysis/script_produccion.py` and `analysis/script_produccion_municipios.py`.

## 3. Piecewise logarithmic treatment-cost model

```
C_fix(T) = C0                            if T < T0
C_fix(T) = C0 · (log2(T / T0) + 1)       if T >= T0
```

with `C0 = 40,000 €/yr` (single-operator minimum viable plant) and `T0 = 5,000 t/yr` (one-person capacity ceiling). Variable cost `v = 0.35 €/t`. The piecewise log form is derived from a doubling-rule argument (each doubling of throughput requires one additional operational unit) and is implemented in `analysis/modelo_coste.py`.

## 4. Total cost decomposition

```
C_trans,i = TT_i · ρ_trans / T_i
C_treat,i = C_fix(T_i) / T_i + v
C_i       = C_trans,i + C_treat,i
```

with `ρ_trans = 0.35 €/t·km` calibrated from the Spanish Ministry of Transport observatory and operational data of the four collaborating CDW companies. See `analysis/recalcula_costes_red_real.py`.

## 5. Multi-level diagnostic

Costs are aggregated at three levels:

- **Plant level** (`n = 32` groups): bimodal pattern, 8 groups below T0.
- **Municipality level** (`n = 383`): unweighted mean €15.82/t.
- **Per-tonne level** (production-weighted): system mean €12.22/t.

See `analysis/genera_figuras_articulo.py` (Fig. 5–11 of the manuscript) and `analysis/genera_mapas_articulo.py` (choropleths).

## 6. Sensitivity and robustness

- `analysis/sensibilidad_dotacion.py` — full scan of `ρ_CDW` over 0.29–0.93 t/inhab·yr.
- `analysis/spatial_sensitivity_analysis.py` — perturbations of the road-network distance matrix.
- `analysis/distributional_robustness_analysis.py` — distributional checks on the cost panel.

## 7. End-to-end pipeline

`replication/run_full_analysis.py` orchestrates the whole sequence and writes outputs to `results/figures/` and `results/tables/`.
