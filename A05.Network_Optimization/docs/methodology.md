# Methodology — A05 Network Rationalization

This document outlines the analytical pipeline implemented in `analysis/` and `replication/`. For the full derivation and discussion, see the manuscript.

## 1. Inputs

- **Distance matrix**: `data/tablas/distancias_reales_plantas_municipios.csv`, 46×383 road-network distances (m) computed from OpenStreetMap with QGIS 3.28 + ORS Tools (companion paper A01).
- **Municipality data**: `data/tablas/datos_voronoi_municipios.csv`, with population, CDW production (`Prod`), and identifiers per municipality.
- **Plant identifiers**: `data/tablas/codigos_plantas.csv`, mapping plant id (1–46) to host municipality.
- **Shapefiles** (`data/shp/`): Extremadura outline, road network, municipal polygons, plant points (EPSG:25830).

The 14 plant pairs/triplets located within 15 km of each other are consolidated into 8 group leaders, yielding **32 effective plant groups**. The consolidation map is hard-coded in the scripts as `CONSOLIDATION`.

## 2. Cost model

Total cost per tonne for municipality `i` assigned to plant group `j`:

```
C_total(i, j) = ρ · d(i, j) + C_fix(T_j) / T_j + v
```

with `ρ = 0.35 €/(t·km)`, `v = 0.35 €/t`, and the piecewise-log fixed-cost model:

```
C_fix(T) = C0                            if T < T0
C_fix(T) = C0 · (log2(T / T0) + 1)       if T ≥ T0
```

with `C0 = 40,000 €/yr` and `T0 = 5,000 t/yr`. Throughput `T_j` is the sum of `Prod_i` over all municipalities assigned to plant group `j`. The model is calibrated in the companion paper A01.

## 3. Greedy progressive pruning (benchmark)

`analysis/analisis_podado.py` implements:

1. Start with all 32 plant groups active.
2. Assign each municipality to its nearest active group (road-network distance).
3. Compute throughputs, unit costs, and system metrics.
4. For each candidate plant, compute the system cost that would result from its removal. Eliminate the plant whose removal causes the smallest cost increase.
5. Repeat until only 3 plants remain.

Outputs:

- `outputs/pareto_podado.csv`: system metrics for each step (32 → 3 plants).
- `outputs/greedy_assignments_24.csv`: per-municipality assignment at the 24-plant pivot (the near-optimal plateau).

The script accepts `--rho`, `--dotacion`, `--scenario`, and `--no-plot` arguments for the sensitivity sweep.

## 4. Iterative cost-based heuristic

`analysis/iterative_cost_assignment.py` implements:

1. Seed: each municipality assigned to its nearest plant group (same as step 2 of greedy).
2. For each municipality, compute total cost to every active plant under the current throughput configuration. Reassign to the lowest-cost option.
3. Recompute throughputs and unit costs.
4. Repeat until no municipality changes assignment.
5. Plants with zero throughput at stabilisation are flagged as eliminated.

The procedure is treated as a fixed-point heuristic; no general convergence proof is claimed. In the Extremadura case the procedure stabilises in **6 iterations**.

Outputs:

- `outputs/iterative_optimization.csv`: iteration-by-iteration history.
- `outputs/optimal_assignments.csv`: per-municipality assignment at stabilisation.

## 5. Sensitivity sweep

`analysis/run_sensitivity.py` is a runner that re-executes the greedy and iterative procedures under six parameter scenarios:

- Transport sensitivity: `ρ ∈ {0.25, 0.35, 0.45}` €/(t·km).
- Demand sensitivity: `ρ_CDW ∈ {0.45, 0.50, 0.55}` t/inhab·yr.

Outputs are written under `outputs/<file>_<scenario>.csv` and aggregated in `sensitivity_summary.csv`, `sensitivity_transport.csv`, `sensitivity_demand.csv`.

## 6. Visualisation: routed flows + cost KDE

`replication/generate_three_panel_a05.py` builds the manuscript Figure 2:

1. Loads the shapefiles and builds a road graph from `carreteras.shp` (≈ 185k nodes, 221k edges).
2. Routes each municipality → assigned plant via NetworkX shortest path on the road graph.
3. Panel (a): baseline routes for all 32 plant groups under nearest-plant assignment.
4. Panel (b): only the routes for the **65** municipalities whose plant assignment changes under the iterative procedure; the unchanged routes are left blank to make the local nature of the change visible.
5. Panel (c): kernel density estimates (`scipy.stats.gaussian_kde`) of the municipal unit-cost distribution under baseline vs iterative.

Output: `results/figures/fig_three_panel_a05.png`.
