# Data Description — A05

All inputs in `data/` are derived from public sources or from the GARNOCEX collaborating companies; the files released here are aggregated to municipal / plant level and contain no individual or commercially sensitive information.

## Shapefiles (`data/shp/`)

| File | Geometry | Notes |
|---|---|---|
| `Extremadura.shp` | Polygon | Regional administrative outline of Extremadura. |
| `carreteras.shp` | LineString | Road network used for shortest-path routing. |
| `municipios.shp` | Polygon | 383 municipal polygons (key field `NAMEUNIT`). |
| `plantas.shp` | Point | 46 physical CDW plant locations (key field `Id` = 1–46). |

CRS: **ETRS89 / UTM zone 30N (EPSG:25830)**.

## Tables (`data/tablas/`)

| File | Rows | Description |
|---|---:|---|
| `codigos_plantas.csv` | 46 | Plant id → host municipality + company name. |
| `datos_voronoi_municipios.csv` | 383 | Municipality production (`Prod`), population (`HBT`), centroid coordinates, baseline assignment fields. |
| `distancias_reales_plantas_municipios.csv` | 17 618 | 46 × 383 road-network distance matrix in metres (`origin_id` = plant, `destination_id` = municipality, `total_cost` = distance). |

## Outputs (`outputs/`)

| File | Origin | Description |
|---|---|---|
| `pareto_podado.csv` | `analisis_podado.py` | System metrics for each greedy pruning step (32 → 3 plants). |
| `greedy_assignments_24.csv` | `analisis_podado.py` | Per-municipality assignment at the 24-plant pivot. |
| `iterative_optimization.csv` | `iterative_cost_assignment.py` | Iteration-by-iteration history of the iterative heuristic. |
| `optimal_assignments.csv` | `iterative_cost_assignment.py` | Per-municipality assignment at stabilisation (24 active plants, 8 with zero throughput). |
| `sensitivity_summary.csv` | `run_sensitivity.py` | Combined transport and demand scenarios. |
| `sensitivity_transport.csv` | `run_sensitivity.py` | Transport-cost scenarios (`ρ = 0.25, 0.35, 0.45 €/t·km`). |
| `sensitivity_demand.csv` | `run_sensitivity.py` | Demand scenarios (`ρ_CDW = 0.45, 0.50, 0.55 t/inhab·yr`). |
| `*_<scenario>.csv` | `run_sensitivity.py` | Per-scenario intermediates (pareto and assignments). |

## Provenance

- Municipal population and centroids: Instituto Nacional de Estadística (INE), open data.
- Plant register: *Registro de Gestores de Residuos de Construcción y Demolición de Extremadura*, Decreto 20/2011 Junta de Extremadura.
- Road network: OpenStreetMap (ODbL 1.0).
- Cost calibration (`C0`, `T0`, `ρ`, `v`): operational data from the four CDW companies participating in the GARNOCEX project, released here in aggregated form only; underlying commercial records remain confidential.

This repository as a whole is distributed under CC-BY 4.0 (see `LICENSE`).
