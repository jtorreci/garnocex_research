# Data Description — A01

All CSV files in `data/` are derived from public sources or from the GARNOCEX collaborating companies; the files released here are aggregated to the municipal/plant level and contain no individual or commercial sensitive information.

## Files

| File | Rows | Description |
|---|---:|---|
| `coordenadas_municipios.csv` | 383 | Municipality centroids (EPSG:25830) with population (INE 2023). |
| `coordenadas_plantas.csv` | 46 | CDW treatment plant locations (EPSG:25830). |
| `Plantas_con_nombre_municipio.csv` | 46 | Each plant linked to its hosting municipality. |
| `Matriz_Municipios.csv` | 383x46 | Road-network shortest-path distance matrix (km) computed via QGIS 3.28 + ORS Tools on OpenStreetMap. |
| `distancias_reales_penalizadas_simplificado.csv` | 383x46 | Penalised road-network distances (simplified). |
| `distancias_euclideas.csv` | 383x46 | Euclidean reference distance matrix. |
| `asignacion_municipios_real.csv` | 383 | Network-based municipality → plant assignment. |
| `asignacion_municipios_euclidiana.csv` | 383 | Euclidean (Voronoi) reference assignment for comparison only. |
| `datos_voronoi_municipios.csv` | 383 | Per-municipality summary used by the spatial analysis scripts. |
| `plant_anisotropy_coefficients.csv` | 46 | Per-plant anisotropy diagnostics. |

## Coordinate reference system

All spatial data are in **ETRS89 / UTM zone 30N (EPSG:25830)**, consistent with the official cartography of the Junta de Extremadura.

## Reproducing the distance matrix from scratch

The road-network distances are precomputed and shipped as `Matriz_Municipios.csv`. To regenerate them from OpenStreetMap data:

1. Install QGIS 3.28 with the *ORS Tools* plugin and request an OpenRouteService API key.
2. Load `coordenadas_municipios.csv` (origin layer) and `coordenadas_plantas.csv` (destination layer).
3. Run *ORS Tools → Matrix from layers* in driving-car mode and export as CSV.

## Provenance and licensing of upstream data

- Municipal population and centroids: Instituto Nacional de Estadística (INE), open data.
- Plant register: *Registro de Gestores de Residuos de Construcción y Demolición de Extremadura*, Decreto 20/2011 Junta de Extremadura.
- Road network: OpenStreetMap (ODbL 1.0).
- Aggregated cost calibration: operational data from the four CDW management companies participating in the GARNOCEX project, released here in aggregated form only.

This repository as a whole is distributed under CC-BY 4.0 (see `LICENSE`).
