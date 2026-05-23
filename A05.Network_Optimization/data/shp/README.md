# Shapefiles — A05

This directory hosts the geospatial layers needed by the analysis and visualisation scripts.

## Files included in the repository

| File set | Geometry | Source |
|---|---|---|
| `Extremadura.{shp,shx,prj,dbf,cpg}` | Polygon | Regional administrative outline. |
| `carreteras.{shp,shx,prj,cpg}` | LineString | Road network restricted to Extremadura, derived from OpenStreetMap. |
| `municipios.{shp,shx,prj,dbf,cpg}` | Polygon | 383 municipal polygons; key field `NAMEUNIT`. |
| `plantas.{shp,shx,prj,dbf,cpg}` | Point | 46 physical CDW plant locations; key field `Id` (1–46). |

CRS: **ETRS89 / UTM zone 30N (EPSG:25830)**.

## Missing file: `carreteras.dbf`

The attribute table for the road-network layer (`carreteras.dbf`) is approximately **591 MB**, exceeding GitHub's single-file size limit (100 MB) and the recommended threshold for normal version control. It is therefore **gitignored** and not bundled with this repository.

The scripts shipped here (`replication/generate_three_panel_a05.py`) only consume the road geometries via `gpd.read_file("carreteras.shp")` and access `road.geometry`; no attribute fields are read. Depending on the GeoPandas / Fiona version installed, reading `carreteras.shp` without its companion `.dbf` may emit a warning but should still return the geometry column.

### How to obtain `carreteras.dbf`

Two options:

1. **Upon request from the authors**: open a GitHub Issue or email the corresponding author (`jtorreci@unex.es`); the file will be shared via a transfer service.
2. **Regenerate from OpenStreetMap**:
   - Export the road layer for Extremadura from OpenStreetMap (e.g., via the *QuickOSM* QGIS plugin or `osmnx`).
   - Reproject to EPSG:25830.
   - Save as ESRI shapefile; the export will include a valid `carreteras.dbf` automatically.

Either source is sufficient to run `generate_three_panel_a05.py`.

### Why so large?

The attribute table contains a row per road segment in Extremadura with all OSM tags preserved (highway class, name, surface, max-speed, lanes, source, etc.). For the present analysis only the geometry matters, but stripping the attributes downstream from the original export was not done before committing. The next iteration of the data preparation will either:

- ship a stripped `carreteras.dbf` (a few MB, only essential columns), or
- replace the shapefile with a GeoPackage / GeoJSON encoding that is more compact.
