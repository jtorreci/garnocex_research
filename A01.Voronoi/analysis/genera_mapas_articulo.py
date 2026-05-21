# -*- coding: utf-8 -*-
"""
genera_mapas_articulo.py
========================
Generates choropleth maps (Figs 2-7) for A01 using recalculated data.
Working versions in geopandas; Blender versions will follow.

Figures:
  Fig 2: Voronoi tessellation (visual reference) + plant locations
  Fig 3: CDW generation by plant catchment area
  Fig 4: Transportation effort (t·km) by plant
  Fig 5: Transportation cost (€/t) by municipality
  Fig 6: Treatment cost (€/t) by municipality
  Fig 7: Total cost (€/t) by municipality
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, MultiPolygon, box
import re
import os

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_DIR = os.path.join(SCRIPT_DIR, "..", "shp")
DATA_DIR = os.path.join(SCRIPT_DIR, "datos_red_real")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Latex", "Imagenes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent palette: warm (low cost = good) to hot (high cost = bad)
# Using a sequential gray-to-red for journal print compatibility
COST_CMAP = "RdYlGn_r"  # Red=high cost, Green=low cost
PROD_CMAP = "YlOrBr"     # Production volume
EFFORT_CMAP = "OrRd"     # Transport effort

RED = "#C03030"
PLANT_COLOR = "black"
PLANT_MARKER = "^"
PLANT_SIZE = 40

# Plant consolidation map (must mirror recalcula_costes_red_real.py).
# Maps absorbed plant Id -> canonical Id for the consolidated group.
PLANT_CONSOLIDATION = {
    2: 1, 12: 1,
    17: 3,
    15: 6,
    40: 19,
    32: 20,
    23: 22, 26: 22, 28: 22,
    34: 27, 35: 27, 37: 27,
    38: 36, 43: 36,
}


# ============================================================
# NAME NORMALIZATION
# ============================================================
def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# LOAD DATA
# ============================================================
print("Loading shapefiles...")
gdf_mun = gpd.read_file(os.path.join(SHP_DIR, "municipios.shp"))
gdf_plantas = gpd.read_file(os.path.join(SHP_DIR, "plantas.shp"))
gdf_ext = gpd.read_file(os.path.join(SHP_DIR, "Extremadura.shp"))
try:
    gdf_roads = gpd.read_file(os.path.join(SHP_DIR, "carreteras.shp"))
    has_roads = True
except:
    has_roads = False

# Ensure consistent CRS
target_crs = "EPSG:25830"
for gdf in [gdf_mun, gdf_plantas, gdf_ext]:
    if gdf.crs != target_crs:
        gdf.set_crs(target_crs, inplace=True, allow_override=True)
if has_roads and gdf_roads.crs != target_crs:
    gdf_roads = gdf_roads.to_crs(target_crs)

# Load cost data
print("Loading cost data...")
df_mun_costs = pd.read_csv(os.path.join(DATA_DIR, "datos_red_real_municipios.csv"))
df_plant_costs = pd.read_csv(os.path.join(DATA_DIR, "datos_red_real_plantas.csv"))

# Merge costs into municipality polygons
gdf_mun["key"] = gdf_mun["NAMEUNIT"].apply(norm)
df_mun_costs["key"] = df_mun_costs["municipio"].apply(norm)
gdf_mun = gdf_mun.merge(df_mun_costs, on="key", how="left")

print(f"Municipalities with cost data: {gdf_mun['C_tot'].notna().sum()}/{len(gdf_mun)}")

# Map bounds for consistent framing
bounds = gdf_ext.total_bounds  # [minx, miny, maxx, maxy]
pad = 5000
xlim = (bounds[0] - pad, bounds[2] + pad)
ylim = (bounds[1] - pad, bounds[3] + pad)


# ============================================================
# HELPER: BASE MAP
# ============================================================
def make_basemap(ax, title="", show_roads=True, show_title=False):
    """Draw boundary, roads, and set limits.

    show_title=False (default) suppresses the embedded title so that journal
    captions are the single source of figure description.
    """
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)
    if show_roads and has_roads:
        gdf_roads.plot(ax=ax, color="0.85", linewidth=0.3, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if show_title and title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_plants(ax, label=True, with_legend=True):
    """Plot plant locations and optionally add a legend entry."""
    gdf_plantas.plot(ax=ax, color=PLANT_COLOR, marker=PLANT_MARKER,
                     markersize=PLANT_SIZE, zorder=5, edgecolor="white",
                     linewidth=0.5)
    if with_legend:
        legend_el = Line2D([0], [0], color=PLANT_COLOR, marker=PLANT_MARKER,
                           ls="None", ms=7, mec="white", mew=0.5,
                           label="CDW treatment plant")
        ax.legend(handles=[legend_el], loc="lower right", fontsize=8,
                  frameon=True, framealpha=0.85)


def add_colorbar(fig, ax, mappable, label=""):
    """Add a nice colorbar."""
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label(label, fontsize=10)
    return cbar


# ============================================================
# FIG 2: VORONOI TESSELLATION (over consolidated plant groups)
# ============================================================
def generate_voronoi_map():
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    make_basemap(ax, "Voronoi tessellation of consolidated CDW plant groups\n(visual reference; assignments based on road network distances)")

    # Resolve canonical group Id for each physical plant
    gdf_p = gdf_plantas.copy()
    gdf_p["group_id"] = gdf_p["Id"].map(lambda i: PLANT_CONSOLIDATION.get(i, i))

    # Compute group centroid as the representative point for the tessellation
    group_centroids = (gdf_p.dissolve(by="group_id")
                            .geometry.centroid
                            .reset_index(name="geometry"))
    centroid_coords = np.array([(p.x, p.y) for p in group_centroids["geometry"]])

    # Voronoi on the 32 group centroids, with far points to bound it
    far = 500_000
    extra = np.array([[-far, -far], [far, -far], [-far, far], [far, far]])
    all_pts = np.vstack([centroid_coords, extra])
    vor = Voronoi(all_pts)

    # Clip Voronoi cells to Extremadura boundary
    ext_boundary = gdf_ext.unary_union
    for region_idx in vor.point_region[:len(centroid_coords)]:
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        polygon = Polygon([vor.vertices[i] for i in region])
        clipped = polygon.intersection(ext_boundary)
        if clipped.is_empty:
            continue
        if isinstance(clipped, (Polygon, MultiPolygon)):
            gpd.GeoSeries([clipped], crs=target_crs).boundary.plot(
                ax=ax, color="0.5", linewidth=0.6, zorder=2)

    # Municipality boundaries (light)
    gdf_mun.boundary.plot(ax=ax, color="0.9", linewidth=0.15, zorder=1)

    # Identify plants belonging to multi-plant groups (those that were consolidated)
    counts = gdf_p["group_id"].value_counts()
    multi_groups = set(counts[counts > 1].index)
    gdf_p["is_grouped"] = gdf_p["group_id"].isin(multi_groups)

    # Draw a thin line from each consolidated plant to its group centroid
    centroid_lookup = {row["group_id"]: row["geometry"]
                       for _, row in group_centroids.iterrows()}
    for _, plant in gdf_p[gdf_p["is_grouped"]].iterrows():
        c = centroid_lookup[plant["group_id"]]
        ax.plot([plant.geometry.x, c.x], [plant.geometry.y, c.y],
                color="0.55", lw=0.7, ls="--", zorder=3)

    # Plot the 46 physical plants (smaller, hollow markers)
    gdf_p.plot(ax=ax, marker=PLANT_MARKER, markersize=22,
               facecolor="white", edgecolor=PLANT_COLOR, linewidth=0.9, zorder=4)

    # Plot the 32 group centroids on top (larger filled markers)
    gpd.GeoSeries(group_centroids["geometry"], crs=target_crs).plot(
        ax=ax, marker=PLANT_MARKER, markersize=PLANT_SIZE,
        color=PLANT_COLOR, edgecolor="white", linewidth=0.5, zorder=5)

    # Legend
    legend_els = [
        Line2D([0], [0], color="0.5", lw=0.8,
               label="Voronoi cells over group centroids (Euclidean)"),
        Line2D([0], [0], color=PLANT_COLOR, marker=PLANT_MARKER, ls="None",
               ms=8, label=f"Consolidated plant group (n={len(centroid_coords)})"),
        Line2D([0], [0], color=PLANT_COLOR, marker=PLANT_MARKER, ls="None",
               ms=5, mfc="white", mew=0.9,
               label=f"Individual treatment plant (n={len(gdf_p)})"),
        Line2D([0], [0], color="0.55", lw=0.7, ls="--",
               label="Plant--group centroid link"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8)

    path = os.path.join(OUTPUT_DIR, "Voronoi.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIG 3: CDW GENERATION BY PLANT
# ============================================================
def generate_cdw_map():
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    make_basemap(ax, "CDW generation by treatment plant catchment area")

    # Color municipalities by assigned plant's production
    gdf_plot = gdf_mun[gdf_mun["produccion_t"].notna()].copy()
    plant_prod = df_plant_costs.set_index("planta_id")["produccion_total"].to_dict()
    gdf_plot["plant_prod"] = gdf_plot["planta_id"].map(plant_prod)

    mappable = gdf_plot.plot(ax=ax, column="plant_prod", cmap=PROD_CMAP,
                             edgecolor="white", linewidth=0.2, zorder=2,
                             legend=False, vmin=0,
                             vmax=gdf_plot["plant_prod"].quantile(0.95))
    add_plants(ax)

    sm = plt.cm.ScalarMappable(cmap=PROD_CMAP,
                                norm=plt.Normalize(0, gdf_plot["plant_prod"].quantile(0.95)))
    add_colorbar(fig, ax, sm, "Plant throughput (t/year)")

    path = os.path.join(OUTPUT_DIR, "CDW_plant.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIG 4: TRANSPORT EFFORT
# ============================================================
def generate_transport_effort_map():
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    make_basemap(ax, "Transportation effort by municipality")

    gdf_plot = gdf_mun[gdf_mun["tkm"].notna()].copy()
    vmax = gdf_plot["tkm"].quantile(0.95)
    gdf_plot.plot(ax=ax, column="tkm", cmap=EFFORT_CMAP,
                  edgecolor="white", linewidth=0.2, zorder=2,
                  legend=False, vmin=0, vmax=vmax)
    add_plants(ax)

    sm = plt.cm.ScalarMappable(cmap=EFFORT_CMAP, norm=plt.Normalize(0, vmax))
    add_colorbar(fig, ax, sm, "Transport effort (t\u00b7km/year)")

    path = os.path.join(OUTPUT_DIR, "Transport_effort_plant.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGS 5-7: COST MAPS (transport, treatment, total)
# ============================================================
def generate_cost_map(column, title, filename, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    make_basemap(ax, title)

    gdf_plot = gdf_mun[gdf_mun[column].notna()].copy()
    if vmax is None:
        vmax = gdf_plot[column].quantile(0.95)
    if vmin is None:
        vmin = gdf_plot[column].min()

    gdf_plot.plot(ax=ax, column=column, cmap=COST_CMAP,
                  edgecolor="white", linewidth=0.2, zorder=2,
                  legend=False, vmin=vmin, vmax=vmax)
    add_plants(ax)

    sm = plt.cm.ScalarMappable(cmap=COST_CMAP, norm=plt.Normalize(vmin, vmax))
    add_colorbar(fig, ax, sm, "\u20ac/t")

    # Mark municipalities above vmax with hatching
    above = gdf_plot[gdf_plot[column] > vmax]
    if len(above) > 0:
        above.plot(ax=ax, facecolor="none", edgecolor=RED, linewidth=1.0,
                   hatch="///", zorder=4)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print()
    print("Generating maps...")
    print()

    generate_voronoi_map()
    generate_cdw_map()
    generate_transport_effort_map()

    # Use consistent scale for cost maps
    all_ctot = gdf_mun["C_tot"].dropna()
    cost_vmax = all_ctot.quantile(0.95)

    generate_cost_map("C_trans", "Transportation cost per tonne by municipality",
                      "Transport_cost_plant.png", vmin=0, vmax=cost_vmax * 0.6)
    generate_cost_map("C_trat", "Treatment cost per tonne by municipality",
                      "treatment_cost.png", vmin=0, vmax=cost_vmax * 0.6)
    generate_cost_map("C_tot", "Total cost per tonne by municipality",
                      "Total_cost.png", vmin=0, vmax=cost_vmax)

    print()
    print("All maps generated in:", OUTPUT_DIR)
