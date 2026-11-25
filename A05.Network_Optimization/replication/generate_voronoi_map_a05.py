# -*- coding: utf-8 -*-
"""
generate_voronoi_map_a05.py
===========================
Figure 1 of A05: Euclidean Voronoi tessellation over the 32 consolidated
plant-group locations, with the merged plants linked to their group centroid.

Self-contained: reads only A05.Network_Optimization/data/shp/. The geometry
follows A01's genera_mapas_articulo.py (same consolidation table, same
centroid rule) but the figure is drawn at its printed size (11 cm wide) with
7-8 pt text, the legend placed outside the map, and the merged plants shown
separately from the single-plant groups.

Output (600 dpi PNG):
    results/figures/Voronoi.png
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Polygon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SHP_DIR = os.path.join(ROOT, "data", "shp")
OUT_PATHS = [
    os.path.join(ROOT, "results", "figures", "Voronoi.png"),
]

FIG_WIDTH_IN = 11.0 / 2.54  # printed width in the manuscript
TARGET_CRS = "EPSG:25830"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "legend.fontsize": 7,
    "savefig.dpi": 600,
})

# Physical plant Id -> canonical Id of its consolidated group (15 km rule);
# mirrors A01 and the other A05 scripts.
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


def load_layers():
    layers = {}
    for name in ("municipios", "plantas", "Extremadura"):
        gdf = gpd.read_file(os.path.join(SHP_DIR, f"{name}.shp"))
        if gdf.crs is None or gdf.crs.to_string() != TARGET_CRS:
            gdf = gdf.to_crs(TARGET_CRS)
        layers[name] = gdf
    return layers["municipios"], layers["plantas"], layers["Extremadura"]


def voronoi_cells(points, boundary):
    """Voronoi cells of `points`, clipped to `boundary`."""
    far = 500_000
    extra = np.array([[-far, -far], [far, -far], [-far, far], [far, far]])
    vor = Voronoi(np.vstack([points, extra]))
    cells = []
    for region_idx in vor.point_region[:len(points)]:
        region = vor.regions[region_idx]
        if -1 in region or not region:
            continue
        clipped = Polygon([vor.vertices[i] for i in region]).intersection(boundary)
        if not clipped.is_empty and isinstance(clipped, (Polygon, MultiPolygon)):
            cells.append(clipped)
    return cells


def add_scale_bar(ax, length_km=50):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    mx, my = x0 + 0.04 * (x1 - x0), y0 + 0.03 * (y1 - y0)
    ax.plot([mx, mx + length_km * 1000], [my, my], color="black", lw=2.0,
            solid_capstyle="butt", zorder=7)
    ax.text(mx + length_km * 500, my + 0.012 * (y1 - y0), f"{length_km} km",
            ha="center", va="bottom", fontsize=7)


def add_north_arrow(ax):
    x, y, dy = 0.95, 0.90, 0.06
    ax.annotate("", xy=(x, y + dy), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0), zorder=7)
    ax.text(x, y + dy + 0.01, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.5, fontweight="bold")


def main():
    gdf_mun, gdf_p, gdf_ext = load_layers()
    gdf_p = gdf_p.copy()
    gdf_p["group_id"] = gdf_p["Id"].map(lambda i: PLANT_CONSOLIDATION.get(i, i))

    centroids = gdf_p.dissolve(by="group_id").geometry.centroid
    centroid_xy = np.array([(p.x, p.y) for p in centroids])
    counts = gdf_p["group_id"].value_counts()
    multi = set(counts[counts > 1].index)
    merged = gdf_p[gdf_p["group_id"].isin(multi)]
    n_groups, n_plants = len(centroid_xy), len(gdf_p)
    n_merged, n_multi = len(merged), len(multi)

    minx, miny, maxx, maxy = gdf_ext.total_bounds
    pad = 4000
    map_w, map_h = maxx - minx + 2 * pad, maxy - miny + 2 * pad
    legend_h_in = 0.36
    map_h_in = FIG_WIDTH_IN * map_h / map_w
    fig_h_in = map_h_in + legend_h_in
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in))
    ax = fig.add_axes([0, legend_h_in / fig_h_in, 1, map_h_in / fig_h_in])

    gdf_mun.boundary.plot(ax=ax, color="0.88", linewidth=0.15, zorder=1)
    for cell in voronoi_cells(centroid_xy, gdf_ext.union_all()):
        gpd.GeoSeries([cell], crs=TARGET_CRS).boundary.plot(
            ax=ax, color="0.35", linewidth=0.6, zorder=2)
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=3)

    for _, plant in merged.iterrows():
        c = centroids.loc[plant["group_id"]]
        ax.plot([plant.geometry.x, c.x], [plant.geometry.y, c.y],
                color="0.3", lw=0.6, ls=(0, (2.5, 1.5)), zorder=4)
    ax.plot(merged.geometry.x, merged.geometry.y, ls="None", marker="^", ms=3.6,
            mfc="white", mec="black", mew=0.7, zorder=5)
    ax.plot(centroid_xy[:, 0], centroid_xy[:, 1], ls="None", marker="^", ms=5,
            mfc="black", mec="white", mew=0.4, zorder=6)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    add_scale_bar(ax)
    add_north_arrow(ax)

    handles = [
        Line2D([0], [0], color="0.35", lw=0.8,
               label="Euclidean Voronoi cell"),
        Line2D([0], [0], ls="None", marker="^", ms=5, mfc="black", mec="white",
               mew=0.4, label=f"Plant group (n = {n_groups})"),
        Line2D([0], [0], ls="None", marker="^", ms=3.6, mfc="white", mec="black",
               mew=0.7, label=f"Merged plant (n = {n_merged})"),
        Line2D([0], [0], color="0.3", lw=0.6, ls=(0, (2.5, 1.5)),
               label="Link to group centroid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.5)

    for path in OUT_PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=600)
        print(f"Saved: {path}")
    plt.close(fig)
    print(f"{n_plants} plants, {n_groups} groups, {n_merged} merged plants "
          f"in {n_multi} multi-plant groups")


if __name__ == "__main__":
    main()
