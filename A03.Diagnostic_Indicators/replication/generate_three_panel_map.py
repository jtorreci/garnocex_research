# -*- coding: utf-8 -*-
"""
generate_three_panel_map.py
===========================
Three-panel flow map for A03 (Figure 2 of the WM&R submission).

Panels: (a) S1 proximity baseline, (b) S2 observed use with conservative
closure, (c) S3 cost-based fixed assignment. The fourth cell of the 2 x 2 grid
holds the shared colour bar and the legend.

Data sources (same as compute_three_scenarios.py):
  S1: data/s1_proximity_baseline.csv (A01 municipality table, nearest plant by
      road distance, consolidated).
  S2: data/s2_observed_flows_public.csv (audited flows normalised to municipal
      production plus the residual closed on the S1 plant), revalued with the
      same transport + treatment model.
  S3: ../A05.Network_Optimization/outputs/optimal_assignments.csv.
Shapefiles (region outline, roads, municipalities, plants) are read from
../A05.Network_Optimization/data/shp/. The road attribute table carreteras.dbf
is not in the repository (see that folder's README.md); only road geometries
are used. All routes are shortest paths on the road graph; origins and plants
snap to the largest connected component of that graph, so no route falls back
to a straight segment because it started on an isolated road fragment.

Output: outputs/fig_three_panel_routed.jpg, 260 mm wide at 300 dpi (twice the
130 mm printed width), greyscale-safe sequential colour map.
"""

from __future__ import annotations

import math
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SHP_DIR = os.path.join(SCRIPT_DIR, "..", "..", "A05.Network_Optimization", "data", "shp")
A05_DATA = os.path.join(SCRIPT_DIR, "..", "..", "A05.Network_Optimization", "outputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

# 2x printed size: 130 mm print -> 260 mm (10.24 in); 15 pt here prints at 7.5 pt.
FIG_WIDTH_IN = 260 / 25.4
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "savefig.dpi": 300,
})

CONSOLIDATION = {2: 1, 12: 1, 17: 3, 15: 6, 40: 19, 32: 20, 23: 22, 26: 22, 28: 22,
                 34: 27, 35: 27, 37: 27, 38: 36, 43: 36}
# Audited S2 plants outside the 46-plant registry use synthetic ids in
# compute_three_scenarios.py. Map them to their physical registry plant when one
# exists; Coria has no registry entry and is drawn at its municipality centroid.
S2_SYNTHETIC_PLANTS = {-1: 23, -3: 26, -4: 28}
S2_SYNTHETIC_TOWNS = {-2: "coria"}
SNAP = 50

# Cost model (same as compute_three_scenarios.py / A01-A05)
C0, T0, RHO, V = 40_000, 5_000, 0.35, 0.35


def _fixed_cost(T: float) -> float:
    return C0 if T <= 0 else max(C0, C0 * (np.log2(T / T0) + 1))


def _unit_treatment_cost(T: float) -> float:
    return np.inf if T <= 0 else _fixed_cost(T) / T + V


def norm(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def snap_coord(x: float, y: float) -> tuple[int, int]:
    return (round(x / SNAP) * SNAP, round(y / SNAP) * SNAP)


# ============================================================
# LOAD SHAPEFILES
# ============================================================
print("Loading shapefiles...")
gdf_ext = gpd.read_file(os.path.join(SHP_DIR, "Extremadura.shp"))
gdf_roads = gpd.read_file(os.path.join(SHP_DIR, "carreteras.shp"))
gdf_mun = gpd.read_file(os.path.join(SHP_DIR, "municipios.shp"))
gdf_plantas = gpd.read_file(os.path.join(SHP_DIR, "plantas.shp"))

for gdf in [gdf_ext, gdf_roads, gdf_mun, gdf_plantas]:
    if gdf.crs and gdf.crs.to_epsg() != 25830:
        gdf.to_crs(epsg=25830, inplace=True)

gdf_mun["centroid"] = gdf_mun.geometry.centroid
gdf_mun["key"] = gdf_mun["NAMEUNIT"].apply(norm)
mun_centroids = {r["key"]: (r["centroid"].x, r["centroid"].y) for _, r in gdf_mun.iterrows()}

plant_coords: dict[int, tuple[float, float]] = {}
for _, r in gdf_plantas.iterrows():
    pid = int(r["Id"]) if "Id" in r else 0
    plant_coords[pid] = (r.geometry.x, r.geometry.y)

# ============================================================
# BUILD ROAD GRAPH
# ============================================================
print("Building road graph...")
G = nx.Graph()
node_coords: dict[tuple[int, int], tuple[float, float]] = {}
for _, road in gdf_roads.iterrows():
    geom = road.geometry
    if geom is None or geom.is_empty:
        continue
    lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            n1 = snap_coord(coords[i][0], coords[i][1])
            n2 = snap_coord(coords[i + 1][0], coords[i + 1][1])
            if n1 == n2:
                continue
            d = math.dist(coords[i][:2], coords[i + 1][:2])
            node_coords[n1] = (coords[i][0], coords[i][1])
            node_coords[n2] = (coords[i + 1][0], coords[i + 1][1])
            if not G.has_edge(n1, n2) or d < G[n1][n2]["weight"]:
                G.add_edge(n1, n2, weight=d)

# Snap origins and plants to the main connected component only: the shapefile has small
# disconnected road fragments, and snapping to one of them would force a straight segment.
main_component = max(nx.connected_components(G), key=len)
graph_nodes = np.array([n for n in node_coords if n in main_component])
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"{nx.number_connected_components(G)} components")


def find_nearest_node(x: float, y: float) -> tuple[int, int]:
    snapped = snap_coord(x, y)
    if snapped in main_component:
        return snapped
    dists = (graph_nodes[:, 0] - snapped[0]) ** 2 + (graph_nodes[:, 1] - snapped[1]) ** 2
    return tuple(graph_nodes[np.argmin(dists)])


FALLBACKS: list[tuple[str, str, float]] = []


def resolve_routes(requests: list[dict]) -> None:
    """Fill request["coords"] with road shortest paths, one Dijkstra per plant.

    Requests are grouped by plant node so that each predecessor map is computed once
    and freed before the next (keeps memory flat). A disconnected pair gets a
    straight segment and is recorded in FALLBACKS.
    """
    by_plant: dict[tuple[int, int], list[dict]] = {}
    for req in requests:
        req["n_mun"] = find_nearest_node(*req["xy_mun"])
        by_plant.setdefault(find_nearest_node(*req["xy_plant"]), []).append(req)
    for n_plant, reqs in by_plant.items():
        pred, _ = nx.dijkstra_predecessor_and_distance(G, n_plant, weight="weight")
        for req in reqs:
            node = req["n_mun"]
            if node not in pred:
                FALLBACKS.append((req["label"],
                                  f"{req['xy_mun'][0]:.0f},{req['xy_mun'][1]:.0f}",
                                  math.dist(req["xy_mun"], req["xy_plant"]) / 1000))
                req["coords"] = [req["xy_mun"], req["xy_plant"]]
                continue
            path = [node]
            while path[-1] != n_plant:
                path.append(pred[path[-1]][0])
            req["coords"] = [node_coords.get(n, n) for n in path]
        del pred


# ============================================================
# COMPUTE ROUTES
# ============================================================
print("Computing S1 routes...")
df_s1 = pd.read_csv(os.path.join(DATA_DIR, "s1_proximity_baseline.csv"))
df_s1["key"] = df_s1["municipio"].apply(norm)
routes_s1 = []
for _, r in df_s1.iterrows():
    mxy = mun_centroids.get(r["key"])
    pid = CONSOLIDATION.get(int(r["planta_id"]), int(r["planta_id"]))
    pxy = plant_coords.get(pid)
    if mxy and pxy:
        routes_s1.append({"xy_mun": mxy, "xy_plant": pxy, "label": f"S1 {r['key']}->{pid}",
                          "cost": r["C_tot"], "prod": r["produccion_t"], "plant_id": pid})
print(f"  S1: {len(routes_s1)} routes")

print("Computing S2 routes (observed use with conservative closure)...")
df_s2 = pd.read_csv(os.path.join(DATA_DIR, "s2_observed_flows_public.csv"))
s2_throughput = df_s2.groupby("plant_id")["prod_t"].sum()
s2_extra_plants: dict[int, tuple[float, float]] = {}
for sid, phys in S2_SYNTHETIC_PLANTS.items():
    s2_extra_plants[sid] = plant_coords[phys]
for sid, town in S2_SYNTHETIC_TOWNS.items():
    s2_extra_plants[sid] = mun_centroids[town]
routes_s2 = []
s2_missing = 0
for _, r in df_s2.iterrows():
    # compute_three_scenarios.py keeps trailing articles ("granja la"); drop them here.
    mkey = re.sub(r"\s+(la|el|los|las)$", "", str(r["mun_norm"]))
    mxy = mun_centroids.get(mkey)
    pid = int(r["plant_id"])
    pxy = s2_extra_plants.get(pid) if pid < 0 else plant_coords.get(pid)
    if mxy and pxy:
        cost_per_t = RHO * r["dist_km"] + _unit_treatment_cost(float(s2_throughput[pid]))
        routes_s2.append({"xy_mun": mxy, "xy_plant": pxy, "label": f"S2 {r['mun_norm']}->{pid}",
                          "cost": cost_per_t, "prod": r["prod_t"], "plant_id": pid})
    else:
        s2_missing += 1
print(f"  S2: {len(routes_s2)} routes ({s2_missing} rows without coordinates)")

print("Computing S3 routes...")
df_s3 = pd.read_csv(os.path.join(A05_DATA, "optimal_assignments.csv"))
df_s3["key"] = df_s3["name"].apply(norm) if "name" in df_s3.columns else df_s3["mun_key"]
routes_s3 = []
for _, r in df_s3.iterrows():
    mxy = mun_centroids.get(r["key"])
    pid = CONSOLIDATION.get(int(r["plant_id"]), int(r["plant_id"]))
    pxy = plant_coords.get(pid)
    if mxy and pxy:
        routes_s3.append({"xy_mun": mxy, "xy_plant": pxy, "label": f"S3 {r['key']}->{pid}",
                          "cost": r["cost"], "prod": r["prod"], "plant_id": pid})
print(f"  S3: {len(routes_s3)} routes")

print("Routing on the road graph...")
resolve_routes(routes_s1 + routes_s2 + routes_s3)
print(f"\nStraight-line fallbacks (no road path): {len(FALLBACKS)}")
for label, xy, km in FALLBACKS:
    print(f"  {label}  municipality centroid at {xy}  straight distance {km:.1f} km")

# ============================================================
# PLOT: 2 x 2 GRID (three maps + key)
# ============================================================
print("\nGenerating map figure...")
bounds = gdf_ext.total_bounds
PAD = 2000
map_aspect = (bounds[3] - bounds[1] + 2 * PAD) / (bounds[2] - bounds[0] + 2 * PAD)
cell_w = FIG_WIDTH_IN / 2
fig_h = 2 * (cell_w * map_aspect) + 0.15
fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, fig_h),
                         gridspec_kw={"wspace": 0.02, "hspace": 0.02,
                                      "left": 0.005, "right": 0.995,
                                      "top": 0.995, "bottom": 0.005})
ax1, ax2, ax3, ax_key = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

all_costs = [r["cost"] for r in routes_s1 + routes_s2 + routes_s3]
vmin, vmax = np.percentile(all_costs, [5, 95])
print(f"  Colour scale clipped at 5th-95th percentiles: {vmin:.2f}-{vmax:.2f} EUR/t "
      f"(data range {min(all_costs):.2f}-{max(all_costs):.2f})")
# Sequential light-to-dark map (cividis), lightest 12 % dropped so cheap routes stay
# visible on white: low cost = light, high cost = dark, also in greyscale.
_base = plt.get_cmap("cividis_r")
cmap = LinearSegmentedColormap.from_list("cividis_trunc", _base(np.linspace(0.12, 1.0, 256)))
norm_c = plt.Normalize(vmin=vmin, vmax=vmax)
MINOR = 750


def draw_panel(ax, routes, title, extra_active=None, show_eliminated=False):
    gdf_roads.plot(ax=ax, color="0.90", linewidth=0.3, zorder=1)
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)
    for route in sorted(routes, key=lambda r: r["prod"]):
        coords = route["coords"]
        if len(coords) < 2:
            continue
        c = cmap(norm_c(min(max(route["cost"], vmin), vmax)))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        if route["prod"] < MINOR:
            ax.plot(xs, ys, color=c, linewidth=0.9, alpha=0.6, zorder=2,
                    linestyle=(0, (4, 3)))
        else:
            lw = max(1.2, min(5.0, 0.6 + 0.6 * math.log(max(route["prod"], 1))))
            ax.plot(xs, ys, color=c, linewidth=lw, alpha=0.75, zorder=2,
                    solid_capstyle="round")
    active = {r["plant_id"] for r in routes}
    for pid, (x, y) in plant_coords.items():
        if CONSOLIDATION.get(pid, pid) in active:
            ax.plot(x, y, marker="^", color="black", mec="white", ms=9, mew=0.8, ls="None",
                    zorder=5)
        elif show_eliminated:
            ax.plot(x, y, marker="x", color="0.25", ms=8, mew=2.0, ls="None", zorder=4)
    for (x, y) in (extra_active or {}).values():
        ax.plot(x, y, marker="^", color="black", mec="white", ms=9, mew=0.8, ls="None",
                zorder=5)
    ax.set_xlim(bounds[0] - PAD, bounds[2] + PAD)
    ax.set_ylim(bounds[1] - PAD, bounds[3] + PAD)
    ax.set_aspect("equal")
    # Title inside the empty north-west corner of the frame (saves vertical space).
    ax.text(0.0, 1.0, title, transform=ax.transAxes, ha="left", va="top", fontsize=15,
            zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                            "pad": 2})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


draw_panel(ax1, routes_s1, "(a) S1 proximity baseline")
draw_panel(ax2, routes_s2, "(b) S2 observed use\n(conservative closure)",
           extra_active=s2_extra_plants)
draw_panel(ax3, routes_s3, "(c) S3 cost-based\nfixed assignment", show_eliminated=True)

n_elim = len({CONSOLIDATION.get(p, p) for p in plant_coords}
             - {r["plant_id"] for r in routes_s3})
print(f"  Plant groups without municipalities in S3: {n_elim}")

# Key cell: colour bar + legend
ax_key.axis("off")
cbar_ax = ax_key.inset_axes([0.08, 0.66, 0.84, 0.07])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
cbar.set_label("Unit cost (€ t$^{-1}$)", fontsize=15)
cbar.ax.tick_params(labelsize=14)
legend = [
    Line2D([0], [0], color="black", marker="^", mec="white", ls="None", ms=11,
           label="Active plant"),
    Line2D([0], [0], color="0.25", marker="x", ls="None", ms=10, mew=2.0,
           label="Plant without flows in S3"),
    Line2D([0], [0], color="0.3", lw=3.0, label=r"Flow $\geq$ 750 t yr$^{-1}$ (width $\propto$ tonnes)"),
    Line2D([0], [0], color="0.3", ls=(0, (4, 3)), lw=1.2, label="Flow < 750 t yr$^{-1}$"),
]
ax_key.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.42),
              fontsize=14, frameon=False, handlelength=2.4)

os.makedirs(OUTPUT_DIR, exist_ok=True)
path = os.path.join(OUTPUT_DIR, "fig_three_panel_routed.jpg")
fig.savefig(path, dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)
print(f"Saved: {path}")
print("Done.")
