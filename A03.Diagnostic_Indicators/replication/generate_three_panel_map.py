# -*- coding: utf-8 -*-
"""
generate_three_panel_map.py
===========================
Three-panel flow map: S1 + S2 on top row, S3 centered below.
S2 uses REAL operational flows from Excel data (736 flows, 23 plants).
All routes computed on the road network.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math, re, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_DIR = os.path.join(SCRIPT_DIR, "..", "..", "A01.Voronoi", "shp")
A01_DATA = os.path.join(SCRIPT_DIR, "..", "..", "A01.Voronoi", "scripts", "datos_red_real")
A05_DATA = os.path.join(SCRIPT_DIR, "..", "..", "A05.Network_Optimization", "outputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Articulo_latex", "figures")

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})

CONSOLIDATION = {2:1,12:1,17:3,15:6,40:19,32:20,23:22,26:22,28:22,34:27,35:27,37:27,38:36,43:36}
SNAP = 50

# Cost model (same as compute_three_scenarios.py / A01-A05): unit cost = RHO*d + fixed_cost(T)/T + V
C0, T0, RHO, V = 40_000, 5_000, 0.35, 0.35
def _fixed_cost(T):
    return C0 if T <= 0 else max(C0, C0 * (np.log2(T / T0) + 1))
def _unit_treatment_cost(T):
    return np.inf if T <= 0 else _fixed_cost(T) / T + V

def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().replace("\n"," ")
    s = s.encode("ascii",errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$","",s,flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+","",s,flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]","",s.lower())
    return re.sub(r"\s+"," ",s).strip()

def snap_coord(x,y): return (round(x/SNAP)*SNAP, round(y/SNAP)*SNAP)

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

# Centroids and plant coords
gdf_mun["centroid"] = gdf_mun.geometry.centroid
gdf_mun["key"] = gdf_mun["NAMEUNIT"].apply(norm)
mun_centroids = {r["key"]: (r["centroid"].x, r["centroid"].y) for _, r in gdf_mun.iterrows()}

plant_coords = {}
for _, r in gdf_plantas.iterrows():
    pid = int(r["Id"]) if "Id" in r else 0
    plant_coords[pid] = (r.geometry.x, r.geometry.y)

# Plant name -> ID mapping for S2
cod = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "A01.Voronoi", "scripts",
                                "tablas_distancias", "codigos_plantas.csv"))
plant_name_to_id = {}
for _, r in cod.iterrows():
    plant_name_to_id[norm(r["MUNICIPIO"])] = int(r["Id"])

# ============================================================
# BUILD ROAD GRAPH
# ============================================================
print("Building road graph...")
G = nx.Graph()
node_coords = {}

for _, road in gdf_roads.iterrows():
    geom = road.geometry
    if geom is None or geom.is_empty: continue
    lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords)-1):
            n1 = snap_coord(coords[i][0], coords[i][1])
            n2 = snap_coord(coords[i+1][0], coords[i+1][1])
            if n1 == n2: continue
            d = math.sqrt((coords[i+1][0]-coords[i][0])**2 + (coords[i+1][1]-coords[i][1])**2)
            node_coords[n1] = (coords[i][0], coords[i][1])
            node_coords[n2] = (coords[i+1][0], coords[i+1][1])
            if not G.has_edge(n1, n2) or d < G[n1][n2]["weight"]:
                G.add_edge(n1, n2, weight=d)

graph_nodes = np.array(list(node_coords.keys()))
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def find_nearest_node(x, y):
    snapped = snap_coord(x, y)
    if snapped in node_coords: return snapped
    dists = (graph_nodes[:,0]-snapped[0])**2 + (graph_nodes[:,1]-snapped[1])**2
    return tuple(graph_nodes[np.argmin(dists)])

# ============================================================
# COMPUTE ROUTES
# ============================================================
def route_between(xy1, xy2):
    n1 = find_nearest_node(xy1[0], xy1[1])
    n2 = find_nearest_node(xy2[0], xy2[1])
    try:
        path = nx.shortest_path(G, n1, n2, weight="weight")
        return [node_coords.get(n, n) for n in path]
    except:
        return [xy1, xy2]

# S1
print("Computing S1 routes...")
df_s1 = pd.read_csv(os.path.join(A01_DATA, "datos_red_real_municipios.csv"))
df_s1["key"] = df_s1["municipio"].apply(norm)
routes_s1 = []
for _, r in df_s1.iterrows():
    mxy = mun_centroids.get(r["key"])
    pid = CONSOLIDATION.get(int(r["planta_id"]), int(r["planta_id"]))
    pxy = plant_coords.get(pid)
    if mxy and pxy:
        routes_s1.append({"coords": route_between(mxy, pxy), "cost": r["C_tot"],
                          "prod": r["produccion_t"], "plant_id": pid})
print(f"  S1: {len(routes_s1)} routes")

# S2 (real flows from Excel)
print("Computing S2 routes...")
df_s2 = pd.read_csv(os.path.join(SCRIPT_DIR, "s2_real_flows.csv"))
def _resolve_pid(pkey):
    pid = plant_name_to_id.get(pkey)
    if pid is None:
        for k, v in plant_name_to_id.items():
            if pkey in k or k in pkey:
                pid = v
                break
    if pid is None:
        return None
    return CONSOLIDATION.get(int(pid), int(pid))

# First pass: resolve consolidated plant id per flow and accumulate S2 throughput
df_s2["pid"] = df_s2["plant"].apply(lambda p: _resolve_pid(norm(p)))
s2_throughput = df_s2.dropna(subset=["pid"]).groupby("pid")["tn"].sum()
routes_s2 = []
for _, r in df_s2.iterrows():
    mkey = norm(r["mun"])
    mxy = mun_centroids.get(mkey)
    pid = r["pid"]
    pid = int(pid) if pid is not None and not pd.isna(pid) else None
    pxy = plant_coords.get(pid) if pid is not None else None
    if mxy and pxy:
        # Model unit cost: transport + treatment, same model as S1/S3
        T_pid = float(s2_throughput.get(pid, 0.0))
        cost_per_t = RHO * r["dist_km"] + _unit_treatment_cost(T_pid)
        routes_s2.append({"coords": route_between(mxy, pxy), "cost": cost_per_t,
                          "prod": r["tn"], "plant_id": pid})
print(f"  S2: {len(routes_s2)} routes")

# S3
print("Computing S3 routes...")
df_s3 = pd.read_csv(os.path.join(A05_DATA, "optimal_assignments.csv"))
df_s3["key"] = df_s3["name"].apply(norm) if "name" in df_s3.columns else df_s3["mun_key"]
routes_s3 = []
for _, r in df_s3.iterrows():
    mxy = mun_centroids.get(r["key"])
    pid = CONSOLIDATION.get(int(r["plant_id"]), int(r["plant_id"]))
    pxy = plant_coords.get(pid)
    if mxy and pxy:
        routes_s3.append({"coords": route_between(mxy, pxy), "cost": r["cost"],
                          "prod": r["prod"], "plant_id": pid})
print(f"  S3: {len(routes_s3)} routes")

# ============================================================
# PLOT: 2+1 LAYOUT
# ============================================================
print("\nGenerating 2+1 panel map...")

fig = plt.figure(figsize=(16, 18))
# Top row: S1 (left), S2 (right)
ax1 = fig.add_axes([0.02, 0.42, 0.47, 0.55])
ax2 = fig.add_axes([0.51, 0.42, 0.47, 0.55])
# Bottom: S3 centered
ax3 = fig.add_axes([0.26, 0.02, 0.47, 0.38])

# Shared color scale across all three
all_costs = [r["cost"] for r in routes_s1 + routes_s2 + routes_s3]
vmin, vmax = np.percentile(all_costs, [5, 95])
cmap = plt.cm.RdYlGn_r
norm_c = plt.Normalize(vmin=vmin, vmax=vmax)

MINOR = 750

def draw_panel(ax, routes, title, show_eliminated=False):
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)
    gdf_roads.plot(ax=ax, color="0.92", linewidth=0.15, zorder=1)
    gdf_mun.boundary.plot(ax=ax, color="0.95", linewidth=0.08, zorder=1)

    for route in sorted(routes, key=lambda r: r["prod"]):
        coords = route["coords"]
        if len(coords) < 2: continue
        c = cmap(norm_c(min(route["cost"], vmax)))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        if route["prod"] < MINOR:
            ax.plot(xs, ys, color=c, linewidth=0.5, alpha=0.35, zorder=2,
                    linestyle=(0,(4,3)), solid_capstyle="round")
        else:
            lw = max(0.7, min(3.0, 0.4 + 0.45 * math.log(max(route["prod"],1))))
            ax.plot(xs, ys, color=c, linewidth=lw, alpha=0.55, zorder=2,
                    solid_capstyle="round")

    active = set(r["plant_id"] for r in routes)
    for pid, (x,y) in plant_coords.items():
        cpid = CONSOLIDATION.get(pid, pid)
        if cpid in active:
            ax.plot(x, y, "k^", ms=4.5, zorder=5, markeredgewidth=0.4)
        elif show_eliminated:
            ax.plot(x, y, "x", color="gray", ms=3.5, zorder=4, markeredgewidth=1.0)

    bounds = gdf_ext.total_bounds
    pad = 5000
    ax.set_xlim(bounds[0]-pad, bounds[2]+pad)
    ax.set_ylim(bounds[1]-pad, bounds[3]+pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

draw_panel(ax1, routes_s1, "(a) Administrative assignment (S1)")
draw_panel(ax2, routes_s2, "(b) Real operational flows (S2)")
draw_panel(ax3, routes_s3, "(c) Cost-optimal assignment (S3)", show_eliminated=True)

# Colorbar horizontal below S3
cbar_ax = fig.add_axes([0.30, 0.005, 0.40, 0.012])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Unit cost (EUR/t)", fontsize=9)

# Legend on S3
legend = [
    Line2D([0],[0], color="black", marker="^", ls="None", ms=6, label="Active plant"),
    Line2D([0],[0], color="gray", marker="x", ls="None", ms=5, mew=1.2, label="Eliminated"),
    Line2D([0],[0], color="0.5", ls=(0,(4,3)), lw=0.8, label="<750 t/y (dashed)"),
    Line2D([0],[0], color="green", ls="-", lw=1.5, alpha=0.6, label="Low cost"),
    Line2D([0],[0], color="red", ls="-", lw=1.5, alpha=0.6, label="High cost"),
]
ax3.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.9)

path = os.path.join(OUTPUT_DIR, "fig_three_panel_routed.png")
plt.savefig(path)
plt.close()
print(f"Saved: {path}")
print("Done.")
