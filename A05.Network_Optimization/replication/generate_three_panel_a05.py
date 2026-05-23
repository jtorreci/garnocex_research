# -*- coding: utf-8 -*-
"""
generate_three_panel_a05.py
===========================
Three-panel routed-flow map for A05 (Network Rationalization paper).

Panels:
    (a) Baseline: 32-plant nearest-plant assignment.
    (b) Greedy pilot: 24-plant nearest-plant assignment after greedy pruning.
    (c) Iterative cost-optimal: 24-plant assignment from the fixed-point heuristic.

Each panel shows routed flows on the real road network, coloured by per-tonne
unit cost; triangles mark active plants and crosses mark eliminated ones.

Self-contained: reads shapefiles from A05.Network_Optimization/data/shp/
and CSVs from A05.Network_Optimization/data/tablas/ and outputs/.
Adapted from A03's generate_three_panel_map.py.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import os
import re

# ============================================================
# PATHS (self-contained: everything under A05/)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SHP_DIR = os.path.join(ROOT, "data", "shp")
TABLAS_DIR = os.path.join(ROOT, "data", "tablas")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "Latex", "Imagenes")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "savefig.dpi": 300,
})

# Physical plant -> consolidated group mapping (same convention as A01/A03)
CONSOLIDATION = {
    2: 1, 12: 1,
    17: 3,
    15: 6,
    40: 19,
    32: 20,
    23: 22, 26: 22, 28: 22,
    34: 27, 35: 27, 37: 27,
    38: 36, 43: 36,
}
SNAP = 50  # road graph node snapping resolution (m)


def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().replace("\n", " ")
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def snap_coord(x, y):
    return (round(x / SNAP) * SNAP, round(y / SNAP) * SNAP)


# ============================================================
# LOAD SHAPEFILES
# ============================================================
print("Loading shapefiles...")
gdf_ext = gpd.read_file(os.path.join(SHP_DIR, "Extremadura.shp"))
gdf_roads = gpd.read_file(os.path.join(SHP_DIR, "carreteras.shp"))
gdf_mun = gpd.read_file(os.path.join(SHP_DIR, "municipios.shp"))
gdf_plantas = gpd.read_file(os.path.join(SHP_DIR, "plantas.shp"))

for gdf in (gdf_ext, gdf_roads, gdf_mun, gdf_plantas):
    if gdf.crs and gdf.crs.to_epsg() != 25830:
        gdf.to_crs(epsg=25830, inplace=True)

gdf_mun["centroid"] = gdf_mun.geometry.centroid
gdf_mun["key"] = gdf_mun["NAMEUNIT"].apply(norm)
mun_centroids_key = {r["key"]: (r["centroid"].x, r["centroid"].y)
                     for _, r in gdf_mun.iterrows()}
# Also a name -> centroid table for greedy CSV (which carries display names)
mun_centroids_name = {norm(r["NAMEUNIT"]): (r["centroid"].x, r["centroid"].y)
                      for _, r in gdf_mun.iterrows()}

plant_coords = {}
for _, r in gdf_plantas.iterrows():
    pid = int(r["Id"]) if "Id" in r else 0
    plant_coords[pid] = (r.geometry.x, r.geometry.y)

# ============================================================
# BUILD ROAD GRAPH (for shortest-path routing per flow line)
# ============================================================
print("Building road graph...")
G = nx.Graph()
node_coords = {}

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
            d = math.sqrt((coords[i + 1][0] - coords[i][0]) ** 2 +
                          (coords[i + 1][1] - coords[i][1]) ** 2)
            node_coords[n1] = (coords[i][0], coords[i][1])
            node_coords[n2] = (coords[i + 1][0], coords[i + 1][1])
            if not G.has_edge(n1, n2) or d < G[n1][n2]["weight"]:
                G.add_edge(n1, n2, weight=d)

graph_nodes = np.array(list(node_coords.keys()))
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def find_nearest_node(x, y):
    snapped = snap_coord(x, y)
    if snapped in node_coords:
        return snapped
    dists = (graph_nodes[:, 0] - snapped[0]) ** 2 + (graph_nodes[:, 1] - snapped[1]) ** 2
    return tuple(graph_nodes[np.argmin(dists)])


def route_between(xy1, xy2):
    n1 = find_nearest_node(xy1[0], xy1[1])
    n2 = find_nearest_node(xy2[0], xy2[1])
    try:
        path = nx.shortest_path(G, n1, n2, weight="weight")
        return [node_coords.get(n, n) for n in path]
    except Exception:
        return [xy1, xy2]


# ============================================================
# COMPUTE PANEL (a): BASELINE 32-PLANT NEAREST-PLANT
# ============================================================
print("Computing baseline (32-plant nearest) assignment...")
# Read distance matrix in 32-group space (apply CONSOLIDATION to the physical
# plant IDs in origin_id, keep min distance per group).
df_dist = pd.read_csv(os.path.join(TABLAS_DIR, "distancias_reales_plantas_municipios.csv"))
df_dist = df_dist.rename(columns={"origin_id": "plant_id",
                                  "destination_id": "municipio",
                                  "total_cost": "distance_m"})
df_dist["group_id"] = df_dist["plant_id"].map(lambda x: CONSOLIDATION.get(int(x), int(x)))
df_dist["distance_km"] = df_dist["distance_m"] / 1000.0
df_group = (df_dist
            .groupby(["municipio", "group_id"], as_index=False)["distance_km"]
            .min())

# Per-municipality production from datos_voronoi_municipios.csv
df_mun_data = pd.read_csv(os.path.join(TABLAS_DIR, "datos_voronoi_municipios.csv"),
                          encoding="utf-8", on_bad_lines="warn")
df_mun_data["municipio"] = df_mun_data["NOMBRE"].astype(str)
prod_lookup = dict(zip(df_mun_data["municipio"], df_mun_data["Prod"]))

# Cost-model parameters (must match the manuscript / piecewise log)
C0 = 40_000.0  # EUR/year
T0 = 5_000.0   # t/year viability threshold
V = 0.35       # EUR/t variable treatment cost
RHO = 0.35     # EUR/(t.km) transport cost


def c_fix(T):
    if T <= 0:
        return C0
    if T < T0:
        return C0
    return C0 * (math.log(T / T0, 2) + 1)


def baseline_assignment_panel():
    """Assign each municipality to its nearest of the 32 plant groups,
    then compute per-tonne treatment cost using piecewise-log model."""
    idx = df_group.groupby("municipio")["distance_km"].idxmin()
    chosen = df_group.loc[idx].copy()
    chosen["produccion"] = chosen["municipio"].map(prod_lookup).fillna(0.0)
    # Aggregate throughput per group
    throughput = chosen.groupby("group_id")["produccion"].sum().to_dict()
    chosen["C_trans"] = chosen["distance_km"] * RHO
    chosen["C_trat"] = chosen["group_id"].apply(
        lambda g: c_fix(throughput.get(g, 0)) / throughput.get(g, 1) + V
        if throughput.get(g, 0) > 0 else 0
    )
    chosen["C_tot"] = chosen["C_trans"] + chosen["C_trat"]
    return chosen.rename(columns={"group_id": "plant_id"})


df_baseline = baseline_assignment_panel()
print(f"  Baseline: {len(df_baseline)} municipalities, "
      f"{df_baseline['plant_id'].nunique()} active groups")

# ============================================================
# LOAD ITERATIVE ASSIGNMENT
# ============================================================
print("Loading iterative cost-based assignment...")
df_iter_raw = pd.read_csv(os.path.join(OUT_DIR, "optimal_assignments.csv"))
df_iter = df_iter_raw.rename(columns={"name": "municipio",
                                       "prod": "produccion",
                                       "cost": "C_tot"})


# ============================================================
# BUILD ROUTES (a) BASELINE AND (b) CHANGED-FLOWS-ONLY
# ============================================================
def build_routes(df, col_municipio, col_plant, col_prod, col_cost,
                 subset_keys=None):
    """Build routed flow records. If subset_keys is provided, restrict to
    municipalities whose normalised key is in that set."""
    routes = []
    for _, r in df.iterrows():
        mun_name = norm(str(r[col_municipio]))
        if subset_keys is not None and mun_name not in subset_keys:
            continue
        mxy = mun_centroids_name.get(mun_name) or mun_centroids_key.get(mun_name)
        if mxy is None:
            continue
        pid = int(r[col_plant])
        # Assignment CSVs use 32-group IDs; map to a physical plant for
        # geographic placement (representative member of the group).
        if pid in plant_coords:
            pxy = plant_coords[pid]
        else:
            members = [k for k, v in CONSOLIDATION.items() if v == pid]
            pxy = plant_coords.get(members[0]) if members else None
        if pxy is None:
            continue
        routes.append({
            "coords": route_between(mxy, pxy),
            "cost": float(r[col_cost]),
            "prod": float(r[col_prod]),
            "plant_id": pid,
            "mun_name": mun_name,
        })
    return routes


print("Computing panel (a) baseline routes (all 32-plant nearest)...")
routes_a = build_routes(df_baseline, "municipio", "plant_id", "produccion", "C_tot")
print(f"  {len(routes_a)} flows")

# Identify municipalities whose assignment CHANGES under the iterative procedure.
df_baseline["mun_key"] = df_baseline["municipio"].apply(norm)
df_iter["mun_key"] = df_iter["municipio"].apply(norm)
df_join = df_baseline[["mun_key", "plant_id"]].rename(columns={"plant_id": "plant_baseline"}) \
            .merge(df_iter[["mun_key", "plant_id", "C_tot"]].rename(
                columns={"plant_id": "plant_iter", "C_tot": "C_iter"}),
                  on="mun_key", how="inner")
changed_keys = set(df_join.loc[df_join["plant_baseline"] != df_join["plant_iter"],
                                "mun_key"])
print(f"  Municipalities that change plant under iterative: {len(changed_keys)}")

print("Computing panel (b) delta routes (changed-only, iterative routing)...")
routes_b = build_routes(df_iter, "municipio", "plant_id", "produccion", "C_tot",
                         subset_keys=changed_keys)
print(f"  {len(routes_b)} flows")

# ============================================================
# PLOT 3-PANEL LAYOUT (2 + 1)
# ============================================================
print("\nRendering three-panel figure (baseline, deltas, cost KDE)...")
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(16, 18))
ax_a = fig.add_axes([0.02, 0.50, 0.47, 0.45])  # top-left: baseline map
ax_b = fig.add_axes([0.51, 0.50, 0.47, 0.45])  # top-right: delta map
ax_c = fig.add_axes([0.10, 0.06, 0.80, 0.34])  # bottom: KDE plot (wider, lower)

all_costs = [r["cost"] for r in routes_a] + list(df_iter["C_tot"])
vmin, vmax = np.percentile(all_costs, [5, 95])
cmap = plt.cm.RdYlGn_r
norm_c = plt.Normalize(vmin=vmin, vmax=vmax)

MINOR = 750  # minor-flow throughput threshold (t/y) for dashed line


def draw_panel(ax, routes, title, show_eliminated=True,
               active_groups_override=None):
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)
    gdf_roads.plot(ax=ax, color="0.92", linewidth=0.15, zorder=1)
    gdf_mun.boundary.plot(ax=ax, color="0.95", linewidth=0.08, zorder=1)

    for route in sorted(routes, key=lambda r: r["prod"]):
        coords = route["coords"]
        if len(coords) < 2:
            continue
        c = cmap(norm_c(min(route["cost"], vmax)))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        if route["prod"] < MINOR:
            ax.plot(xs, ys, color=c, linewidth=0.5, alpha=0.35, zorder=2,
                    linestyle=(0, (4, 3)), solid_capstyle="round")
        else:
            lw = max(0.7, min(3.0, 0.4 + 0.45 * math.log(max(route["prod"], 1))))
            ax.plot(xs, ys, color=c, linewidth=lw, alpha=0.55, zorder=2,
                    solid_capstyle="round")

    active_groups = (active_groups_override if active_groups_override is not None
                     else set(r["plant_id"] for r in routes))
    for pid, (x, y) in plant_coords.items():
        cpid = CONSOLIDATION.get(pid, pid)
        if cpid in active_groups:
            ax.plot(x, y, "k^", ms=4.5, zorder=5, markeredgewidth=0.4)
        elif show_eliminated:
            ax.plot(x, y, "x", color="gray", ms=3.5, zorder=4, markeredgewidth=1.0)

    bounds = gdf_ext.total_bounds
    pad = 5000
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# Panel (a): all 32 plants active
draw_panel(ax_a, routes_a,
           "(a) Baseline: 32-plant nearest-plant assignment",
           show_eliminated=False)

# Panel (b): only the changed flows under the iterative procedure.
# Active set for the triangles = 24 plants of the iterative configuration.
iter_active = set(df_iter["plant_id"].astype(int).unique())
draw_panel(ax_b, routes_b,
           f"(b) Delta: {len(changed_keys)} reassigned flows under iterative cost-based assignment",
           show_eliminated=True,
           active_groups_override=iter_active)

# Shared colorbar across the two maps (well above panel c title to avoid clash)
cbar_ax = fig.add_axes([0.30, 0.46, 0.40, 0.010])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Unit cost (EUR/t)", fontsize=9)
cbar.ax.tick_params(labelsize=7)

# Panel (c): KDE overlay of municipal cost distribution under each configuration.
# The KDE itself is over the municipality-level cost (one observation per
# municipality). The legend labels and the vertical reference lines mark the
# production-weighted system means; for the baseline we read the canonical figure
# from the greedy pruning output (pareto_podado.csv at n_plants=32), so that the
# value displayed on the figure agrees exactly with what is reported in the
# manuscript tables.
costs_baseline = df_baseline["C_tot"].to_numpy()
costs_iter = df_iter["C_tot"].to_numpy()
prod_iter = df_iter["produccion"].to_numpy()

# Canonical baseline cost: read directly from pareto_podado.csv (row n_plants=32).
df_pareto = pd.read_csv(os.path.join(OUT_DIR, "pareto_podado.csv"))
w_baseline = float(df_pareto.loc[df_pareto["n_plants"] == 32, "cost_per_tonne"].iloc[0])
# Iterative production-weighted mean (matches optimal_assignments.csv).
w_iter = (costs_iter * prod_iter).sum() / prod_iter.sum()

xs = np.linspace(min(costs_baseline.min(), costs_iter.min()) - 1,
                 max(costs_baseline.max(), costs_iter.max()) + 1, 600)
kde_b = gaussian_kde(costs_baseline)
kde_i = gaussian_kde(costs_iter)
ax_c.fill_between(xs, kde_b(xs), alpha=0.30, color="#4a6fa5",
                  label=(f"Baseline (32 plants, "
                         f"production-weighted $\\bar{{C}}={w_baseline:.2f}$~EUR/t)"))
ax_c.fill_between(xs, kde_i(xs), alpha=0.30, color="#c44e52",
                  label=(f"Iterative cost-based assignment (24 plants, "
                         f"production-weighted $\\bar{{C}}={w_iter:.2f}$~EUR/t)"))
ax_c.plot(xs, kde_b(xs), color="#4a6fa5", linewidth=1.5)
ax_c.plot(xs, kde_i(xs), color="#c44e52", linewidth=1.5)
ax_c.axvline(w_baseline, color="#4a6fa5", linestyle="--", linewidth=1.0)
ax_c.axvline(w_iter, color="#c44e52", linestyle="--", linewidth=1.0)
ax_c.set_xlabel("Total unit cost per municipality (EUR/t)", fontsize=10)
ax_c.set_ylabel("Density", fontsize=10)
ax_c.set_title("(c) Distribution of municipal unit cost: baseline vs iterative cost-based",
               fontsize=11, fontweight="bold", pad=10)
ax_c.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_c.grid(True, alpha=0.3, linestyle=":")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

# Legend for the two map panels (placed on panel b for visibility)
legend_map = [
    Line2D([0], [0], color="black", marker="^", ls="None", ms=6, label="Active plant"),
    Line2D([0], [0], color="gray", marker="x", ls="None", ms=5, mew=1.2,
           label="Eliminated"),
    Line2D([0], [0], color="0.5", ls=(0, (4, 3)), lw=0.8,
           label="<750 t/y (dashed)"),
    Line2D([0], [0], color="green", ls="-", lw=1.5, alpha=0.6, label="Low cost"),
    Line2D([0], [0], color="red", ls="-", lw=1.5, alpha=0.6, label="High cost"),
]
ax_b.legend(handles=legend_map, loc="lower right", fontsize=7, framealpha=0.9)

out_path = os.path.join(FIG_DIR, "fig_three_panel_a05.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")
print("Done.")
