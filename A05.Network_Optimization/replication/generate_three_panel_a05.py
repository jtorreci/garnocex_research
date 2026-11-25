# -*- coding: utf-8 -*-
"""
generate_three_panel_a05.py
===========================
Three-panel routed-flow map for A05 (Network Rationalization paper).

Panels:
    (a) Baseline: 32-plant nearest-plant assignment, all routed flows.
    (b) Routed flows of the municipalities whose plant changes under the
        iterative cost-based (fixed-point) assignment, 24 active plants.
    (c) Kernel density of municipal unit cost (equal weight per municipality)
        under both configurations, with production-weighted means.

Map flows follow the real road network and are coloured by per-tonne unit
cost (reversed viridis, clipped at the 5th/95th percentiles). The figure is
drawn at its printed size (6.5 in wide) and saved at 600 dpi to
results/figures/fig_three_panel_a05.png.

Production per municipality is looked up by normalised name: the distance
table carries some names with garbled accents (e.g. "Talav n"), and an
exact-name lookup gave those municipalities zero tonnage.

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
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "savefig.dpi": 600,
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
# Key production by the normalised name: the distance matrix carries some
# names with mangled accents (e.g. "Talav n"), so an exact-name lookup drops
# their production and inflates the unit cost of their plant.
prod_lookup = {norm(n): p for n, p in zip(df_mun_data["NOMBRE"], df_mun_data["Prod"])}

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
    chosen["produccion"] = chosen["municipio"].map(lambda n: prod_lookup.get(norm(n), 0.0))
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
_w = (df_baseline["C_tot"] * df_baseline["produccion"]).sum() / df_baseline["produccion"].sum()
print(f"  Baseline: {len(df_baseline)} municipalities, "
      f"{df_baseline['plant_id'].nunique()} active groups, "
      f"{(df_baseline['produccion'] <= 0).sum()} without production, "
      f"max cost {df_baseline['C_tot'].max():.2f}, weighted mean {_w:.4f} EUR/t")

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
# PLOT 3-PANEL LAYOUT (2 maps + KDE), drawn at printed size
# ============================================================
print("\nRendering three-panel figure (baseline, deltas, cost KDE)...")
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

FIG_W = 6.5  # inches; printed at \textwidth (16.6 cm) without scaling
bounds = gdf_ext.total_bounds
PAD = 4000
map_aspect = (bounds[3] - bounds[1] + 2 * PAD) / (bounds[2] - bounds[0] + 2 * PAD)
MAP_W = 2.8
MAP_H = MAP_W * map_aspect
TITLE_H = 0.3   # space above each map for the panel label
CBAR_H = 0.62    # colour bar + map legend row
KDE_H = 1.05     # axes height of panel (c)
KDE_PAD = 0.62   # x label + panel title of (c)
FIG_H = TITLE_H + MAP_H + CBAR_H + KDE_H + KDE_PAD

fig = plt.figure(figsize=(FIG_W, FIG_H))


def axes_in(left, bottom, width, height):
    """Add axes from inch coordinates (origin bottom-left)."""
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


map_bottom = KDE_PAD + KDE_H + CBAR_H
ax_a = axes_in(0.05, map_bottom, MAP_W, MAP_H)
ax_b = axes_in(FIG_W - MAP_W - 0.05, map_bottom, MAP_W, MAP_H)
ax_c = axes_in(0.75, 0.38, FIG_W - 1.0, KDE_H)

all_costs = [r["cost"] for r in routes_a] + list(df_iter["C_tot"])
CLIP_PCT = (5, 95)
vmin, vmax = np.percentile(all_costs, CLIP_PCT)
# Reversed viridis without its palest end: low cost light, high cost dark,
# monotonic in lightness so the scale survives greyscale printing.
cmap = ListedColormap(plt.cm.viridis_r(np.linspace(0.12, 1.0, 256)))
norm_c = plt.Normalize(vmin=vmin, vmax=vmax)
print(f"  Colour scale clipped at percentiles {CLIP_PCT}: {vmin:.2f}-{vmax:.2f} EUR/t")

MINOR = 750  # minor-flow throughput threshold (t/y) for dashed line


def draw_panel(ax, routes, title, show_eliminated=True,
               active_groups_override=None):
    gdf_mun.boundary.plot(ax=ax, color="0.93", linewidth=0.08, zorder=1)
    gdf_roads.plot(ax=ax, color="0.88", linewidth=0.12, zorder=1)
    gdf_ext.boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=3)

    for route in sorted(routes, key=lambda r: r["prod"]):
        coords = route["coords"]
        if len(coords) < 2:
            continue
        c = cmap(norm_c(route["cost"]))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        if route["prod"] < MINOR:
            ax.plot(xs, ys, color=c, linewidth=0.35, alpha=0.6, zorder=2,
                    linestyle=(0, (3, 2)), solid_capstyle="round")
        else:
            lw = max(0.45, min(1.5, 0.2 + 0.17 * math.log(max(route["prod"], 1))))
            ax.plot(xs, ys, color=c, linewidth=lw, alpha=0.8, zorder=2,
                    solid_capstyle="round")

    active_groups = (active_groups_override if active_groups_override is not None
                     else set(r["plant_id"] for r in routes))
    for pid, (x, y) in plant_coords.items():
        cpid = CONSOLIDATION.get(pid, pid)
        if cpid in active_groups:
            ax.plot(x, y, "^", color="black", mec="white", ms=3.6, zorder=5,
                    markeredgewidth=0.3)
        elif show_eliminated:
            ax.plot(x, y, "x", color="#c0392b", ms=3.6, zorder=5, markeredgewidth=0.9)

    ax.set_xlim(bounds[0] - PAD, bounds[2] + PAD)
    ax.set_ylim(bounds[1] - PAD, bounds[3] + PAD)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3, loc="left")
    ax.set_axis_off()


# Panel (a): all 32 plants active
draw_panel(ax_a, routes_a, "(a) Baseline: 32 plants, nearest plant",
           show_eliminated=False)

# Panel (b): only the changed flows under the iterative procedure.
# Active set for the triangles = 24 plants of the iterative configuration.
iter_active = set(df_iter["plant_id"].astype(int).unique())
draw_panel(ax_b, routes_b,
           f"(b) {len(changed_keys)} reassigned municipalities, iterative",
           show_eliminated=True,
           active_groups_override=iter_active)

# Shared colour bar (left) and map legend (right) under the two maps
cbar_ax = axes_in(0.3, KDE_PAD + KDE_H + 0.36, 2.6, 0.09)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
cbar.set_label(f"Unit cost (€/t), clipped at the {CLIP_PCT[0]}th and "
               f"{CLIP_PCT[1]}th percentiles", fontsize=7)
cbar.ax.tick_params(labelsize=7, length=2, pad=1.5)

legend_map = [
    Line2D([0], [0], color="black", marker="^", mec="white", mew=0.3, ls="None",
           ms=5, label="Active plant"),
    Line2D([0], [0], color="#c0392b", marker="x", ls="None", ms=4.5, mew=1.0,
           label="Plant with zero throughput (b)"),
    Line2D([0], [0], color="0.35", ls=(0, (3, 2)), lw=0.6,
           label=f"Flow < {MINOR} t/year"),
    Line2D([0], [0], color="0.35", ls="-", lw=1.2, label="Flow ≥ 750 t/year"),
]
fig.legend(handles=legend_map, loc="center left", fontsize=7, frameon=False, ncol=2,
           bbox_to_anchor=(3.35 / FIG_W, (KDE_PAD + KDE_H + 0.36) / FIG_H),
           handlelength=1.8, columnspacing=1.0, labelspacing=0.3)

# Panel (c): KDE of the municipal cost distribution under each configuration.
# The KDE is over municipality-level cost (one observation per municipality,
# equal weights). The vertical reference lines mark the production-weighted
# system means; for the baseline we read the canonical figure from the greedy
# pruning output (pareto_podado.csv at n_plants=32), so that the value displayed
# agrees exactly with the manuscript tables.
costs_baseline = df_baseline["C_tot"].to_numpy()
costs_iter = df_iter["C_tot"].to_numpy()
prod_iter = df_iter["produccion"].to_numpy()

df_pareto = pd.read_csv(os.path.join(OUT_DIR, "pareto_podado.csv"))
w_baseline = float(df_pareto.loc[df_pareto["n_plants"] == 32, "cost_per_tonne"].iloc[0])
w_iter = (costs_iter * prod_iter).sum() / prod_iter.sum()
print(f"  Production-weighted means: baseline {w_baseline:.2f}, iterative {w_iter:.2f}")

lo = min(costs_baseline.min(), costs_iter.min())
hi = max(costs_baseline.max(), costs_iter.max())
xs = np.linspace(lo, hi, 600)
kde_b = gaussian_kde(costs_baseline)
kde_i = gaussian_kde(costs_iter)
C_BASE, C_ITER = "#2b4c7e", "#d17a00"
ax_c.fill_between(xs, kde_b(xs), color=C_BASE, alpha=0.18, lw=0)
ax_c.fill_between(xs, kde_i(xs), facecolor="none", edgecolor=C_ITER, hatch="////",
                  lw=0, alpha=0.5)
ax_c.plot(xs, kde_b(xs), color=C_BASE, lw=1.2,
          label="Baseline, 32 plants (density)")
ax_c.plot(xs, kde_i(xs), color=C_ITER, lw=1.2, ls=(0, (4, 1.5)),
          label="Iterative, 24 plants (density)")
ax_c.axvline(w_baseline, color=C_BASE, ls=(0, (1, 1.2)), lw=1.0,
             label=f"Baseline mean, {w_baseline:.2f} €/t")
ax_c.axvline(w_iter, color=C_ITER, ls=(0, (5, 1.5, 1, 1.5)), lw=1.0,
             label=f"Iterative mean, {w_iter:.2f} €/t")
ax_c.set_xlim(lo, hi)
ax_c.set_ylim(bottom=0)
ax_c.set_xlabel("Total unit cost per municipality (€/t)", fontsize=7.5, labelpad=2)
ax_c.set_ylabel("Density", fontsize=7.5, labelpad=2)
ax_c.tick_params(labelsize=7, length=2.5, pad=2)
ax_c.set_title("(c) Distribution of municipal unit cost", fontsize=8,
               fontweight="bold", pad=3, loc="left")
ax_c.legend(loc="upper right", fontsize=7, frameon=False, handlelength=2.6)
ax_c.grid(True, alpha=0.3, linestyle=":", lw=0.5)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

out_paths = [
    os.path.join(FIG_DIR, "fig_three_panel_a05.png"),
]
for out_path in out_paths:
    fig.savefig(out_path, dpi=600)
    print(f"Saved: {out_path}")
plt.close(fig)
print("Done.")
