# -*- coding: utf-8 -*-
"""
generate_graphical_abstract.py
==============================
Generates a publication-quality Graphical Abstract for A05 (Network Rationalization).
Complies with Elsevier / Resources, Conservation & Recycling (RCR) specifications:
- Landscape layout, 2.6:1 aspect ratio (~13 cm x 5 cm).
- Minimalist design, high legibility in thumbnail size, 2-3 color palette.
- High-resolution raster (300 dpi PNG) and vector (PDF) output.
- Deterministic python-based generation (no generative AI).
"""

import os
import math
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SHP_DIR = os.path.join(ROOT, "data", "shp")
TABLAS_DIR = os.path.join(ROOT, "data", "tablas")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set publication style font settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "savefig.dpi": 300,
})

# Physical plant -> consolidated group mapping
CONSOLIDATION = {
    2: 1, 12: 1, 17: 3, 15: 6, 40: 19, 32: 20,
    23: 22, 26: 22, 28: 22, 34: 27, 35: 27, 37: 27, 38: 36, 43: 36,
}

# Design color palette (Premium, high-contrast, minimalist)
COLOR_BG_EXT = "#f8f9fa"       # Extremadura background
COLOR_BORDER_EXT = "#495057"   # Extremadura boundary
COLOR_MUN_BORDER = "#dee2e6"   # Municipality boundaries (ultra-subtle)
COLOR_VIABLE = "#10ac84"       # Emerald green for viable scale plants (>= 5000 t/yr)
COLOR_INVIABLE = "#ff6b6b"     # Light coral red for unviable plants (< 5000 t/yr)
COLOR_ELIMINATED = "#bdc3c7"   # Light gray for eliminated plants (marks/crosses)
COLOR_TEXT_MAIN = "#2c3e50"    # Dark navy for text
COLOR_ARROW = "#2980b9"        # Blue for transition arrow

# ============================================================
# LOAD DATA AND COMPUTE THROUGHPUTS
# ============================================================
print("Loading spatial data...")
gdf_ext = gpd.read_file(os.path.join(SHP_DIR, "Extremadura.shp"))
gdf_mun = gpd.read_file(os.path.join(SHP_DIR, "municipios.shp"))
gdf_plantas = gpd.read_file(os.path.join(SHP_DIR, "plantas.shp"))

# Reproject to EPSG:25830 (ETRS89 / UTM zone 30N) for metric calculations and accurate coordinates
for gdf in (gdf_ext, gdf_mun, gdf_plantas):
    if gdf.crs and gdf.crs.to_epsg() != 25830:
        gdf.to_crs(epsg=25830, inplace=True)

# Map physical plants to coords
plant_coords = {}
for _, r in gdf_plantas.iterrows():
    pid = int(r["Id"]) if "Id" in r else 0
    plant_coords[pid] = (r.geometry.x, r.geometry.y)

def get_group_coord(gid):
    # Returns coordinates of the representative plant of the group
    if gid in plant_coords:
        return plant_coords[gid]
    members = [k for k, v in CONSOLIDATION.items() if v == gid]
    if members and members[0] in plant_coords:
        return plant_coords[members[0]]
    return None

# Calculate Baseline Throughputs (32 plants)
df_dist = pd.read_csv(os.path.join(TABLAS_DIR, "distancias_reales_plantas_municipios.csv"))
df_dist = df_dist.rename(columns={"origin_id": "plant_id", "destination_id": "municipio", "total_cost": "distance_m"})
df_dist["group_id"] = df_dist["plant_id"].map(lambda x: CONSOLIDATION.get(int(x), int(x)))
df_dist["distance_km"] = df_dist["distance_m"] / 1000.0
df_group = df_dist.groupby(["municipio", "group_id"], as_index=False)["distance_km"].min()

df_mun_data = pd.read_csv(os.path.join(TABLAS_DIR, "datos_voronoi_municipios.csv"))
df_mun_data["municipio"] = df_mun_data["NOMBRE"].astype(str)
prod_lookup = dict(zip(df_mun_data["municipio"], df_mun_data["Prod"]))

idx = df_group.groupby("municipio")["distance_km"].idxmin()
chosen_baseline = df_group.loc[idx].copy()
chosen_baseline["produccion"] = chosen_baseline["municipio"].map(prod_lookup).fillna(0.0)
throughput_baseline = chosen_baseline.groupby("group_id")["produccion"].sum().to_dict()

# Load Iterative Throughputs (24 plants active, 8 eliminated)
df_iter_raw = pd.read_csv(os.path.join(OUT_DIR, "optimal_assignments.csv"))
df_iter = df_iter_raw.rename(columns={"name": "municipio", "prod": "produccion"})
throughput_iter = df_iter.groupby("plant_id")["produccion"].sum().to_dict()

# ============================================================
# GRAPHICAL ABSTRACT FIGURE LAYOUT (13 cm x 5 cm = ~2.6:1 aspect ratio)
# ============================================================
# Set figure dimensions in inches
# SHOW_TITLE = True adds the paper title centered at the top.
# For the official RCR/Elsevier submission, set it to False to avoid redundancy.
SHOW_TITLE = True
PAPER_TITLE = "Network Rationalization of Construction and Demolition Waste Treatment Plants:\nCost-Based Assignment and the Scale–Circularity Nexus — A Case Study of Extremadura, Spain"

fig_w = 7.8  # 19.8 cm
fig_h = 3.25 if SHOW_TITLE else 3.0  # Slightly taller if showing the title
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

# Adjust heights and vertical positioning dynamically depending on SHOW_TITLE
if SHOW_TITLE:
    y_bottom = 0.12
    y_height = 0.74
    y_frame_bottom = 0.09
    y_frame_height = 0.81
else:
    y_bottom = 0.14
    y_height = 0.78
    y_frame_bottom = 0.11
    y_frame_height = 0.84

# 1. Background subplots to draw the panel frames (zorder=1)
ax_bg_left = fig.add_axes([0.015, y_frame_bottom, 0.46, y_frame_height], zorder=1)
ax_bg_right = fig.add_axes([0.525, y_frame_bottom, 0.46, y_frame_height], zorder=1)

for ax in (ax_bg_left, ax_bg_right):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Panel 1 frame: Baseline (soft blue-grey)
frame_left = patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                     boxstyle="round,pad=0.0,rounding_size=0.015",
                                     facecolor="#f4f6f9", edgecolor="#dee2e6", linewidth=0.5, zorder=1)
# Panel 2 frame: Rationalized (soft mint-green)
frame_right = patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                     boxstyle="round,pad=0.0,rounding_size=0.015",
                                     facecolor="#eefaf6", edgecolor="#c2ecc1", linewidth=0.5, zorder=1)

ax_bg_left.add_patch(frame_left)
ax_bg_right.add_patch(frame_right)

# 2. Content subplots (zorder=2, transparent)
ax_left_info = fig.add_axes([0.03, y_bottom, 0.17, y_height], zorder=2)
ax_left_map = fig.add_axes([0.21, y_bottom, 0.25, y_height], zorder=2)
ax_center = fig.add_axes([0.47, y_bottom, 0.06, y_height], zorder=2)
ax_right_map = fig.add_axes([0.54, y_bottom, 0.25, y_height], zorder=2)
ax_right_info = fig.add_axes([0.80, y_bottom, 0.17, y_height], zorder=2)

for ax in (ax_left_info, ax_left_map, ax_center, ax_right_map, ax_right_info):
    ax.patch.set_visible(False)  # Make background transparent

# Background maps limits (standard padding to center and enlarge maps)
bounds = gdf_ext.total_bounds
pad_x = 8000
pad_y = 8000
xlims = (bounds[0] - pad_x, bounds[2] + pad_x)
ylims = (bounds[1] - pad_y, bounds[3] + pad_y)

# Sizing function for plant markers (sqrt relationship for volume area scaling)
MAX_T = 90000.0
def get_marker_size(t):
    if t <= 0:
        return 0
    return 6 + 100 * math.sqrt(t / MAX_T)

# ============================================================
# LEFT PANEL - BASELINE INFO & MAP
# ============================================================
print("Plotting baseline panel...")
ax_left_info.axis("off")
ax_left_info.set_xlim(0, 1)
ax_left_info.set_ylim(0, 1)

# Title for baseline
ax_left_info.text(0.0, 0.94, "BASELINE\nCONFIGURATION", fontsize=9.0, fontweight="bold", color=COLOR_TEXT_MAIN, va="top", zorder=2)
ax_left_info.plot([0, 0.95], [0.81, 0.81], color="#dee2e6", linewidth=0.8, transform=ax_left_info.transAxes, zorder=2)

# Metrics
ax_left_info.text(0.0, 0.74, "32", fontsize=12.5, fontweight="bold", color=COLOR_TEXT_MAIN, va="top", zorder=2)
ax_left_info.text(0.0, 0.64, "plant groups in the\nphysical network", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

ax_left_info.text(0.0, 0.49, "€12.22", fontsize=12.5, fontweight="bold", color=COLOR_TEXT_MAIN, va="top", zorder=2)
ax_left_info.text(0.0, 0.39, "per ton average\nunit cost", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

ax_left_info.text(0.0, 0.24, "Low Scale", fontsize=12.5, fontweight="bold", color=COLOR_TEXT_MAIN, va="top", zorder=2)
ax_left_info.text(0.0, 0.14, "high fragmentation\nand inefficiency", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

# Plot baseline map
gdf_ext.plot(ax=ax_left_map, color=COLOR_BG_EXT, edgecolor=COLOR_BORDER_EXT, linewidth=0.8, zorder=1)
gdf_mun.boundary.plot(ax=ax_left_map, color=COLOR_MUN_BORDER, linewidth=0.08, zorder=2)

for gid, t in throughput_baseline.items():
    if t <= 0:
        continue
    pxy = get_group_coord(gid)
    if pxy is None:
        continue
    color = COLOR_VIABLE if t >= 5000 else COLOR_INVIABLE
    size = get_marker_size(t)
    ax_left_map.scatter(pxy[0], pxy[1], s=size, color=color, edgecolor=COLOR_TEXT_MAIN, linewidths=0.6, zorder=5, alpha=0.9)

ax_left_map.set_xlim(xlims)
ax_left_map.set_ylim(ylims)
ax_left_map.set_aspect("equal")
ax_left_map.axis("off")

# ============================================================
# CENTRAL PANEL - TRANSITION ARROW
# ============================================================
print("Plotting transition panel...")
ax_center.axis("off")
ax_center.set_xlim(0, 100)
ax_center.set_ylim(0, 100)

arrow = patches.FancyArrowPatch((5, 50), (95, 50),
                                 arrowstyle="Simple, tail_width=3, head_width=7, head_length=7",
                                 color=COLOR_ARROW, zorder=2)
ax_center.add_patch(arrow)

ax_center.text(50, 68, "Iterative\ncost\nrealloc.",
               horizontalalignment="center", verticalalignment="center",
               fontsize=6.5, fontweight="bold", color=COLOR_TEXT_MAIN, zorder=3)

ax_center.text(50, 32, "Scale–\ncircularity\nnexus",
               horizontalalignment="center", verticalalignment="center",
               fontsize=6.5, fontweight="bold", color=COLOR_ARROW, zorder=3)

# ============================================================
# RIGHT PANEL - RATIONALIZED INFO & MAP
# ============================================================
print("Plotting rationalized panel...")
ax_right_info.axis("off")
ax_right_info.set_xlim(0, 1)
ax_right_info.set_ylim(0, 1)

# Title for rationalized
ax_right_info.text(0.0, 0.94, "RATIONALIZED\nCONFIGURATION", fontsize=9.0, fontweight="bold", color=COLOR_VIABLE, va="top", zorder=2)
ax_right_info.plot([0, 0.95], [0.81, 0.81], color="#dee2e6", linewidth=0.8, transform=ax_right_info.transAxes, zorder=2)

# Metrics
ax_right_info.text(0.0, 0.74, "24", fontsize=12.5, fontweight="bold", color=COLOR_VIABLE, va="top", zorder=2)
ax_right_info.text(0.0, 0.64, "active plant groups\n(8 eliminated)", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

ax_right_info.text(0.0, 0.49, "€11.34", fontsize=12.5, fontweight="bold", color=COLOR_VIABLE, va="top", zorder=2)
ax_right_info.text(0.0, 0.39, "per ton average\nunit cost", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

ax_right_info.text(0.0, 0.24, "−7.2%", fontsize=12.5, fontweight="bold", color=COLOR_VIABLE, va="top", zorder=2)
ax_right_info.text(0.0, 0.14, "average unit\ncost reduction", fontsize=7.0, color="#555555", va="top", linespacing=1.2, zorder=2)

# Plot rationalized map
gdf_ext.plot(ax=ax_right_map, color=COLOR_BG_EXT, edgecolor=COLOR_BORDER_EXT, linewidth=0.8, zorder=1)
gdf_mun.boundary.plot(ax=ax_right_map, color=COLOR_MUN_BORDER, linewidth=0.08, zorder=2)

for gid in range(1, 47):
    if gid in CONSOLIDATION:
         continue
    pxy = get_group_coord(gid)
    if pxy is None:
         continue
    t = throughput_iter.get(gid, 0.0)
    if t > 0:
        color = COLOR_VIABLE if t >= 5000 else COLOR_INVIABLE
        size = get_marker_size(t)
        ax_right_map.scatter(pxy[0], pxy[1], s=size, color=color, edgecolor=COLOR_TEXT_MAIN, linewidths=0.6, zorder=5, alpha=0.9)
    else:
        if throughput_baseline.get(gid, 0) > 0:
             ax_right_map.scatter(pxy[0], pxy[1], marker="x", s=18, color=COLOR_ELIMINATED, linewidths=0.8, zorder=4)

ax_right_map.set_xlim(xlims)
ax_right_map.set_ylim(ylims)
ax_right_map.set_aspect("equal")
ax_right_map.axis("off")

# ============================================================
# LOWER BANNER
# ============================================================
# Draw neutral background banner
banner_rect = patches.Rectangle((0, 0), 1, 0.08, transform=fig.transFigure,
                                facecolor="#f1f3f5", edgecolor="none", zorder=0)
fig.patches.extend([banner_rect])

fig.text(0.5, 0.035, "23 of 24 plants > 5,000 t/yr  •  viable scale unlocks circularity",
         horizontalalignment="center", verticalalignment="center",
         fontsize=9.0, fontweight="bold", color=COLOR_TEXT_MAIN)

# ============================================================
# OPTIONAL OVERHEAD PAPER TITLE
# ============================================================
if SHOW_TITLE:
    fig.text(0.5, 0.955, PAPER_TITLE, horizontalalignment="center", verticalalignment="center",
             fontsize=8.0, fontweight="bold", color=COLOR_TEXT_MAIN)

# ============================================================
# SAVE OUTPUTS (PNG at 300 dpi and vector PDF)
# ============================================================
out_png = os.path.join(FIG_DIR, "graphical_abstract.png")
out_pdf = os.path.join(FIG_DIR, "graphical_abstract.pdf")

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"Graphical Abstract generated successfully!")
print(f"  Raster (300 dpi): {out_png}")
print(f"  Vectorial (PDF):  {out_pdf}")
