"""Generate the final showcase map for the manuscript:

    extremadura_safety_bands.pdf

Layers (bottom to top):
    1. Extremadura boundary (light fill)
    2. Municipality boundaries (thin grey)
    3. Voronoi cells of the 32 effective plants (after consolidation),
       coloured categorically, semi-transparent
    4. Safety band polygons at q* in {0.20, 0.10, 0.05}, increasing red
       opacity for tighter risk levels (using s_b = 0.326)
    5. Plant points (triangles, dark)
    6. Misallocated municipality centroids (red dots, opaque)

The figure ties the framework together: real Extremadura geometry,
real plant locations, the Euclidean Voronoi assignment those imply,
and the safety bands that flag the territorial strips where the
Euclidean assignment is unreliable under the empirical lognormal
calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.spatial import Voronoi
from scipy.stats import norm
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parent
DATA = ANALYSES / "data"
SHP = DATA / "shp"
SUB_FIG = ANALYSES.parent / "submission" / "figures"
SUB_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plant consolidation (15 km rule, see CHANGELOG and §4.2)
# ---------------------------------------------------------------------------
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

# Regional anisotropy s (muni-to-muni domain): the appropriate scale for
# territorial safety-band visualization, where bands represent the
# terrain-level dispersion of network detour. The predictor s_b = 0.326
# (muni-to-facility) operates per-route and is used for the closed-form
# misallocation count, not for the geographic visualization.
S_REGIONAL = 0.093
S_PREDICTOR = S_REGIONAL  # alias kept for the rest of the script

# Risk levels (smaller q* = tighter safety; rendered in increasing opacity)
RISK_LEVELS = [0.20, 0.10, 0.05]


def consolidate_plant(pid: int) -> int:
    return PLANT_CONSOLIDATION.get(pid, pid)


def load_geo():
    extremadura = gpd.read_file(SHP / "Extremadura.shp")
    munis = gpd.read_file(SHP / "municipios.shp")
    plantas = gpd.read_file(SHP / "plantas.shp")
    print(f"  Extremadura: {extremadura.crs}, {len(extremadura)} polygons")
    print(f"  Munis: {len(munis)} polygons")
    print(f"  Plants (raw): {len(plantas)} points")

    # Consolidate plants: keep only those whose Id maps to itself (canonical)
    plantas["canonical_id"] = plantas["Id"].apply(consolidate_plant)
    canonical_ids = sorted(plantas[plantas["Id"] == plantas["canonical_id"]]["Id"].tolist())
    plantas_canonical = plantas[plantas["Id"].isin(canonical_ids)].copy().reset_index(drop=True)
    print(f"  Plants (canonical, post 15-km consolidation): {len(plantas_canonical)}")

    return extremadura, munis, plantas_canonical


def load_canonical_csv():
    """Load per-muni assignment, beta, misallocation status."""
    df = pd.read_csv(DATA / "municipios_canonical.csv")
    print(f"  Canonical CSV: {len(df)} munis "
          f"({df['misallocated'].sum()} misallocated)")
    return df


def voronoi_polygons(plant_pts: np.ndarray, boundary):
    """Return one polygon per plant in plant_pts order, clipped to boundary.

    plant_pts : (n, 2) UTM coordinates
    boundary  : shapely Polygon/MultiPolygon (the union of munis or Extremadura)
    """
    # Pad with farpoints to close the unbounded Voronoi cells
    minx, miny, maxx, maxy = boundary.bounds
    span = max(maxx - minx, maxy - miny)
    far = 5 * span
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    pad = np.array([
        [cx - far, cy], [cx + far, cy],
        [cx, cy - far], [cx, cy + far],
        [cx - far, cy - far], [cx + far, cy + far],
        [cx - far, cy + far], [cx + far, cy - far],
    ])
    all_pts = np.vstack([plant_pts, pad])

    vor = Voronoi(all_pts)
    polys = [None] * len(plant_pts)
    for i, region_idx in enumerate(vor.point_region[:len(plant_pts)]):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            polys[i] = None
            continue
        poly = Polygon([vor.vertices[v] for v in region])
        polys[i] = poly.intersection(boundary)
    return polys, vor


def _variable_width_band(seg_pts: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                          s: float, q_star: float, n_samples: int = 80) -> Polygon | None:
    """Build a variable-width band polygon along the ridge segment seg_pts
    (between A1 and A2), with t*(Q) = sqrt(2)*s/kappa(Q) * Phi^{-1}(1-q*)
    and kappa(Q) = 2 sin(theta(Q)/2) / d(Q, A1), per Lemma 1.

    seg_pts: (2, 2) array — the two endpoints of the Voronoi ridge.
             (Voronoi ridges are straight; intermediate Q lies on the
             perpendicular bisector of A1A2.)
    Returns a shapely Polygon (or None if degenerate).
    """
    p0, p1 = seg_pts[0], seg_pts[1]
    L = np.linalg.norm(p1 - p0)
    if L < 1.0:
        return None

    # Sample along the ridge
    ts = np.linspace(0.0, 1.0, n_samples)
    Q = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]  # (n, 2)

    # On the perpendicular bisector: d(Q, A1) = d(Q, A2)
    v1 = A1[None, :] - Q
    v2 = A2[None, :] - Q
    d = np.linalg.norm(v1, axis=1)  # (n,)
    # Avoid div0 if Q coincides with a facility (shouldn't on a ridge)
    d = np.where(d < 1.0, 1.0, d)
    cos_theta = np.sum(v1 * v2, axis=1) / (d * np.linalg.norm(v2, axis=1))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    kappa = 2.0 * np.sin(theta / 2.0) / d  # Lemma 1
    # Numerical floor: avoid kappa~0 at far ends (would blow up t*).
    # Cap at the cell-scale distance equivalent.
    kappa = np.maximum(kappa, 1e-7)
    t_star = (np.sqrt(2.0) * s / kappa) * norm.ppf(1.0 - q_star)

    # Local tangent and normal at each Q (forward diff, last sample copies prev)
    tangent = np.diff(Q, axis=0)
    tangent = np.vstack([tangent, tangent[-1:]])
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tnorm = np.where(tnorm < 1e-9, 1.0, tnorm)
    tangent = tangent / tnorm
    # Rotate tangent 90 deg CCW to get the normal
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    upper = Q + normal * t_star[:, None]
    lower = Q - normal * t_star[:, None]

    poly_pts = np.vstack([upper, lower[::-1]])
    poly = Polygon(poly_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def safety_band_polygons(vor: Voronoi, n_canonical: int, boundary, q_levels):
    """Variable-width safety bands along each Voronoi ridge, computed with
    the per-point Lemma 1 curvature (kappa varies along the bisector, so
    the band is narrowest at the ridge midpoint and widens toward the
    Voronoi vertices).
    """
    bands = {q: [] for q in q_levels}
    pts = vor.points

    for (i, j), ridge_v in zip(vor.ridge_points, vor.ridge_vertices):
        if i >= n_canonical or j >= n_canonical:
            continue
        if -1 in ridge_v:
            continue
        if len(ridge_v) < 2:
            continue
        v0 = vor.vertices[ridge_v[0]]
        v1 = vor.vertices[ridge_v[-1]]
        seg = LineString([v0, v1])
        if not seg.intersects(boundary):
            continue

        A1 = pts[i]
        A2 = pts[j]
        seg_pts = np.array([v0, v1])

        for q in q_levels:
            band = _variable_width_band(seg_pts, A1, A2, S_PREDICTOR, q)
            if band is None:
                continue
            band_clipped = band.intersection(boundary)
            if band_clipped.is_empty:
                continue
            bands[q].append(band_clipped)

    out = {}
    for q in q_levels:
        if bands[q]:
            merged = unary_union(bands[q]).intersection(boundary)
            out[q] = merged
        else:
            out[q] = None
    return out


def _normalize_name(s: str) -> str:
    """Normalize a municipality name for matching across CSVs/SHPs with
    different encodings (UTF-8 vs Latin-1, accents stripped, articles
    moved)."""
    if not isinstance(s, str):
        return ""
    out = s.strip().lower()
    out = out.encode("ascii", errors="ignore").decode("ascii")
    out = out.replace(",", "").replace(".", "")
    out = " ".join(out.split())
    return out


def match_misallocated_munis(munis: gpd.GeoDataFrame,
                              df_mis: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return the subset of munis polygons whose normalized name matches one
    of the misallocated municipalities."""
    munis = munis.copy()
    munis["_norm"] = munis["NAMEUNIT"].apply(_normalize_name)
    targets = set(df_mis["municipio"].apply(_normalize_name))
    sel = munis["_norm"].isin(targets)
    matched = munis[sel].copy()
    print(f"  Misallocated polygon match: {len(matched)} of {len(df_mis)} "
          f"target munis matched against {len(munis)} polygons")
    return matched


def main():
    print("Loading geo data...")
    extremadura, munis, plantas = load_geo()
    df = load_canonical_csv()

    # Bounding region: union of all munis (canonical Extremadura outline)
    boundary = unary_union(munis.geometry)

    # --- Plant centroids ---
    plant_pts = np.array([(p.x, p.y) for p in plantas.geometry])
    plant_ids = plantas["Id"].tolist()
    print(f"  Plant centroids ready: {plant_pts.shape}")

    # --- Voronoi cells ---
    print("Computing Voronoi tessellation of canonical plants...")
    cells, vor = voronoi_polygons(plant_pts, boundary)
    print(f"  Done ({sum(1 for c in cells if c is not None)} cells)")

    # --- Safety bands ---
    print(f"Computing safety bands at q ∈ {RISK_LEVELS} with s = {S_PREDICTOR}...")
    bands = safety_band_polygons(vor, n_canonical=len(plant_pts),
                                  boundary=boundary, q_levels=RISK_LEVELS)
    for q, geom in bands.items():
        if geom is None:
            print(f"  q={q}: empty")
        else:
            area_km2 = geom.area / 1e6
            print(f"  q={q}: area = {area_km2:.0f} km^2")

    # --- Misallocated munis: polygons + centroids ---
    df_mis = df[df["misallocated"] == 1].copy()
    print(f"  Misallocated munis: {len(df_mis)}")
    munis_mis = match_misallocated_munis(munis, df_mis)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # 1. Extremadura outline (background)
    extremadura.plot(ax=ax, facecolor="#f7f7f7",
                      edgecolor="#222222", linewidth=1.0, zorder=1)

    # 2. Municipality boundaries
    munis.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.3, zorder=2)

    # 3. Voronoi cells (categorical colors, semi-transparent)
    cmap = plt.get_cmap("tab20", len(plant_pts))
    cell_records = []
    for idx, (cell, pid) in enumerate(zip(cells, plant_ids)):
        if cell is None or cell.is_empty:
            continue
        c = cmap(idx % 20)
        c_rgba = (c[0], c[1], c[2], 0.18)
        cell_records.append({"plant_id": pid, "geometry": cell, "color": c_rgba})
    gdf_cells = gpd.GeoDataFrame(cell_records, crs=plantas.crs)
    gdf_cells.plot(ax=ax, color=gdf_cells["color"],
                    edgecolor="#444444", linewidth=0.6, zorder=3)

    # 4. Safety bands (concentric, variable-width, computed via Lemma 1).
    # Each smaller-q* band is a superset of the larger-q* one (smaller q*
    # = stricter tolerance = wider unreliable strip). Plot widest first
    # (q*=5%, pale) and narrowest last (q*=20%, dark) so the result reads
    # as concentric layers of increasing risk.
    # High-contrast palette: pale gold -> orange -> dark red.
    band_colors = {0.05: ("#fff2b3", 0.42),  # outermost: pale gold
                   0.10: ("#f4751a", 0.55),  # middle: pure orange
                   0.20: ("#7a0316", 0.85)}  # core: very dark red
    for zorder_offset, q in enumerate([0.05, 0.10, 0.20]):
        geom = bands.get(q)
        if geom is None or geom.is_empty:
            continue
        color, alpha = band_colors[q]
        gpd.GeoSeries([geom], crs=plantas.crs).plot(
            ax=ax, facecolor=color, edgecolor="none",
            alpha=alpha, zorder=4 + zorder_offset
        )

    # 5. Municipality population centres (points): the predictor operates
    # on these, not on the muni polygons. Blue = correctly assigned,
    # red = misallocated (Voronoi nearest != network nearest).
    df_ok = df[df["misallocated"] == 0]
    df_mis = df[df["misallocated"] == 1]
    ax.scatter(df_ok["utm_x"].values, df_ok["utm_y"].values,
               marker="o", s=22, c="#1f77b4",
               edgecolors="white", linewidths=0.5, zorder=7,
               label="Correctly assigned municipalities (322)")
    ax.scatter(df_mis["utm_x"].values, df_mis["utm_y"].values,
               marker="o", s=30, c="#d62728",
               edgecolors="white", linewidths=0.7, zorder=7.5,
               label="Misallocated municipalities (61)")

    # 6. Plants (triangles)
    ax.scatter(plant_pts[:, 0], plant_pts[:, 1],
               marker="^", s=140, c="#1a1a1a",
               edgecolors="white", linewidths=1.3, zorder=8,
               label="Treatment facilities (32)")

    # --- Cosmetics ---
    ax.set_aspect("equal", adjustable="datalim")
    minx, miny, maxx, maxy = boundary.bounds
    pad = 5000
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # Axis in km
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{int(t/1000)}" for t in xticks])
    ax.set_yticklabels([f"{int(t/1000)}" for t in yticks])
    ax.set_xlabel("UTM east (km, EPSG:25830)")
    ax.set_ylabel("UTM north (km)")

    # Legend for safety bands + symbols
    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1a1a1a",
                markersize=12, markeredgecolor="white",
                label="Treatment facilities (32 effective)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                markersize=8, markeredgecolor="white",
                label="Municipalities, correctly assigned (322)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                markersize=9, markeredgecolor="white",
                label="Municipalities, misallocated (61)"),
        Patch(facecolor=band_colors[0.20][0], alpha=band_colors[0.20][1],
                label=r"Risk $> 20\%$ (inner core)"),
        Patch(facecolor=band_colors[0.10][0], alpha=band_colors[0.10][1],
                label=r"Risk $> 10\%$"),
        Patch(facecolor=band_colors[0.05][0], alpha=band_colors[0.05][1],
                label=r"Risk $> 5\%$ (outer band)"),
        Patch(facecolor="#cccccc", alpha=0.4,
                label="Voronoi cells of effective facilities"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              framealpha=0.95, fontsize=10)

    ax.set_title(
        r"Extremadura: Euclidean Voronoi assignment, safety bands and misallocations"
        f"\n(regional anisotropy $s = {S_REGIONAL:.3f}$; "
        f"61/383 misallocated)",
        fontsize=12,
    )

    fig.tight_layout()
    out = SUB_FIG / "extremadura_safety_bands.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.name} and PNG variant.")


if __name__ == "__main__":
    main()
