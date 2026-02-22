#!/usr/bin/env python3
"""
generate_network_voronoi_map.py

Comparison map: roads coloured by network-nearest plant assignment,
with Euclidean Voronoi boundaries overlaid.  Uses graph colouring
(4-colour theorem) so that 46 plant territories are shown with only
4-6 distinguishable colours.

Pipeline:
  1. Parse carreteras.geojson -> weighted undirected graph
  2. Snap 46 plant coordinates to nearest graph nodes
  3. Multi-source Dijkstra -> assign every node to its network-nearest plant
  4. Build plant adjacency graph -> greedy 4-colouring
  5. Render: coloured roads + Euclidean Voronoi boundaries + boundary

Inputs (relative to support_material/):
  - carreteras.geojson          (road network, EPSG:25830)
  - codigo/coordenadas_plantas.csv
  - extremadura.geojson

Output:
  - figuras_clean/network_voronoi_comparison.pdf
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.spatial import Voronoi, KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.stats import norm as sp_norm

from shapely.geometry import shape, LineString, box
from shapely.ops import unary_union
from shapely import prepare

import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT    = os.path.dirname(SCRIPT_DIR)
BASE_DIR   = os.path.dirname(SUPPORT)
CODE_DIR   = os.path.join(SUPPORT, 'codigo')

ROADS      = os.path.join(SUPPORT, 'carreteras.geojson')
DIST_EUCL  = os.path.join(CODE_DIR, 'distancias_euclideas.csv')
MUNIC_CSV  = os.path.join(CODE_DIR, 'coordenadas_municipios.csv')
BOUNDARY   = os.path.join(SUPPORT, 'extremadura.geojson')
OUTPUT     = os.path.join(SCRIPT_DIR, 'network_voronoi_comparison.pdf')

# ── Colour palette (Tol muted, colourblind-friendly) ──────────────────
PALETTE = [
    '#4477AA',  # blue
    '#EE6677',  # rose
    '#228833',  # green
    '#CCBB44',  # yellow
    '#AA3377',  # purple
    '#66CCEE',  # cyan
    '#BBBBBB',  # grey (fallback)
]

COORD_ROUND = 1   # round XY to 0.1 m for node matching


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_boundary(path):
    with open(path, encoding='utf-8') as f:
        gj = json.load(f)
    geoms = [shape(feat['geometry']) for feat in gj['features']]
    return unary_union(geoms)


def trilaterate_plants(dist_path, munic_path):
    """
    Recover 46 plant positions from Euclidean distance matrix +
    municipality coordinates, via least-squares trilateration.

    Returns (N, 2) array of plant coordinates in UTM, and list of names.
    """
    from scipy.optimize import least_squares as _lstsq
    import unicodedata, re, difflib

    def _strip(s):
        s = str(s)
        try:
            s = s.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        return re.sub(r'[^A-Za-z0-9]', '', s).upper()

    # Load municipality coordinates
    df_m = pd.read_csv(munic_path, encoding='utf-8')
    muni_xy = {}
    for _, r in df_m.iterrows():
        muni_xy[_strip(r['NOMBRE'])] = np.array([r['X'], r['Y']])

    def find_muni(name):
        key = _strip(name)
        if key in muni_xy:
            return muni_xy[key]
        best, best_r = None, 0
        for k, v in muni_xy.items():
            ratio = difflib.SequenceMatcher(None, key, k).ratio()
            if ratio > best_r:
                best_r = ratio
                best = v
        return best if best_r >= 0.75 else None

    # Load euclidean distances (origin=plant, destination=municipality)
    df_d = pd.read_csv(dist_path, sep=';', decimal=',', encoding='utf-8')
    plant_names = sorted(df_d['origin_id'].unique())

    positions = []
    names = []
    for pname in plant_names:
        sub = df_d[df_d['origin_id'] == pname]
        refs = []
        for _, row in sub.iterrows():
            pos = find_muni(row['destination_id'])
            if pos is not None:
                refs.append((pos, row['distance_m']))
        if len(refs) < 3:
            continue
        refs.sort(key=lambda x: x[1])
        refs = refs[:25]
        pts = np.array([r[0] for r in refs])
        dists = np.array([r[1] for r in refs])
        w = 1.0 / (dists + 1)
        x0 = np.average(pts, weights=w, axis=0)

        def residuals(p, pts=pts, dists=dists):
            return np.sqrt(((pts - p) ** 2).sum(axis=1)) - dists

        result = _lstsq(residuals, x0, method='lm')
        positions.append(result.x)
        names.append(pname)

    return np.array(positions), names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GRAPH CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_graph(roads_path):
    """
    Parse road GeoJSON -> weighted undirected graph.

    Returns
    -------
    nodes      : (N, 2) float64 array of UTM coordinates
    graph      : CSR sparse matrix (N×N), weights = distance in metres
    seg_lines  : list of (Mx2 ndarray, representative_node_idx)
                 for rendering each road polyline
    """
    print("  Loading JSON (this may take 30-60 s for 276 MB)...")
    t = time.time()
    with open(roads_path, encoding='utf-8') as f:
        data = json.load(f)
    n_feat = len(data['features'])
    print(f"  JSON loaded in {time.time()-t:.1f} s  ({n_feat:,} features)")

    node_map  = {}       # (rx, ry) -> index
    node_list = []       # [(x, y), ...]
    rows, cols, wts = [], [], []
    seg_lines = []

    def get_node(x, y):
        key = (round(x, COORD_ROUND), round(y, COORD_ROUND))
        idx = node_map.get(key)
        if idx is None:
            idx = len(node_list)
            node_map[key] = idx
            node_list.append((x, y))
        return idx

    print("  Extracting edges...")
    t = time.time()
    for feat in data['features']:
        for linestring in feat['geometry']['coordinates']:
            coords = []
            first_node = None
            prev = None
            for pt in linestring:
                x, y = pt[0], pt[1]
                idx = get_node(x, y)
                coords.append((x, y))
                if first_node is None:
                    first_node = idx
                if prev is not None and prev != idx:
                    px, py = node_list[prev]
                    d = np.hypot(x - px, y - py)
                    rows.append(prev); cols.append(idx); wts.append(d)
                    rows.append(idx);  cols.append(prev); wts.append(d)
                prev = idx
            if len(coords) >= 2 and first_node is not None:
                seg_lines.append((np.array(coords), first_node))

    del data
    N = len(node_list)
    nodes = np.array(node_list, dtype=np.float64)
    print(f"  Graph: {N:,} nodes, {len(rows)//2:,} edges  "
          f"({time.time()-t:.1f} s)")

    graph = csr_matrix(
        (np.array(wts, dtype=np.float64),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(N, N)
    )
    return nodes, graph, seg_lines


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. NETWORK VORONOI (multi-source Dijkstra)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def network_voronoi(graph, plant_node_ids, plant_coords, all_nodes):
    """
    Assign every graph node to its network-nearest plant.

    Unreachable nodes (disconnected components) fall back to
    Euclidean nearest plant.
    """
    n_plants = len(plant_node_ids)
    print(f"  Dijkstra from {n_plants} sources...")
    t = time.time()
    dist = sp_dijkstra(graph, directed=False, indices=plant_node_ids)
    print(f"  Dijkstra done in {time.time()-t:.1f} s")

    assignment = np.argmin(dist, axis=0)
    min_d      = np.min(dist, axis=0)

    unreachable = np.isinf(min_d)
    n_unr = unreachable.sum()
    if n_unr:
        print(f"  {n_unr:,} unreachable nodes -> Euclidean fallback")
        tree = KDTree(plant_coords)
        _, eucl_idx = tree.query(all_nodes[unreachable])
        assignment[unreachable] = eucl_idx

    return assignment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. GRAPH COLOURING  (4-colour theorem)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def four_colour(plant_coords):
    """
    Colour Voronoi adjacency graph of plant positions with few colours.
    Returns dict  plant_index -> colour_index.
    """
    vor = Voronoi(plant_coords)
    G = nx.Graph()
    G.add_nodes_from(range(len(plant_coords)))
    for p1, p2 in vor.ridge_points:
        if p1 >= 0 and p2 >= 0:
            G.add_edge(p1, p2)
    coloring = nx.coloring.greedy_color(G, strategy='DSATUR')
    n_col = max(coloring.values()) + 1
    print(f"  Greedy colouring: {n_col} colours for "
          f"{len(plant_coords)} plants")
    return coloring, n_col


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. EUCLIDEAN VORONOI OVERLAY (clipped to boundary)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def voronoi_edges_clipped(plant_coords, boundary):
    """Voronoi ridges clipped to the regional boundary."""
    vor = Voronoi(plant_coords)
    center = plant_coords.mean(axis=0)
    lines = []

    for pidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            seg = LineString(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]
            t = vor.vertices[i]
            midpoint = plant_coords[pidx].mean(axis=0)
            tangent  = plant_coords[pidx[1]] - plant_coords[pidx[0]]
            normal   = np.array([-tangent[1], tangent[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len == 0:
                continue
            normal /= norm_len
            if np.dot(midpoint - center, normal) < 0:
                normal = -normal
            seg = LineString([t, t + normal * 300_000])

        clipped = seg.intersection(boundary)
        if clipped.is_empty:
            continue
        if clipped.geom_type == 'MultiLineString':
            for part in clipped.geoms:
                lines.append(np.array(part.coords))
        elif clipped.geom_type == 'LineString':
            lines.append(np.array(clipped.coords))

    return lines


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. MUNICIPALITY CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_municipalities(munic_coords, plant_coords, nodes, assignment,
                            s=0.257, q_star=0.05):
    """
    Classify each municipality into one of three categories:
      0 = yellow : correctly assigned by Voronoi (and by all methods)
      1 = blue   : misallocated by Voronoi, rescued by safety-band framework
      2 = red    : misallocated by Voronoi AND by safety-band framework

    Safety-band criterion: P(mis) = Phi(-ln R / (sqrt(2) s)) > q*
    means the municipality is flagged for network verification.
    """
    n_mun = len(munic_coords)

    # Voronoi assignment (Euclidean nearest plant)
    plant_tree = KDTree(plant_coords)
    euc_dists, euc_idx = plant_tree.query(munic_coords, k=2)
    voronoi_assign = euc_idx[:, 0]

    # Network assignment (snap municipality to nearest graph node)
    node_tree = KDTree(nodes)
    _, snap_ids = node_tree.query(munic_coords)
    network_assign = assignment[snap_ids]

    # Misallocation probability per municipality (Theorem 1)
    d1 = euc_dists[:, 0]
    d2 = euc_dists[:, 1]
    R = d2 / np.maximum(d1, 1e-6)
    p_mis = sp_norm.cdf(-np.log(R) / (np.sqrt(2) * s))

    voronoi_correct = (voronoi_assign == network_assign)
    in_band = (p_mis > q_star)

    classes = np.zeros(n_mun, dtype=int)          # 0 = yellow
    classes[~voronoi_correct & in_band] = 1       # 1 = blue (rescued)
    classes[~voronoi_correct & ~in_band] = 2      # 2 = red  (missed)

    n_y = (classes == 0).sum()
    n_b = (classes == 1).sum()
    n_r = (classes == 2).sum()
    print(f"  Municipality classification: "
          f"{n_y} correct (yellow), {n_b} rescued (blue), {n_r} missed (red)")
    return classes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _clip_segments(seg_lines, boundary):
    """Keep only road segments whose midpoint lies inside the boundary."""
    prepare(boundary)
    from shapely.geometry import Point
    kept = []
    for coords_arr, first_node in seg_lines:
        mid = coords_arr[len(coords_arr) // 2]
        if boundary.contains(Point(mid[0], mid[1])):
            kept.append((coords_arr, first_node))
    return kept


def render(nodes, seg_lines, assignment, coloring, n_colours,
           plant_coords, boundary, vor_lines, output_path,
           munic_coords=None, munic_classes=None):

    palette = PALETTE[:n_colours]

    # Clip roads to Extremadura
    print("  Clipping roads to boundary...")
    seg_lines = _clip_segments(seg_lines, boundary)
    print(f"  {len(seg_lines):,} road segments inside boundary")

    fig, ax = plt.subplots(figsize=(10, 12))

    # Light fill for Extremadura
    if boundary.geom_type == 'MultiPolygon':
        polys = list(boundary.geoms)
    else:
        polys = [boundary]
    for poly in polys:
        from matplotlib.patches import Polygon as MplPoly
        ring = np.array(poly.exterior.coords)
        bg = MplPoly(ring[:, :2], closed=True,
                     facecolor='#F5F5F0', edgecolor='none', zorder=0)
        ax.add_patch(bg)

    # Group road polylines by colour
    groups = {c: [] for c in range(n_colours)}
    for coords_arr, first_node in seg_lines:
        plant_idx  = assignment[first_node]
        colour_idx = coloring[plant_idx]
        groups[colour_idx].append(coords_arr)

    # Draw roads (rasterised to keep PDF small)
    for ci in range(n_colours):
        if groups[ci]:
            lc = LineCollection(groups[ci], colors=palette[ci],
                                linewidths=2.0, alpha=1.0,
                                rasterized=True, zorder=1)
            ax.add_collection(lc)

    # Euclidean Voronoi boundaries
    if vor_lines:
        vlc = LineCollection(vor_lines, colors='#333333', linewidths=0.8,
                             linestyles=(0, (6, 4)), alpha=0.6, zorder=2)
        ax.add_collection(vlc)

    # Extremadura boundary
    for poly in polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='#222222', linewidth=1.6, zorder=3)
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color='#222222', linewidth=0.8, zorder=3)

    # Municipality centroids coloured by classification
    MUNIC_STYLE = {
        #  cls: (colour,      size, marker, edgecolour, edgewidth, zorder, alpha)
        0: ('#228833',   40, 'o', '#0a4a0a', 0.3, 4, 0.70),   # green, subtle
        1: ('#4477AA',   40, 's', 'black',   0.6, 6, 1.00),   # blue square, prominent
        2: ('#CC3311',   55, 'X', 'black',   0.7, 7, 1.00),   # red X, very prominent
    }
    MUNIC_LABELS = {
        0: 'Correct (all methods)',
        1: 'Rescued by safety bands',
        2: 'Missed (still misallocated)',
    }
    if munic_coords is not None and munic_classes is not None:
        for cls in [0, 1, 2]:
            mask = (munic_classes == cls)
            if not mask.any():
                continue
            col, sz, mk, ec, ew, zo, al = MUNIC_STYLE[cls]
            ax.scatter(munic_coords[mask, 0], munic_coords[mask, 1],
                       c=col, s=sz, zorder=zo, marker=mk,
                       edgecolors=ec, linewidths=ew, alpha=al)
    elif munic_coords is not None:
        ax.scatter(munic_coords[:, 0], munic_coords[:, 1],
                   c='black', s=8, zorder=4, marker='o',
                   edgecolors='white', linewidths=0.3, alpha=0.85)

    # Plant locations (red asterisks)
    ax.scatter(plant_coords[:, 0], plant_coords[:, 1],
               c='#CC3311', s=120, zorder=8, marker='*',
               edgecolors='black', linewidths=0.3)

    # Legend (no territory groups -- same colour repeats across zones)
    handles = []
    handles.append(Line2D([0], [0], color='#333333', linestyle='dashed',
                          linewidth=0.8, label='Euclidean Voronoi'))
    handles.append(Line2D([0], [0], color='#222222', linewidth=1.6,
                          label='Region boundary'))
    handles.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='#CC3311', markeredgecolor='black',
                          markersize=10, markeredgewidth=0.3,
                          label='CDW plant'))
    if munic_classes is not None:
        legend_sz = {0: 5, 1: 6, 2: 7}
        for cls in [0, 1, 2]:
            col, _, mk, ec, ew, _, _ = MUNIC_STYLE[cls]
            handles.append(Line2D([0], [0], marker=mk, color='w',
                                  markerfacecolor=col,
                                  markeredgecolor=ec,
                                  markersize=legend_sz[cls],
                                  markeredgewidth=ew,
                                  label=MUNIC_LABELS[cls]))
    ax.legend(handles=handles, loc='lower left', fontsize=7,
              framealpha=0.95, edgecolor='#CCCCCC')

    ax.set_aspect('equal')
    ax.set_xlabel('UTM Easting (m)', fontsize=9)
    ax.set_ylabel('UTM Northing (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor('white')

    minx, miny, maxx, maxy = boundary.bounds
    ax.set_xlim(minx - 5000, maxx + 5000)
    ax.set_ylim(miny - 5000, maxy + 5000)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Network Voronoi comparison map")
    print("=" * 60)

    # 1 ── Plants (trilaterated from Euclidean distance matrix)
    print("\n[1/6] Trilaterating plant positions from distance matrix...")
    plant_coords, plant_names = trilaterate_plants(DIST_EUCL, MUNIC_CSV)
    print(f"  {len(plant_coords)} plants trilaterated")

    # 2 ── Road graph
    print("\n[2/6] Building road network graph...")
    nodes, graph, seg_lines = build_graph(ROADS)

    # 3 ── Snap plants to network
    print("\n[3/6] Snapping plants to nearest network nodes...")
    tree = KDTree(nodes)
    snap_d, plant_node_ids = tree.query(plant_coords)
    print(f"  Snap distance: min {snap_d.min():.0f} m, "
          f"max {snap_d.max():.0f} m, mean {snap_d.mean():.0f} m")

    # 4 ── Multi-source Dijkstra
    print("\n[4/6] Computing network Voronoi (shortest paths)...")
    assignment = network_voronoi(graph, plant_node_ids,
                                 plant_coords, nodes)

    # 5 ── 4-colour
    print("\n[5/6] Graph colouring...")
    coloring, n_colours = four_colour(plant_coords)

    # 6 ── Municipality classification
    print("\n[6/7] Classifying municipalities...")
    df_m = pd.read_csv(MUNIC_CSV, encoding='utf-8')
    munic_coords = df_m[['X', 'Y']].values
    munic_classes = classify_municipalities(
        munic_coords, plant_coords, nodes, assignment)

    # 7 ── Render
    print("\n[7/7] Rendering...")
    boundary  = load_boundary(BOUNDARY)
    vor_lines = voronoi_edges_clipped(plant_coords, boundary)

    render(nodes, seg_lines, assignment, coloring, n_colours,
           plant_coords, boundary, vor_lines, OUTPUT,
           munic_coords=munic_coords, munic_classes=munic_classes)

    print(f"\nTotal time: {time.time()-t0:.0f} s")


if __name__ == '__main__':
    main()
