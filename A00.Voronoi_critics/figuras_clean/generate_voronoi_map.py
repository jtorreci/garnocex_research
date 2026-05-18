#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate two-panel Voronoi + Safety Bands map figure.

Panel (a): Euclidean Voronoi tessellation of 46 CDW plants in Extremadura.
Panel (b): Safety bands overlay — municipalities in green (safe, no verification
           needed) or red (within the safety band, require network verification).

The safety band is defined by Theorem 1:
    P(mis) = Phi(-ln(R) / (sqrt(2)*s))  where R = d2/d1
A municipality is flagged if P(mis) > q*.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.spatial import Voronoi, ConvexHull
from scipy.optimize import least_squares
from scipy import stats
import unicodedata
import difflib
import re
import os

# ------------------------------------------------------------------ #
# Parameters
# ------------------------------------------------------------------ #
S_HAT = 0.257       # Global Extremadura dispersion (k=5 nearest)
Q_STAR = 0.05       # Risk tolerance (5%)
GRID_RES = 500      # Grid resolution for contour

DUAL_PLANT_MUNIS = {'Miajadas', 'Ribera del Fresno', 'Trujillo'}


# ------------------------------------------------------------------ #
# Name matching
# ------------------------------------------------------------------ #

def _strip(s):
    """Aggressively normalize: NFKD, strip combining, keep only A-Z."""
    s = str(s)
    try:
        s = s.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^A-Za-z]', '', s).upper()


def build_muni_lookup(df_coords):
    """Build a robust lookup: raw_name -> (X, Y) for all 383 municipalities."""
    # Primary index by stripped name
    by_strip = {}
    raw_list = []
    for _, row in df_coords.iterrows():
        key = _strip(row['NOMBRE'])
        by_strip[key] = np.array([row['X'], row['Y']])
        raw_list.append((row['NOMBRE'], key, row['X'], row['Y']))

    def lookup(name):
        """Return (X, Y) or None."""
        key = _strip(name)
        if key in by_strip:
            return by_strip[key]
        # Fallback: match by longest common prefix (>= 5 chars)
        for plen in range(min(len(key), 12), 4, -1):
            prefix = key[:plen]
            cands = [(k, v) for k, v in by_strip.items() if k[:plen] == prefix]
            if len(cands) == 1:
                return cands[0][1]
        # Fallback 2: fuzzy match via SequenceMatcher
        best, best_ratio = None, 0
        for k, v in by_strip.items():
            ratio = difflib.SequenceMatcher(None, key, k).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = v
        if best_ratio >= 0.80:
            return best
        return None

    return lookup


# ------------------------------------------------------------------ #
# Plant trilateration
# ------------------------------------------------------------------ #

def trilaterate_plants(df_dist, muni_lookup):
    """Recover 46 plant positions from Euclidean distances via trilateration."""
    # Assign unique plant IDs: for dual-plant municipalities, distinguish by distance
    df = df_dist.sort_values(['InputID', 'TargetID', 'Distance']).copy()
    df['sub'] = df.groupby(['InputID', 'TargetID']).cumcount()
    df['PlantID'] = df.apply(
        lambda r: f"{r['TargetID']}_{r['sub']+1}"
        if r['TargetID'] in DUAL_PLANT_MUNIS else r['TargetID'],
        axis=1)

    plant_ids = sorted(df['PlantID'].unique())

    # Build reference municipality positions
    muni_pos = {}
    for m in df['InputID'].unique():
        pos = muni_lookup(m)
        if pos is not None:
            muni_pos[m] = pos

    positions = {}
    for pid in plant_ids:
        sub = df[df['PlantID'] == pid]
        refs = [(muni_pos[r['InputID']], r['Distance'])
                for _, r in sub.iterrows() if r['InputID'] in muni_pos]
        if len(refs) < 3:
            continue
        refs.sort(key=lambda x: x[1])
        refs = refs[:25]  # Use 25 closest municipalities

        pts = np.array([r[0] for r in refs])
        dists = np.array([r[1] for r in refs])

        # Initial guess: distance-weighted centroid
        w = 1.0 / (dists + 1)
        x0 = np.average(pts, weights=w, axis=0)

        def residuals(p):
            return np.sqrt(((pts - p) ** 2).sum(axis=1)) - dists

        result = least_squares(residuals, x0, method='lm')
        positions[pid] = result.x

    return positions, df


# ------------------------------------------------------------------ #
# Voronoi edge rendering
# ------------------------------------------------------------------ #

def voronoi_segments(vor, bbox):
    """Yield line segments for all Voronoi ridges (finite + extended infinite)."""
    center = vor.points.mean(axis=0)
    diag = max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 1.5
    for pidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            yield vor.vertices[simplex]
        else:
            i = simplex[simplex >= 0][0]
            t = vor.points[pidx[1]] - vor.points[pidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            mid = vor.points[pidx].mean(axis=0)
            direction = np.sign(np.dot(mid - center, n)) * n
            yield np.array([vor.vertices[i], vor.vertices[i] + direction * diag])


def load_boundary(base_dir):
    """Load Extremadura boundary from GeoJSON, return matplotlib Path."""
    gj_path = os.path.join(base_dir, 'extremadura.geojson')
    with open(gj_path, encoding='utf-8') as f:
        gj = json.load(f)
    feat = gj['features'][0]
    # MultiPolygon → take first polygon, outer ring
    coords = np.array(feat['geometry']['coordinates'][0][0])
    return Path(coords[:, :2])  # x, y only


def boundary_mask(boundary_path, xg, yg):
    """Mask grid to Extremadura GeoJSON boundary."""
    Xg, Yg = np.meshgrid(xg, yg)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    return boundary_path.contains_points(pts).reshape(Xg.shape)


# ------------------------------------------------------------------ #
# Main figure
# ------------------------------------------------------------------ #

def generate_figure():
    """Generate the two-panel Voronoi + safety bands map."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'axes.titlesize': 12,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')

    # Load data
    df_coords = pd.read_csv(
        os.path.join(base_dir, 'codigo', 'coordenadas_municipios.csv'),
        encoding='utf-8')
    df_dist = pd.read_csv(
        os.path.join(base_dir, 'tables', 'D_euclidea_plantas_clean.csv'),
        encoding='latin-1')
    df_netw = pd.read_csv(
        os.path.join(base_dir, 'codigo', 'tablas',
                     'D_real_plantas_clean_corrected.csv'),
        encoding='latin-1')

    muni_lookup = build_muni_lookup(df_coords)

    # Trilaterate all 46 plants
    plant_pos, df_full = trilaterate_plants(df_dist, muni_lookup)
    plants = np.array([plant_pos[k] for k in sorted(plant_pos.keys())])
    n_plants = len(plants)
    print(f'  Plants trilaterated: {n_plants}')

    # Municipality data: coordinates + actual misallocation status
    # Use original row indices to distinguish plants within dual-plant municipalities
    # (both CSVs are row-aligned: row i in Euclidean ↔ row i in Network = same plant)
    df_dist_ri = df_dist.reset_index()   # 'index' col = original row number
    df_netw_ri = df_netw.reset_index()
    muni_list = []
    for muni_name in df_dist['InputID'].unique():
        pos = muni_lookup(muni_name)
        if pos is None:
            continue
        sub_e = df_dist_ri[df_dist_ri['InputID'] == muni_name].sort_values('Distance')
        sub_r = df_netw_ri[df_netw_ri['origin_id'] == muni_name].sort_values('total_cost')
        if len(sub_e) < 2 or len(sub_r) < 1:
            continue
        d1, d2 = sub_e.iloc[0]['Distance'], sub_e.iloc[1]['Distance']
        if d1 <= 0:
            continue
        R = d2 / d1
        p_mis = stats.norm.cdf(-np.log(R) / (np.sqrt(2) * S_HAT))
        # Actual misallocation: Euclidean-nearest ≠ Network-nearest
        # Compare original row indices (not municipality names) to handle
        # dual-plant municipalities (Miajadas, Ribera del Fresno, Trujillo)
        e_nearest_row = sub_e.iloc[0]['index']
        r_nearest_row = sub_r.iloc[0]['index']
        wrong = (e_nearest_row != r_nearest_row)
        muni_list.append({
            'x': pos[0], 'y': pos[1],
            'R': R, 'p_mis': p_mis,
            'wrong': wrong,
        })
    munis = pd.DataFrame(muni_list)
    n_munis = len(munis)
    n_wrong = int(munis['wrong'].sum())
    n_correct = n_munis - n_wrong
    print(f'  Municipalities: {n_munis}')
    print(f'  Correctly assigned: {n_correct}, Misallocated: {n_wrong}')

    # Extremadura boundary from GeoJSON
    boundary_path = load_boundary(base_dir)
    bnd_verts = boundary_path.vertices
    bx_min, by_min = bnd_verts.min(axis=0)
    bx_max, by_max = bnd_verts.max(axis=0)

    # Voronoi
    vor = Voronoi(plants)
    margin = 8000
    xmin, xmax = bx_min - margin, bx_max + margin
    ymin, ymax = by_min - margin, by_max + margin
    bbox = [xmin, xmax, ymin, ymax]

    # Probability grid
    print('  Computing misallocation probability grid...')
    xg = np.linspace(xmin, xmax, GRID_RES)
    yg = np.linspace(ymin, ymax, GRID_RES)
    Xg, Yg = np.meshgrid(xg, yg)
    gpts = np.column_stack([Xg.ravel(), Yg.ravel()])
    dists = np.sqrt(((gpts[:, None, :] - plants[None, :, :]) ** 2).sum(axis=2))
    idx = np.argpartition(dists, 2, axis=1)[:, :2]
    d1g = dists[np.arange(len(gpts)), idx[:, 0]]
    d2g = dists[np.arange(len(gpts)), idx[:, 1]]
    d_min, d_max = np.minimum(d1g, d2g), np.maximum(d1g, d2g)
    Rg = np.clip(d_max / d_min, 1.0001, None)
    P_mis = stats.norm.cdf(-np.log(Rg) / (np.sqrt(2) * S_HAT)).reshape(GRID_RES, GRID_RES)
    mask = boundary_mask(boundary_path, xg, yg)
    P_masked = np.where(mask, P_mis, np.nan)

    segments = list(voronoi_segments(vor, bbox))

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # -- Helper: draw Extremadura boundary + clip exterior --
    def cookie_cutter(ax):
        """White mask covering everything OUTSIDE the boundary."""
        M = 50000
        # Outer rectangle (CCW in math coords)
        outer = np.array([[xmin - M, ymin - M], [xmax + M, ymin - M],
                          [xmax + M, ymax + M], [xmin - M, ymax + M],
                          [xmin - M, ymin - M]])
        # Inner hole: boundary is CW (checked), use as-is for hole
        inner = bnd_verts  # already CW → opposite to outer CCW
        verts = np.concatenate([outer, inner])
        codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                 + [Path.MOVETO] + [Path.LINETO] * (len(inner) - 2)
                 + [Path.CLOSEPOLY])
        mask_path = Path(verts, codes)
        mask_patch = mpatches.PathPatch(mask_path, facecolor='white',
                                        edgecolor='none', zorder=7)
        ax.add_patch(mask_patch)
        # Boundary outline on top
        ax.plot(bnd_verts[:, 0], bnd_verts[:, 1],
                color='#333333', lw=1.2, zorder=8)

    def setup_panel(ax):
        """Set limits, hide axes."""
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ============================================================ #
    # Panel (a): Plain Voronoi
    # ============================================================ #
    setup_panel(ax1)
    # Voronoi edges (clipped to boundary visually by region fill)
    for seg in segments:
        ax1.plot(*seg.T, color='#555555', lw=0.7, alpha=0.6, zorder=1)
    ax1.scatter(*plants.T, c='black', s=40, marker='^', zorder=5,
                label=f'CDW plants ({n_plants})', edgecolors='white', linewidth=0.3)
    ax1.scatter(munis['x'], munis['y'], c='#888888', s=6, alpha=0.5, zorder=3,
                label=f'Municipalities ({n_munis})')
    cookie_cutter(ax1)
    ax1.set_title('(a) Euclidean Voronoi tessellation')
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)

    # ============================================================ #
    # Panel (b): Misallocations + safety band background
    # ============================================================ #
    setup_panel(ax2)
    # Safety band contours at 5%, 10%, 20% probability levels
    ax2.contourf(Xg, Yg, P_masked,
                 levels=[0.20, 1.0],
                 colors=['#FFCDD2'], alpha=0.45, zorder=1)   # 20%+ : light red
    ax2.contourf(Xg, Yg, P_masked,
                 levels=[0.10, 0.20],
                 colors=['#FFE0B2'], alpha=0.45, zorder=1)   # 10-20%: light orange
    ax2.contourf(Xg, Yg, P_masked,
                 levels=[0.05, 0.10],
                 colors=['#FFF9C4'], alpha=0.45, zorder=1)   # 5-10% : light yellow
    # Dashed contour lines at each threshold
    ax2.contour(Xg, Yg, P_masked, levels=[0.05],
                colors=['#F57F17'], linewidths=1.0, linestyles='--',
                alpha=0.7, zorder=2)
    ax2.contour(Xg, Yg, P_masked, levels=[0.10],
                colors=['#E65100'], linewidths=0.8, linestyles=':',
                alpha=0.7, zorder=2)
    ax2.contour(Xg, Yg, P_masked, levels=[0.20],
                colors=['#B71C1C'], linewidths=0.8, linestyles='-.',
                alpha=0.7, zorder=2)
    # Legend patches for the bands
    from matplotlib.lines import Line2D
    band_handles = [
        mpatches.Patch(facecolor='#FFF9C4', alpha=0.45, edgecolor='#F57F17',
                       linestyle='--', label=r'$5\%< P_{\mathrm{mis}} \leq 10\%$'),
        mpatches.Patch(facecolor='#FFE0B2', alpha=0.45, edgecolor='#E65100',
                       linestyle=':', label=r'$10\%< P_{\mathrm{mis}} \leq 20\%$'),
        mpatches.Patch(facecolor='#FFCDD2', alpha=0.45, edgecolor='#B71C1C',
                       linestyle='-.', label=r'$P_{\mathrm{mis}} > 20\%$'),
    ]
    # Voronoi edges
    for seg in segments:
        ax2.plot(*seg.T, color='#999999', lw=0.5, alpha=0.4, zorder=1)
    # Municipalities: correct (green) vs misallocated (red)
    correct = munis[~munis['wrong']]
    wrong_df = munis[munis['wrong']]
    ax2.scatter(correct['x'], correct['y'], c='#2E7D32', s=10, zorder=3,
                alpha=0.6, label=f'Correctly assigned ({n_correct})')
    ax2.scatter(wrong_df['x'], wrong_df['y'], c='#D32F2F', s=30, marker='x',
                zorder=5, alpha=0.9, linewidths=1.5,
                label=f'Misallocated ({n_wrong})')
    ax2.scatter(*plants.T, c='black', s=40, marker='^', zorder=5,
                edgecolors='white', linewidth=0.3)
    cookie_cutter(ax2)
    ax2.set_title(f'(b) Misallocations and safety bands ($\\hat{{s}} = {S_HAT}$)')
    # Combine scatter and band legend handles
    scatter_handles, scatter_labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=scatter_handles + band_handles,
               loc='lower left', fontsize=8, framealpha=0.9)

    plt.tight_layout(w_pad=2)

    # Save
    for d in [script_dir, os.path.join(script_dir, '..', '..', 'figures')]:
        if os.path.isdir(d):
            p = os.path.join(d, 'voronoi_safety_bands_map.pdf')
            fig.savefig(p); print(f'  Saved: {p}')
    plt.close()


if __name__ == '__main__':
    print('=' * 60)
    print('GENERATING VORONOI + SAFETY BANDS MAP (46 plants, 383 munis)')
    print('=' * 60)
    generate_figure()
    print('\nDone.')
