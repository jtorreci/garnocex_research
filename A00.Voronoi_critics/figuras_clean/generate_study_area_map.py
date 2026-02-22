#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Study Area Map (Figure 6)
====================================

Three-panel figure:
  (A) Study area location — Extremadura boundary silhouette with Spain inset
  (B) Municipalities and CDW plants — scatter + Voronoi edges + boundary
  (C) Topographic stratification — zones by latitude + boundary

Uses trilateration for 46 plant positions (same as generate_voronoi_map.py).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from scipy.spatial import Voronoi
from scipy.optimize import least_squares
import unicodedata
import difflib
import re
import os

# ------------------------------------------------------------------ #
# Parameters
# ------------------------------------------------------------------ #
DUAL_PLANT_MUNIS = {'Miajadas', 'Ribera del Fresno', 'Trujillo'}

# Topographic zone thresholds (UTM Northing, m)
# Yields 139 plains / 104 piedmont / 140 mountain = 383 total
ZONE_PLAIN_PIED = 4_323_000   # Plains < this < Piedmont
ZONE_PIED_MOUNT = 4_400_000   # Piedmont < this < Mountain

# Updated s-hat values from k=5 terrain stratification
S_PLAINS = 0.180
S_PIEDMONT = 0.201
S_MOUNTAINS = 0.301


# ------------------------------------------------------------------ #
# Name matching (reused from generate_voronoi_map.py)
# ------------------------------------------------------------------ #

def _strip(s):
    s = str(s)
    try:
        s = s.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^A-Za-z]', '', s).upper()


def build_muni_lookup(df_coords):
    by_strip = {}
    for _, row in df_coords.iterrows():
        key = _strip(row['NOMBRE'])
        by_strip[key] = np.array([row['X'], row['Y']])

    def lookup(name):
        key = _strip(name)
        if key in by_strip:
            return by_strip[key]
        for plen in range(min(len(key), 12), 4, -1):
            prefix = key[:plen]
            cands = [(k, v) for k, v in by_strip.items() if k[:plen] == prefix]
            if len(cands) == 1:
                return cands[0][1]
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
# Trilateration (reused from generate_voronoi_map.py)
# ------------------------------------------------------------------ #

def trilaterate_plants(df_dist, muni_lookup):
    df = df_dist.sort_values(['InputID', 'TargetID', 'Distance']).copy()
    df['sub'] = df.groupby(['InputID', 'TargetID']).cumcount()
    df['PlantID'] = df.apply(
        lambda r: f"{r['TargetID']}_{r['sub']+1}"
        if r['TargetID'] in DUAL_PLANT_MUNIS else r['TargetID'],
        axis=1)

    plant_ids = sorted(df['PlantID'].unique())

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
        refs = refs[:25]

        pts = np.array([r[0] for r in refs])
        dists = np.array([r[1] for r in refs])

        w = 1.0 / (dists + 1)
        x0 = np.average(pts, weights=w, axis=0)

        def residuals(p):
            return np.sqrt(((pts - p) ** 2).sum(axis=1)) - dists

        result = least_squares(residuals, x0, method='lm')
        positions[pid] = result.x

    return positions


# ------------------------------------------------------------------ #
# Voronoi edge rendering
# ------------------------------------------------------------------ #

def voronoi_segments(vor, bbox):
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


# ------------------------------------------------------------------ #
# Load boundary
# ------------------------------------------------------------------ #

def load_boundary(base_dir):
    gj_path = os.path.join(base_dir, 'extremadura.geojson')
    with open(gj_path, encoding='utf-8') as f:
        gj = json.load(f)
    coords = np.array(gj['features'][0]['geometry']['coordinates'][0][0])
    return coords[:, 0], coords[:, 1]


# ------------------------------------------------------------------ #
# Cookie cutter
# ------------------------------------------------------------------ #

def cookie_cutter(ax, bnd_x, bnd_y, xmin, xmax, ymin, ymax):
    M = 50000
    outer = np.array([[xmin - M, ymin - M], [xmax + M, ymin - M],
                      [xmax + M, ymax + M], [xmin - M, ymax + M],
                      [xmin - M, ymin - M]])
    inner = np.column_stack([bnd_x, bnd_y])
    verts = np.concatenate([outer, inner])
    codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * (len(inner) - 2)
             + [Path.CLOSEPOLY])
    mask_path = Path(verts, codes)
    mask_patch = mpatches.PathPatch(mask_path, facecolor='white',
                                    edgecolor='none', zorder=7)
    ax.add_patch(mask_patch)
    ax.plot(bnd_x, bnd_y, color='#333333', lw=1.2, zorder=8)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def generate_figure():
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

    muni_lookup = build_muni_lookup(df_coords)

    # Trilaterate plants
    plant_pos = trilaterate_plants(df_dist, muni_lookup)
    plants = np.array([plant_pos[k] for k in sorted(plant_pos.keys())])
    n_plants = len(plants)
    print(f'  Plants: {n_plants}')

    # Municipality positions
    muni_x = df_coords['X'].values
    muni_y = df_coords['Y'].values
    n_munis = len(df_coords)
    print(f'  Municipalities: {n_munis}')

    # Boundary
    bnd_x, bnd_y = load_boundary(base_dir)
    bx_min, bx_max = bnd_x.min(), bnd_x.max()
    by_min, by_max = bnd_y.min(), bnd_y.max()

    # Voronoi
    vor = Voronoi(plants)
    margin = 8000
    xmin, xmax = bx_min - margin, bx_max + margin
    ymin, ymax = by_min - margin, by_max + margin
    bbox = [xmin, xmax, ymin, ymax]
    segments = list(voronoi_segments(vor, bbox))

    # Topographic zones
    zones = np.where(muni_y < ZONE_PLAIN_PIED, 'Plains',
             np.where(muni_y < ZONE_PIED_MOUNT, 'Piedmont', 'Mountain'))
    n_plains = (zones == 'Plains').sum()
    n_piedmont = (zones == 'Piedmont').sum()
    n_mountain = (zones == 'Mountain').sum()
    print(f'  Zones: Plains={n_plains}, Piedmont={n_piedmont}, Mountain={n_mountain}')

    # ---- Figure: 3 panels with unequal widths ----
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 2], wspace=0.08)

    # ============================================================ #
    # Panel (A): Study area location — Extremadura silhouette
    # ============================================================ #
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.fill(bnd_x / 1000, bnd_y / 1000, color='#E8E8E8', zorder=1)
    ax_a.plot(bnd_x / 1000, bnd_y / 1000, color='#333333', lw=1.5, zorder=2)

    # Province dividing line (approximate: horizontal at ~4390 km)
    prov_y = 4390
    ax_a.axhline(prov_y, color='#999999', lw=0.8, ls='--', zorder=3,
                 xmin=0.05, xmax=0.95)
    ax_a.text(210, prov_y + 25, 'CÁCERES', ha='center', fontsize=9,
              color='#555555', fontstyle='italic', zorder=4)
    ax_a.text(210, prov_y - 35, 'BADAJOZ', ha='center', fontsize=9,
              color='#555555', fontstyle='italic', zorder=4)

    ax_a.set_aspect('equal')
    ax_a.set_xlim(bx_min / 1000 - 5, bx_max / 1000 + 5)
    ax_a.set_ylim(by_min / 1000 - 5, by_max / 1000 + 5)
    ax_a.set_xlabel('UTM Easting (km)')
    ax_a.set_ylabel('UTM Northing (km)')
    ax_a.set_title('(A) Study area location', fontweight='bold')

    # Scale bar
    sb_x, sb_y = bx_min / 1000 + 10, by_min / 1000 + 5
    ax_a.plot([sb_x, sb_x + 50], [sb_y, sb_y], 'k-', lw=3, zorder=5)
    ax_a.text(sb_x + 25, sb_y + 4, '50 km', ha='center', fontsize=8, zorder=5)

    # ============================================================ #
    # Panel (B): Municipalities and CDW plants
    # ============================================================ #
    ax_b = fig.add_subplot(gs[0, 1])

    # Voronoi edges
    for seg in segments:
        ax_b.plot(seg[:, 0] / 1000, seg[:, 1] / 1000,
                  color='#BBBBBB', lw=0.5, alpha=0.5, zorder=1)

    ax_b.scatter(muni_x / 1000, muni_y / 1000, c='#5B9BD5', s=15,
                 alpha=0.6, zorder=3, label=f'Municipalities')
    ax_b.scatter(plants[:, 0] / 1000, plants[:, 1] / 1000,
                 c='#2E7D32', s=80, marker='^', zorder=5,
                 edgecolors='black', linewidth=0.5,
                 label=f'CDW Plants')

    # Province labels
    ax_b.text(210, 4415 / 1000 * 1000, 'CÁCERES', ha='center', fontsize=10,
              color='#777777', fontstyle='italic', zorder=4,
              transform=ax_b.transData)

    cookie_cutter(ax_b, bnd_x / 1000, bnd_y / 1000,
                  xmin / 1000, xmax / 1000, ymin / 1000, ymax / 1000)

    ax_b.set_aspect('equal')
    ax_b.set_xlim(bx_min / 1000 - 5, bx_max / 1000 + 5)
    ax_b.set_ylim(by_min / 1000 - 5, by_max / 1000 + 5)
    ax_b.set_xlabel('UTM Easting (km)')
    ax_b.set_ylabel('UTM Northing (km)')
    ax_b.set_title('(B) Municipalities and CDW plants', fontweight='bold')
    ax_b.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Scale bar
    sb_x, sb_y = bx_min / 1000 + 10, by_min / 1000 + 5
    ax_b.plot([sb_x, sb_x + 50], [sb_y, sb_y], 'k-', lw=3, zorder=9)
    ax_b.text(sb_x + 25, sb_y + 4, '50 km', ha='center', fontsize=8, zorder=9)

    # ============================================================ #
    # Panel (C): Topographic stratification
    # ============================================================ #
    ax_c = fig.add_subplot(gs[0, 2])

    zone_colors = {
        'Mountain': '#8B6914',   # brown
        'Piedmont': '#E5A520',   # gold
        'Plains':   '#7CB342',   # green
    }

    for zone, color in zone_colors.items():
        mask = zones == zone
        count = mask.sum()
        pct = 100 * count / n_munis
        ax_c.scatter(muni_x[mask] / 1000, muni_y[mask] / 1000,
                     c=color, s=25, alpha=0.7, zorder=3,
                     label=f'{zone} ({count}, {pct:.0f}%)')

    # Zone dividing lines
    ax_c.axhline(ZONE_PLAIN_PIED / 1000, color='#888888', lw=0.8,
                 ls=':', zorder=2)
    ax_c.axhline(ZONE_PIED_MOUNT / 1000, color='#888888', lw=0.8,
                 ls=':', zorder=2)

    # Zone labels on right side
    right_x = bx_max / 1000 + 2
    ax_c.text(right_x, (by_max / 1000 + ZONE_PIED_MOUNT / 1000) / 2,
              'Sistema\nCentral', ha='left', va='center', fontsize=8,
              color='#8B6914', fontstyle='italic')
    ax_c.text(right_x, (ZONE_PIED_MOUNT / 1000 + ZONE_PLAIN_PIED / 1000) / 2,
              'Piedmont\nzone', ha='left', va='center', fontsize=8,
              color='#E5A520', fontstyle='italic')
    ax_c.text(right_x, (ZONE_PLAIN_PIED / 1000 + by_min / 1000) / 2,
              'Tagus-Guadiana\nbasins', ha='left', va='center', fontsize=8,
              color='#7CB342', fontstyle='italic')

    # s-hat annotations on left side
    left_x = bx_min / 1000 + 5
    ax_c.text(left_x, (by_max / 1000 + ZONE_PIED_MOUNT / 1000) / 2,
              f'1,000\u20132,400 m\n$\\hat{{s}} = {S_MOUNTAINS}$',
              ha='left', va='center', fontsize=7.5, color='#8B6914')
    ax_c.text(left_x, (ZONE_PIED_MOUNT / 1000 + ZONE_PLAIN_PIED / 1000) / 2,
              f'200\u2013600 m\n$\\hat{{s}} = {S_PIEDMONT}$',
              ha='left', va='center', fontsize=7.5, color='#E5A520')
    ax_c.text(left_x, (ZONE_PLAIN_PIED / 1000 + by_min / 1000) / 2,
              f'<300 m\n$\\hat{{s}} = {S_PLAINS}$',
              ha='left', va='center', fontsize=7.5, color='#7CB342')

    # Boundary
    ax_c.plot(bnd_x / 1000, bnd_y / 1000, color='#333333', lw=1.2, zorder=6)

    ax_c.set_aspect('equal')
    ax_c.set_xlim(bx_min / 1000 - 5, bx_max / 1000 + 15)
    ax_c.set_ylim(by_min / 1000 - 5, by_max / 1000 + 5)
    ax_c.set_xlabel('UTM Easting (km)')
    ax_c.set_ylabel('UTM Northing (km)')
    ax_c.set_title('(C) Topographic stratification', fontweight='bold')
    ax_c.legend(loc='lower right', fontsize=8, framealpha=0.9,
                title='Topographic zones', title_fontsize=9)

    plt.tight_layout()

    # Save
    for d in [script_dir, os.path.join(script_dir, '..', '..', 'figures')]:
        if os.path.isdir(d):
            p = os.path.join(d, 'study_area_map.pdf')
            fig.savefig(p)
            print(f'  Saved: {p}')
    plt.close()


if __name__ == '__main__':
    print('=' * 60)
    print('GENERATING STUDY AREA MAP')
    print('=' * 60)
    generate_figure()
    print('\nDone.')
