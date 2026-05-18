#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Study Area Location Figure
=====================================

Two-panel figure:
  Left  — Iberian Peninsula (Location.png), dominant panel
  Right — Extremadura silhouette (UTM) with key facts box inside
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def load_boundary(base_dir):
    gj_path = os.path.join(base_dir, 'extremadura.geojson')
    with open(gj_path, encoding='utf-8') as f:
        gj = json.load(f)
    coords = np.array(gj['features'][0]['geometry']['coordinates'][0][0])
    return coords[:, 0], coords[:, 1]


def generate_figure():
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.titlesize': 13,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')
    fig_dir = os.path.join(script_dir, '..', '..', 'figures')

    # Load boundary
    bnd_x, bnd_y = load_boundary(base_dir)
    bx_min, bx_max = bnd_x.min(), bnd_x.max()
    by_min, by_max = bnd_y.min(), bnd_y.max()

    # Load peninsula image
    location_path = os.path.join(fig_dir, 'Location.png')
    if not os.path.exists(location_path):
        print(f'  ERROR: {location_path} not found')
        return
    img = mpimg.imread(location_path)
    print(f'  Peninsula image loaded from {location_path}')

    # Figure: two panels, peninsula dominant
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.05)

    # ---- Left panel: Iberian Peninsula ----
    ax_pen = fig.add_subplot(gs[0, 0])
    ax_pen.imshow(img)
    ax_pen.set_xticks([])
    ax_pen.set_yticks([])
    for spine in ax_pen.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.0)
    ax_pen.set_title('(A) Location in the Iberian Peninsula', fontweight='bold')

    # ---- Right panel: Extremadura silhouette ----
    ax_ext = fig.add_subplot(gs[0, 1])
    ax_ext.fill(bnd_x / 1000, bnd_y / 1000, color='#E8E8E8', zorder=1)
    ax_ext.plot(bnd_x / 1000, bnd_y / 1000, color='#333333', lw=1.8, zorder=2)

    ax_ext.set_aspect('equal')
    ax_ext.set_xlim(bx_min / 1000 - 8, bx_max / 1000 + 8)
    ax_ext.set_ylim(by_min / 1000 - 8, by_max / 1000 + 8)
    ax_ext.set_xlabel('UTM Easting (km)', fontsize=11)
    ax_ext.set_ylabel('UTM Northing (km)', fontsize=11)
    ax_ext.set_title('(B) Extremadura, Spain', fontweight='bold')

    # Scale bar
    sb_x = bx_min / 1000 + 8
    sb_y = by_min / 1000 + 3
    ax_ext.plot([sb_x, sb_x + 50], [sb_y, sb_y], 'k-', lw=3, zorder=5)
    ax_ext.text(sb_x + 25, sb_y + 3, '50 km', ha='center', fontsize=9,
                zorder=5)

    # Key facts box inside the silhouette
    facts = ('Area: 41,635 km$^2$\n'
             'Elevation: 200\u20132,400 m\n'
             '383 municipalities\n'
             '46 CDW treatment plants')
    ax_ext.text((bx_min + bx_max) / 2000, (by_min + by_max) / 2000,
                facts, ha='center', va='center', fontsize=10,
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.85, edgecolor='#999999'),
                zorder=4)

    plt.tight_layout()

    # Save
    for d in [script_dir, fig_dir]:
        if os.path.isdir(d):
            p = os.path.join(d, 'study_area_location.pdf')
            fig.savefig(p)
            print(f'  Saved: {p}')
    plt.close()


if __name__ == '__main__':
    print('=' * 60)
    print('GENERATING STUDY AREA LOCATION FIGURE')
    print('=' * 60)
    generate_figure()
    print('\nDone.')
