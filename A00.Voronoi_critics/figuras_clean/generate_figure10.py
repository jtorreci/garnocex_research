#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Safety Bands Figure (Corollary 1)
==========================================

Based on the exact Corollary formula:
    t*(q*) = (sqrt(2) * s / kappa) * Phi^{-1}(1 - q*)

Panel (a): Effect of terrain impedance s at fixed risk tolerance q* = 5%
Panel (b): Effect of risk tolerance q* at fixed Extremadura s = 0.257

Empirical s values from k=5 terrain stratification:
  - Plains (south):      s_hat = 0.180
  - Piedmont (central):  s_hat = 0.201
  - Global Extremadura:  s_hat = 0.257
  - Mountains (north):   s_hat = 0.301
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'text.usetex': False,
    })


def t_star(s, q_star, kappa):
    """Corollary formula: t*(q*) = (sqrt(2) * s / kappa) * Phi^{-1}(1 - q*)."""
    return (np.sqrt(2) * s / kappa) * stats.norm.ppf(1 - q_star)


def generate_safety_bands():
    """Generate two-panel safety bands figure."""
    set_publication_style()

    # Geometric parameter range (km^-1), realistic for regional Voronoi cells
    # kappa = 2*sin(theta/2)/d  with d ~ 5..50 km, theta ~ pi/3..pi
    kappa = np.linspace(0.03, 0.50, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------------ #
    # Panel (a): Effect of terrain impedance s  (fixed q* = 5%)
    # ------------------------------------------------------------------ #
    q_fixed = 0.05

    terrain = [
        (0.180, 'Southern plains',      '#2196F3', '-',  1.6),
        (0.201, 'Central piedmont',     '#FF9800', '--', 1.6),
        (0.257, 'Extremadura (global)', '#2E7D32', '-',  2.4),
        (0.301, 'Northern mountains',   '#D32F2F', '-.', 1.6),
    ]

    for s_val, label, color, ls, lw in terrain:
        t_vals = t_star(s_val, q_fixed, kappa)
        ax1.plot(kappa, t_vals, color=color, ls=ls, lw=lw,
                 label=fr'{label} ($\hat{{s}}={s_val:.3f}$)')

    ax1.set_xlabel(r'Geometric parameter $\kappa$ (km$^{-1}$)')
    ax1.set_ylabel(r'Safety band width $t^*$ (km)')
    ax1.set_title(f'(a) Effect of terrain impedance  ($q^* = {q_fixed*100:.0f}$%)')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='0.7')
    ax1.set_xlim(0.03, 0.50)
    ax1.set_ylim(0, 25)

    # ------------------------------------------------------------------ #
    # Panel (b): Effect of risk tolerance q*  (fixed s = 0.257)
    # ------------------------------------------------------------------ #
    s_fixed = 0.257

    risks = [
        (0.01, r'$q^* = 1\%$',  '#7B1FA2', '-',  1.8),
        (0.05, r'$q^* = 5\%$',  '#2E7D32', '-',  2.4),
        (0.10, r'$q^* = 10\%$', '#FF9800', '--', 1.8),
        (0.20, r'$q^* = 20\%$', '#2196F3', '-.', 1.8),
    ]

    for q_val, label, color, ls, lw in risks:
        t_vals = t_star(s_fixed, q_val, kappa)
        ax2.plot(kappa, t_vals, color=color, ls=ls, lw=lw, label=label)

    ax2.set_xlabel(r'Geometric parameter $\kappa$ (km$^{-1}$)')
    ax2.set_ylabel(r'Safety band width $t^*$ (km)')
    ax2.set_title(fr'(b) Effect of risk tolerance  ($\hat{{s}} = {s_fixed}$, Extremadura)')
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='0.7')
    ax2.set_xlim(0.03, 0.50)
    ax2.set_ylim(0, 30)

    plt.tight_layout()

    # Save to figuras_clean/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir  # already in figuras_clean/
    os.makedirs(out_dir, exist_ok=True)
    path_clean = os.path.join(out_dir, 'safety_bands_voronoi_risk.pdf')
    fig.savefig(path_clean)
    print(f'  Saved: {path_clean}')

    # Also save to figures/ for the manuscript
    figures_dir = os.path.join(script_dir, '..', '..', 'figures')
    if os.path.isdir(figures_dir):
        path_fig = os.path.join(figures_dir, 'safety_bands_voronoi_risk.pdf')
        fig.savefig(path_fig)
        print(f'  Saved: {path_fig}')

    plt.close()

    # Print reference values for verification
    print('\n  Reference values (s=0.257, q*=5%):')
    for kv in [0.05, 0.10, 0.20, 0.30]:
        tv = t_star(0.257, 0.05, kv)
        print(f'    kappa={kv:.2f} => t*={tv:.1f} km')


if __name__ == '__main__':
    print('='*60)
    print('GENERATING SAFETY BANDS FIGURE (Corollary 1)')
    print('='*60)
    generate_safety_bands()
    print('\nDone.')
