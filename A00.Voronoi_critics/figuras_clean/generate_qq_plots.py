#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Q-Q distributional robustness figure (single row, 3 panels).

Focuses on the critical zone (beta <= 2.0) where misallocations occur.
Uses k=5 nearest-plant data (n=1,915).
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent          # support_material/
TABLES = BASE / "tables"
CODIGO = BASE / "codigo" / "tablas"
FIGCLEAN = BASE / "figuras_clean"
FIGPAPER = BASE.parent / "figures"


def load_k5_betas():
    euc = pd.read_csv(TABLES / "D_euclidea_plantas_clean.csv", encoding="latin-1")
    euc.columns = ["municipality", "plant_name", "d_euclidean"]

    net = pd.read_csv(CODIGO / "D_real_plantas_clean_corrected.csv", encoding="latin-1")
    net.columns = ["municipality", "plant_name",
                   "entry_cost", "network_cost", "exit_cost", "d_network"]

    df = euc.merge(net[["municipality", "plant_name", "d_network"]],
                   on=["municipality", "plant_name"])
    df["beta"] = df["d_network"] / df["d_euclidean"]
    df["euc_rank"] = (df.groupby("municipality")["d_euclidean"]
                        .rank(method="first").astype(int))

    betas = df[df["euc_rank"] <= 5]["beta"].values
    print(f"  Loaded k=5 nearest: n={len(betas)}, "
          f"mean={betas.mean():.3f}, std={betas.std():.3f}")
    return betas


def generate_figure(betas):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })

    # Fit distributions
    ln_b = np.log(betas)
    m, s = ln_b.mean(), ln_b.std(ddof=1)
    dists = {
        'Log-Normal': stats.lognorm(s=s, scale=np.exp(m)),
        'Gamma':      stats.gamma(*stats.gamma.fit(betas, floc=0)),
        'Weibull':    stats.weibull_min(*stats.weibull_min.fit(betas, floc=0)),
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    sorted_b = np.sort(betas)
    n = len(sorted_b)
    p = (np.arange(1, n + 1) - 0.5) / n

    # Critical zone cutoff
    beta_crit = 2.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for idx, (name, dist) in enumerate(dists.items()):
        ax = axes[idx]
        col = colors[idx]

        theo_q = dist.ppf(p)

        # Focus on critical zone
        mask_crit = sorted_b <= beta_crit
        theo_crit = theo_q[mask_crit]
        obs_crit = sorted_b[mask_crit]

        # Full range (faded background)
        mask_tail = sorted_b > beta_crit
        theo_tail = theo_q[mask_tail]
        obs_tail = sorted_b[mask_tail]

        # Plot tail (faded)
        hi_disp = np.percentile(betas, 99)
        if mask_tail.sum() > 0:
            ax.scatter(theo_tail, obs_tail, alpha=0.12, s=6, color='gray',
                       zorder=1)

        # Plot critical zone (prominent)
        ax.scatter(theo_crit, obs_crit, alpha=0.5, s=12, color=col, zorder=3,
                   label=f'Critical zone ($\\beta \\leq {beta_crit}$)')

        # Perfect fit line
        lo = min(theo_q.min(), sorted_b.min())
        ax.plot([lo, hi_disp], [lo, hi_disp], 'r--', lw=1.5,
                label='Perfect fit', zorder=2)

        ax.set_xlim(lo - 0.05, hi_disp)
        ax.set_ylim(lo - 0.05, hi_disp)
        ax.set_xlabel(f'Theoretical ({name})', fontsize=12)
        ax.set_ylabel('Observed', fontsize=12)
        ax.set_title(f'({"ABC"[idx]}) Q-Q: {name}', fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.25)

        # MAD in critical zone
        mad_crit = np.mean(np.abs(theo_crit - obs_crit))
        # MAD in near-boundary zone (beta <= 1.3)
        mask_nb = sorted_b <= 1.3
        if mask_nb.sum() > 10:
            mad_nb = np.mean(np.abs(theo_q[mask_nb] - sorted_b[mask_nb]))
            text = (f'MAD ($\\beta \\leq {beta_crit}$) = {mad_crit:.3f}\n'
                    f'MAD ($\\beta \\leq 1.3$) = {mad_nb:.3f}')
        else:
            text = f'MAD ($\\beta \\leq {beta_crit}$) = {mad_crit:.3f}'
        ax.text(0.97, 0.05, text, transform=ax.transAxes, ha='right',
                fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.9))

    plt.tight_layout()

    out = FIGCLEAN / "qq_distributional_robustness_municipality_plant.pdf"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    shutil.copy2(out, FIGPAPER / out.name)
    plt.close()
    print(f"  Saved: {out.name}")


if __name__ == '__main__':
    print('=' * 60)
    print('GENERATING Q-Q DISTRIBUTIONAL ROBUSTNESS FIGURE')
    print('=' * 60)
    betas = load_k5_betas()
    generate_figure(betas)
    print('\nDone.')
