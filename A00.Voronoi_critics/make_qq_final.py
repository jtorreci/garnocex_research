#!/usr/bin/env python3
"""
Final Q-Q figure for the paper using REAL data.

Two panels:
  (a) Overlaid Q-Q for the three distributions in the critical zone [1.0, 1.8]
  (b) Q-Q residuals (empirical - theoretical) showing deviation from perfect fit

Uses Municipality-to-Plant data (the calibration dataset).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE, 'tables')
FIG_DIR = os.path.join(BASE, 'figures')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
})

COLORS = {'Log-Normal': '#1f77b4', 'Gamma': '#ff7f0e', 'Weibull': '#2ca02c'}


def load_beta_plants():
    """Load municipality-to-plant beta values from real data."""
    de = pd.read_csv(os.path.join(TABLE_DIR, 'D_euclidea_plantas_clean.csv'))
    dr = pd.read_csv(os.path.join(TABLE_DIR, 'D_real_plantas_clean.csv'))

    de = de.rename(columns={'InputID': 'origin', 'TargetID': 'dest', 'Distance': 'd_eucl'})
    dr = dr.rename(columns={'origin_id': 'origin', 'destination_id': 'dest', 'total_cost': 'd_real'})

    merged = pd.merge(de, dr[['origin', 'dest', 'd_real']], on=['origin', 'dest'], how='inner')
    merged['beta'] = merged['d_real'] / merged['d_eucl']

    beta = merged['beta'].values
    beta = beta[np.isfinite(beta) & (beta >= 1.0)]
    return np.sort(beta)


def load_beta_municipalities():
    """Load municipality-to-municipality beta values from real data."""
    de = pd.read_csv(os.path.join(TABLE_DIR, 'D_euclidea_municipios_clean.csv'))
    dr = pd.read_csv(os.path.join(TABLE_DIR, 'D_real_municipios_clean.csv'))

    de = de.rename(columns={'InputID': 'origin', 'TargetID': 'dest', 'Distance': 'd_eucl'})
    dr = dr.rename(columns={'origin_id': 'origin', 'destination_id': 'dest', 'total_cost': 'd_real'})

    merged = pd.merge(de, dr[['origin', 'dest', 'd_real']], on=['origin', 'dest'], how='inner')
    merged['beta'] = merged['d_real'] / merged['d_eucl']

    beta = merged['beta'].values
    beta = beta[np.isfinite(beta) & (beta >= 1.0)]
    return np.sort(beta)


def qq_quantiles(data, dist_name, params):
    """Compute theoretical quantiles for Q-Q plot."""
    n = len(data)
    p = (np.arange(1, n+1) - 0.5) / n

    if dist_name == 'Log-Normal':
        m, s = params
        return stats.lognorm.ppf(p, s, scale=np.exp(m))
    elif dist_name == 'Gamma':
        k, _, sc = params
        return stats.gamma.ppf(p, k, scale=sc)
    elif dist_name == 'Weibull':
        k, _, sc = params
        return stats.weibull_min.ppf(p, k, scale=sc)


def main():
    print("Loading real data...")
    beta_plant = load_beta_plants()
    beta_mun = load_beta_municipalities()
    print(f"  Municipality-to-Plant: n={len(beta_plant)}")
    print(f"  Municipality-to-Municipality: n={len(beta_mun)}")

    # Use both datasets
    datasets = {
        'Municipality-to-Plant': beta_plant,
        'Municipality-to-Municipality': beta_mun,
    }

    for dset_name, data in datasets.items():
        print(f"\n=== {dset_name} ===")

        # Fit distributions
        lns = np.log(data)
        m_fit, s_fit = np.mean(lns), np.std(lns, ddof=0)
        gfit = stats.gamma.fit(data, floc=0)
        wfit = stats.weibull_min.fit(data, floc=0)

        fitted = {
            'Log-Normal': (m_fit, s_fit),
            'Gamma': gfit,
            'Weibull': wfit,
        }

        # Compute Q-Q quantiles
        theo = {}
        for name in COLORS:
            theo[name] = qq_quantiles(data, name, fitted[name])

        # Q-Q residuals
        resid = {name: data - theo[name] for name in COLORS}

        # Critical zone mask (beta <= 1.3 in empirical data)
        crit = data <= 1.3
        pct_crit = np.sum(crit) / len(data) * 100

        # MAD in critical zone (mean absolute Q-Q deviation)
        print(f"  Critical zone: {pct_crit:.0f}% of data")
        print(f"  {'Distribution':15s} {'MAD(crit)':>10s} {'MAD(full)':>10s} {'MaxDev(crit)':>12s}")
        for name in COLORS:
            mad_c = np.mean(np.abs(resid[name][crit]))
            mad_f = np.mean(np.abs(resid[name]))
            maxd_c = np.max(np.abs(resid[name][crit]))
            print(f"  {name:15s} {mad_c:10.4f} {mad_f:10.4f} {maxd_c:12.4f}")

        # ── Figure ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel (a): Overlaid Q-Q, zoomed to critical zone [1.0, 1.8]
        ax = axes[0]
        for name in COLORS:
            mask = (data <= 1.8) & (theo[name] <= 1.8)
            ax.scatter(theo[name][mask], data[mask], s=3, alpha=0.3,
                       color=COLORS[name], label=name, rasterized=True)

        ax.plot([0.9, 1.8], [0.9, 1.8], 'k-', lw=0.8, alpha=0.5)
        ax.axhspan(1.0, 1.3, alpha=0.06, color='gray')
        ax.axvspan(1.0, 1.3, alpha=0.06, color='gray')
        ax.set_xlim(0.9, 1.8)
        ax.set_ylim(0.9, 1.8)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
        ax.set_title('(a) Q-Q plot (critical zone)')
        ax.legend(loc='lower right', framealpha=0.9, markerscale=3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Panel (b): Q-Q residuals in the critical zone
        ax = axes[1]
        for name in COLORS:
            mask = crit & (theo[name] <= 1.5)
            ax.scatter(theo[name][mask], resid[name][mask], s=2, alpha=0.3,
                       color=COLORS[name], label=name, rasterized=True)
        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_xlabel('Theoretical quantile')
        ax.set_ylabel('Residual (empirical $-$ theoretical)')
        ax.set_title('(b) Q-Q residuals ($\\beta \\leq 1.3$)')
        ax.legend(loc='best', framealpha=0.9, markerscale=3)
        ax.grid(True, alpha=0.2)

        # Panel (c): MAD comparison bar chart
        ax = axes[2]
        names_list = list(COLORS.keys())
        mad_crit = [np.mean(np.abs(resid[n][crit])) for n in names_list]
        mad_full = [np.mean(np.abs(resid[n])) for n in names_list]

        x = np.arange(len(names_list))
        w = 0.35
        bars1 = ax.bar(x - w/2, mad_crit, w, label='Critical zone ($\\beta \\leq 1.3$)',
                        color=[COLORS[n] for n in names_list], alpha=0.8)
        bars2 = ax.bar(x + w/2, mad_full, w, label='Full range',
                        color=[COLORS[n] for n in names_list], alpha=0.35,
                        edgecolor=[COLORS[n] for n in names_list], linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(names_list, fontsize=9)
        ax.set_ylabel('Mean absolute Q-Q deviation')
        ax.set_title('(c) Fit quality (MAD)')
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5,
                    color='#666666')

        fig.tight_layout()
        suffix = dset_name.lower().replace('-to-', '_').replace(' ', '_')
        path = os.path.join(FIG_DIR, f'qq_distributional_robustness_{suffix}.pdf')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close(fig)

    print("\nDone.")


if __name__ == '__main__':
    main()
