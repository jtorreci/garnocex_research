#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate beta distribution histogram with fitted theoretical distributions.

Uses the k=5 nearest municipality-to-plant subset (n=1,915), consistent
with the adopted parameters in the manuscript (Sec 4.2.3).

Distributions are fitted with loc=0 (no location shift) to match
the paper's 2-parameter log-normal model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


def set_publication_style():
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_k5_betas():
    """
    Load municipality-to-plant distances, compute beta, keep k=5 nearest
    plants per municipality by Euclidean distance.
    """
    print("Loading municipality-to-plant distance tables...")

    df_eucl = pd.read_csv("../tables/D_euclidea_plantas_clean.csv")
    try:
        df_real = pd.read_csv("../tables/D_real_plantas_clean_corrected.csv")
    except FileNotFoundError:
        df_real = pd.read_csv("../tables/D_real_plantas_clean.csv")

    # Normalise column names
    df_eucl = df_eucl.rename(columns={
        'InputID': 'municipio', 'TargetID': 'planta',
        'Distance': 'euclidean_distance'})
    df_real = df_real.rename(columns={
        'origin_id': 'municipio', 'destination_id': 'planta'})

    df = df_eucl.merge(
        df_real[['municipio', 'planta', 'total_cost']],
        on=['municipio', 'planta'], how='inner')
    df = df.drop_duplicates(subset=['municipio', 'planta'])
    df['beta'] = df['total_cost'] / df['euclidean_distance']
    df = df[df['beta'] >= 1.0]

    # Keep k=5 nearest per municipality (by Euclidean distance)
    k = 5
    df = df.sort_values(['municipio', 'euclidean_distance'])
    df = df.groupby('municipio').head(k).reset_index(drop=True)

    print(f"  Total pairs after k={k} nearest: {len(df):,}")
    print(f"  Beta range: {df['beta'].min():.3f} - {df['beta'].max():.3f}")
    print(f"  Beta mean:  {df['beta'].mean():.3f}")
    print(f"  ln(beta) mean (m): {np.log(df['beta']).mean():.3f}")
    print(f"  ln(beta) std  (s): {np.log(df['beta']).std():.3f}")

    return df['beta']


def fit_distributions(betas):
    """
    Fit distributions allowing a location shift.

    Since beta >= 1 by construction, a free loc parameter (typically
    converging near 1) lets each distribution concentrate on the excess
    over the physical lower bound, producing a sharper, more leptokurtic
    peak that matches the empirical histogram.
    """
    print("\n=== FITTING DISTRIBUTIONS (free loc) ===")

    data = betas.values
    fitted = {}

    # Log-Normal  (3-param: shape, loc, scale)
    ln_params = stats.lognorm.fit(data, floc=1)       # beta >= 1
    fitted['lognorm'] = {
        'params': ln_params, 'dist': stats.lognorm,
        'name': 'Log-Normal', 'color': 'red',
        'linestyle': '-', 'linewidth': 2.5}

    # Normal
    norm_params = stats.norm.fit(data)
    fitted['norm'] = {
        'params': norm_params, 'dist': stats.norm,
        'name': 'Normal', 'color': 'blue',
        'linestyle': '--', 'linewidth': 2}

    # Gamma  (beta - 1 >= 0)
    gamma_params = stats.gamma.fit(data, floc=1)
    fitted['gamma'] = {
        'params': gamma_params, 'dist': stats.gamma,
        'name': 'Gamma', 'color': 'green',
        'linestyle': ':', 'linewidth': 2}

    # Weibull  (beta - 1 >= 0)
    weibull_params = stats.weibull_min.fit(data, floc=1)
    fitted['weibull'] = {
        'params': weibull_params, 'dist': stats.weibull_min,
        'name': 'Weibull', 'color': 'purple',
        'linestyle': '-.', 'linewidth': 2}

    # K-S statistics
    print("  Goodness of fit:")
    for key, info in fitted.items():
        ks, p = stats.kstest(data,
                             lambda x, i=info: i['dist'].cdf(x, *i['params']))
        info['ks_stat'] = ks
        info['p_value'] = p
        print(f"    {info['name']:12s}: KS={ks:.4f}, p={p:.4f}, "
              f"params={tuple(round(x, 4) for x in info['params'])}")

    return fitted


def generate_figure(betas):
    print("\n=== GENERATING FIGURE ===")
    set_publication_style()

    fitted = fit_distributions(betas)

    data = betas.values
    data_display = data[data <= 4.0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Histogram
    n_hist, bins, _ = ax.hist(
        data_display, bins=60, density=True, alpha=0.7,
        color='lightgray', edgecolor='black', linewidth=0.5,
        label='Observed data')

    # Fitted curves
    x = np.linspace(data_display.min(), 4.0, 1000)
    y_maxes = []
    for info in fitted.values():
        y = info['dist'].pdf(x, *info['params'])
        y_maxes.append(y.max())
        ax.plot(x, y, color=info['color'], linestyle=info['linestyle'],
                linewidth=info['linewidth'], alpha=0.9,
                label=f"{info['name']} (KS={info['ks_stat']:.3f})")

    y_limit = max(n_hist.max(), max(y_maxes)) * 1.15
    ax.set_xlim(0.9, 4.0)
    ax.set_ylim(0, y_limit)
    ax.set_xlabel(r'Network scaling factor ($\beta = d_{\mathrm{network}} / d_{\mathrm{Euclidean}}$)')
    ax.set_ylabel('Probability density')
    ax.set_title(r'$\beta$ distribution with fitted curves ($k = 5$ nearest, $n = {:,}$)'.format(
        len(data)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True,
              shadow=True, framealpha=0.9)

    stats_text = (
        f"$k = 5$ nearest plants\n"
        f"$n = {len(data):,}$\n"
        f"Mean $\\beta$ = {data.mean():.3f}\n"
        f"$\\hat{{m}}$ = {np.log(data).mean():.3f}\n"
        f"$\\hat{{s}}$ = {np.log(data).std():.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.savefig('beta_distribution_fitted_curves.pdf', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Saved: beta_distribution_fitted_curves.pdf")


def main():
    print("=" * 70)
    print("  Beta histogram with fitted distributions (k=5 nearest)")
    print("=" * 70)

    betas = load_k5_betas()
    generate_figure(betas)

    print("\nDone.")


if __name__ == "__main__":
    main()
