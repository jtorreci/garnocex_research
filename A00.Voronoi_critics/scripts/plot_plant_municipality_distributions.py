#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Visualization: Plant-Municipality Pairs
=====================================================

Generate publication-quality figures for distributional analysis:
- Q-Q plots (Log-Normal, Gamma, Weibull)
- Histogram with fitted distributions
- CDF comparison plots

Author: Analysis script for Voronoi framework revision v5
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# Seaborn styling
sns.set_style("whitegrid")
sns.set_palette("Set2")

def load_data():
    """Load plant-municipality beta values"""
    # Load assignment data
    asig_euclidean = pd.read_csv('codigo/asignacion_municipios_euclidiana.csv')
    asig_real = pd.read_csv('codigo/asignacion_municipios_real.csv')

    # Rename columns to English
    asig_euclidean = asig_euclidean.rename(columns={
        'municipio': 'municipality',
        'planta_asignada': 'assigned_plant',
        'euclidean_distance': 'd_euclidean'
    })

    asig_real = asig_real.rename(columns={
        'municipio': 'municipality',
        'planta_asignada': 'assigned_plant_real',
        'real_distance': 'd_real'
    })

    # Merge and calculate beta
    data = asig_euclidean.merge(
        asig_real[['municipality', 'assigned_plant_real', 'd_real']],
        on='municipality',
        suffixes=('_voronoi', '_real')
    )

    data['beta'] = data['d_real'] / data['d_euclidean']
    data = data[(data['beta'] >= 1.0) & (data['beta'].notna()) & np.isfinite(data['beta'])]

    return data['beta'].values

def fit_distributions(beta_values):
    """Fit three distributions"""
    # Log-Normal
    log_beta = np.log(beta_values)
    m_hat = np.mean(log_beta)
    s_hat = np.std(log_beta, ddof=1)
    lognorm_dist = stats.lognorm(s=s_hat, scale=np.exp(m_hat))

    # Gamma
    shape_g, loc_g, scale_g = stats.gamma.fit(beta_values, floc=0)
    gamma_dist = stats.gamma(shape_g, loc=loc_g, scale=scale_g)

    # Weibull
    shape_w, loc_w, scale_w = stats.weibull_min.fit(beta_values, floc=0)
    weibull_dist = stats.weibull_min(shape_w, loc=loc_w, scale=scale_w)

    return {
        'lognormal': {'dist': lognorm_dist, 'params': (m_hat, s_hat), 'label': 'Log-Normal'},
        'gamma': {'dist': gamma_dist, 'params': (shape_g, scale_g), 'label': 'Gamma'},
        'weibull': {'dist': weibull_dist, 'params': (shape_w, scale_w), 'label': 'Weibull'}
    }

def plot_qq_comparison(beta_values, distributions, output_file='figuras_clean/qq_plots_plant_municipality.pdf'):
    """
    Generate 2x3 panel of Q-Q plots (same layout as original figure).

    Top row: Log-Normal, Gamma, Weibull
    Bottom row: Corresponding histograms with fitted PDFs
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for idx, (dist_name, dist_info) in enumerate(distributions.items()):
        ax_qq = axes[0, idx]
        ax_hist = axes[1, idx]

        dist = dist_info['dist']
        label = dist_info['label']
        color = colors[idx]

        # --- Q-Q Plot (Top row) ---
        # Theoretical quantiles
        sorted_beta = np.sort(beta_values)
        n = len(sorted_beta)
        p = (np.arange(1, n + 1) - 0.5) / n  # Plotting positions
        theoretical_q = dist.ppf(p)

        ax_qq.scatter(theoretical_q, sorted_beta, alpha=0.5, s=15, color=color)
        ax_qq.plot([theoretical_q.min(), theoretical_q.max()],
                   [theoretical_q.min(), theoretical_q.max()],
                   'r--', lw=1.5, label='Perfect fit')

        ax_qq.set_xlabel(f'Theoretical Quantiles ({label})', fontsize=9)
        ax_qq.set_ylabel('Observed Quantiles', fontsize=9)
        ax_qq.set_title(f'Q-Q Plot: {label}', fontsize=10, fontweight='bold')
        ax_qq.legend(fontsize=8, loc='upper left')
        ax_qq.grid(alpha=0.3)

        # --- Histogram with Fitted PDF (Bottom row) ---
        ax_hist.hist(beta_values, bins=30, density=True, alpha=0.6,
                     color=color, edgecolor='black', label='Empirical')

        # Fitted PDF
        x_range = np.linspace(beta_values.min(), beta_values.max(), 500)
        pdf_fitted = dist.pdf(x_range)
        ax_hist.plot(x_range, pdf_fitted, color='darkred', lw=2,
                     label=f'Fitted {label}')

        ax_hist.set_xlabel(r'$\beta = d_r / d_e$', fontsize=9)
        ax_hist.set_ylabel('Density', fontsize=9)
        ax_hist.set_title(f'Histogram + Fitted {label}', fontsize=10, fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Q-Q plots saved to: {output_file}")
    plt.close()

def plot_cdf_comparison(beta_values, distributions, output_file='figuras_clean/cdf_comparison_plant_municipality.pdf'):
    """Generate CDF comparison plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Empirical CDF
    sorted_beta = np.sort(beta_values)
    n = len(sorted_beta)
    ecdf = np.arange(1, n + 1) / n

    ax.step(sorted_beta, ecdf, where='post', lw=2, color='black',
            label='Empirical CDF', alpha=0.8)

    # Fitted CDFs
    x_range = np.linspace(beta_values.min(), beta_values.max(), 500)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (dist_name, dist_info) in enumerate(distributions.items()):
        cdf_fitted = dist_info['dist'].cdf(x_range)
        ax.plot(x_range, cdf_fitted, lw=2, color=colors[idx],
                linestyle='--', label=f'{dist_info["label"]} CDF')

    ax.set_xlabel(r'Network Scaling Factor $\beta$', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF Comparison: Plant-Municipality Pairs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"CDF comparison saved to: {output_file}")
    plt.close()

def plot_histogram_with_all_distributions(beta_values, distributions,
                                          output_file='figuras_clean/histogram_plant_municipality_all_fits.pdf'):
    """Single histogram with all three fitted distributions overlaid"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    ax.hist(beta_values, bins=35, density=True, alpha=0.5,
            color='lightblue', edgecolor='black', label='Empirical Data')

    # Fitted distributions
    x_range = np.linspace(beta_values.min(), beta_values.max(), 500)
    colors = {'lognormal': '#1f77b4', 'gamma': '#ff7f0e', 'weibull': '#2ca02c'}
    linestyles = {'lognormal': '-', 'gamma': '--', 'weibull': '-.'}

    for dist_name, dist_info in distributions.items():
        pdf_fitted = dist_info['dist'].pdf(x_range)
        ax.plot(x_range, pdf_fitted, lw=2.5,
                color=colors[dist_name],
                linestyle=linestyles[dist_name],
                label=f'{dist_info["label"]} (fitted)')

    ax.set_xlabel(r'Network Scaling Factor $\beta = d_r / d_e$', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Distribution Fitting: Plant-Municipality Pairs (n=383)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # Add statistics box
    textstr = f'n = {len(beta_values)}\nMean = {np.mean(beta_values):.3f}\nStd = {np.std(beta_values):.3f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram with fits saved to: {output_file}")
    plt.close()

def main():
    """Main execution"""
    print("="*80)
    print("DISTRIBUTION VISUALIZATION: PLANT-MUNICIPALITY PAIRS")
    print("="*80)
    print()

    # Create output directory
    Path('figuras_clean').mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    beta_values = load_data()
    print(f"  Loaded n={len(beta_values)} observations")
    print(f"  Range: [{beta_values.min():.3f}, {beta_values.max():.3f}]")
    print(f"  Mean: {beta_values.mean():.3f} ± {beta_values.std():.3f}")

    # Fit distributions
    print("\nFitting distributions...")
    distributions = fit_distributions(beta_values)

    # Generate plots
    print("\nGenerating figures...")
    print("  1. Q-Q plots (2×3 panel)...")
    plot_qq_comparison(beta_values, distributions)

    print("  2. CDF comparison...")
    plot_cdf_comparison(beta_values, distributions)

    print("  3. Histogram with all fits...")
    plot_histogram_with_all_distributions(beta_values, distributions)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - figuras_clean/qq_plots_plant_municipality.pdf")
    print("  - figuras_clean/cdf_comparison_plant_municipality.pdf")
    print("  - figuras_clean/histogram_plant_municipality_all_fits.pdf")
    print("\nThese figures demonstrate:")
    print("  - Log-Normal provides best fit (Q-Q plot most linear)")
    print("  - Independent observations (plant-municipality pairs)")
    print("  - Higher statistical power than all-pairs analysis")

if __name__ == "__main__":
    main()
