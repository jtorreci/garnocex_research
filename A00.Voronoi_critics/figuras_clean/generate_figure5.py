#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 5: Q-Q Plots of Beta Coefficients vs Theoretical Distributions (2x3)
===================================================================================

Figure 5 shows Q-Q plots comparing beta coefficient distributions with theoretical
distributions (Log-Normal, Gamma, Weibull) in a 2x3 layout:

Top row (Municipality-to-Municipality):
- Panel A: Q-Q vs Log-Normal
- Panel B: Q-Q vs Gamma
- Panel C: Q-Q vs Weibull

Bottom row (Municipality-to-Assigned-Plant):
- Panel D: Q-Q vs Log-Normal
- Panel E: Q-Q vs Gamma
- Panel F: Q-Q vs Weibull

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def set_publication_style():
    """Set matplotlib style for publication-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def load_clean_data():
    """Load cleaned distance tables"""
    print("Loading clean distance tables...")

    # Load plant distances (using corrected data)
    df_eucl_plant = pd.read_csv("../tables/D_euclidea_plantas_clean.csv")
    # Try corrected version first, fallback to standard

    try:

        df_real_plant = pd.read_csv("../tables/D_real_plantas_clean_corrected.csv")

    except FileNotFoundError:

        df_real_plant = pd.read_csv("../tables/D_real_plantas_clean.csv")

    # Load municipality distances
    df_eucl_mun = pd.read_csv("../tables/D_euclidea_municipios_clean.csv")
    df_real_mun = pd.read_csv("../tables/D_real_municipios_clean.csv")

    print(f"  Plant data: {len(df_eucl_plant):,} euclidean, {len(df_real_plant):,} real")
    print(f"  Municipality data: {len(df_eucl_mun):,} euclidean, {len(df_real_mun):,} real")

    return df_eucl_plant, df_real_plant, df_eucl_mun, df_real_mun

def calculate_municipality_betas(df_eucl_mun, df_real_mun):
    """Calculate beta ratios for municipality-to-municipality distances"""
    print("\n=== CALCULATING MUNICIPALITY-TO-MUNICIPALITY BETAS ===")

    # Standardize column names
    df_eucl = df_eucl_mun.copy()
    df_real = df_real_mun.copy()

    # Check column names and standardize
    if 'InputID' in df_eucl.columns:
        df_eucl = df_eucl.rename(columns={'InputID': 'origin', 'TargetID': 'destination', 'Distance': 'euclidean_distance'})

    if 'origin_id' in df_real.columns:
        df_real = df_real.rename(columns={'origin_id': 'origin', 'destination_id': 'destination'})

    # Merge on origin-destination pairs
    df_merged = df_eucl.merge(df_real[['origin', 'destination', 'total_cost']],
                             on=['origin', 'destination'], how='inner')

    # Calculate beta ratios
    df_merged['beta_ratio'] = df_merged['total_cost'] / df_merged['euclidean_distance']

    # Remove duplicates by keeping first occurrence
    df_merged = df_merged.drop_duplicates(subset=['origin', 'destination'])

    print(f"  Pairs processed: {len(df_merged):,}")
    print(f"  Beta range: {df_merged['beta_ratio'].min():.3f} - {df_merged['beta_ratio'].max():.3f}")
    print(f"  Beta mean: {df_merged['beta_ratio'].mean():.3f}")

    return df_merged['beta_ratio']

def calculate_plant_assignment_betas(df_eucl_plant, df_real_plant):
    """Calculate beta ratios for municipality-to-assigned-plant distances"""
    print("\n=== CALCULATING MUNICIPALITY-TO-ASSIGNED-PLANT BETAS ===")

    # Standardize column names
    df_eucl = df_eucl_plant.rename(columns={'InputID': 'municipio', 'TargetID': 'planta', 'Distance': 'euclidean_distance'})
    df_real = df_real_plant.rename(columns={'origin_id': 'municipio', 'destination_id': 'planta'})

    # Merge on municipio-planta pairs
    df_merged = df_eucl.merge(df_real[['municipio', 'planta', 'total_cost']],
                             on=['municipio', 'planta'], how='inner')

    # Calculate beta ratios
    df_merged['beta_ratio'] = df_merged['total_cost'] / df_merged['euclidean_distance']

    # Remove duplicates by keeping first occurrence
    df_merged = df_merged.drop_duplicates(subset=['municipio', 'planta'])

    # Calculate Voronoi assignments (nearest plant for each municipality)
    voronoi_assignments = df_eucl.loc[df_eucl.groupby('municipio')['euclidean_distance'].idxmin()]
    assigned_pairs = set(zip(voronoi_assignments['municipio'], voronoi_assignments['planta']))

    # Filter to only assigned pairs
    df_assigned = df_merged[df_merged.apply(lambda row: (row['municipio'], row['planta']) in assigned_pairs, axis=1)]

    print(f"  Total municipality-plant pairs: {len(df_merged):,}")
    print(f"  Assigned pairs (Voronoi): {len(df_assigned):,}")
    print(f"  Beta range (assigned): {df_assigned['beta_ratio'].min():.3f} - {df_assigned['beta_ratio'].max():.3f}")
    print(f"  Beta mean (assigned): {df_assigned['beta_ratio'].mean():.3f}")

    return df_assigned['beta_ratio']

def create_qq_plot(ax, data, dist_name, dist_func, title, fit_params=None):
    """Create a Q-Q plot for given data and distribution"""

    # Fit the distribution if no parameters provided
    if fit_params is None:
        if dist_name == 'lognorm':
            # For lognormal, fit returns (s, loc, scale)
            fit_params = stats.lognorm.fit(data)
        elif dist_name == 'gamma':
            # For gamma, fit returns (a, loc, scale)
            fit_params = stats.gamma.fit(data)
        elif dist_name == 'weibull_min':
            # For Weibull, fit returns (c, loc, scale)
            fit_params = stats.weibull_min.fit(data)

    # Create Q-Q plot
    stats.probplot(data, dist=dist_func, sparams=fit_params, plot=ax)

    # Customize appearance
    ax.get_lines()[0].set_markerfacecolor('darkgray')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Calculate R-squared for goodness of fit
    theoretical_quantiles = ax.get_lines()[1].get_xdata()
    sample_quantiles = ax.get_lines()[1].get_ydata()

    if len(theoretical_quantiles) == len(sample_quantiles) and len(theoretical_quantiles) > 1:
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        r_squared = correlation ** 2

        # Add R-squared text
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=9)

    return fit_params

def generate_figure5(beta_municipalities, beta_plants):
    """Generate Figure 5: 2x3 Q-Q plots comparing beta distributions"""
    print("\n=== GENERATING FIGURE 5 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Create figure with 2x3 subplots
    from matplotlib import gridspec

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Top row: Municipality-to-Municipality betas
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Bottom row: Municipality-to-Plant betas
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    print("  Creating Q-Q plots for municipality-to-municipality betas...")

    # Top row: Municipality-to-Municipality Q-Q plots
    create_qq_plot(ax1, beta_municipalities, 'lognorm', stats.lognorm,
                   'A) Municipality-Municipality vs Log-Normal')

    create_qq_plot(ax2, beta_municipalities, 'gamma', stats.gamma,
                   'B) Municipality-Municipality vs Gamma')

    create_qq_plot(ax3, beta_municipalities, 'weibull_min', stats.weibull_min,
                   'C) Municipality-Municipality vs Weibull')

    print("  Creating Q-Q plots for municipality-to-plant betas...")

    # Bottom row: Municipality-to-Plant Q-Q plots
    create_qq_plot(ax4, beta_plants, 'lognorm', stats.lognorm,
                   'D) Municipality-Plant vs Log-Normal')

    create_qq_plot(ax5, beta_plants, 'gamma', stats.gamma,
                   'E) Municipality-Plant vs Gamma')

    create_qq_plot(ax6, beta_plants, 'weibull_min', stats.weibull_min,
                   'F) Municipality-Plant vs Weibull')

    # Set common axis labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles (Municipality-Municipality)')

    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles (Municipality-Plant)')

    # Overall title
    fig.suptitle('Q-Q Plots of Beta Coefficients vs Theoretical Distributions',
                 fontsize=16, y=0.96)

    plt.savefig('../figuras_clean/qq_plots_beta_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/qq_plots_beta_distributions.pdf")
    print(f"  Municipality-municipality data: n={len(beta_municipalities):,}")
    print(f"  Municipality-plant data: n={len(beta_plants):,}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 5: Q-Q PLOTS BETA DISTRIBUTIONS (2x3)")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant, df_eucl_mun, df_real_mun = load_clean_data()

    # Calculate beta ratios
    beta_municipalities = calculate_municipality_betas(df_eucl_mun, df_real_mun)
    beta_plants = calculate_plant_assignment_betas(df_eucl_plant, df_real_plant)

    # Generate figure
    generate_figure5(beta_municipalities, beta_plants)

    print("\n" + "=" * 80)
    print("FIGURE 5 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Top row: Municipality-to-Municipality Q-Q plots")
    print("    A) vs Log-Normal, B) vs Gamma, C) vs Weibull")
    print("  - Bottom row: Municipality-to-Plant Q-Q plots")
    print("    D) vs Log-Normal, E) vs Gamma, F) vs Weibull")
    print("  - Gray points with red reference lines")
    print("  - R-squared values for goodness of fit")

if __name__ == "__main__":
    main()