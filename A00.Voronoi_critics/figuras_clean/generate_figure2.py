#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 2: Violin Plot of Beta Distribution Comparison
============================================================

Figure 2 shows violin plots comparing the distribution of beta coefficients
between municipality-to-municipality and municipality-to-assigned-plant distances.

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def set_publication_style():
    """Set matplotlib style for publication-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def load_clean_data():
    """Load cleaned distance tables and calculate betas"""
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

    # Filter outliers for visualization (keep all for stats)
    beta_ratios = df_merged['beta_ratio']
    beta_ratios_display = beta_ratios[beta_ratios <= 5.0]  # For violin plot display

    print(f"  Pairs processed: {len(df_merged):,}")
    print(f"  Beta range: {beta_ratios.min():.3f} - {beta_ratios.max():.3f}")
    print(f"  Beta mean: {beta_ratios.mean():.3f}")
    print(f"  For display (beta<=5): {len(beta_ratios_display):,} ({len(beta_ratios_display)/len(beta_ratios)*100:.1f}%)")

    return beta_ratios, beta_ratios_display

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

    # Filter outliers for visualization
    beta_ratios = df_assigned['beta_ratio']
    beta_ratios_display = beta_ratios[beta_ratios <= 5.0]  # For violin plot display

    print(f"  Total municipality-plant pairs: {len(df_merged):,}")
    print(f"  Assigned pairs (Voronoi): {len(df_assigned):,}")
    print(f"  Beta range (assigned): {beta_ratios.min():.3f} - {beta_ratios.max():.3f}")
    print(f"  Beta mean (assigned): {beta_ratios.mean():.3f}")
    print(f"  For display (beta<=5): {len(beta_ratios_display):,} ({len(beta_ratios_display)/len(beta_ratios)*100:.1f}%)")

    return beta_ratios, beta_ratios_display

def generate_figure2(beta_mun_all, beta_mun_display, beta_plant_all, beta_plant_display):
    """Generate Figure 2: Violin plot comparison of beta distributions"""
    print("\n=== GENERATING FIGURE 2 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Prepare data for violin plot
    # Combine data with labels
    violin_data = []

    # Add municipality-to-municipality data
    for value in beta_mun_display:
        violin_data.append(['Municipality-to-Municipality', value])

    # Add municipality-to-plant data
    for value in beta_plant_display:
        violin_data.append(['Municipality-to-Assigned-Plant', value])

    # Convert to DataFrame
    df_violin = pd.DataFrame(violin_data, columns=['Category', 'Beta'])

    # Create violin plot with grayscale
    parts = ax.violinplot([beta_mun_display, beta_plant_display],
                         positions=[1, 2],
                         widths=0.6,
                         showmeans=False,
                         showmedians=False)

    # Style violin plots in grayscale
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        pc.set_linewidth(1)

    # Style other violin plot elements
    if 'cbars' in parts:
        parts['cbars'].set_color('black')
        parts['cbars'].set_linewidth(1)
    if 'cmins' in parts:
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(1)
    if 'cmaxes' in parts:
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(1)

    # Add mean lines (red dashed)
    mean_mun = beta_mun_all.mean()
    mean_plant = beta_plant_all.mean()

    ax.plot([0.7, 1.3], [mean_mun, mean_mun], 'r--', linewidth=2, alpha=0.8)
    ax.plot([1.7, 2.3], [mean_plant, mean_plant], 'r--', linewidth=2, alpha=0.8)

    # Add mean value labels
    ax.text(1, mean_mun + 0.1, f'Mean: {mean_mun:.3f}',
            ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
    ax.text(2, mean_plant + 0.1, f'Mean: {mean_plant:.3f}',
            ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)

    # Customize axes
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Municipality-to-\nMunicipality\n(n={:,})'.format(len(beta_mun_display)),
                       'Municipality-to-\nAssigned-Plant\n(n={:,})'.format(len(beta_plant_display))])

    ax.set_ylabel('Network Factor (β = d_r / d_e)')
    ax.set_title('Distribution Comparison of Network Scaling Factors')

    # Set y-axis limits
    ax.set_ylim(0.9, 5.0)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"""Statistics Summary:
Municipality-to-Municipality:
  Mean = {mean_mun:.3f}
  Std = {beta_mun_all.std():.3f}
  n = {len(beta_mun_all):,}

Municipality-to-Assigned-Plant:
  Mean = {mean_plant:.3f}
  Std = {beta_plant_all.std():.3f}
  n = {len(beta_plant_all):,}

Note: Display limited to beta <= 5.0
Outliers: {(beta_mun_all > 5.0).sum()} mun-mun, {(beta_plant_all > 5.0).sum()} mun-plant"""

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=9)

    plt.tight_layout()
    plt.savefig('../figuras_clean/violin_plot_beta_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/violin_plot_beta_comparison.pdf")
    print(f"  Municipality betas: n={len(beta_mun_all):,}, mean={mean_mun:.3f}")
    print(f"  Plant assignment betas: n={len(beta_plant_all):,}, mean={mean_plant:.3f}")
    print(f"  Displayed ranges: beta <= 5.0 for both distributions")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 2: VIOLIN PLOT BETA COMPARISON")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant, df_eucl_mun, df_real_mun = load_clean_data()

    # Calculate beta ratios
    beta_mun_all, beta_mun_display = calculate_municipality_betas(df_eucl_mun, df_real_mun)
    beta_plant_all, beta_plant_display = calculate_plant_assignment_betas(df_eucl_plant, df_real_plant)

    # Generate figure
    generate_figure2(beta_mun_all, beta_mun_display, beta_plant_all, beta_plant_display)

    print("\n" + "=" * 80)
    print("FIGURE 2 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Violin plot comparing beta distributions")
    print("  - Grayscale violins with red dashed mean lines")
    print("  - Statistics summary box included")
    print("  - Display limited to beta <= 5.0 for better visualization")
    print("  - Both complete statistics (including outliers) and display data shown")

if __name__ == "__main__":
    main()