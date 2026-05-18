#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 1: Histogram of Network Scaling Factor (Beta)
============================================================

Figure 1 shows the distribution of the network scaling factor (beta = dr/de)
with two subplots:
- Left: Beta for municipality-to-municipality distances
- Right: Beta for municipality-to-assigned-plant distances

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
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

def generate_figure1(beta_municipalities, beta_plants):
    """Generate Figure 1: Two-panel histogram of beta ratios"""
    print("\n=== GENERATING FIGURE 1 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Municipality-to-municipality betas (more bins for this larger dataset)
    n1, bins1, patches1 = ax1.hist(beta_municipalities, bins=80, alpha=0.7,
                                   color='lightgray', edgecolor='black', linewidth=0.5)

    # Add mean line
    mean1 = beta_municipalities.mean()
    ax1.axvline(mean1, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean1:.3f}')

    # Count outliers > 5
    outliers1 = (beta_municipalities > 5.0).sum()

    # Statistics box
    stats_text1 = f"""n = {len(beta_municipalities):,}
Mean = {beta_municipalities.mean():.3f}
Std = {beta_municipalities.std():.3f}
Outliers (β>5): {outliers1}"""

    ax1.text(0.75, 0.85, stats_text1, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10)

    ax1.set_xlabel('Network Factor (β = d_r / d_e)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('A) Municipality-to-Municipality Distances')
    ax1.set_xlim(0.9, 5.0)  # Limit x-axis to maximum 5
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Municipality-to-assigned-plant betas
    n2, bins2, patches2 = ax2.hist(beta_plants, bins=50, alpha=0.7,
                                   color='lightgray', edgecolor='black', linewidth=0.5)

    # Add mean line
    mean2 = beta_plants.mean()
    ax2.axvline(mean2, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean2:.3f}')

    # Count outliers > 5
    outliers2 = (beta_plants > 5.0).sum()

    # Statistics box
    stats_text2 = f"""n = {len(beta_plants):,}
Mean = {beta_plants.mean():.3f}
Std = {beta_plants.std():.3f}
Outliers (β>5): {outliers2}"""

    ax2.text(0.75, 0.85, stats_text2, transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10)

    ax2.set_xlabel('Network Factor (β = d_r / d_e)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('B) Municipality-to-Assigned-Plant Distances')
    ax2.set_xlim(0.9, 5.0)  # Limit x-axis to maximum 5
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Histogram of the Network Scaling Factor (β = d_r/d_e)',
                 fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle

    plt.savefig('../figuras_clean/histogram_ratio_dr_de.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/histogram_ratio_dr_de.pdf")
    print(f"  Municipality betas: n={len(beta_municipalities):,}, mean={mean1:.3f}")
    print(f"  Plant assignment betas: n={len(beta_plants):,}, mean={mean2:.3f}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 1: NETWORK SCALING FACTOR HISTOGRAMS")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant, df_eucl_mun, df_real_mun = load_clean_data()

    # Calculate beta ratios
    beta_municipalities = calculate_municipality_betas(df_eucl_mun, df_real_mun)
    beta_plants = calculate_plant_assignment_betas(df_eucl_plant, df_real_plant)

    # Generate figure
    generate_figure1(beta_municipalities, beta_plants)

    print("\n" + "=" * 80)
    print("FIGURE 1 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Left panel: Beta distribution for municipality-to-municipality distances")
    print("  - Right panel: Beta distribution for municipality-to-assigned-plant distances")
    print("  - Both panels include statistics box and mean line")
    print("  - Grayscale histograms with red mean line as requested")

if __name__ == "__main__":
    main()