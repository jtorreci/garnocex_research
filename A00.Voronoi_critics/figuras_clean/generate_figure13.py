#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 13: Distribution of Distance Ratio Improvements
==============================================================

Figure 13 shows the distribution of improvements in distance ratios when switching
from Voronoi (Euclidean-based) assignments to network-based assignments.

The improvement is calculated as:
improvement = (voronoi_network_distance - optimal_network_distance) / voronoi_network_distance

This shows the relative benefit of using network-based assignment compared to
Voronoi tessellation for each municipality.

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
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
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

    print(f"  Plant data: {len(df_eucl_plant):,} euclidean, {len(df_real_plant):,} real")

    return df_eucl_plant, df_real_plant

def calculate_distance_ratio_improvements(df_eucl_plant, df_real_plant):
    """Calculate distance ratio improvements from Voronoi vs optimal assignment"""
    print("\n=== CALCULATING DISTANCE RATIO IMPROVEMENTS ===")

    # Standardize column names
    df_eucl = df_eucl_plant.rename(columns={'InputID': 'municipio', 'TargetID': 'planta', 'Distance': 'euclidean_distance'})
    df_real = df_real_plant.rename(columns={'origin_id': 'municipio', 'destination_id': 'planta'})

    # Remove duplicates from df_real BEFORE merging to avoid row multiplication
    # (Some municipality-plant pairs have duplicate routes with slightly different costs)
    df_real = df_real.drop_duplicates(subset=['municipio', 'planta'], keep='first')

    # Merge to get both distances
    df_merged = df_eucl.merge(df_real[['municipio', 'planta', 'total_cost']],
                             on=['municipio', 'planta'], how='inner')

    # Remove any remaining duplicates
    df_merged = df_merged.drop_duplicates(subset=['municipio', 'planta'])

    # Calculate Voronoi assignments (nearest plant by Euclidean distance)
    print("  Calculating Voronoi assignments...")
    voronoi_assignments = df_eucl.loc[df_eucl.groupby('municipio')['euclidean_distance'].idxmin()]
    voronoi_assignments = voronoi_assignments[['municipio', 'planta', 'euclidean_distance']].copy()
    voronoi_assignments.columns = ['municipio', 'voronoi_plant', 'voronoi_eucl_distance']

    # Get network distance for Voronoi assignments
    voronoi_with_network = voronoi_assignments.merge(
        df_real[['municipio', 'planta', 'total_cost']],
        left_on=['municipio', 'voronoi_plant'],
        right_on=['municipio', 'planta'],
        how='left'
    )
    voronoi_with_network = voronoi_with_network.rename(columns={'total_cost': 'voronoi_network_distance'})

    # Calculate network-based optimal assignments (nearest plant by network distance)
    print("  Calculating optimal network assignments...")
    optimal_assignments = df_real.loc[df_real.groupby('municipio')['total_cost'].idxmin()]
    optimal_assignments = optimal_assignments[['municipio', 'planta', 'total_cost']].copy()
    optimal_assignments.columns = ['municipio', 'optimal_plant', 'optimal_network_distance']

    # Merge for comparison
    comparison = pd.merge(
        voronoi_with_network[['municipio', 'voronoi_plant', 'voronoi_network_distance']],
        optimal_assignments,
        on='municipio'
    )

    print(f"  Total municipalities analyzed: {len(comparison)}")

    # Calculate improvement ratios (relative improvement)
    # improvement = (voronoi_distance - optimal_distance) / voronoi_distance
    comparison['absolute_improvement'] = (
        comparison['voronoi_network_distance'] - comparison['optimal_network_distance']
    )
    comparison['improvement_ratio'] = (
        comparison['absolute_improvement'] / comparison['voronoi_network_distance']
    )

    # Identify cases where assignments differ
    comparison['different_assignment'] = (
        comparison['voronoi_plant'] != comparison['optimal_plant']
    )

    different_assignments = comparison[comparison['different_assignment']]
    same_assignments = comparison[~comparison['different_assignment']]

    print(f"  Different assignments: {len(different_assignments)} ({len(different_assignments)/len(comparison)*100:.1f}%)")
    print(f"  Same assignments: {len(same_assignments)} ({len(same_assignments)/len(comparison)*100:.1f}%)")

    # For same assignments, improvement should be zero
    if len(different_assignments) > 0:
        print(f"  Mean improvement ratio (different assignments): {different_assignments['improvement_ratio'].mean():.4f}")
        print(f"  Max improvement ratio: {different_assignments['improvement_ratio'].max():.4f}")
        print(f"  Min improvement ratio: {different_assignments['improvement_ratio'].min():.4f}")

    # Return improvement ratios for all municipalities
    # Same assignments have 0 improvement, different assignments have calculated improvement
    all_improvements = comparison['improvement_ratio'].values

    improvement_data = {
        'improvement_ratios': all_improvements,
        'mean_improvement': all_improvements.mean(),
        'std_improvement': all_improvements.std(),
        'total_municipalities': len(comparison),
        'different_assignments': len(different_assignments),
        'misallocation_rate': len(different_assignments)/len(comparison)*100
    }

    return improvement_data

def generate_figure13():
    """Generate Figure 13: Distribution of distance ratio improvements"""
    print("\n=== GENERATING FIGURE 13 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Load clean data
    df_eucl_plant, df_real_plant = load_clean_data()

    # Calculate improvements
    data = calculate_distance_ratio_improvements(df_eucl_plant, df_real_plant)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Extract improvement ratios
    improvements = data['improvement_ratios']
    mean_improvement = data['mean_improvement']

    # Create histogram with gray bars
    n, bins, patches = ax.hist(improvements, bins=40, alpha=0.8, color='lightgray',
                              edgecolor='black', linewidth=0.5, density=False)

    # Adjust y-axis to exclude the zero bar for better visualization
    # Find the height of the zero bar (first bar containing value 0)
    zero_bar_height = 0
    for i, (bin_left, bin_right) in enumerate(zip(bins[:-1], bins[1:])):
        if bin_left <= 0 <= bin_right:
            zero_bar_height = n[i]
            break

    # Set y-axis limit to focus on non-zero improvements
    if zero_bar_height > 0:
        non_zero_heights = [height for height in n if height != zero_bar_height]
        if len(non_zero_heights) > 0:
            max_non_zero = max(non_zero_heights)
            ax.set_ylim(0, max_non_zero * 1.1)

    # Mark the mean with a red dashed line
    ax.axvline(mean_improvement, color='red', linestyle='--', linewidth=3,
               label=f'Mean: {mean_improvement:.4f}')

    # Add zero line for reference (no improvement)
    ax.axvline(0, color='darkblue', linestyle='-', linewidth=1.5, alpha=0.8,
               label='No improvement')

    # Add annotation for the zero bar that's cut off
    if zero_bar_height > 0:
        ax.text(0.02, 0.85, f'Zero bar height: {int(zero_bar_height)}\\n(no improvement cases)',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=9)

    # Add statistics text box
    stats_text = f"""Statistics:
Mean improvement: {mean_improvement:.4f}
Std deviation: {data['std_improvement']:.4f}
Total municipalities: {data['total_municipalities']}
Misallocation rate: {data['misallocation_rate']:.1f}%
Different assignments: {data['different_assignments']}"""

    ax.text(0.02, 0.70, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=9)

    # Set labels and title
    ax.set_xlabel('Distance Ratio Improvement')
    ax.set_ylabel('Number of Municipalities')
    ax.set_title('Distribution of Distance Ratio Improvements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig('../figuras_clean/distance_ratio_improvements.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/distance_ratio_improvements.pdf")
    print(f"  Mean improvement ratio: {mean_improvement:.4f}")
    print(f"  Misallocation rate: {data['misallocation_rate']:.1f}%")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 13: DISTRIBUTION OF DISTANCE RATIO IMPROVEMENTS")
    print("=" * 80)

    # Generate figure
    generate_figure13()

    print("\n" + "=" * 80)
    print("FIGURE 13 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Histogram of distance ratio improvements (gray bars)")
    print("  - Red dashed line marking the mean improvement")
    print("  - Blue line at zero (no improvement reference)")
    print("  - Improvement = (Voronoi_distance - Optimal_distance) / Voronoi_distance")
    print("  - Positive values indicate network-based assignment is better")
    print("  - Zero values indicate both methods assign to the same plant")

if __name__ == "__main__":
    main()