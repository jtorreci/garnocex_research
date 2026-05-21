#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 12: Distance Improvement Analysis (1x2)
=======================================================

Figure 12 shows distance improvement analysis comparing network-based vs Voronoi assignment
in a 1x2 layout:

- Panel A (0,0): Histogram of distance improvement (network vs Voronoi) with mean marked
- Panel B (0,1): Bar chart of correct vs incorrect assignments with counts and percentages

For correctly assigned municipalities, distance improvement is zero since both methods
yield the same assignment.

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

def calculate_assignment_improvements(df_eucl_plant, df_real_plant):
    """Calculate distance improvements from network-based vs Voronoi assignments"""
    print("\n=== CALCULATING ASSIGNMENT IMPROVEMENTS ===")

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

    # Calculate network-based assignments (nearest plant by network distance)
    print("  Calculating network-based assignments...")
    network_assignments = df_real.loc[df_real.groupby('municipio')['total_cost'].idxmin()]
    network_assignments = network_assignments[['municipio', 'planta', 'total_cost']].copy()
    network_assignments.columns = ['municipio', 'network_plant', 'network_distance']

    # Merge assignments for comparison
    assignment_comparison = pd.merge(
        voronoi_with_network[['municipio', 'voronoi_plant', 'voronoi_network_distance']],
        network_assignments,
        on='municipio'
    )

    print(f"  Total municipalities analyzed: {len(assignment_comparison)}")

    # Calculate distance improvements
    assignment_comparison['distance_improvement'] = (
        assignment_comparison['voronoi_network_distance'] - assignment_comparison['network_distance']
    )

    # Identify correct vs incorrect assignments
    assignment_comparison['is_correct'] = (
        assignment_comparison['voronoi_plant'] == assignment_comparison['network_plant']
    )

    # For correctly assigned municipalities, improvement should be zero
    # (since both methods assign to the same plant)
    correctly_assigned = assignment_comparison[assignment_comparison['is_correct']]
    incorrectly_assigned = assignment_comparison[~assignment_comparison['is_correct']]

    print(f"  Correctly assigned municipalities: {len(correctly_assigned)} ({len(correctly_assigned)/len(assignment_comparison)*100:.1f}%)")
    print(f"  Incorrectly assigned municipalities: {len(incorrectly_assigned)} ({len(incorrectly_assigned)/len(assignment_comparison)*100:.1f}%)")

    # Distance improvements (only for incorrectly assigned)
    if len(incorrectly_assigned) > 0:
        print(f"  Average distance improvement (incorrect assignments): {incorrectly_assigned['distance_improvement'].mean():.3f}")
        print(f"  Max distance improvement: {incorrectly_assigned['distance_improvement'].max():.3f}")
        print(f"  Min distance improvement: {incorrectly_assigned['distance_improvement'].min():.3f}")

    # Create combined improvement data
    # Correctly assigned have 0 improvement, incorrectly assigned have calculated improvement
    all_improvements = []

    # Add zeros for correctly assigned
    all_improvements.extend([0.0] * len(correctly_assigned))

    # Add actual improvements for incorrectly assigned
    all_improvements.extend(incorrectly_assigned['distance_improvement'].tolist())

    improvement_data = {
        'all_improvements': np.array(all_improvements),
        'correctly_assigned_count': len(correctly_assigned),
        'incorrectly_assigned_count': len(incorrectly_assigned),
        'correctly_assigned_pct': len(correctly_assigned)/len(assignment_comparison)*100,
        'incorrectly_assigned_pct': len(incorrectly_assigned)/len(assignment_comparison)*100,
        'total_municipalities': len(assignment_comparison)
    }

    return improvement_data

def generate_figure12():
    """Generate Figure 12: Distance improvement analysis"""
    print("\n=== GENERATING FIGURE 12 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Load clean data
    df_eucl_plant, df_real_plant = load_clean_data()

    # Calculate assignment improvements
    data = calculate_assignment_improvements(df_eucl_plant, df_real_plant)

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Histogram of distance improvements
    improvements = data['all_improvements']
    mean_improvement = improvements.mean()

    # Create histogram
    n, bins, patches = ax1.hist(improvements, bins=30, alpha=0.7, color='lightgray',
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
            ax1.set_ylim(0, max_non_zero * 1.1)

    # Mark the mean with a vertical line
    ax1.axvline(mean_improvement, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_improvement:.3f}')

    # Add zero line for reference
    ax1.axvline(0, color='darkblue', linestyle='-', linewidth=1.5, alpha=0.8,
               label='No improvement (correct assignments)')

    # Add annotation for the zero bar that's cut off
    if zero_bar_height > 0:
        ax1.text(0.02, 0.75, f'Zero bar height: {int(zero_bar_height)}\n(correctly assigned municipalities)',
                transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=9)

    # Add statistics text box
    stats_text = f"""Statistics:
Mean improvement: {mean_improvement:.3f}
Std deviation: {improvements.std():.3f}
Total municipalities: {data['total_municipalities']}
Correctly assigned: {data['correctly_assigned_count']}
Incorrectly assigned: {data['incorrectly_assigned_count']}"""

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=9)

    ax1.set_xlabel('Distance Improvement (Network - Voronoi)')
    ax1.set_ylabel('Number of Municipalities')
    ax1.set_title('A) Distance Improvement Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Bar chart of correct vs incorrect assignments
    categories = ['Correctly\nAssigned', 'Incorrectly\nAssigned']
    counts = [data['correctly_assigned_count'], data['incorrectly_assigned_count']]
    percentages = [data['correctly_assigned_pct'], data['incorrectly_assigned_pct']]

    # Use different colors - green for correct, red for incorrect
    colors = ['lightgreen', 'lightcoral']

    bars = ax2.bar(categories, counts, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Add count and percentage labels inside bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{count}\n({pct:.1f}%)', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Remove duplicate labels on top of bars - counts are already inside bars

    ax2.set_ylabel('Number of Municipalities')
    ax2.set_title('B) Assignment Accuracy Analysis')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add total municipalities annotation
    ax2.text(0.5, 0.95, f'Total: {data["total_municipalities"]} municipalities',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=11, fontweight='bold')

    # Overall title
    fig.suptitle('Distance Improvement Analysis (Network vs Voronoi)',
                fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout()

    plt.savefig('../figuras_clean/distance_improvement_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/distance_improvement_analysis.pdf")
    print(f"  Mean distance improvement: {mean_improvement:.3f}")
    print(f"  Assignment accuracy: {data['correctly_assigned_pct']:.1f}%")
    print(f"  Misassignment rate: {data['incorrectly_assigned_pct']:.1f}%")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 12: DISTANCE IMPROVEMENT ANALYSIS (1x2)")
    print("=" * 80)

    # Generate figure
    generate_figure12()

    print("\n" + "=" * 80)
    print("FIGURE 12 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Panel A: Histogram of distance improvements with mean marked")
    print("  - Panel B: Bar chart of correct vs incorrect assignments")
    print("  - Correctly assigned municipalities have zero improvement")
    print("  - Incorrectly assigned show actual distance savings from network-based assignment")
    print("  - Green bars: Correct assignments, Red bars: Incorrect assignments")

if __name__ == "__main__":
    main()