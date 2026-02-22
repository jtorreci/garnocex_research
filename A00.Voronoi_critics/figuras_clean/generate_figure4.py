#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 4: Municipality Assignment Changes (Voronoi vs Real)
===================================================================

Figure 4 shows a bar chart where each bar represents the number of municipalities
that each plant gains (+) or loses (-) when comparing real network-based assignments
versus Voronoi (Euclidean) assignments.

- Red bars: Plants that gain municipalities
- Gray bars: Plants that lose municipalities
- X-axis: Plant ID (1, 2, 3, ...)
- Y-axis: Net change in municipality assignments

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
        'legend.fontsize': 11,
        'figure.titlesize': 16,
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

def calculate_assignment_changes(df_eucl_plant, df_real_plant):
    """Calculate changes in municipality assignments between Voronoi and real assignments"""
    print("\n=== CALCULATING ASSIGNMENT CHANGES ===")

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
    voronoi_assignments = voronoi_assignments[['municipio', 'planta']].copy()
    voronoi_assignments.columns = ['municipio', 'voronoi_assigned_plant']

    # Calculate real assignments (nearest plant by network distance)
    print("  Calculating real network assignments...")
    real_assignments = df_real.loc[df_real.groupby('municipio')['total_cost'].idxmin()]
    real_assignments = real_assignments[['municipio', 'planta']].copy()
    real_assignments.columns = ['municipio', 'real_assigned_plant']

    # Merge assignments
    assignment_comparison = pd.merge(voronoi_assignments, real_assignments, on='municipio')

    print(f"  Total municipalities analyzed: {len(assignment_comparison)}")

    # Count assignments per plant for each method
    voronoi_counts = assignment_comparison['voronoi_assigned_plant'].value_counts()
    real_counts = assignment_comparison['real_assigned_plant'].value_counts()

    # Get all plants that appear in either assignment
    all_plants = set(voronoi_counts.index) | set(real_counts.index)

    # Calculate changes for each plant
    plant_changes = []
    for plant in all_plants:
        voronoi_count = voronoi_counts.get(plant, 0)
        real_count = real_counts.get(plant, 0)
        net_change = real_count - voronoi_count  # Positive = gain, Negative = loss

        plant_changes.append({
            'plant': plant,
            'voronoi_assignments': voronoi_count,
            'real_assignments': real_count,
            'net_change': net_change
        })

    df_changes = pd.DataFrame(plant_changes)

    # Sort by plant name for consistent ordering
    df_changes = df_changes.sort_values('plant').reset_index(drop=True)

    # Add plant_id (1-indexed)
    df_changes['plant_id'] = range(1, len(df_changes) + 1)

    print(f"  Plants with changes: {len(df_changes)}")
    print(f"  Plants gaining municipalities: {(df_changes['net_change'] > 0).sum()}")
    print(f"  Plants losing municipalities: {(df_changes['net_change'] < 0).sum()}")
    print(f"  Plants with no change: {(df_changes['net_change'] == 0).sum()}")

    # Show top gainers and losers
    top_gainers = df_changes.nlargest(3, 'net_change')
    top_losers = df_changes.nsmallest(3, 'net_change')

    print("  Top 3 gainers:")
    for _, row in top_gainers.iterrows():
        print(f"    {row['plant']}: +{row['net_change']} municipalities")

    print("  Top 3 losers:")
    for _, row in top_losers.iterrows():
        if row['net_change'] < 0:
            print(f"    {row['plant']}: {row['net_change']} municipalities")

    return df_changes

def generate_figure4(df_changes):
    """Generate Figure 4: Bar chart of municipality assignment changes"""
    print("\n=== GENERATING FIGURE 4 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Prepare data
    x_positions = df_changes['plant_id']
    net_changes = df_changes['net_change']

    # Create colors: red for gains, gray for losses
    colors = ['red' if change > 0 else 'lightgray' for change in net_changes]

    # Create bar chart
    bars = ax.bar(x_positions, net_changes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.8)

    # Customize axes
    ax.set_xlabel('Plant ID')
    ax.set_ylabel('Net Change in Municipality Assignments')
    ax.set_title('Municipality Assignment Changes (Real vs Voronoi)')

    # Set x-axis to show all plant IDs
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_positions)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Plants gaining municipalities'),
        Patch(facecolor='lightgray', alpha=0.8, label='Plants losing municipalities')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add value labels on bars for significant changes
    for bar, change in zip(bars, net_changes):
        if abs(change) >= 2:  # Only label bars with significant changes
            height = bar.get_height()
            label_y = height + (0.5 if height > 0 else -0.5)
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{int(change):+d}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    # Add statistics text box
    total_gains = df_changes[df_changes['net_change'] > 0]['net_change'].sum()
    total_losses = df_changes[df_changes['net_change'] < 0]['net_change'].sum()
    net_zero_check = total_gains + total_losses

    stats_text = f"""Summary:
Plants gaining: {(df_changes['net_change'] > 0).sum()}
Plants losing: {(df_changes['net_change'] < 0).sum()}
Plants unchanged: {(df_changes['net_change'] == 0).sum()}

Total gains: +{total_gains}
Total losses: {total_losses}
Net balance: {net_zero_check}"""

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    plt.savefig('../figuras_clean/municipality_assignment_changes.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/municipality_assignment_changes.pdf")
    print(f"  Plants analyzed: {len(df_changes)}")
    print(f"  Total municipalities: {df_changes['real_assignments'].sum()}")
    print(f"  Balance check: {net_zero_check} (should be 0)")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 4: MUNICIPALITY ASSIGNMENT CHANGES")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant = load_clean_data()

    # Calculate assignment changes
    df_changes = calculate_assignment_changes(df_eucl_plant, df_real_plant)

    # Generate figure
    generate_figure4(df_changes)

    print("\n" + "=" * 80)
    print("FIGURE 4 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Bar chart of municipality assignment changes")
    print("  - Red bars: Plants gaining municipalities (Real vs Voronoi)")
    print("  - Gray bars: Plants losing municipalities")
    print("  - X-axis: Plant ID (1, 2, 3, ...)")
    print("  - Y-axis: Net change in assignments")
    print("  - Value labels on significant changes (>=2 municipalities)")

if __name__ == "__main__":
    main()