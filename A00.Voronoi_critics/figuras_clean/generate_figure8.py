#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 8: Spatial Analysis of Beta Coefficients (2x2)
===============================================================

Figure 8 analyzes the spatial distribution of beta coefficients in a 2x2 layout:

- Panel A (0,0): Spatial distribution of beta values (UTM coordinates with equal scale)
- Panel B (0,1): Moran's I scatter plot of spatial autocorrelation
- Panel C (1,0): Histogram of beta coefficients (local significance patterns)
- Panel D (1,1): Spatial clustering of beta coefficients (UTM with quartile categories)

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import os
import warnings
warnings.filterwarnings('ignore')


def load_boundary():
    """Load Extremadura boundary from GeoJSON, return (x, y) arrays."""
    gj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'extremadura.geojson')
    with open(gj_path, encoding='utf-8') as f:
        gj = json.load(f)
    coords = np.array(gj['features'][0]['geometry']['coordinates'][0][0])
    return coords[:, 0], coords[:, 1]

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

    print(f"  Plant data: {len(df_eucl_plant):,} euclidean, {len(df_real_plant):,} real")

    return df_eucl_plant, df_real_plant

def load_municipality_coordinates():
    """Load real UTM coordinates for municipalities from coordenadas_municipios.csv"""
    print("\n=== LOADING REAL MUNICIPALITY COORDINATES ===")

    # Load real coordinates
    coords_df = pd.read_csv("../codigo/coordenadas_municipios.csv")

    # Standardize column names
    coords_df = coords_df.rename(columns={'NOMBRE': 'municipality', 'X': 'utm_x', 'Y': 'utm_y'})

    # Remove trailing comma from municipality names if present
    coords_df['municipality'] = coords_df['municipality'].str.strip().str.rstrip(',')

    # Get unique municipalities from the data
    df_eucl_plant = pd.read_csv("../tables/D_euclidea_plantas_clean.csv")
    municipalities_in_data = set(df_eucl_plant['InputID'].unique())

    # Filter coordinates to only municipalities in our data
    coords_filtered = coords_df[coords_df['municipality'].isin(municipalities_in_data)].copy()

    print(f"  Total coordinates available: {len(coords_df)}")
    print(f"  Municipalities in analysis data: {len(municipalities_in_data)}")
    print(f"  Matched coordinates: {len(coords_filtered)}")
    print(f"  UTM X range: {coords_filtered['utm_x'].min():.0f} - {coords_filtered['utm_x'].max():.0f}")
    print(f"  UTM Y range: {coords_filtered['utm_y'].min():.0f} - {coords_filtered['utm_y'].max():.0f}")

    if len(coords_filtered) < len(municipalities_in_data):
        missing = municipalities_in_data - set(coords_filtered['municipality'])
        print(f"  Missing coordinates for {len(missing)} municipalities:")
        for i, muni in enumerate(sorted(missing)):
            if i < 5:  # Show first 5
                print(f"    {muni}")
            elif i == 5:
                print(f"    ... and {len(missing)-5} more")
                break

    return coords_filtered

def calculate_municipality_beta_averages(df_eucl_plant, df_real_plant, municipality_coords):
    """Calculate average beta coefficient for each municipality"""
    print("\n=== CALCULATING MUNICIPALITY BETA AVERAGES ===")

    # Standardize column names
    df_eucl = df_eucl_plant.rename(columns={'InputID': 'municipio', 'TargetID': 'planta', 'Distance': 'euclidean_distance'})
    df_real = df_real_plant.rename(columns={'origin_id': 'municipio', 'destination_id': 'planta'})

    # Merge to get both distances
    df_merged = df_eucl.merge(df_real[['municipio', 'planta', 'total_cost']],
                             on=['municipio', 'planta'], how='inner')

    # Calculate beta ratios
    df_merged['beta_ratio'] = df_merged['total_cost'] / df_merged['euclidean_distance']

    # Remove duplicates
    df_merged = df_merged.drop_duplicates(subset=['municipio', 'planta'])

    # Calculate average beta per municipality
    municipality_betas = df_merged.groupby('municipio')['beta_ratio'].agg(['mean', 'std', 'count']).reset_index()
    municipality_betas.columns = ['municipality', 'beta_mean', 'beta_std', 'beta_count']

    # Merge with coordinates
    spatial_data = municipality_coords.merge(municipality_betas, on='municipality', how='inner')

    print(f"  Municipalities with beta data: {len(spatial_data)}")
    print(f"  Beta mean range: {spatial_data['beta_mean'].min():.3f} - {spatial_data['beta_mean'].max():.3f}")
    print(f"  Average connections per municipality: {spatial_data['beta_count'].mean():.1f}")

    return spatial_data

def calculate_morans_i(spatial_data):
    """Calculate Moran's I for spatial autocorrelation"""
    print("\n=== CALCULATING MORAN'S I ===")

    coords = spatial_data[['utm_x', 'utm_y']].values
    values = spatial_data['beta_mean'].values

    # Calculate distance matrix
    distances = squareform(pdist(coords))

    # Create spatial weights matrix (inverse distance with cutoff)
    max_distance = np.percentile(distances[distances > 0], 25)  # Use 25th percentile as cutoff
    weights = np.zeros_like(distances)

    for i in range(len(distances)):
        for j in range(len(distances)):
            if i != j and distances[i, j] <= max_distance:
                weights[i, j] = 1.0 / distances[i, j]

    # Normalize weights
    row_sums = weights.sum(axis=1)
    for i in range(len(weights)):
        if row_sums[i] > 0:
            weights[i] = weights[i] / row_sums[i]

    # Calculate Moran's I
    n = len(values)
    mean_val = np.mean(values)

    # Numerator: spatial covariance
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)

    # Denominator: variance
    denominator = np.sum((values - mean_val) ** 2)

    # Moran's I
    morans_i = numerator / denominator if denominator > 0 else 0

    # Calculate standardized values for scatter plot
    z_values = (values - mean_val) / np.std(values)
    spatial_lag = np.zeros(n)

    for i in range(n):
        neighbors = weights[i] > 0
        if neighbors.sum() > 0:
            spatial_lag[i] = np.sum(weights[i, neighbors] * z_values[neighbors])

    print(f"  Moran's I: {morans_i:.4f}")
    print(f"  Distance cutoff: {max_distance:.0f} m")
    print(f"  Average neighbors per municipality: {(weights > 0).sum(axis=1).mean():.1f}")

    return morans_i, z_values, spatial_lag, weights

def calculate_quartile_categories(spatial_data):
    """Calculate quartile categories for spatial clustering"""
    betas = spatial_data['beta_mean']

    q1 = betas.quantile(0.25)
    q2 = betas.quantile(0.50)
    q3 = betas.quantile(0.75)

    def categorize_beta(beta):
        if beta <= q1:
            return 'Low (Q1)'
        elif beta <= q2:
            return 'Medium-Low (Q2)'
        elif beta <= q3:
            return 'Medium-High (Q3)'
        else:
            return 'High (Q4)'

    spatial_data['quartile_category'] = spatial_data['beta_mean'].apply(categorize_beta)

    print(f"  Quartile thresholds: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")

    return spatial_data

def generate_figure8(spatial_data):
    """Generate Figure 8: 2x2 spatial analysis of beta coefficients"""
    print("\n=== GENERATING FIGURE 8 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Calculate Moran's I
    morans_i, z_values, spatial_lag, weights = calculate_morans_i(spatial_data)

    # Calculate quartile categories
    spatial_data = calculate_quartile_categories(spatial_data)

    # Create figure with 2x2 subplots
    from matplotlib import gridspec
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Load Extremadura boundary
    bnd_x, bnd_y = load_boundary()

    # Panel A (0,0): Spatial distribution with color scale
    ax1 = fig.add_subplot(gs[0, 0])

    # Extremadura boundary
    ax1.plot(bnd_x, bnd_y, color='#333333', lw=1.0, zorder=0)

    # Create scatter plot with color mapping
    scatter = ax1.scatter(spatial_data['utm_x'], spatial_data['utm_y'],
                         c=spatial_data['beta_mean'], cmap='viridis',
                         s=60, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Ensure equal aspect ratio and set consistent axis limits
    ax1.set_aspect('equal')

    # Calculate common axis limits for both spatial panels
    x_min, x_max = spatial_data['utm_x'].min(), spatial_data['utm_x'].max()
    y_min, y_max = spatial_data['utm_y'].min(), spatial_data['utm_y'].max()

    # Add some padding (5%)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Average Beta Coefficient')

    ax1.set_xlabel('UTM X (m)')
    ax1.set_ylabel('UTM Y (m)')
    ax1.set_title('A) Spatial Distribution of Beta Coefficients')
    ax1.grid(True, alpha=0.3)

    # Panel B (0,1): Moran's I scatter plot
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.scatter(z_values, spatial_lag, alpha=0.6, s=50, color='darkblue', edgecolor='black', linewidth=0.5)

    # Add regression line
    if len(z_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(z_values, spatial_lag)
        line_x = np.linspace(z_values.min(), z_values.max(), 100)
        line_y = slope * line_x + intercept
        ax2.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.8)

        # Add Moran's I text
        ax2.text(0.05, 0.95, f"Moran's I = {morans_i:.4f}\nSlope = {slope:.4f}\nR² = {r_value**2:.4f}",
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=9)

    # Add quadrant lines
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Standardized Beta Values')
    ax2.set_ylabel('Spatial Lag')
    ax2.set_title("B) Moran's I Scatter Plot")
    ax2.grid(True, alpha=0.3)

    # Panel C (1,0): Histogram of beta coefficients
    ax3 = fig.add_subplot(gs[1, 0])

    # Create histogram with quartile coloring
    betas = spatial_data['beta_mean']
    n, bins, patches = ax3.hist(betas, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color bars by quartiles
    q1, q2, q3 = betas.quantile([0.25, 0.5, 0.75])

    for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
        if bin_center <= q1:
            patch.set_facecolor('lightblue')
        elif bin_center <= q2:
            patch.set_facecolor('lightgreen')
        elif bin_center <= q3:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('lightcoral')

    # Add quartile lines
    ax3.axvline(q1, color='blue', linestyle='--', alpha=0.8, label=f'Q1 = {q1:.3f}')
    ax3.axvline(q2, color='green', linestyle='--', alpha=0.8, label=f'Q2 = {q2:.3f}')
    ax3.axvline(q3, color='orange', linestyle='--', alpha=0.8, label=f'Q3 = {q3:.3f}')

    ax3.set_xlabel('Beta Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('C) Beta Distribution (Local Significance Patterns)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D (1,1): Spatial clustering by quartiles
    ax4 = fig.add_subplot(gs[1, 1])

    # Extremadura boundary
    ax4.plot(bnd_x, bnd_y, color='#333333', lw=1.0, zorder=0)

    # Define colors for quartiles
    quartile_colors = {
        'Low (Q1)': 'lightblue',
        'Medium-Low (Q2)': 'lightgreen',
        'Medium-High (Q3)': 'yellow',
        'High (Q4)': 'lightcoral'
    }

    # Plot points by quartile
    for category, color in quartile_colors.items():
        mask = spatial_data['quartile_category'] == category
        if mask.sum() > 0:
            ax4.scatter(spatial_data.loc[mask, 'utm_x'], spatial_data.loc[mask, 'utm_y'],
                       c=color, s=60, alpha=0.8, edgecolor='black', linewidth=0.5,
                       label=f'{category} (n={mask.sum()})')

    # Ensure equal aspect ratio and same axis limits as Panel A
    ax4.set_aspect('equal')

    # Use the same axis limits as Panel A for consistent scale
    ax4.set_xlim(x_min - x_padding, x_max + x_padding)
    ax4.set_ylim(y_min - y_padding, y_max + y_padding)

    ax4.set_xlabel('UTM X (m)')
    ax4.set_ylabel('UTM Y (m)')
    ax4.set_title('D) Spatial Clustering (Quartile Categories)')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Spatial Analysis of Beta Coefficients', fontsize=16, y=0.96)

    plt.savefig('../figuras_clean/spatial_analysis_beta_coefficients.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/spatial_analysis_beta_coefficients.pdf")
    print(f"  Moran's I: {morans_i:.4f}")
    print(f"  Municipalities analyzed: {len(spatial_data)}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 8: SPATIAL ANALYSIS BETA COEFFICIENTS (2x2)")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant = load_clean_data()

    # Load real municipality coordinates
    municipality_coords = load_municipality_coordinates()

    # Calculate municipality beta averages
    spatial_data = calculate_municipality_beta_averages(df_eucl_plant, df_real_plant, municipality_coords)

    # Generate figure
    generate_figure8(spatial_data)

    print("\n" + "=" * 80)
    print("FIGURE 8 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Panel A: Spatial distribution with color scale (UTM coordinates)")
    print("  - Panel B: Moran's I scatter plot for spatial autocorrelation")
    print("  - Panel C: Histogram with quartile-based coloring")
    print("  - Panel D: Spatial clustering by quartile categories")
    print("  - Equal scale UTM coordinates in spatial panels")

if __name__ == "__main__":
    main()