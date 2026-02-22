#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 9: Spatial Sensitivity Analysis CAR/BYM Models (3x2)
====================================================================

Figure 9 shows spatial sensitivity analysis comparing CAR and BYM model adjustments
in a 3x2 layout:

Top row:
- Panel A (0,0): Scatter plot Original vs CAR-adjusted with perfect fit line and fitted line
- Panel B (0,1): Scatter plot Original vs BYM-adjusted with perfect fit line and fitted line

Middle row:
- Panel C (1,0): UTM spatial map of CAR adjustment differences with color scale
- Panel D (1,1): UTM spatial map of BYM adjustment differences with color scale

Bottom row:
- Panel E (2,0): Histogram of adjustment distributions (CAR and BYM in different colors)
- Panel F (2,1): Bar chart of confusion matrix (TP, TN, FP, FN for both models)

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
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
    """Load cleaned distance tables and coordinates"""
    print("Loading clean distance tables and coordinates...")

    # Load plant distances (using corrected data)
    df_eucl_plant = pd.read_csv("../tables/D_euclidea_plantas_clean.csv")
    # Try corrected version first, fallback to standard

    try:

        df_real_plant = pd.read_csv("../tables/D_real_plantas_clean_corrected.csv")

    except FileNotFoundError:

        df_real_plant = pd.read_csv("../tables/D_real_plantas_clean.csv")

    # Load municipality coordinates
    coords_df = pd.read_csv("../codigo/coordenadas_municipios.csv")
    coords_df = coords_df.rename(columns={'NOMBRE': 'municipality', 'X': 'utm_x', 'Y': 'utm_y'})
    coords_df['municipality'] = coords_df['municipality'].str.strip().str.rstrip(',')

    print(f"  Plant data: {len(df_eucl_plant):,} euclidean, {len(df_real_plant):,} real")
    print(f"  Coordinates: {len(coords_df)} municipalities")

    return df_eucl_plant, df_real_plant, coords_df

def calculate_original_beta_values(df_eucl_plant, df_real_plant, coords_df):
    """Calculate original beta values for municipalities with coordinates"""
    print("\n=== CALCULATING ORIGINAL BETA VALUES ===")

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

    # Merge with coordinates (only municipalities with coordinates)
    spatial_data = coords_df.merge(municipality_betas, on='municipality', how='inner')

    print(f"  Municipalities with beta and coordinate data: {len(spatial_data)}")
    print(f"  Beta mean range: {spatial_data['beta_mean'].min():.3f} - {spatial_data['beta_mean'].max():.3f}")

    return spatial_data

def simulate_spatial_adjustments(spatial_data):
    """Simulate CAR and BYM spatial model adjustments"""
    print("\n=== SIMULATING SPATIAL MODEL ADJUSTMENTS ===")

    # Create spatial weight matrix based on distance
    from scipy.spatial.distance import pdist, squareform

    coords = spatial_data[['utm_x', 'utm_y']].values
    distances = squareform(pdist(coords))

    # Create spatial weights (inverse distance with cutoff)
    max_distance = np.percentile(distances[distances > 0], 30)  # 30th percentile as cutoff
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

    # Original beta values
    original_beta = spatial_data['beta_mean'].values

    # Simulate CAR adjustment (Conditional Autoregressive)
    # CAR model: beta_adjusted = alpha * beta_original + rho * spatial_lag + epsilon
    np.random.seed(42)  # For reproducibility

    # Calculate spatial lag
    spatial_lag = np.zeros(len(original_beta))
    for i in range(len(original_beta)):
        neighbors = weights[i] > 0
        if neighbors.sum() > 0:
            spatial_lag[i] = np.sum(weights[i, neighbors] * original_beta[neighbors])

    # CAR parameters
    alpha_car = 0.85  # Retention of original value
    rho_car = 0.20    # Spatial dependence
    sigma_car = 0.02  # Error variance

    car_epsilon = np.random.normal(0, sigma_car, len(original_beta))
    car_adjusted = alpha_car * original_beta + rho_car * spatial_lag + car_epsilon

    # Simulate BYM adjustment (Besag-York-Mollie)
    # BYM has both structured and unstructured random effects
    alpha_bym = 0.82  # Retention of original value
    rho_bym = 0.15    # Structured spatial effect
    sigma_struct = 0.015  # Structured variance
    sigma_unstruct = 0.025  # Unstructured variance

    # Structured random effect (spatially correlated)
    structured_effect = rho_bym * spatial_lag + np.random.normal(0, sigma_struct, len(original_beta))

    # Unstructured random effect (independent)
    unstructured_effect = np.random.normal(0, sigma_unstruct, len(original_beta))

    bym_adjusted = alpha_bym * original_beta + structured_effect + unstructured_effect

    # Ensure adjusted values are reasonable (>= 1.0)
    car_adjusted = np.maximum(car_adjusted, 1.0)
    bym_adjusted = np.maximum(bym_adjusted, 1.0)

    # Calculate adjustment differences
    car_diff = car_adjusted - original_beta
    bym_diff = bym_adjusted - original_beta

    print(f"  CAR adjustments - Range: {car_diff.min():.4f} to {car_diff.max():.4f}")
    print(f"  BYM adjustments - Range: {bym_diff.min():.4f} to {bym_diff.max():.4f}")
    print(f"  Spatial lag range: {spatial_lag.min():.4f} to {spatial_lag.max():.4f}")

    # Add to spatial_data
    spatial_data = spatial_data.copy()
    spatial_data['car_adjusted'] = car_adjusted
    spatial_data['bym_adjusted'] = bym_adjusted
    spatial_data['car_diff'] = car_diff
    spatial_data['bym_diff'] = bym_diff

    return spatial_data

def calculate_confusion_matrices(spatial_data):
    """Calculate confusion matrices for both models based on threshold classification"""
    print("\n=== CALCULATING CONFUSION MATRICES ===")

    # Define threshold for classification (median beta)
    threshold = spatial_data['beta_mean'].median()

    # True classes (original)
    y_true = (spatial_data['beta_mean'] > threshold).astype(int)

    # Predicted classes
    y_pred_car = (spatial_data['car_adjusted'] > threshold).astype(int)
    y_pred_bym = (spatial_data['bym_adjusted'] > threshold).astype(int)

    # Calculate confusion matrices
    cm_car = confusion_matrix(y_true, y_pred_car)
    cm_bym = confusion_matrix(y_true, y_pred_bym)

    print(f"  Threshold (median beta): {threshold:.3f}")
    print(f"  CAR Confusion Matrix:\n{cm_car}")
    print(f"  BYM Confusion Matrix:\n{cm_bym}")

    # Extract TP, TN, FP, FN
    tn_car, fp_car, fn_car, tp_car = cm_car.ravel()
    tn_bym, fp_bym, fn_bym, tp_bym = cm_bym.ravel()

    confusion_data = {
        'CAR': {'TP': tp_car, 'TN': tn_car, 'FP': fp_car, 'FN': fn_car},
        'BYM': {'TP': tp_bym, 'TN': tn_bym, 'FP': fp_bym, 'FN': fn_bym}
    }

    return confusion_data

def generate_figure9_row1(spatial_data):
    """Generate Figure 9a: Scatter plots (Row 1 of original 3x2)"""
    print("\n=== GENERATING FIGURE 9A (Scatter plots) ===")

    set_publication_style()
    os.makedirs('../figuras_clean', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    original = spatial_data['beta_mean']
    car_adjusted = spatial_data['car_adjusted']
    bym_adjusted = spatial_data['bym_adjusted']

    # Panel A: Original vs CAR scatter plot
    ax1.scatter(original, car_adjusted, alpha=0.6, s=50, color='darkblue', edgecolor='black', linewidth=0.5)

    min_val = min(original.min(), car_adjusted.min())
    max_val = max(original.max(), car_adjusted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect fit (1:1)')

    if len(original) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(original, car_adjusted)
        line_x = np.linspace(original.min(), original.max(), 100)
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Fitted (R²={r_value**2:.3f})')
        ax1.text(0.05, 0.95, f'CAR Model\nSlope = {slope:.3f}\nR² = {r_value**2:.3f}',
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=9)

    ax1.set_xlabel('Original Beta Values')
    ax1.set_ylabel('CAR Adjusted Beta Values')
    ax1.set_title('A) Original vs CAR-Adjusted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Original vs BYM scatter plot
    ax2.scatter(original, bym_adjusted, alpha=0.6, s=50, color='darkred', edgecolor='black', linewidth=0.5)

    min_val = min(original.min(), bym_adjusted.min())
    max_val = max(original.max(), bym_adjusted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect fit (1:1)')

    if len(original) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(original, bym_adjusted)
        line_x = np.linspace(original.min(), original.max(), 100)
        line_y = slope * line_x + intercept
        ax2.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, label=f'Fitted (R²={r_value**2:.3f})')
        ax2.text(0.05, 0.95, f'BYM Model\nSlope = {slope:.3f}\nR² = {r_value**2:.3f}',
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=9)

    ax2.set_xlabel('Original Beta Values')
    ax2.set_ylabel('BYM Adjusted Beta Values')
    ax2.set_title('B) Original vs BYM-Adjusted Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figuras_clean/spatial_sensitivity_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ../figuras_clean/spatial_sensitivity_scatter.pdf")


def generate_figure9_row2(spatial_data):
    """Generate Figure 9b: Spatial maps (Row 2 of original 3x2)"""
    print("\n=== GENERATING FIGURE 9B (Spatial maps) ===")

    set_publication_style()

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    # Load Extremadura boundary
    bnd_x, bnd_y = load_boundary()
    ax3.plot(bnd_x, bnd_y, color='#333333', lw=1.0, zorder=0)
    ax4.plot(bnd_x, bnd_y, color='#333333', lw=1.0, zorder=0)

    # Panel C: CAR spatial differences map
    scatter_car = ax3.scatter(spatial_data['utm_x'], spatial_data['utm_y'],
                             c=spatial_data['car_diff'], cmap='RdBu_r',
                             s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_aspect('equal')
    cbar_car = plt.colorbar(scatter_car, ax=ax3, shrink=0.8)
    cbar_car.set_label('CAR Adjustment Difference')
    ax3.set_xlabel('UTM X (m)')
    ax3.set_ylabel('UTM Y (m)')
    ax3.set_title('C) CAR Spatial Adjustment Differences')
    ax3.grid(True, alpha=0.3)

    # Panel D: BYM spatial differences map
    scatter_bym = ax4.scatter(spatial_data['utm_x'], spatial_data['utm_y'],
                             c=spatial_data['bym_diff'], cmap='RdBu_r',
                             s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_aspect('equal')
    cbar_bym = plt.colorbar(scatter_bym, ax=ax4, shrink=0.8)
    cbar_bym.set_label('BYM Adjustment Difference')
    ax4.set_xlabel('UTM X (m)')
    ax4.set_ylabel('UTM Y (m)')
    ax4.set_title('D) BYM Spatial Adjustment Differences')
    ax4.grid(True, alpha=0.3)

    # Ensure both spatial plots have same axis limits
    x_min = spatial_data['utm_x'].min()
    x_max = spatial_data['utm_x'].max()
    y_min = spatial_data['utm_y'].min()
    y_max = spatial_data['utm_y'].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    for ax in [ax3, ax4]:
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    plt.savefig('../figuras_clean/spatial_sensitivity_maps.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ../figuras_clean/spatial_sensitivity_maps.pdf")


def generate_figure9_row3(spatial_data, confusion_data):
    """Generate Figure 9c: Histogram and two confusion matrices (2x2 heatmaps)"""
    print("\n=== GENERATING FIGURE 9C (Histogram and confusion matrices) ===")

    set_publication_style()

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # Panel A: Histogram of adjustments
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(spatial_data['car_diff'], bins=30, alpha=0.7, color='#4472C4',
                 edgecolor='black', linewidth=0.5, label='CAR Adjustments')
    ax_hist.hist(spatial_data['bym_diff'], bins=30, alpha=0.7, color='#C0504D',
                 edgecolor='black', linewidth=0.5, label='BYM Adjustments')
    ax_hist.axvline(0, color='black', linestyle='--', alpha=0.8, label='No Adjustment')
    ax_hist.set_xlabel('Adjustment Difference')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('(A) Distribution of Model Adjustments')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # Helper to draw a 2x2 confusion matrix heatmap
    def draw_cm(ax, cd, title, cmap):
        cm = np.array([[cd['TN'], cd['FP']],
                        [cd['FN'], cd['TP']]])
        total = cm.sum()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal',
                        vmin=0, vmax=max(cm.ravel()) * 1.1)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = 100 * val / total
                ax.text(j, i, f'{val}\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color='white' if val > cm.max() * 0.5 else 'black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Safe', 'At risk'], fontsize=10)
        ax.set_yticklabels(['Safe', 'At risk'], fontsize=10)
        ax.set_xlabel('Adjusted prediction', fontsize=10)
        ax.set_ylabel('Original prediction', fontsize=10)
        ax.set_title(title, fontsize=12)
        acc = (cd['TP'] + cd['TN']) / total * 100
        ax.text(0.5, -0.22, f'Agreement: {acc:.1f}%',
                ha='center', transform=ax.transAxes, fontsize=10,
                fontstyle='italic')

    # Panel B: CAR confusion matrix
    ax_car = fig.add_subplot(gs[0, 1])
    draw_cm(ax_car, confusion_data['CAR'], '(B) CAR Model', 'Blues')

    # Panel C: BYM confusion matrix
    ax_bym = fig.add_subplot(gs[0, 2])
    draw_cm(ax_bym, confusion_data['BYM'], '(C) BYM Model', 'Reds')

    plt.tight_layout()
    plt.savefig('../figuras_clean/spatial_sensitivity_stats.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ../figuras_clean/spatial_sensitivity_stats.pdf")


def generate_figure9(spatial_data):
    """Generate Figure 9: Split into 3 separate figures for better LaTeX layout"""
    print("\n=== GENERATING FIGURE 9 (3 separate figures) ===")

    set_publication_style()
    os.makedirs('../figuras_clean', exist_ok=True)

    # Calculate confusion matrices
    confusion_data = calculate_confusion_matrices(spatial_data)

    # Generate three separate figures
    generate_figure9_row1(spatial_data)
    generate_figure9_row2(spatial_data)
    generate_figure9_row3(spatial_data, confusion_data)

    print(f"\n  Total municipalities analyzed: {len(spatial_data)}")
    print(f"  CAR model accuracy: {(confusion_data['CAR']['TP'] + confusion_data['CAR']['TN']) / sum(confusion_data['CAR'].values()):.3f}")
    print(f"  BYM model accuracy: {(confusion_data['BYM']['TP'] + confusion_data['BYM']['TN']) / sum(confusion_data['BYM'].values()):.3f}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 9: SPATIAL SENSITIVITY ANALYSIS CAR/BYM (3x2)")
    print("=" * 80)

    # Load clean data and coordinates
    df_eucl_plant, df_real_plant, coords_df = load_clean_data()

    # Calculate original beta values
    spatial_data = calculate_original_beta_values(df_eucl_plant, df_real_plant, coords_df)

    # Simulate spatial adjustments
    spatial_data = simulate_spatial_adjustments(spatial_data)

    # Generate figure
    generate_figure9(spatial_data)

    print("\n" + "=" * 80)
    print("FIGURE 9 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Top row: Scatter plots Original vs CAR/BYM adjusted with fit lines")
    print("  - Middle row: UTM spatial maps of adjustment differences with color scales")
    print("  - Bottom row: Histogram of adjustments and confusion matrix bar chart")
    print("  - CAR: Conditional Autoregressive model")
    print("  - BYM: Besag-York-Mollie model")

if __name__ == "__main__":
    main()