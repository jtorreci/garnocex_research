#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 3: Plant Anisotropy Analysis (2x2 subplots)
==========================================================

Figure 3 analyzes plant anisotropy coefficients with four panels:
- Top left: Histogram of anisotropy coefficients by plant
- Top right: Boxplots of anisotropy by number of assigned municipalities (1-5, 6-10, 11-20, >20)
- Bottom left: Scatter plot of municipalities served vs anisotropy coefficient with trend line
- Bottom right: Scatter plot of min beta vs max beta per plant with 1:1 isotropy line

Author: Voronoi Framework Team
Date: September 30, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
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

    print(f"  Plant data: {len(df_eucl_plant):,} euclidean, {len(df_real_plant):,} real")

    return df_eucl_plant, df_real_plant

def calculate_plant_anisotropy(df_eucl_plant, df_real_plant):
    """Calculate anisotropy coefficients for each plant"""
    print("\n=== CALCULATING PLANT ANISOTROPY ANALYSIS ===")

    # Standardize column names
    df_eucl = df_eucl_plant.rename(columns={'InputID': 'municipio', 'TargetID': 'planta', 'Distance': 'euclidean_distance'})
    df_real = df_real_plant.rename(columns={'origin_id': 'municipio', 'destination_id': 'planta'})

    # Merge on municipio-planta pairs
    df_merged = df_eucl.merge(df_real[['municipio', 'planta', 'total_cost']],
                             on=['municipio', 'planta'], how='inner')

    # Calculate beta ratios
    df_merged['beta_ratio'] = df_merged['total_cost'] / df_merged['euclidean_distance']

    # Remove duplicates
    df_merged = df_merged.drop_duplicates(subset=['municipio', 'planta'])

    # Calculate Voronoi assignments to get assigned municipalities per plant
    voronoi_assignments = df_eucl.loc[df_eucl.groupby('municipio')['euclidean_distance'].idxmin()]
    municipalities_per_plant = voronoi_assignments.groupby('planta').size().reset_index(name='num_municipalities')

    # Calculate anisotropy statistics per plant
    plant_stats = []
    for planta in df_merged['planta'].unique():
        plant_data = df_merged[df_merged['planta'] == planta]

        if len(plant_data) > 1:
            beta_min = plant_data['beta_ratio'].min()
            beta_max = plant_data['beta_ratio'].max()
            anisotropy = beta_max / beta_min
            beta_mean = plant_data['beta_ratio'].mean()
            beta_std = plant_data['beta_ratio'].std()

            plant_stats.append({
                'planta': planta,
                'anisotropy': anisotropy,
                'beta_min': beta_min,
                'beta_max': beta_max,
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'total_routes': len(plant_data)
            })

    df_plant_stats = pd.DataFrame(plant_stats)

    # Merge with municipalities per plant (Voronoi assignments)
    df_plant_stats = df_plant_stats.merge(municipalities_per_plant, on='planta', how='left')
    df_plant_stats['num_municipalities'] = df_plant_stats['num_municipalities'].fillna(0)

    # Create municipality groups
    def categorize_municipalities(num_munis):
        if num_munis <= 5:
            return '1-5'
        elif num_munis <= 10:
            return '6-10'
        elif num_munis <= 20:
            return '11-20'
        else:
            return '>20'

    df_plant_stats['muni_category'] = df_plant_stats['num_municipalities'].apply(categorize_municipalities)

    # Filter out outliers with anisotropy > 10
    outliers = df_plant_stats[df_plant_stats['anisotropy'] > 10.0]
    df_plant_stats_filtered = df_plant_stats[df_plant_stats['anisotropy'] <= 10.0].copy()

    print(f"  Plants analyzed (total): {len(df_plant_stats)}")
    print(f"  Outliers removed (anisotropy > 10): {len(outliers)}")
    if len(outliers) > 0:
        for _, outlier in outliers.iterrows():
            print(f"    {outlier['planta']}: anisotropy = {outlier['anisotropy']:.3f}")
    print(f"  Plants after filtering: {len(df_plant_stats_filtered)}")
    print(f"  Anisotropy range (filtered): {df_plant_stats_filtered['anisotropy'].min():.3f} - {df_plant_stats_filtered['anisotropy'].max():.3f}")
    print(f"  Mean anisotropy (filtered): {df_plant_stats_filtered['anisotropy'].mean():.3f}")
    print(f"  Municipality categories distribution:")
    print(df_plant_stats_filtered['muni_category'].value_counts().sort_index())

    return df_plant_stats_filtered

def generate_figure3(df_plant_stats):
    """Generate Figure 3: 2x2 plant anisotropy analysis"""
    print("\n=== GENERATING FIGURE 3 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Create figure with 2x2 subplots using GridSpec for better control
    from matplotlib import gridspec

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel A (top-left): Histogram of anisotropy coefficients
    anisotropy_values = df_plant_stats['anisotropy']

    n, bins, patches = ax1.hist(anisotropy_values, bins=15, alpha=0.7,
                               color='lightgray', edgecolor='black', linewidth=0.5)

    # Add vertical lines for mean and median
    mean_aniso = anisotropy_values.mean()
    median_aniso = anisotropy_values.median()

    ax1.axvline(mean_aniso, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_aniso:.3f}')
    ax1.axvline(median_aniso, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Median: {median_aniso:.3f}')

    # Statistics text box
    stats_text = f"""n = {len(anisotropy_values)}
Mean = {mean_aniso:.3f}
Median = {median_aniso:.3f}
Std = {anisotropy_values.std():.3f}
Max = {anisotropy_values.max():.3f}"""

    ax1.text(0.75, 0.85, stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=9)

    # Adjust x-axis limits to filtered data
    ax1.set_xlim(anisotropy_values.min() * 0.95, anisotropy_values.max() * 1.05)

    ax1.set_xlabel('Anisotropy Coefficient')
    ax1.set_ylabel('Number of Plants')
    ax1.set_title('A) Distribution of Plant Anisotropy Coefficients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B (top-right): Boxplots by municipality categories
    categories = ['1-5', '6-10', '11-20', '>20']
    category_data = [df_plant_stats[df_plant_stats['muni_category'] == cat]['anisotropy'].values
                     for cat in categories]

    # Remove empty categories
    non_empty_categories = []
    non_empty_data = []
    for i, (cat, data) in enumerate(zip(categories, category_data)):
        if len(data) > 0:
            non_empty_categories.append(cat)
            non_empty_data.append(data)

    bp = ax2.boxplot(non_empty_data, labels=non_empty_categories, patch_artist=True)

    # Style boxplots in grayscale
    for patch in bp['boxes']:
        patch.set_facecolor('lightgray')
        patch.set_alpha(0.7)

    # Style other elements in black
    for element in ['whiskers', 'fliers', 'caps']:
        if element in bp:
            for item in bp[element]:
                item.set_color('black')

    # Style medians in red and thick
    if 'medians' in bp:
        for median_line in bp['medians']:
            median_line.set_color('red')
            median_line.set_linewidth(3)

    # Add sample sizes
    for i, (cat, data) in enumerate(zip(non_empty_categories, non_empty_data)):
        ax2.text(i+1, ax2.get_ylim()[1]*0.95, f'n={len(data)}',
                ha='center', va='top', fontsize=8)

    # Adjust y-axis limits to filtered data
    all_boxplot_data = np.concatenate(non_empty_data)
    ax2.set_ylim(all_boxplot_data.min() * 0.95, all_boxplot_data.max() * 1.05)

    ax2.set_xlabel('Number of Assigned Municipalities')
    ax2.set_ylabel('Anisotropy Coefficient')
    ax2.set_title('B) Anisotropy by Municipality Assignment Groups')
    ax2.grid(True, alpha=0.3)

    # Panel C (bottom-left): Scatter plot with trend line
    x = df_plant_stats['num_municipalities']
    y = df_plant_stats['anisotropy']

    ax3.scatter(x, y, alpha=0.6, s=50, color='darkgray', edgecolor='black', linewidth=0.5)

    # Add trend line
    if len(x) > 1:
        # Remove any potential NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) > 1:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax3.plot(x_trend, p(x_trend), 'r-', linewidth=2, alpha=0.8, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')

            # Calculate correlation
            corr, p_value = stats.pearsonr(x_clean, y_clean)
            ax3.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.3f}',
                    transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    verticalalignment='top', fontsize=9)

    # Adjust axis limits to filtered data
    ax3.set_xlim(x.min() * 0.95, x.max() * 1.05)
    ax3.set_ylim(y.min() * 0.95, y.max() * 1.05)

    # Set x-axis to show only integer values (number of municipalities)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax3.set_xlabel('Number of Assigned Municipalities')
    ax3.set_ylabel('Anisotropy Coefficient')
    ax3.set_title('C) Municipalities Served vs Anisotropy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D (bottom-right): Min beta vs Max beta with 1:1 line
    x_beta = df_plant_stats['beta_min']
    y_beta = df_plant_stats['beta_max']

    ax4.scatter(x_beta, y_beta, alpha=0.6, s=50, color='darkgray', edgecolor='black', linewidth=0.5)

    # Set axis limits starting from (0,0) to show perfect isotropy line
    max_val = max(x_beta.max(), y_beta.max()) * 1.05
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)

    # Add 1:1 isotropy line from (0,0)
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.8,
             label='Perfect Isotropy (1:1)')

    # Add some diagonal reference lines for common anisotropy values

    # Get actual plot limits
    xlims = ax4.get_xlim()
    ylims = ax4.get_ylim()

    for aniso in [2, 3]:
        # Calculate line within plot boundaries
        x_start = max(xlims[0], ylims[0]/aniso)
        x_end = min(xlims[1], ylims[1]/aniso)

        if x_start < x_end:
            x_ref = np.array([x_start, x_end])
            y_ref = x_ref * aniso

            ax4.plot(x_ref, y_ref, 'r:', alpha=0.4, linewidth=1)

            # Place label at a safe position within plot bounds
            mid_x = (x_start + x_end) / 2
            mid_y = mid_x * aniso

            # Ensure label is within plot bounds
            if xlims[0] <= mid_x <= xlims[1] and ylims[0] <= mid_y <= ylims[1]:
                ax4.text(mid_x, mid_y, f'{aniso}:1',
                        rotation=np.degrees(np.arctan(aniso)), fontsize=7, alpha=0.7, color='red',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))

    ax4.set_xlabel('Minimum Beta Coefficient per Plant')
    ax4.set_ylabel('Maximum Beta Coefficient per Plant')
    ax4.set_title('D) Min vs Max Beta Coefficients by Plant')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Plant Anisotropy Analysis', fontsize=16, y=0.95)

    plt.savefig('../figuras_clean/plant_anisotropy_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/plant_anisotropy_analysis.pdf")
    print(f"  Panel A: Histogram with mean={mean_aniso:.3f}, median={median_aniso:.3f}")
    print(f"  Panel B: Boxplots for {len(non_empty_categories)} municipality categories")
    print(f"  Panel C: Scatter plot with trend line")
    print(f"  Panel D: Min vs Max beta with isotropy reference lines")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 3: PLANT ANISOTROPY ANALYSIS (2x2)")
    print("=" * 80)

    # Load clean data
    df_eucl_plant, df_real_plant = load_clean_data()

    # Calculate plant anisotropy statistics
    df_plant_stats = calculate_plant_anisotropy(df_eucl_plant, df_real_plant)

    # Generate figure
    generate_figure3(df_plant_stats)

    print("\n" + "=" * 80)
    print("FIGURE 3 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  - Panel A: Histogram of anisotropy coefficients with mean, median, std, max")
    print("  - Panel B: Boxplots by municipality assignment groups (1-5, 6-10, 11-20, >20)")
    print("  - Panel C: Scatter plot of municipalities served vs anisotropy with trend line")
    print("  - Panel D: Min vs Max beta scatter plot with 1:1 isotropy line")
    print("  - All data in grayscale, reference lines in red")

if __name__ == "__main__":
    main()