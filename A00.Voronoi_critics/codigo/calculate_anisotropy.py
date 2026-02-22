#!/usr/bin/env python3
"""
Script to calculate complete anisotropy coefficients for all 388 municipalities
and generate explanatory plots for both statistical tables.

Author: Claude Code Analysis
Date: September 16, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up paths
codigo_dir = Path(__file__).parent
output_dir = codigo_dir.parent / "imagenes"
tables_dir = codigo_dir.parent / "tables"

# Create output directories
output_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)

def load_distance_data():
    """Load and process the distance matrix data."""
    print("Loading distance data...")

    # Load the main distance matrix (European format: semicolon separator, comma decimal)
    matriz_file = codigo_dir / "Matriz_Municipios.csv"
    if not matriz_file.exists():
        raise FileNotFoundError(f"Distance matrix not found: {matriz_file}")

    # Read the European format CSV
    df_raw = pd.read_csv(matriz_file, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"Loaded raw distance data: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")

    # Process the data - we need total_cost (network distance)
    df_processed = df_raw[['origin_id', 'destination_id', 'total_cost']].copy()

    # Remove rows with missing costs
    df_processed = df_processed.dropna(subset=['total_cost'])

    # Convert to pivot table (municipality x plant matrix)
    df_matrix = df_processed.pivot(index='origin_id', columns='destination_id', values='total_cost')
    print(f"Created distance matrix: {df_matrix.shape}")

    return df_matrix

def calculate_ratios_from_matrix(df_distances):
    """Calculate beta ratios from the distance matrix."""
    print("Calculating beta ratios...")

    # Load Euclidean distances for comparison
    euclidean_file = codigo_dir / "distancias_euclideas.csv"
    if euclidean_file.exists():
        # Load Euclidean distances (same European format)
        df_euclidean_raw = pd.read_csv(euclidean_file, sep=';', decimal=',', encoding='utf-8-sig')
        print(f"Loaded Euclidean raw data: {df_euclidean_raw.shape}")

        # Convert to pivot table
        df_euclidean = df_euclidean_raw.pivot(index='origin_id', columns='destination_id', values='distance_m')
        print(f"Created Euclidean distance matrix: {df_euclidean.shape}")

        # Calculate ratios where both distances exist
        ratios = []
        for muni in df_distances.index:
            if muni in df_euclidean.index:
                for plant in df_distances.columns:
                    if plant in df_euclidean.columns:
                        d_network = df_distances.loc[muni, plant]
                        d_euclidean = df_euclidean.loc[muni, plant]

                        # Only include valid positive distances
                        if pd.notna(d_network) and pd.notna(d_euclidean) and d_euclidean > 0:
                            ratio = d_network / d_euclidean
                            ratios.append({
                                'Municipality': muni,
                                'Plant': plant,
                                'NetworkDistance': d_network,
                                'EuclideanDistance': d_euclidean,
                                'Ratio': ratio
                            })

        return pd.DataFrame(ratios)

    else:
        # Fallback: assume matrix contains ratios directly
        print("Euclidean distances not found, assuming matrix contains ratios")
        ratios = []
        for muni in df_distances.index:
            for plant in df_distances.columns:
                ratio = df_distances.loc[muni, plant]
                if pd.notna(ratio) and ratio > 0:
                    ratios.append({
                        'Municipality': muni,
                        'Plant': plant,
                        'Ratio': ratio
                    })

        return pd.DataFrame(ratios)

def calculate_anisotropy_coefficients(df_ratios):
    """Calculate anisotropy coefficient for each municipality."""
    print("Calculating anisotropy coefficients...")

    anisotropy_results = []

    for municipality in df_ratios['Municipality'].unique():
        muni_data = df_ratios[df_ratios['Municipality'] == municipality]

        if len(muni_data) >= 2:  # Need at least 2 plants for anisotropy
            ratios = muni_data['Ratio'].values
            max_ratio = np.max(ratios)
            min_ratio = np.min(ratios)

            anisotropy_coeff = max_ratio / min_ratio

            anisotropy_results.append({
                'Municipality': municipality,
                'MaxRatio': max_ratio,
                'MinRatio': min_ratio,
                'AnisotropyCoefficient': anisotropy_coeff,
                'NumPlants': len(muni_data)
            })

    return pd.DataFrame(anisotropy_results)

def save_latex_table(stats_series, filename, caption, label):
    """Save pandas describe() output as LaTeX table."""
    stats_df = stats_series.to_frame().reset_index()
    stats_df.columns = ['Statistic', 'Value']

    # Rename for clarity
    metric_map = {
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Std. Dev.',
        'min': 'Minimum',
        '25%': '25\\% (Q1)',
        '50%': 'Median (50\\%)',
        '75%': '75\\% (Q3)',
        'max': 'Maximum'
    }

    stats_df['Statistic'] = stats_df['Statistic'].map(metric_map).fillna(stats_df['Statistic'])

    # Format values
    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.6f}")

    # Create LaTeX table
    latex_content = f"""\\begin{{table}}[htbp]
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lr}}
\\toprule
Statistic & Value \\\\
\\midrule
"""

    for _, row in stats_df.iterrows():
        latex_content += f"{row['Statistic']} & {row['Value']} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"Saved LaTeX table: {filename}")

def create_explanatory_plots(df_ratios, df_anisotropy):
    """Create explanatory plots for both statistical tables."""
    print("Creating explanatory plots...")

    # Set style for better readability and grayscale palette
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

    # Figure 1: Ratio distribution histogram with detailed statistics
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Histogram of beta ratios
    ax1.hist(df_ratios['Ratio'], bins=50, density=True, alpha=0.8, color='lightgray', edgecolor='black')
    ax1.set_xlabel(r'Network-to-Euclidean Distance Ratio (beta = d_r/d_e)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Distance Ratios\\n(n = {len(df_ratios):,} municipality-plant pairs)')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Mean: {df_ratios['Ratio'].mean():.3f}
Median: {df_ratios['Ratio'].median():.3f}
Std: {df_ratios['Ratio'].std():.3f}
Min: {df_ratios['Ratio'].min():.3f}
Max: {df_ratios['Ratio'].max():.3f}"""
    ax1.text(0.7, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Box plot of β ratios
    ax2.boxplot(df_ratios['Ratio'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.8),
                medianprops=dict(color='red', linewidth=3))
    ax2.set_ylabel(r'Network-to-Euclidean Distance Ratio (beta)')
    ax2.set_title('Box Plot of Distance Ratios')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels(['All Pairs'])

    plt.tight_layout()
    plt.savefig(output_dir / 'ratio_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Anisotropy coefficients analysis
    fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram of anisotropy coefficients
    ax3.hist(df_anisotropy['AnisotropyCoefficient'], bins=30, density=True, alpha=0.8,
             color='lightgray', edgecolor='black')

    # Add mean line in red
    mean_aniso = df_anisotropy['AnisotropyCoefficient'].mean()
    ax3.axvline(mean_aniso, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_aniso:.2f}')

    ax3.set_xlabel('Anisotropy Coefficient (max β / min β)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Distribution of Anisotropy Coefficients\\n(n = {len(df_anisotropy)} municipalities)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add statistics
    aniso_stats = f"""Mean: {df_anisotropy['AnisotropyCoefficient'].mean():.3f}
Median: {df_anisotropy['AnisotropyCoefficient'].median():.3f}
Std: {df_anisotropy['AnisotropyCoefficient'].std():.3f}"""
    ax3.text(0.7, 0.95, aniso_stats, transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Box plot of anisotropy
    ax4.boxplot(df_anisotropy['AnisotropyCoefficient'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.8),
                medianprops=dict(color='red', linewidth=3))
    ax4.set_ylabel('Anisotropy Coefficient')
    ax4.set_title('Box Plot of Anisotropy Coefficients')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels(['All Municipalities'])

    # Scatter plot: Number of plants vs Anisotropy
    ax5.scatter(df_anisotropy['NumPlants'], df_anisotropy['AnisotropyCoefficient'],
                alpha=0.7, s=60, color='darkgray', edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Number of Plants Accessible')
    ax5.set_ylabel('Anisotropy Coefficient')
    ax5.set_title('Anisotropy vs Number of Accessible Plants')
    ax5.grid(True, alpha=0.3)

    # Max vs Min ratio scatter
    ax6.scatter(df_anisotropy['MinRatio'], df_anisotropy['MaxRatio'],
                alpha=0.7, s=60, color='darkgray', edgecolor='black', linewidth=0.5)
    ax6.plot([df_anisotropy['MinRatio'].min(), df_anisotropy['MinRatio'].max()],
             [df_anisotropy['MinRatio'].min(), df_anisotropy['MinRatio'].max()],
             'r--', linewidth=2, alpha=0.8, label='Equal ratios (no anisotropy)')
    ax6.set_xlabel('Minimum beta Ratio')
    ax6.set_ylabel('Maximum beta Ratio')
    ax6.set_title('Maximum vs Minimum beta Ratios by Municipality')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'anisotropy_analysis_complete.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Created explanatory plots:")
    print(f"  - {output_dir / 'ratio_distribution_analysis.png'}")
    print(f"  - {output_dir / 'anisotropy_analysis_complete.png'}")

def main():
    """Main execution function."""
    print("=== ANISOTROPY COEFFICIENT CALCULATION ===")
    print("Starting complete analysis of transport network anisotropy...")

    try:
        # Load data
        df_distances = load_distance_data()

        # Calculate ratios
        df_ratios = calculate_ratios_from_matrix(df_distances)
        print(f"Calculated {len(df_ratios)} beta ratios")

        # Calculate anisotropy coefficients
        df_anisotropy = calculate_anisotropy_coefficients(df_ratios)
        print(f"Calculated anisotropy for {len(df_anisotropy)} municipalities")

        # Generate updated statistics tables
        ratio_stats = df_ratios['Ratio'].describe()
        anisotropy_stats = df_anisotropy['AnisotropyCoefficient'].describe()

        # Save updated LaTeX tables
        save_latex_table(
            ratio_stats,
            tables_dir / "table_stats_ratio_updated.tex",
            "Descriptive statistics of the network-to-Euclidean distance ratio ($d_r/d_e$) - Complete Analysis.",
            "tab:stats_ratio_updated"
        )

        save_latex_table(
            anisotropy_stats,
            tables_dir / "table_stats_anisotropy_updated.tex",
            "Descriptive statistics of the transport network anisotropy coefficient - Complete Analysis.",
            "tab:stats_anisotropy_updated"
        )

        # Create explanatory plots
        create_explanatory_plots(df_ratios, df_anisotropy)

        # Save detailed CSV results
        df_ratios.to_csv(codigo_dir / "detailed_ratios_analysis.csv", index=False)
        df_anisotropy.to_csv(codigo_dir / "complete_anisotropy_coefficients.csv", index=False)

        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Beta Ratios Analysis:")
        print(f"  - Total pairs: {len(df_ratios):,}")
        print(f"  - Mean ratio: {ratio_stats['mean']:.3f}")
        print(f"  - Std deviation: {ratio_stats['std']:.3f}")
        print(f"  - Range: {ratio_stats['min']:.3f} - {ratio_stats['max']:.3f}")

        print(f"\nAnisotropy Analysis:")
        print(f"  - Total municipalities: {len(df_anisotropy)}")
        print(f"  - Mean anisotropy: {anisotropy_stats['mean']:.3f}")
        print(f"  - Median anisotropy: {anisotropy_stats['50%']:.3f}")
        print(f"  - Range: {anisotropy_stats['min']:.3f} - {anisotropy_stats['max']:.3f}")

        if len(df_anisotropy) == 388:
            print("SUCCESS: Calculated anisotropy for all 388 municipalities!")
        else:
            print(f"WARNING: Only {len(df_anisotropy)}/388 municipalities calculated")

        print("\n=== FILES GENERATED ===")
        print(f"Updated tables:")
        print(f"  - {tables_dir / 'table_stats_ratio_updated.tex'}")
        print(f"  - {tables_dir / 'table_stats_anisotropy_updated.tex'}")
        print(f"Explanatory plots:")
        print(f"  - {output_dir / 'ratio_distribution_analysis.png'}")
        print(f"  - {output_dir / 'anisotropy_analysis_complete.png'}")
        print(f"Data files:")
        print(f"  - {codigo_dir / 'detailed_ratios_analysis.csv'}")
        print(f"  - {codigo_dir / 'complete_anisotropy_coefficients.csv'}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()