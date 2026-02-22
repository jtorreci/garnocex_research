#!/usr/bin/env python3
"""
Script to calculate anisotropy coefficients for plants based on assignment analysis.
This corrects the previous approach by focusing on plant-based anisotropy rather than municipality-based.

UPDATED: Now includes filtering of physically invalid beta < 1 cases for methodological rigor.

Author: Claude Code Analysis
Date: September 16, 2025
Updated: December 2024 - Added beta filtering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from data_filtering import apply_standard_filter, get_filtering_summary, print_methodology_text

# Set up paths
codigo_dir = Path(__file__).parent
output_dir = codigo_dir.parent / "imagenes"
tables_dir = codigo_dir.parent / "tables"

# Create output directories
output_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)

def load_assignment_data():
    """Load municipality assignment data for both Euclidean and network distances."""
    print("Loading assignment data...")

    # Load Euclidean assignments
    euclidean_file = codigo_dir / "asignacion_municipios_euclidiana.csv"
    network_file = codigo_dir / "asignacion_municipios_real.csv"

    if not euclidean_file.exists() or not network_file.exists():
        raise FileNotFoundError("Assignment files not found")

    df_euclidean = pd.read_csv(euclidean_file)
    df_network = pd.read_csv(network_file)

    print(f"Loaded Euclidean assignments: {len(df_euclidean)} municipalities")
    print(f"Loaded network assignments: {len(df_network)} municipalities")

    return df_euclidean, df_network

def load_full_distance_matrices():
    """Load the complete distance matrices for detailed analysis."""
    print("Loading complete distance matrices...")

    # Load network distances (municipality-plant pairs)
    matriz_file = codigo_dir / "Matriz_Municipios.csv"
    df_network_raw = pd.read_csv(matriz_file, sep=';', decimal=',', encoding='utf-8-sig')

    # Filter only municipality-plant pairs (exclude municipality-municipality)
    # Assuming plants have specific naming pattern or are identified by fewer entries
    df_network_processed = df_network_raw[['origin_id', 'destination_id', 'total_cost']].copy()
    df_network_processed = df_network_processed.dropna(subset=['total_cost'])

    print(f"Network distance matrix: {len(df_network_processed)} pairs")

    # Load Euclidean distances between municipalities
    euclidean_file = codigo_dir / "distancias_euclideas.csv"
    df_euclidean_raw = pd.read_csv(euclidean_file, sep=';', decimal=',', encoding='utf-8-sig')

    print(f"Euclidean distance matrix: {len(df_euclidean_raw)} pairs")

    return df_network_processed, df_euclidean_raw

def identify_plants_from_assignments(df_euclidean, df_network):
    """Identify plant IDs and their assignments."""
    print("Identifying plant assignments...")

    # Get unique plants from assignments
    plants_euclidean = set(df_euclidean['planta_asignada'].unique())
    plants_network = set(df_network['planta_asignada'].unique())
    plants_common = plants_euclidean.intersection(plants_network)

    print(f"Plants in Euclidean assignments: {len(plants_euclidean)}")
    print(f"Plants in network assignments: {len(plants_network)}")
    print(f"Common plants: {len(plants_common)}")

    return plants_common, plants_euclidean, plants_network

def load_filtered_ratios_data():
    """Load the main filtered ratios dataset for consistency."""
    print("Loading main filtered ratios dataset...")

    # Load the same filtered data used by other analyses
    ratios_file = "detailed_ratios_analysis.csv"
    if not os.path.exists(ratios_file):
        raise FileNotFoundError(f"Main ratios file {ratios_file} not found. Run generate_basic_plots.py first.")

    df_ratios_raw = pd.read_csv(ratios_file)
    print(f"Loaded {len(df_ratios_raw)} raw ratio records")

    # Apply same filtering as other analyses
    print("\n=== APPLYING BETA FILTERING (consistency with main analysis) ===")
    df_ratios_filtered = apply_standard_filter(df_ratios_raw)

    # Store filtering summary for methodology reporting
    filtering_summary = get_filtering_summary(df_ratios_raw, df_ratios_filtered)

    print(f"Using {len(df_ratios_filtered)} filtered records for plant anisotropy analysis")
    return df_ratios_filtered, filtering_summary

def calculate_plant_ratios(df_euclidean, df_network):
    """Calculate beta ratios for each plant using FILTERED main dataset."""
    print("Calculating plant-based beta ratios using filtered main dataset...")

    # CRITICAL FIX: Use the same filtered dataset as other analyses
    try:
        df_plant_ratios_filtered, filtering_summary = load_filtered_ratios_data()

        # For compatibility with existing code, also analyze assignment differences
        df_merged = pd.merge(df_euclidean, df_network, on='municipio', suffixes=('_euc', '_net'))

        print(f"Plant anisotropy analysis using {len(df_plant_ratios_filtered)} filtered β ratios")
        print(f"β range: {df_plant_ratios_filtered['Ratio'].min():.6f} to {df_plant_ratios_filtered['Ratio'].max():.6f}")

        # Ensure column naming consistency
        if 'Plant' not in df_plant_ratios_filtered.columns:
            # Map from the detailed_ratios_analysis.csv format if needed
            print("Mapping column names for compatibility...")
            # This might need adjustment based on the actual column structure
            # For now, we'll work with the data as-is

        return df_plant_ratios_filtered, df_merged, filtering_summary

    except FileNotFoundError:
        print("WARNING: Main filtered dataset not found. Falling back to assignment-based calculation.")
        print("This may produce β > 4 values. Run generate_basic_plots.py first for proper filtering.")

        # Fallback to original method (with warning)
        df_merged = pd.merge(df_euclidean, df_network, on='municipio', suffixes=('_euc', '_net'))
        same_assignment = df_merged[df_merged['planta_asignada_euc'] == df_merged['planta_asignada_net']].copy()
        same_assignment['ratio'] = same_assignment['real_distance'] / same_assignment['euclidean_distance']

        # Create plant ratios dataframe
        plant_ratios = []
        for _, row in same_assignment.iterrows():
            plant_ratios.append({
                'Municipality': row['municipio'],
                'Plant': row['planta_asignada_euc'],
                'NetworkDistance': row['real_distance'],
                'EuclideanDistance': row['euclidean_distance'],
                'Ratio': row['ratio']
            })

        df_plant_ratios_raw = pd.DataFrame(plant_ratios)

        # Apply beta filtering
        print("\n=== APPLYING BETA FILTERING (fallback method) ===")
        df_plant_ratios_filtered = apply_standard_filter(df_plant_ratios_raw)
        filtering_summary = get_filtering_summary(df_plant_ratios_raw, df_plant_ratios_filtered)

        return df_plant_ratios_filtered, df_merged, filtering_summary

def calculate_plant_anisotropy(df_plant_ratios):
    """Calculate anisotropy coefficient for each plant."""
    print("Calculating plant anisotropy coefficients...")

    anisotropy_results = []

    for plant in df_plant_ratios['Plant'].unique():
        plant_data = df_plant_ratios[df_plant_ratios['Plant'] == plant]

        if len(plant_data) >= 2:  # Need at least 2 municipalities for anisotropy
            ratios = plant_data['Ratio'].values
            max_ratio = np.max(ratios)
            min_ratio = np.min(ratios)

            anisotropy_coeff = max_ratio / min_ratio

            anisotropy_results.append({
                'Plant': plant,
                'MaxRatio': max_ratio,
                'MinRatio': min_ratio,
                'AnisotropyCoefficient': anisotropy_coeff,
                'NumMunicipalities': len(plant_data),
                'MeanRatio': np.mean(ratios),
                'StdRatio': np.std(ratios)
            })

    return pd.DataFrame(anisotropy_results)

def analyze_assignment_differences(df_euclidean, df_network):
    """Analyze differences between Euclidean and network assignments."""
    print("Analyzing assignment differences...")

    df_merged = pd.merge(df_euclidean, df_network, on='municipio', suffixes=('_euc', '_net'))

    # Count misallocations
    misallocated = df_merged[df_merged['planta_asignada_euc'] != df_merged['planta_asignada_net']].copy()

    print(f"Total municipalities: {len(df_merged)}")
    print(f"Misallocated municipalities: {len(misallocated)}")
    print(f"Misallocation rate: {len(misallocated)/len(df_merged)*100:.1f}%")

    # Calculate distance improvements
    misallocated.loc[:, 'distance_improvement'] = misallocated['euclidean_distance'] - misallocated['real_distance']
    misallocated.loc[:, 'ratio_improvement'] = misallocated['euclidean_distance'] / misallocated['real_distance']

    return df_merged, misallocated

def create_plant_analysis_plots(df_plant_ratios, df_plant_anisotropy, df_misallocated):
    """Create comprehensive plots for plant-based analysis."""
    print("Creating plant analysis plots...")

    # Set style for grayscale with strategic color highlights and improved readability
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Figure 1: Plant anisotropy analysis
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram of plant anisotropy coefficients
    ax1.hist(df_plant_anisotropy['AnisotropyCoefficient'], bins=20, density=True, alpha=0.7,
             color='lightgray', edgecolor='black')

    # Add mean line in red
    mean_aniso = df_plant_anisotropy['AnisotropyCoefficient'].mean()
    ax1.axvline(mean_aniso, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_aniso:.2f}')

    ax1.set_xlabel('Plant Anisotropy Coefficient (max β / min β)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Plant Anisotropy Coefficients\\n(n = {len(df_plant_anisotropy)} plants)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"""Median: {df_plant_anisotropy['AnisotropyCoefficient'].median():.3f}
Std: {df_plant_anisotropy['AnisotropyCoefficient'].std():.3f}
Max: {df_plant_anisotropy['AnisotropyCoefficient'].max():.3f}"""
    ax1.text(0.7, 0.85, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Box plot of anisotropy by number of municipalities served
    df_plant_anisotropy['MuniGroup'] = pd.cut(df_plant_anisotropy['NumMunicipalities'],
                                              bins=[0, 5, 10, 20, 100],
                                              labels=['1-5', '6-10', '11-20', '20+'])

    box_data = [df_plant_anisotropy[df_plant_anisotropy['MuniGroup'] == group]['AnisotropyCoefficient'].values
                for group in ['1-5', '6-10', '11-20', '20+'] if not df_plant_anisotropy[df_plant_anisotropy['MuniGroup'] == group].empty]
    box_labels = [group for group in ['1-5', '6-10', '11-20', '20+']
                  if not df_plant_anisotropy[df_plant_anisotropy['MuniGroup'] == group].empty]

    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        # Color boxes in grayscale
        for patch in bp['boxes']:
            patch.set_facecolor('lightgray')
            patch.set_alpha(0.7)
        # Highlight medians in red
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        ax2.set_xlabel('Number of Municipalities Served')
        ax2.set_ylabel('Anisotropy Coefficient')
        ax2.set_title('Plant Anisotropy vs Service Area Size')
        ax2.grid(True, alpha=0.3)

    # Scatter: Number of municipalities vs anisotropy
    ax3.scatter(df_plant_anisotropy['NumMunicipalities'], df_plant_anisotropy['AnisotropyCoefficient'],
                alpha=0.6, s=60, color='gray', edgecolor='black')

    # Add trend line in green
    if len(df_plant_anisotropy) > 2:
        z = np.polyfit(df_plant_anisotropy['NumMunicipalities'], df_plant_anisotropy['AnisotropyCoefficient'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_plant_anisotropy['NumMunicipalities'].min(),
                             df_plant_anisotropy['NumMunicipalities'].max(), 100)
        ax3.plot(x_trend, p(x_trend), "g-", linewidth=2, alpha=0.8, label='Trend')
        ax3.legend()

    ax3.set_xlabel('Number of Municipalities Served')
    ax3.set_ylabel('Anisotropy Coefficient')
    ax3.set_title('Plant Service Area Size vs Anisotropy')
    ax3.grid(True, alpha=0.3)

    # Max vs Min ratio for plants
    ax4.scatter(df_plant_anisotropy['MinRatio'], df_plant_anisotropy['MaxRatio'],
                alpha=0.6, s=60, color='gray', edgecolor='black')

    # No anisotropy line in red
    ax4.plot([df_plant_anisotropy['MinRatio'].min(), df_plant_anisotropy['MaxRatio'].max()],
             [df_plant_anisotropy['MinRatio'].min(), df_plant_anisotropy['MaxRatio'].max()],
             'r--', linewidth=2, alpha=0.8, label='Perfect isotropy')
    ax4.set_xlabel('Minimum β Ratio to Plant')
    ax4.set_ylabel('Maximum β Ratio to Plant')
    ax4.set_title('Plant Accessibility Anisotropy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'plant_anisotropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Assignment comparison
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 8))

    # Misallocation rate by plant
    plant_misalloc = df_misallocated.groupby('planta_asignada_euc').size().reset_index(name='misallocations')
    total_assignments = df_plant_ratios.groupby('Plant').size().reset_index(name='total')

    if not plant_misalloc.empty and not total_assignments.empty:
        plant_rates = pd.merge(plant_misalloc, total_assignments,
                              left_on='planta_asignada_euc', right_on='Plant', how='outer').fillna(0)
        plant_rates['misalloc_rate'] = plant_rates['misallocations'] / plant_rates['total'] * 100

        # Create gradient bar colors from gray to red based on rate
        colors = ['lightgray' if rate < 20 else 'red' for rate in plant_rates['misalloc_rate']]

        ax5.bar(range(len(plant_rates)), plant_rates['misalloc_rate'],
                alpha=0.7, color=colors, edgecolor='black')
        ax5.axhline(y=plant_rates['misalloc_rate'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {plant_rates["misalloc_rate"].mean():.1f}%')
        ax5.set_xlabel('Plant ID')
        ax5.set_ylabel('Misallocation Rate (%)')
        ax5.set_title('Misallocation Rate by Plant')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        # Show overall misallocation rate
        ax5.text(0.5, 0.5, f'Overall Misallocation Rate:\\n{len(df_misallocated)/383*100:.1f}%',
                transform=ax5.transAxes, ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.set_title('Municipality Assignment Analysis')

    # Distance ratio distribution - simple histogram in grayscale
    if not df_plant_ratios.empty:
        ax6.hist(df_plant_ratios['Ratio'], bins=30, alpha=0.7, color='lightgray',
                edgecolor='black', density=True, label='All Ratios')

        # Add mean and median lines
        mean_ratio = df_plant_ratios['Ratio'].mean()
        median_ratio = df_plant_ratios['Ratio'].median()
        ax6.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
        ax6.axvline(median_ratio, color='green', linestyle=':', linewidth=2, label=f'Median: {median_ratio:.2f}')

    ax6.set_xlabel('β Ratio (Network/Euclidean)')
    ax6.set_ylabel('Density')
    ax6.set_title('Distance Ratio Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'plant_assignment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Created plant analysis plots:")
    print(f"  - {output_dir / 'plant_anisotropy_analysis.png'}")
    print(f"  - {output_dir / 'plant_assignment_analysis.png'}")

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

def main():
    """Main execution function."""
    print("=== PLANT-BASED ANISOTROPY ANALYSIS ===")
    print("Analyzing transport network anisotropy from plant perspective...")

    try:
        # Load assignment data
        df_euclidean, df_network = load_assignment_data()

        # Analyze assignment differences
        df_merged, df_misallocated = analyze_assignment_differences(df_euclidean, df_network)

        # Calculate plant-based ratios using assignment data (includes filtering)
        df_plant_ratios, df_assignment_ratios, filtering_summary = calculate_plant_ratios(df_euclidean, df_network)

        # Print methodology text for paper
        print_methodology_text(filtering_summary)

        # Calculate plant anisotropy coefficients
        df_plant_anisotropy = calculate_plant_anisotropy(df_plant_ratios)

        print(f"\\nCalculated ratios for {len(df_plant_ratios)} municipality-plant pairs")
        print(f"Calculated anisotropy for {len(df_plant_anisotropy)} plants")

        # Generate statistics
        if not df_plant_anisotropy.empty:
            anisotropy_stats = df_plant_anisotropy['AnisotropyCoefficient'].describe()

            # Save updated LaTeX table
            save_latex_table(
                anisotropy_stats,
                tables_dir / "table_stats_plant_anisotropy.tex",
                "Descriptive statistics of plant-based transport network anisotropy coefficients.",
                "tab:stats_plant_anisotropy"
            )

        # Create analysis plots
        if not df_plant_anisotropy.empty:
            create_plant_analysis_plots(df_plant_ratios, df_plant_anisotropy, df_misallocated)

        # Save detailed results
        df_plant_ratios.to_csv(codigo_dir / "plant_ratios_analysis.csv", index=False)
        df_plant_anisotropy.to_csv(codigo_dir / "plant_anisotropy_coefficients.csv", index=False)
        df_misallocated.to_csv(codigo_dir / "misallocated_municipalities.csv", index=False)

        # Print summary
        print("\\n=== PLANT ANALYSIS SUMMARY ===")
        print(f"Assignment Analysis:")
        print(f"  - Total municipalities: {len(df_merged)}")
        print(f"  - Misallocated: {len(df_misallocated)} ({len(df_misallocated)/len(df_merged)*100:.1f}%)")

        if not df_plant_anisotropy.empty:
            print(f"\\nPlant Anisotropy Analysis:")
            print(f"  - Plants analyzed: {len(df_plant_anisotropy)}")
            print(f"  - Mean anisotropy: {anisotropy_stats['mean']:.3f}")
            print(f"  - Median anisotropy: {anisotropy_stats['50%']:.3f}")
            print(f"  - Range: {anisotropy_stats['min']:.3f} - {anisotropy_stats['max']:.3f}")

        print("\\n=== FILES GENERATED ===")
        print(f"LaTeX table: {tables_dir / 'table_stats_plant_anisotropy.tex'}")
        print(f"Analysis plots: {output_dir / 'plant_anisotropy_analysis.png'}")
        print(f"                {output_dir / 'plant_assignment_analysis.png'}")
        print(f"Data files: {codigo_dir / 'plant_ratios_analysis.csv'}")
        print(f"            {codigo_dir / 'plant_anisotropy_coefficients.csv'}")
        print(f"            {codigo_dir / 'misallocated_municipalities.csv'}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()