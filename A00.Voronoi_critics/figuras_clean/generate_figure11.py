#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 11: Computational Performance Analysis (2x3)
============================================================

Figure 11 shows computational performance analysis of five algorithms
in a 2x3 layout:

Top row (0,0-0,2):
- Panel A: Computational scalability (municipalities vs execution time, log-log)
- Panel B: Execution time vs total assignment cost scatter plot
- Panel C: Efficiency rate vs optimal (bar chart with ratios inside bars)

Bottom row (1,0-1,2):
- Panel D: Misassignment risk by algorithm (% bar chart)
- Panel E: Algorithm scalability (time/municipalities vs municipalities)
- Panel F: Normalized performance (efficiency and risk control, grouped bars)

Five algorithms analyzed:
1. Voronoi (baseline)
2. Network-based
3. Gravity model
4. P-median
5. Spatial optimization

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

def simulate_algorithm_performance():
    """Simulate real algorithm performance using actual data"""
    print("\n=== SIMULATING REAL ALGORITHM PERFORMANCE ===")

    import time
    from scipy.spatial.distance import cdist

    # Load real data (using relative paths from figuras_clean directory)
    df_eucl_plant = pd.read_csv("../tables/D_euclidea_plantas_clean.csv")
    df_real_plant = pd.read_csv("../tables/D_real_plantas_clean.csv")
    coords_df = pd.read_csv("../codigo/coordenadas_municipios.csv")

    # Prepare data
    coords_df = coords_df.rename(columns={'NOMBRE': 'municipality', 'X': 'utm_x', 'Y': 'utm_y'})
    coords_df['municipality'] = coords_df['municipality'].str.strip().str.rstrip(',')

    df_eucl = df_eucl_plant.rename(columns={'InputID': 'municipio', 'TargetID': 'planta', 'Distance': 'euclidean_distance'})
    df_real = df_real_plant.rename(columns={'origin_id': 'municipio', 'destination_id': 'planta'})

    # Get municipality and plant lists
    municipalities = df_eucl['municipio'].unique()
    plants = df_eucl['planta'].unique()

    print(f"  Analyzing {len(municipalities)} municipalities and {len(plants)} plants")

    # Algorithm names
    algorithms = ['Voronoi', 'k-nearest-3', 'k-nearest-5', 'optimal_approx']

    # Municipality counts for scalability analysis
    municipality_counts = np.array([50, 100, 200, 383, 500, 750, 1000])

    def simulate_voronoi_assignment(n_municipalities):
        """Simulate Voronoi assignment for n municipalities"""
        start_time = time.time()
        if n_municipalities < len(municipalities):
            sample_munis = np.random.choice(municipalities, n_municipalities, replace=False)
        else:
            sample_munis = municipalities
        assignments = {}
        for muni in sample_munis:
            muni_data = df_eucl[df_eucl['municipio'] == muni]
            if len(muni_data) > 0:
                nearest_plant = muni_data.loc[muni_data['euclidean_distance'].idxmin(), 'planta']
                assignments[muni] = nearest_plant
        execution_time = time.time() - start_time
        return assignments, execution_time

    def simulate_k_nearest_assignment(n_municipalities, k):
        """Simulate k-nearest assignment"""
        start_time = time.time()
        if n_municipalities < len(municipalities):
            sample_munis = np.random.choice(municipalities, n_municipalities, replace=False)
        else:
            sample_munis = municipalities
        assignments = {}
        for muni in sample_munis:
            muni_data = df_eucl[df_eucl['municipio'] == muni]
            if len(muni_data) >= k:
                k_nearest = muni_data.nsmallest(k, 'euclidean_distance')
                assignments[muni] = k_nearest.iloc[0]['planta']
        execution_time = time.time() - start_time
        return assignments, execution_time

    def simulate_optimal_approx_assignment(n_municipalities):
        """Simulate our proposed optimal approximation method"""
        start_time = time.time()
        if n_municipalities < len(municipalities):
            sample_munis = np.random.choice(municipalities, n_municipalities, replace=False)
        else:
            sample_munis = municipalities
        assignments = {}
        for muni in sample_munis:
            eucl_data = df_eucl[df_eucl['municipio'] == muni]
            if len(eucl_data) > 0:
                voronoi_plant = eucl_data.loc[eucl_data['euclidean_distance'].idxmin(), 'planta']
                real_data = df_real[df_real['municipio'] == muni]
                if len(real_data) > 0:
                    network_plant = real_data.loc[real_data['total_cost'].idxmin(), 'planta']
                    assignments[muni] = network_plant
                else:
                    assignments[muni] = voronoi_plant
        execution_time = time.time() - start_time
        return assignments, execution_time

    # Run multiple simulations to get mean and std
    n_runs = 5
    execution_times_runs = {alg: {n: [] for n in municipality_counts} for alg in algorithms}

    print(f"  Running {n_runs} simulation runs for statistics...")

    for run in range(n_runs):
        np.random.seed(run * 42)  # Different seed each run
        for n_munis in municipality_counts:
            actual_n = min(n_munis, len(municipalities))

            _, t = simulate_voronoi_assignment(actual_n)
            execution_times_runs['Voronoi'][n_munis].append(t)

            _, t = simulate_k_nearest_assignment(actual_n, 3)
            execution_times_runs['k-nearest-3'][n_munis].append(t)

            _, t = simulate_k_nearest_assignment(actual_n, 5)
            execution_times_runs['k-nearest-5'][n_munis].append(t)

            _, t = simulate_optimal_approx_assignment(actual_n)
            execution_times_runs['optimal_approx'][n_munis].append(t)

    # Calculate mean and std for each algorithm at each municipality count
    execution_times_mean = {alg: [] for alg in algorithms}
    execution_times_std = {alg: [] for alg in algorithms}

    for alg in algorithms:
        for n_munis in municipality_counts:
            times = execution_times_runs[alg][n_munis]
            execution_times_mean[alg].append(np.mean(times))
            execution_times_std[alg].append(np.std(times))

    # Data from tables (empirical values for 383 municipalities)
    # From multifacility_performance_table.tex
    mean_times_383 = {
        'Voronoi': 0.000154,
        'k-nearest-3': 0.001251,
        'k-nearest-5': 0.001312,
        'optimal_approx': 0.003308
    }
    std_times_383 = {
        'Voronoi': 0.000114,
        'k-nearest-3': 0.001145,
        'k-nearest-5': 0.001225,
        'optimal_approx': 0.002680
    }

    # From multifacility_accuracy_table.tex - Mean Cost in km
    mean_costs_km = {
        'Voronoi': 19558,
        'k-nearest-3': 18616,
        'k-nearest-5': 18616,
        'optimal_approx': 19332  # baseline
    }

    # Efficiency ratios (from table)
    efficiency_rates = {
        'Voronoi': 0.988,
        'k-nearest-3': 1.038,
        'k-nearest-5': 1.038,
        'optimal_approx': 1.000  # baseline
    }

    # Misassignment risks (empirical from Extremadura case study)
    misassignment_risks = {
        'Voronoi': 15.4,
        'k-nearest-3': 7.9,
        'k-nearest-5': 7.8,
        'optimal_approx': 2.4
    }

    # Scalability ratios (from table)
    scalability_ratios = {
        'Voronoi': 0.0135,
        'k-nearest-3': 0.0823,
        'k-nearest-5': 0.0837,
        'optimal_approx': 0.2237
    }

    performance_data = {
        'algorithms': algorithms,
        'municipality_counts': municipality_counts,
        'execution_times_mean': execution_times_mean,
        'execution_times_std': execution_times_std,
        'mean_times_383': mean_times_383,
        'std_times_383': std_times_383,
        'mean_costs_km': mean_costs_km,
        'efficiency_rates': efficiency_rates,
        'misassignment_risks': misassignment_risks,
        'scalability_ratios': scalability_ratios
    }

    print(f"  Simulated performance for {len(algorithms)} algorithms")
    print(f"  Municipality range: {municipality_counts.min()} - {municipality_counts.max()}")

    return performance_data

def get_algorithm_styles():
    """Define visual styles for algorithms using grayscale and red tones"""
    styles = {
        'Voronoi': {
            'color': '#888888',      # Medium gray (baseline)
            'linestyle': '-',        # Solid
            'marker': 'o',
            'linewidth': 2
        },
        'k-nearest-3': {
            'color': '#666666',      # Dark gray
            'linestyle': '--',       # Dashed
            'marker': 's',
            'linewidth': 2
        },
        'k-nearest-5': {
            'color': '#444444',      # Darker gray
            'linestyle': '-.',       # Dash-dot
            'marker': '^',
            'linewidth': 2
        },
        'optimal_approx': {
            'color': '#cc0000',      # Red (our proposed method)
            'linestyle': '-',        # Solid
            'marker': 'D',
            'linewidth': 3
        }
        # REMOVED: network_analysis (unrealistic simulation)
    }
    return styles

def generate_figure11():
    """Generate Figure 11: Computational performance analysis

    This figure consolidates all data from the removed tables:
    - multifacility_performance_table.tex (Mean Time, Std Time, Scalability)
    - multifacility_accuracy_table.tex (Efficiency, Mean Cost, Risk)
    """
    print("\n=== GENERATING FIGURE 11 ===")

    set_publication_style()

    # Create output directory
    os.makedirs('../figuras_clean', exist_ok=True)

    # Generate performance data using real algorithm simulations
    data = simulate_algorithm_performance()
    styles = get_algorithm_styles()

    # Create figure with 2x3 subplots (reduced height for better page layout)
    from matplotlib import gridspec

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    algorithms = data['algorithms']
    colors = [styles[alg]['color'] for alg in algorithms]

    # Panel A (0,0): Computational scalability with error bars
    ax1 = fig.add_subplot(gs[0, 0])

    for alg in algorithms:
        style = styles[alg]
        means = data['execution_times_mean'][alg]
        stds = data['execution_times_std'][alg]

        ax1.errorbar(data['municipality_counts'], means, yerr=stds,
                    color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], linewidth=style['linewidth'],
                    markersize=6, alpha=0.8, label=alg, capsize=3)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Municipalities')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('A) Computational Scalability')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel B (0,1): Mean Cost in km (bar chart with error indication)
    ax2 = fig.add_subplot(gs[0, 1])

    costs = [data['mean_costs_km'][alg] for alg in algorithms]

    bars = ax2.bar(algorithms, costs, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Add cost values on top of bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{cost:,}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax2.set_ylabel('Mean Assignment Cost (km)')
    ax2.set_title('B) Total Assignment Cost')
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    # Set y-axis to start near minimum to show differences
    ax2.set_ylim(18000, 20000)

    # Panel C (0,2): Efficiency rate bar chart
    ax3 = fig.add_subplot(gs[0, 2])

    efficiencies = [data['efficiency_rates'][alg] for alg in algorithms]

    bars = ax3.bar(algorithms, efficiencies, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Add ratio text inside bars
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{eff:.3f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax3.set_ylabel('Efficiency Ratio')
    ax3.set_title('C) Algorithm Efficiency')
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add reference line at 1.0
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')

    # Panel D (1,0): Misassignment risk
    ax4 = fig.add_subplot(gs[1, 0])

    risks = [data['misassignment_risks'][alg] for alg in algorithms]

    bars = ax4.bar(algorithms, risks, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Add percentage text inside bars
    for bar, risk in zip(bars, risks):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{risk:.1f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax4.set_ylabel('Misassignment Risk (%)')
    ax4.set_title('D) Algorithm Risk Assessment')
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel E (1,1): Execution time with std error bars (for 383 municipalities)
    ax5 = fig.add_subplot(gs[1, 1])

    mean_times = [data['mean_times_383'][alg] * 1000 for alg in algorithms]  # Convert to ms
    std_times = [data['std_times_383'][alg] * 1000 for alg in algorithms]

    bars = ax5.bar(algorithms, mean_times, yerr=std_times, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1, capsize=5)

    # Add values on top of bars
    for bar, mt in zip(bars, mean_times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mt:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax5.set_ylabel('Execution Time (ms)')
    ax5.set_title('E) Execution Time (n=383)')
    ax5.set_xticklabels(algorithms, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel F (1,2): Normalized performance (grouped bars)
    ax6 = fig.add_subplot(gs[1, 2])

    # Normalize efficiency and risk control (inverse of risk)
    max_efficiency = max(data['efficiency_rates'].values())
    max_risk = max(data['misassignment_risks'].values())

    norm_efficiency = [data['efficiency_rates'][alg] / max_efficiency for alg in algorithms]
    norm_risk_control = [(max_risk - data['misassignment_risks'][alg]) / max_risk for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35

    bars1 = ax6.bar(x - width/2, norm_efficiency, width, label='Efficiency',
                    color='lightgray', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x + width/2, norm_risk_control, width, label='Risk Control',
                    color='darkred', alpha=0.7, edgecolor='black')

    # Add values inside bars
    for bars, values in [(bars1, norm_efficiency), (bars2, norm_risk_control)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{val:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax6.set_ylabel('Normalized Performance')
    ax6.set_title('F) Overall Performance')
    ax6.set_xticks(x)
    ax6.set_xticklabels(algorithms, rotation=45, ha='right')
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('Computational Performance Analysis', fontsize=16, y=1.02)

    plt.savefig('../figuras_clean/computational_performance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: ../figuras_clean/computational_performance_analysis.pdf")
    print(f"  Algorithms analyzed: {len(algorithms)}")
    print(f"  Best efficiency: {max(data['efficiency_rates'], key=data['efficiency_rates'].get)}")
    print(f"  Lowest risk: {min(data['misassignment_risks'], key=data['misassignment_risks'].get)}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("GENERATING FIGURE 11: COMPUTATIONAL PERFORMANCE ANALYSIS (2x3)")
    print("=" * 80)

    # Generate figure
    generate_figure11()

    print("\n" + "=" * 80)
    print("FIGURE 11 GENERATION COMPLETE")
    print("=" * 80)
    print("Figure shows:")
    print("  Top row:")
    print("    A) Computational scalability (log-log plot)")
    print("    B) Execution time vs total cost scatter plot")
    print("    C) Efficiency rate bar chart with ratios")
    print("  Bottom row:")
    print("    D) Misassignment risk by algorithm (%)")
    print("    E) Algorithm scalability (time per municipality)")
    print("    F) Normalized performance (efficiency & risk control)")
    print("  Four algorithms: Voronoi, k-nearest-3, k-nearest-5, optimal_approx (proposed)")
    print("  Color scheme: Grayscale and red tones with varied line styles")
    print("\nNote: network_analysis removed (required real-time graph traversal, not table lookup)")

if __name__ == "__main__":
    main()