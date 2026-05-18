#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 11B: Qualitative Algorithmic Complexity Analysis
================================================================

This figure shows the THEORETICAL computational complexity of different
assignment methods using Big-O notation, WITHOUT specific timing measurements.

This complements Figure 11A (empirical timings for table-based methods) by
showing the theoretical complexity of full network analysis.

Author: Voronoi Framework Team
Date: November 11, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def create_complexity_figure():
    """
    Create two-panel figure showing algorithmic complexity analysis

    Layout: 1 row with 2 panels (scalability curves and trade-off space)
    NOTE: Table (Panel C) moved to separate LaTeX table for better page layout
    """

    # Create figure with 1x2 layout (panels only, no embedded table)
    fig = plt.figure(figsize=(12, 4))  # Reduced height for single row
    gs = GridSpec(1, 2, figure=fig,
                  width_ratios=[1, 1],
                  wspace=0.30)

    # Two graphs side by side
    ax_scalability = fig.add_subplot(gs[0, 0])
    plot_complexity_curves(ax_scalability)

    ax_tradeoff = fig.add_subplot(gs[0, 1])
    plot_accuracy_complexity_tradeoff(ax_tradeoff)

    # NOTE: Table (Panel C) removed from figure - now a separate LaTeX table
    # See tables/algorithmic_complexity_table.tex

    return fig

def plot_complexity_curves(ax):
    """
    Panel A: Computational scalability with Big-O notation

    Shows theoretical growth curves for different algorithmic complexities
    """

    # Range of municipality counts (n)
    n_values = np.logspace(1, 3.5, 100)  # 10 to ~3162 municipalities

    # Define methods with their theoretical complexities
    methods = {
        'Voronoi (table lookup)': {
            'complexity': lambda n: n,  # O(n)
            'color': '#2c3e50',
            'linestyle': '-',
            'linewidth': 2.0,
            'label': r'$O(n)$ - Voronoi'
        },
        'k-nearest (table sort)': {
            'complexity': lambda n: n * np.log(n),  # O(n log n)
            'color': '#3498db',
            'linestyle': '--',
            'linewidth': 2.0,
            'label': r'$O(n \log n)$ - k-nearest'
        },
        'Optimal approx (dual lookup)': {
            'complexity': lambda n: 2 * n,  # O(2n) = O(n)
            'color': '#e74c3c',
            'linestyle': '-.',
            'linewidth': 2.5,
            'label': r'$O(n)$ - Optimal approx'
        },
        'Network analysis (graph traversal)': {
            'complexity': lambda n: n**2 * np.log(n),  # O(n² log n)
            'color': '#95a5a6',
            'linestyle': ':',
            'linewidth': 2.5,
            'label': r'$O(n^2 \log n)$ - Network analysis'
        }
    }

    # Normalize all curves to start at 1.0 for n=10
    base_n = 10

    for method_name, props in methods.items():
        complexity_values = props['complexity'](n_values)
        base_value = props['complexity'](base_n)
        normalized_values = complexity_values / base_value

        ax.loglog(n_values, normalized_values,
                 color=props['color'],
                 linestyle=props['linestyle'],
                 linewidth=props['linewidth'],
                 label=props['label'],
                 alpha=0.9)

    # Mark Extremadura scale (383 municipalities)
    ax.axvline(383, color='green', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.text(383, 0.8, 'Extremadura\n(n=383)', ha='center', va='top',
           fontsize=7, color='green', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax.set_xlabel('Number of municipalities (n)', fontweight='bold')
    ax.set_ylabel('Relative computational cost\n(normalized to n=10)', fontweight='bold')
    ax.set_title('(A) Algorithmic Complexity Scalability', fontweight='bold', pad=10)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=7)  # Smaller legend
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_xlim(10, 3200)
    ax.set_ylim(0.5, 5000)

def plot_accuracy_complexity_tradeoff(ax):
    """
    Panel B: Accuracy vs Complexity trade-off space

    Qualitative positioning of methods in accuracy-complexity space
    """

    # Define methods with qualitative positioning
    methods = {
        'Voronoi': {
            'accuracy': 84.6,  # 100 - 15.4 (misassignment risk)
            'complexity_class': 1,  # O(n)
            'color': '#2c3e50',
            'marker': 'o',
            'size': 120,
            'label': 'Voronoi\n(baseline)'
        },
        'k-nearest-3': {
            'accuracy': 92.1,  # 100 - 7.9
            'complexity_class': 2,  # O(n log n)
            'color': '#3498db',
            'marker': 's',
            'size': 120,
            'label': 'k-nearest-3'
        },
        'k-nearest-5': {
            'accuracy': 92.2,  # 100 - 7.8
            'complexity_class': 2.2,  # O(n log n) slightly higher
            'color': '#9b59b6',
            'marker': 's',
            'size': 120,
            'label': 'k-nearest-5'
        },
        'Optimal approx': {
            'accuracy': 97.6,  # 100 - 2.4
            'complexity_class': 1.5,  # O(n) with dual lookup
            'color': '#e74c3c',
            'marker': '*',
            'size': 300,
            'label': 'Optimal approx\n(proposed)'
        },
        'Network analysis': {
            'accuracy': 100.0,  # True optimal
            'complexity_class': 10,  # O(n² log n) - much higher
            'color': '#95a5a6',
            'marker': 'D',
            'size': 150,
            'label': 'Network analysis\n(impractical)'
        }
    }

    # Plot methods
    for method_name, props in methods.items():
        ax.scatter(props['complexity_class'], props['accuracy'],
                  color=props['color'],
                  marker=props['marker'],
                  s=props['size'],
                  alpha=0.8,
                  edgecolors='black',
                  linewidths=1.5,
                  zorder=3,
                  label=props['label'])

    # Add Pareto frontier zone (CONVEX: Voronoi → optimal_approx → network_analysis)
    # Connect only non-dominated solutions: (1.0, 84.6), (1.5, 97.6), (10.0, 100.0)
    pareto_x = [1.0, 1.5, 10.0]
    pareto_y = [84.6, 97.6, 100.0]
    ax.fill_between(pareto_x, 80, pareto_y, alpha=0.15, color='orange', zorder=1)
    ax.plot(pareto_x, pareto_y, color='orange', linestyle='--', linewidth=1.5,
            alpha=0.6, zorder=2, label='Pareto frontier')
    ax.text(3.5, 93, 'Pareto Frontier', fontsize=8, color='orange',
           fontweight='bold', rotation=8, alpha=0.7)

    # Mark "practical" zone
    ax.axvline(3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.text(1.8, 82, 'Practical\nmethods', ha='center', fontsize=8,
           color='green', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))
    ax.text(6.5, 82, 'Prohibitively\nexpensive', ha='center', fontsize=8,
           color='darkred', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.6))

    ax.set_xlabel('Computational Complexity Class\n' +
                  r'(1=$O(n)$, 2=$O(n \log n)$, 10=$O(n^2 \log n)$)',
                  fontweight='bold', fontsize=8)
    ax.set_ylabel('Assignment Accuracy (%)', fontweight='bold')
    ax.set_title('(B) Accuracy vs Computational Cost Trade-off', fontweight='bold', pad=10)
    ax.set_xlim(0.5, 12)
    ax.set_ylim(80, 102)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=6)  # Smaller legend
    ax.grid(True, alpha=0.3, linestyle=':')

def plot_method_comparison_table(ax):
    """
    Panel C: Comparative summary table

    Tabular comparison of methods across key dimensions
    """

    ax.axis('off')

    # Table data
    methods = ['Voronoi', 'k-nearest-3', 'k-nearest-5', 'Optimal\napprox', 'Network\nanalysis']

    # Complexity classes
    complexity = [
        r'$O(n)$',
        r'$O(n \log k)$',
        r'$O(n \log k)$',
        r'$O(n)$',
        r'$O(n^2 \log n)$'
    ]

    # Accuracy (from empirical data)
    accuracy = ['84.6%', '92.1%', '92.2%', '97.6%', '100.0%']

    # Practicality (using ASCII characters)
    practicality = ['+++', '+++', '++', '+++', 'X']

    # Data requirements
    data_req = ['Euclidean', 'Euclidean', 'Euclidean', 'Both tables', 'Road graph']

    # Combine into table
    table_data = [
        ['Method', 'Complexity', 'Accuracy', 'Practical', 'Data Required'],
        ['', '', '', '', ''],  # Separator
    ]

    for i, method in enumerate(methods):
        table_data.append([method, complexity[i], accuracy[i], practicality[i], data_req[i]])

    # Create table (full width, better spacing)
    table = ax.table(cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    bbox=[0.05, 0.15, 0.90, 0.70])  # Better positioning for readability

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly larger for readability

    # Header row styling
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.12)

    # Separator row
    for i in range(5):
        cell = table[(1, i)]
        cell.set_facecolor('#ecf0f1')
        cell.set_height(0.02)

    # Data rows styling
    colors = ['#bdc3c7', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']
    for i in range(5):  # 5 methods
        for j in range(5):  # 5 columns
            cell = table[(i+2, j)]
            if j == 0:  # Method name column
                cell.set_facecolor(colors[i])
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('white')
            cell.set_height(0.14)

            # Highlight optimal approx (proposed method)
            if i == 3:
                cell.set_edgecolor('red')
                cell.set_linewidth(2.0)

    # Add title
    ax.text(0.5, 0.98, '(C) Method Comparison Summary',
           ha='center', va='top', fontsize=10, fontweight='bold',
           transform=ax.transAxes)

    # Add note
    note_text = ('Note: Complexity classes indicate asymptotic growth rate.\n'
                'Network analysis is theoretically optimal but computationally\n'
                'prohibitive for real-time planning (requires road network graph).')
    ax.text(0.5, -0.08, note_text,
           ha='center', va='top', fontsize=6, style='italic',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

def main():
    """Generate and save the figure"""

    print("=" * 80)
    print("GENERATING FIGURE 11B: QUALITATIVE ALGORITHMIC COMPLEXITY ANALYSIS")
    print("=" * 80)

    # Create figure
    fig = create_complexity_figure()

    # Save as PDF (for manuscript)
    pdf_path = "../figuras_clean/algorithmic_complexity_analysis.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nSaved PDF: {pdf_path}")

    # Save as PNG (for preview)
    png_path = "../figuras_clean/algorithmic_complexity_analysis.png"
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved PNG: {png_path}")

    print("\n" + "=" * 80)
    print("FIGURE 11B GENERATION COMPLETE")
    print("=" * 80)
    print("\nThis figure shows:")
    print("  Panel A: Big-O complexity curves (log-log scale)")
    print("  Panel B: Accuracy vs complexity trade-off space")
    print("  Panel C: Method comparison table")
    print("\nKey insight:")
    print("  Network analysis is O(n² log n) - computationally prohibitive")
    print("  Optimal approximation achieves 97.6% accuracy with O(n) complexity")
    print("\nComplements Figure 11A (empirical timings for practical methods)")

if __name__ == "__main__":
    main()
