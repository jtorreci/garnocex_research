#!/usr/bin/env python3
"""
Safety Bands Analysis - Calibration Curves for Practitioners
===========================================================

This script generates calibration curves for safety bands around Voronoi boundaries.
Creates the critical |t*| = f(κ, s, q*) relationships that allow practitioners
to determine when Euclidean approximations are reliable vs. when network analysis is needed.

These curves will be HIGHLY CITABLE as they provide actionable guidance.

Author: Claude Code Enhancement
Date: September 16, 2025
Purpose: Generate practitioner-ready calibration tool for Q1 submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import erfinv

def set_publication_style():
    """Set matplotlib style for publication-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2
    })

def inverse_phi(q):
    """Inverse of standard normal CDF"""
    return np.sqrt(2) * erfinv(2*q - 1)

def compute_critical_distance(kappa, s, q_star):
    """
    Compute critical distance |t*| for given parameters

    From paper: q(P) ≈ Φ(-κ|t|/(√2 s))
    Setting q(P) = q* and solving for |t|:
    |t*| = -Φ^(-1)(q*) * √2 * s / κ
    """
    phi_inv_q = inverse_phi(q_star)
    t_critical = -phi_inv_q * np.sqrt(2) * s / kappa
    return np.abs(t_critical)

def generate_calibration_curves():
    """Generate calibration curves for different parameter ranges"""

    # Parameter ranges based on paper and realistic scenarios
    kappa_range = np.linspace(0.1, 2.0, 100)  # Geometric parameter
    s_values = [0.05, 0.08, 0.093, 0.12, 0.15, 0.20]  # Different geographic contexts
    q_star_values = [0.10, 0.20, 0.30]  # Risk thresholds

    # Geographic context mapping
    s_contexts = {
        0.05: 'Urban Dense',
        0.08: 'Flat Plains',
        0.093: 'Extremadura (Moderate)',
        0.12: 'Mountainous',
        0.15: 'Complex Terrain',
        0.20: 'Island/Archipelago'
    }

    # Colors for different contexts
    colors = plt.cm.viridis(np.linspace(0, 1, len(s_values)))

    # Create figure with subplots for each risk threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, q_star in enumerate(q_star_values):
        ax = axes[i]

        for j, (s, color) in enumerate(zip(s_values, colors)):
            t_critical = compute_critical_distance(kappa_range, s, q_star)

            # Plot curve
            label = f's = {s} ({s_contexts[s]})'
            ax.plot(kappa_range, t_critical, color=color, label=label, linewidth=2.5)

        ax.set_xlabel('Geometric Parameter κ')
        ax.set_ylabel('Critical Distance |t*|')
        ax.set_title(f'Safety Bands for q* = {q_star*100:.0f}% Risk')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)

    # Create unified legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)

    plt.suptitle('Safety Band Calibration Curves for Voronoi Risk Assessment', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for unified legend
    plt.savefig('safety_bands_calibration_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: safety_bands_calibration_curves.png")

    return kappa_range, s_values, q_star_values

def create_contour_plot():
    """Create 2D contour plot showing |t*| as function of κ and s"""

    # Create meshgrid
    kappa = np.linspace(0.1, 2.0, 50)
    s = np.linspace(0.03, 0.25, 50)
    K, S = np.meshgrid(kappa, s)

    # Create figure with subplots for different risk levels (larger for better readability)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    q_star_values = [0.10, 0.20, 0.30]

    # Store legend handles for unified legend
    legend_handles = []
    legend_labels = []

    for i, q_star in enumerate(q_star_values):
        ax = axes[i]

        # Compute |t*| for all combinations
        T_critical = compute_critical_distance(K, S, q_star)

        # Create contour plot
        contour = ax.contour(K, S, T_critical, levels=20, colors='black', alpha=0.6, linewidths=0.8)
        contourf = ax.contourf(K, S, T_critical, levels=20, cmap='viridis', alpha=0.8)

        # Add contour labels (larger font)
        ax.clabel(contour, inline=True, fontsize=12, fmt='%.1f')

        # Colorbar (larger font)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Critical Distance |t*|', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Mark Extremadura case (only collect legend info once)
        extremadura_point = ax.plot(0.5, 0.093, 'r*', markersize=18)
        if i == 0:  # Only add to legend for first subplot
            legend_handles.append(extremadura_point[0])
            legend_labels.append('Extremadura Case')

        ax.set_xlabel('Geometric Parameter κ', fontsize=14)
        ax.set_ylabel('Geographic Parameter s', fontsize=14)
        ax.set_title(f'|t*| Contours for q* = {q_star*100:.0f}%', fontsize=16)
        ax.tick_params(labelsize=12)

    # Add unified legend below the plots
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=12)

    plt.suptitle('Safety Band Contour Maps: When to Use Network Analysis', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig('safety_bands_contour_maps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: safety_bands_contour_maps.png")

def create_lookup_table():
    """Create lookup table for practitioners"""

    # Selected practical values
    kappa_values = [0.2, 0.5, 1.0, 1.5, 2.0]
    s_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    q_star_values = [0.10, 0.20, 0.30]

    # Geographic contexts
    s_contexts = {
        0.05: 'Urban Dense',
        0.08: 'Flat Plains',
        0.10: 'Moderate Hills',
        0.12: 'Mountainous',
        0.15: 'Complex Terrain',
        0.20: 'Island/Archipelago'
    }

    # Create comprehensive table
    lookup_data = []

    for q_star in q_star_values:
        for s in s_values:
            context = s_contexts[s]
            for kappa in kappa_values:
                t_critical = compute_critical_distance(kappa, s, q_star)

                lookup_data.append({
                    'Risk_Threshold': f'{q_star*100:.0f}%',
                    'Geographic_Context': context,
                    's_Parameter': s,
                    'Kappa_Parameter': kappa,
                    'Critical_Distance_t': f'{t_critical:.2f}'
                })

    df_lookup = pd.DataFrame(lookup_data)

    # Save to CSV for practitioners
    df_lookup.to_csv('safety_bands_lookup_table.csv', index=False)
    print("Saved: safety_bands_lookup_table.csv")

    # Create formatted LaTeX table for paper (sample)
    # Show just 10% risk threshold for space
    df_sample = df_lookup[df_lookup['Risk_Threshold'] == '10%'].copy()
    df_sample_pivot = df_sample.pivot(index=['Geographic_Context', 's_Parameter'],
                                     columns='Kappa_Parameter',
                                     values='Critical_Distance_t')

    print("\n" + "="*80)
    print("SAMPLE LOOKUP TABLE (10% Risk Threshold)")
    print("="*80)
    print(df_sample_pivot.to_string())
    print("="*80)

    # Generate LaTeX table
    latex_table = df_sample_pivot.to_latex(float_format='%.2f',
                                          caption="Critical distance |t*| lookup table for 10\\% risk threshold",
                                          label="tab:safety_bands_lookup")

    with open('safety_bands_lookup_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("Saved: safety_bands_lookup_table.tex")

    return df_lookup

def create_practical_examples():
    """Create practical examples for different scenarios"""

    examples = [
        {
            'scenario': 'Urban Waste Management',
            'context': 'Dense urban area with grid roads',
            's': 0.05,
            'typical_kappa': 1.0,
            'description': 'High road density, minimal topographic barriers'
        },
        {
            'scenario': 'Rural Emergency Services',
            'context': 'Moderate topography (Extremadura-like)',
            's': 0.093,
            'typical_kappa': 0.5,
            'description': 'Mixed terrain with moderate impedance'
        },
        {
            'scenario': 'Mountain Rescue Operations',
            'context': 'Complex mountainous terrain',
            's': 0.15,
            'typical_kappa': 0.3,
            'description': 'Significant elevation changes, winding roads'
        },
        {
            'scenario': 'Island Logistics',
            'context': 'Archipelago with ferry connections',
            's': 0.20,
            'typical_kappa': 0.2,
            'description': 'Water barriers, complex routing'
        }
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    q_star_values = [0.10, 0.20, 0.30]
    colors = ['green', 'orange', 'red']

    for i, (q_star, color) in enumerate(zip(q_star_values, colors)):
        t_values = []
        scenario_names = []

        for example in examples:
            s = example['s']
            kappa = example['typical_kappa']
            t_critical = compute_critical_distance(kappa, s, q_star)
            t_values.append(t_critical)
            scenario_names.append(example['scenario'])

        # Create bar plot
        x_pos = np.arange(len(scenario_names))
        bars = ax.bar(x_pos + i*0.25, t_values, 0.25,
                     color=color, alpha=0.7,
                     label=f'{q_star*100:.0f}% Risk Threshold')

        # Add value labels on bars
        for bar, val in zip(bars, t_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Application Scenario')
    ax.set_ylabel('Critical Distance |t*|')
    ax.set_title('Safety Band Thresholds for Real-World Applications')
    ax.set_xticks(x_pos + 0.25)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('safety_bands_practical_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: safety_bands_practical_examples.png")

    # Create practical guidance text
    guidance_text = """
PRACTICAL GUIDANCE FOR USING SAFETY BANDS

1. DETERMINE YOUR CONTEXT:
   - Measure or estimate geographic parameter 's' for your region
   - Identify typical geometric parameter 'κ' from Voronoi analysis
   - Choose acceptable risk threshold q* (10%, 20%, or 30%)

2. FIND CRITICAL DISTANCE:
   - Use lookup table or formula: |t*| = -Φ^(-1)(q*) * √2 * s / κ
   - This gives you the distance threshold around Voronoi boundaries

3. APPLY IN PRACTICE:
   - For municipalities with distance to boundary |t| < |t*|: USE NETWORK ANALYSIS
   - For municipalities with distance to boundary |t| > |t*|: EUCLIDEAN OK
   - This provides optimal computational resource allocation

4. EXAMPLES FROM ANALYSIS:
"""

    for example in examples:
        s = example['s']
        kappa = example['typical_kappa']
        for q_star in [0.10, 0.20, 0.30]:
            t_critical = compute_critical_distance(kappa, s, q_star)
            guidance_text += f"\n   {example['scenario']} (s={s}, κ={kappa}):"
            guidance_text += f"\n   └─ {q_star*100:.0f}% risk → |t*| = {t_critical:.2f}"

    # Save guidance
    with open('safety_bands_practical_guidance.txt', 'w', encoding='utf-8') as f:
        f.write(guidance_text)
    print("Saved: safety_bands_practical_guidance.txt")

def main():
    """Main execution function"""
    print("="*80)
    print("SAFETY BANDS ANALYSIS - CALIBRATION CURVES FOR Q1")
    print("="*80)

    # Set publication style
    set_publication_style()

    # Generate calibration curves
    print("Creating calibration curves...")
    kappa_range, s_values, q_star_values = generate_calibration_curves()

    # Create contour plots
    print("Creating contour maps...")
    create_contour_plot()

    # Create lookup table
    print("Creating lookup table...")
    lookup_df = create_lookup_table()

    # Create practical examples
    print("Creating practical examples...")
    create_practical_examples()

    print("\n" + "="*80)
    print("SAFETY BANDS ANALYSIS COMPLETE")
    print("="*80)
    print("Generated files:")
    print("  - safety_bands_calibration_curves.png")
    print("  - safety_bands_contour_maps.png")
    print("  - safety_bands_lookup_table.csv")
    print("  - safety_bands_lookup_table.tex")
    print("  - safety_bands_practical_examples.png")
    print("  - safety_bands_practical_guidance.txt")

    print("\nKey outcomes for Q1 submission:")
    print("1. HIGHLY CITABLE calibration curves")
    print("2. Practitioner-ready lookup tables")
    print("3. Real-world application examples")
    print("4. Computational resource optimization guidance")
    print("\nThese tools transform theoretical framework into actionable methodology!")

if __name__ == "__main__":
    main()