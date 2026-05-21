#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kolmogorov-Smirnov Test Analysis: Plant-Municipality Pairs
============================================================

This script performs distributional analysis of network scaling factors (β = d_r/d_e)
using PLANT-MUNICIPALITY pairs instead of municipality-municipality pairs.

Key advantages:
- Statistically independent observations (each municipality → its assigned plant)
- Directly addresses research question: "How good is Voronoi assignment?"
- Reduces autocorrelation compared to all-pairs analysis

Author: Analysis script for Voronoi framework revision v5
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def load_plant_municipality_data():
    """
    Load plant-municipality assignment data.

    Returns:
        DataFrame with columns: municipality, assigned_plant, d_euclidean, d_real, beta
    """
    # Load Euclidean assignments (Voronoi-based)
    asig_euclidean = pd.read_csv('codigo/asignacion_municipios_euclidiana.csv')

    # Load real network assignments
    asig_real = pd.read_csv('codigo/asignacion_municipios_real.csv')

    # Load distance data
    # We need D_euclidean and D_real from municipality to assigned plant
    # These should be in the assignment files or separate distance matrices

    # For each municipality, get d_euclidean and d_real to its Voronoi-assigned plant
    # Then calculate beta = d_real / d_euclidean

    # Rename columns to English for consistency
    asig_euclidean = asig_euclidean.rename(columns={
        'municipio': 'municipality',
        'planta_asignada': 'assigned_plant',
        'euclidean_distance': 'd_euclidean'
    })

    asig_real = asig_real.rename(columns={
        'municipio': 'municipality',
        'planta_asignada': 'assigned_plant_real',
        'real_distance': 'd_real'
    })

    # Merge datasets
    data = asig_euclidean.merge(
        asig_real[['municipality', 'assigned_plant_real', 'd_real']],
        on='municipality',
        suffixes=('_voronoi', '_real')
    )

    # Calculate beta for plant-municipality pairs
    data['beta'] = data['d_real'] / data['d_euclidean']

    # Filter: only include valid beta values (>= 1.0, finite)
    data = data[(data['beta'] >= 1.0) & (data['beta'].notna()) & np.isfinite(data['beta'])]

    print(f"Loaded {len(data)} plant-municipality pairs")
    print(f"Beta range: [{data['beta'].min():.4f}, {data['beta'].max():.4f}]")
    print(f"Beta mean: {data['beta'].mean():.4f} ± {data['beta'].std():.4f}")

    return data

def fit_distributions(beta_values):
    """
    Fit Log-Normal, Gamma, and Weibull distributions to beta values.

    Args:
        beta_values: numpy array of beta values

    Returns:
        dict with fitted parameters and goodness-of-fit statistics
    """
    results = {}

    # 1. Log-Normal Distribution
    # Fit to ln(beta) ~ N(m, s^2)
    log_beta = np.log(beta_values)
    m_hat = np.mean(log_beta)
    s_hat = np.std(log_beta, ddof=1)

    # K-S test: compare empirical CDF vs theoretical Log-Normal CDF
    ks_stat_lognorm, p_value_lognorm = stats.kstest(
        beta_values,
        lambda x: stats.lognorm.cdf(x, s=s_hat, scale=np.exp(m_hat))
    )

    results['lognormal'] = {
        'params': {'m': m_hat, 's': s_hat},
        'ks_statistic': ks_stat_lognorm,
        'p_value': p_value_lognorm,
        'distribution': stats.lognorm(s=s_hat, scale=np.exp(m_hat))
    }

    # 2. Gamma Distribution
    # Fit using MLE
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(beta_values, floc=0)

    ks_stat_gamma, p_value_gamma = stats.kstest(
        beta_values,
        lambda x: stats.gamma.cdf(x, shape_gamma, loc=loc_gamma, scale=scale_gamma)
    )

    results['gamma'] = {
        'params': {'shape': shape_gamma, 'scale': scale_gamma},
        'ks_statistic': ks_stat_gamma,
        'p_value': p_value_gamma,
        'distribution': stats.gamma(shape_gamma, loc=loc_gamma, scale=scale_gamma)
    }

    # 3. Weibull Distribution
    # Fit using MLE
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(beta_values, floc=0)

    ks_stat_weibull, p_value_weibull = stats.kstest(
        beta_values,
        lambda x: stats.weibull_min.cdf(x, shape_weibull, loc=loc_weibull, scale=scale_weibull)
    )

    results['weibull'] = {
        'params': {'shape': shape_weibull, 'scale': scale_weibull},
        'ks_statistic': ks_stat_weibull,
        'p_value': p_value_weibull,
        'distribution': stats.weibull_min(shape_weibull, loc=loc_weibull, scale=scale_weibull)
    }

    return results

def generate_latex_table(results, n_obs, output_file='tables/ks_test_plant_municipality.tex'):
    """
    Generate LaTeX table with K-S test results.

    Args:
        results: dict from fit_distributions()
        n_obs: number of observations
        output_file: path to save LaTeX table
    """
    latex_content = r"""\begin{table}[htbp]
\caption{Goodness-of-fit tests for network scaling factor $\beta$ (Plant-Municipality pairs, n=NUM_OBS)}
\label{tab:ks_test_plant_municipality}
\centering
\begin{tabular}{lrrrl}
\toprule
Distribution & K-S Statistic & p-value & Parameters & Fit Quality \\
\midrule
""".replace('NUM_OBS', str(n_obs))

    # Sort by p-value (descending) to show best fit first
    sorted_dists = sorted(results.items(), key=lambda x: x[1]['p_value'], reverse=True)

    for dist_name, res in sorted_dists:
        dist_name_pretty = dist_name.capitalize()
        ks_stat = res['ks_statistic']
        p_val = res['p_value']

        # Format parameters
        if dist_name == 'lognormal':
            params = f"$m={res['params']['m']:.3f}$, $s={res['params']['s']:.3f}$"
        elif dist_name == 'gamma':
            params = f"$k={res['params']['shape']:.3f}$, $\\theta={res['params']['scale']:.3f}$"
        else:  # weibull
            params = f"$k={res['params']['shape']:.3f}$, $\\lambda={res['params']['scale']:.3f}$"

        # Determine fit quality
        if p_val > 0.05:
            quality = "Good"
        elif p_val > 0.01:
            quality = "Acceptable"
        else:
            quality = "Poor"

        latex_content += f"{dist_name_pretty} & {ks_stat:.4f} & {p_val:.4f} & {params} & {quality} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\nLaTeX table saved to: {output_file}")

def print_comparison_with_all_pairs():
    """
    Print comparison between plant-municipality and all-pairs analysis.
    """
    print("\n" + "="*80)
    print("COMPARISON: Plant-Municipality vs All-Pairs Analysis")
    print("="*80)
    print("\nAdvantages of Plant-Municipality analysis:")
    print("  1. Statistical independence: Each observation is independent")
    print("  2. Direct interpretation: beta for assigned Voronoi plant")
    print("  3. Reduced autocorrelation: No spatial clustering effects")
    print("  4. Smaller sample: n=383 vs n=9,112 (more appropriate for K-S test)")
    print("\nExpected outcome:")
    print("  - Higher p-values (better fit)")
    print("  - Log-Normal should remain best fit")
    print("  - Clearer statistical significance")

def main():
    """Main execution"""
    print("="*80)
    print("DISTRIBUTIONAL ANALYSIS: PLANT-MUNICIPALITY PAIRS")
    print("="*80)
    print()

    # Load data
    print("Step 1: Loading plant-municipality assignment data...")
    data = load_plant_municipality_data()
    beta_values = data['beta'].values

    print(f"\nStep 2: Fitting distributions to n={len(beta_values)} observations...")
    results = fit_distributions(beta_values)

    # Print results
    print("\n" + "-"*80)
    print("GOODNESS-OF-FIT RESULTS")
    print("-"*80)

    for dist_name, res in sorted(results.items(), key=lambda x: x[1]['p_value'], reverse=True):
        print(f"\n{dist_name.upper()}:")
        print(f"  Parameters: {res['params']}")
        print(f"  K-S statistic: {res['ks_statistic']:.6f}")
        print(f"  p-value: {res['p_value']:.6f}")

        if res['p_value'] > 0.05:
            verdict = "GOOD FIT (p > 0.05)"
        elif res['p_value'] > 0.01:
            verdict = "ACCEPTABLE FIT (0.01 < p < 0.05)"
        else:
            verdict = "POOR FIT (p < 0.01)"
        print(f"  Verdict: {verdict}")

    # Generate LaTeX table
    print("\nStep 3: Generating LaTeX table...")
    generate_latex_table(results, len(beta_values))

    # Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'distribution': ['Log-Normal', 'Gamma', 'Weibull'],
        'ks_statistic': [results['lognormal']['ks_statistic'],
                         results['gamma']['ks_statistic'],
                         results['weibull']['ks_statistic']],
        'p_value': [results['lognormal']['p_value'],
                   results['gamma']['p_value'],
                   results['weibull']['p_value']],
        'n_observations': len(beta_values)
    })
    results_df.to_csv('codigo/ks_test_plant_municipality_results.csv', index=False)
    print("Results saved to: codigo/ks_test_plant_municipality_results.csv")

    # Print comparison
    print_comparison_with_all_pairs()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run visualization script: python scripts/plot_plant_municipality_distributions.py")
    print("  2. Compare with all-pairs analysis (Table 2b)")
    print("  3. Update manuscript with new Table 2a")

if __name__ == "__main__":
    main()
