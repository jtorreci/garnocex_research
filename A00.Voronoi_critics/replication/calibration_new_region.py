#!/usr/bin/env python3
"""
Geographic Region Calibration Tool
=================================

Apply the Voronoi probabilistic framework to a new geographic region.
This script helps practitioners calibrate parameters and generate safety bands
for their specific geographic context.

Usage:
    python calibration_new_region.py --data new_region_data.csv --config config.json

Author: Voronoi Framework Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import erfinv
from scipy.optimize import minimize_scalar
import argparse
import json
import os
from pathlib import Path

def load_region_data(data_path):
    """
    Load data for a new geographic region

    Expected CSV format:
    - Municipality_ID: Unique identifier
    - Facility_ID: Facility identifier
    - Euclidean_Distance_km: Straight-line distance
    - Network_Distance_km: Actual travel distance
    - Municipality_X, Municipality_Y: Geographic coordinates (optional)
    - Facility_X, Facility_Y: Facility coordinates (optional)
    """
    print(f"Loading region data from: {data_path}")

    df = pd.read_csv(data_path)

    # Validate required columns
    required_cols = ['Municipality_ID', 'Facility_ID', 'Euclidean_Distance_km', 'Network_Distance_km']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute beta factors
    df['Beta_Factor'] = df['Network_Distance_km'] / df['Euclidean_Distance_km']

    # Remove invalid ratios
    valid_mask = (df['Beta_Factor'] > 0.5) & (df['Beta_Factor'] < 10.0)
    n_removed = len(df) - valid_mask.sum()
    if n_removed > 0:
        print(f"  ⚠ Removed {n_removed} invalid distance ratios")

    df = df[valid_mask].copy()

    print(f"  ✓ Loaded {len(df)} municipality-facility pairs")
    print(f"  ✓ Mean β factor: {df['Beta_Factor'].mean():.3f}")
    print(f"  ✓ Std β factor: {df['Beta_Factor'].std():.3f}")

    return df

def estimate_geographic_parameter(beta_values, method='mle'):
    """
    Estimate geographic complexity parameter 's' from beta values

    Methods:
    - 'mle': Maximum likelihood estimation
    - 'robust': Robust estimation using median-based statistics
    """
    print(f"Estimating geographic parameter 's' using {method} method...")

    if method == 'mle':
        # Fit log-normal distribution to beta values
        sigma_ln, loc_ln, scale_ln = stats.lognorm.fit(beta_values, floc=0)
        mu_ln = np.log(scale_ln)

        # Convert to geographic parameter
        # From paper: σ ≈ s (geographic complexity)
        s_estimate = sigma_ln

    elif method == 'robust':
        # Robust estimation using quantiles
        log_beta = np.log(beta_values)
        q75 = np.percentile(log_beta, 75)
        q25 = np.percentile(log_beta, 25)

        # IQR-based estimate of standard deviation
        sigma_robust = (q75 - q25) / (2 * stats.norm.ppf(0.75))
        s_estimate = sigma_robust

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  ✓ Estimated s = {s_estimate:.3f}")
    return s_estimate

def fit_lognormal_parameters(beta_values):
    """Fit log-normal distribution and return parameters"""
    print("Fitting log-normal distribution...")

    # Fit parameters
    sigma_ln, loc_ln, scale_ln = stats.lognorm.fit(beta_values, floc=0)
    mu_ln = np.log(scale_ln)

    # Goodness of fit
    ks_stat, ks_pvalue = stats.kstest(beta_values, stats.lognorm(sigma_ln, loc=loc_ln, scale=scale_ln).cdf)

    print(f"  ✓ μ = {mu_ln:.3f}, σ = {sigma_ln:.3f}")
    print(f"  ✓ KS-test: statistic = {ks_stat:.3f}, p-value = {ks_pvalue:.3f}")

    return {
        'mu': mu_ln,
        'sigma': sigma_ln,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'distribution': stats.lognorm(sigma_ln, loc=loc_ln, scale=scale_ln)
    }

def compute_safety_bands(s_param, kappa_values=None, q_star_values=None):
    """Compute safety bands for given parameters"""
    print("Computing safety bands...")

    if kappa_values is None:
        kappa_values = [0.2, 0.5, 1.0, 1.5, 2.0]

    if q_star_values is None:
        q_star_values = [0.10, 0.20, 0.30]

    def inverse_phi(q):
        """Inverse of standard normal CDF"""
        return np.sqrt(2) * erfinv(2*q - 1)

    def compute_critical_distance(kappa, s, q_star):
        """Critical distance formula from paper"""
        phi_inv_q = inverse_phi(q_star)
        t_critical = -phi_inv_q * np.sqrt(2) * s / kappa
        return np.abs(t_critical)

    safety_bands = {}
    for q_star in q_star_values:
        safety_bands[q_star] = {}
        for kappa in kappa_values:
            t_critical = compute_critical_distance(kappa, s_param, q_star)
            safety_bands[q_star][kappa] = t_critical

    return safety_bands

def create_calibration_plots(df, params, safety_bands, output_dir):
    """Create calibration plots for the new region"""
    print("Creating calibration plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Beta distribution with fit
    ax1 = axes[0, 0]
    beta_values = df['Beta_Factor'].values

    # Histogram
    ax1.hist(beta_values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Fitted distribution
    x = np.linspace(beta_values.min(), beta_values.max(), 1000)
    fitted_dist = params['distribution']
    ax1.plot(x, fitted_dist.pdf(x), 'r-', linewidth=2,
            label=f'Log-Normal(μ={params["mu"]:.3f}, σ={params["sigma"]:.3f})')

    ax1.set_xlabel('β Scaling Factor')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Fit for New Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(beta_values, dist=fitted_dist, plot=ax2)
    ax2.set_title('Q-Q Plot: Goodness of Fit')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Safety bands curves
    ax3 = axes[1, 0]
    kappa_range = np.linspace(0.1, 2.0, 100)

    colors = ['green', 'orange', 'red']
    q_star_values = [0.10, 0.20, 0.30]

    for i, (q_star, color) in enumerate(zip(q_star_values, colors)):
        s_param = params['sigma']  # Use fitted sigma as s parameter
        phi_inv_q = np.sqrt(2) * erfinv(2*q_star - 1)
        t_critical = -phi_inv_q * np.sqrt(2) * s_param / kappa_range

        ax3.plot(kappa_range, np.abs(t_critical), color=color, linewidth=2,
                label=f'{q_star*100:.0f}% Risk Threshold')

    ax3.set_xlabel('Geometric Parameter κ')
    ax3.set_ylabel('Critical Distance |t*|')
    ax3.set_title('Safety Bands for New Region')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 5)

    # Plot 4: Parameter comparison
    ax4 = axes[1, 1]

    # Geographic contexts from paper
    regions = {
        'Urban Dense': 0.05,
        'Flat Plains': 0.08,
        'Extremadura': 0.093,
        'Mountainous': 0.12,
        'Complex': 0.15,
        'New Region': params['sigma']
    }

    region_names = list(regions.keys())
    s_values = list(regions.values())
    colors_bar = ['lightblue'] * (len(region_names) - 1) + ['red']

    bars = ax4.bar(region_names, s_values, color=colors_bar, alpha=0.7, edgecolor='black')

    # Highlight new region
    new_idx = region_names.index('New Region')
    bars[new_idx].set_color('red')
    bars[new_idx].set_alpha(0.9)

    ax4.set_ylabel('Geographic Parameter s')
    ax4.set_title('Regional Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'region_calibration_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved calibration plots: {plot_path}")

def create_lookup_table(safety_bands, output_dir):
    """Create lookup table for the new region"""
    print("Creating lookup table...")

    # Convert to DataFrame
    table_data = []
    for q_star, kappa_dict in safety_bands.items():
        for kappa, t_critical in kappa_dict.items():
            table_data.append({
                'Risk_Threshold': f'{q_star*100:.0f}%',
                'Kappa_Parameter': kappa,
                'Critical_Distance_t': f'{t_critical:.2f}'
            })

    df_lookup = pd.DataFrame(table_data)

    # Save CSV
    csv_path = os.path.join(output_dir, 'new_region_lookup_table.csv')
    df_lookup.to_csv(csv_path, index=False)
    print(f"  ✓ Saved lookup table: {csv_path}")

    # Create formatted table for display
    pivot_table = df_lookup.pivot(index='Kappa_Parameter', columns='Risk_Threshold', values='Critical_Distance_t')

    print("\n" + "="*60)
    print("SAFETY BANDS LOOKUP TABLE FOR NEW REGION")
    print("="*60)
    print(pivot_table.to_string())
    print("="*60)

    return df_lookup

def generate_implementation_guide(params, safety_bands, output_dir):
    """Generate implementation guide for the new region"""
    print("Generating implementation guide...")

    guide_content = f"""# Implementation Guide for New Region

## Calibrated Parameters

**Geographic complexity parameter**: s = {params['sigma']:.3f}
**Log-normal location parameter**: μ = {params['mu']:.3f}
**Log-normal scale parameter**: σ = {params['sigma']:.3f}
**Distribution fit quality**: KS p-value = {params['ks_pvalue']:.3f}

## Regional Classification

Based on the estimated s parameter ({params['sigma']:.3f}), this region is classified as:

"""

    # Classify region
    if params['sigma'] < 0.06:
        classification = "Urban Dense"
        description = "High road density, minimal topographic barriers"
    elif params['sigma'] < 0.09:
        classification = "Flat Plains"
        description = "Agricultural areas with good road connectivity"
    elif params['sigma'] < 0.11:
        classification = "Moderate Hills"
        description = "Mixed terrain with moderate impedance (Extremadura-like)"
    elif params['sigma'] < 0.14:
        classification = "Mountainous"
        description = "Significant elevation changes, winding roads"
    else:
        classification = "Complex Terrain"
        description = "Archipelago or very complex topography"

    guide_content += f"**{classification}**: {description}\n\n"

    guide_content += """## How to Use Safety Bands

1. **Identify your context**:
   - Use the calibrated parameters above
   - Determine typical geometric parameter κ from your Voronoi analysis
   - Choose acceptable risk threshold (10%, 20%, or 30%)

2. **Find critical distance**:
   - Use the lookup table generated for your region
   - Or calculate: |t*| = -Φ^(-1)(q*) × √2 × s / κ

3. **Apply in practice**:
   - For areas with distance to boundary |t| < |t*|: Use network analysis
   - For areas with distance to boundary |t| > |t*|: Euclidean approximation OK
   - This optimizes computational resources while maintaining accuracy

## Example Applications

"""

    # Add specific examples for this region
    kappa_typical = 0.5  # Conservative estimate
    for q_star in [0.10, 0.20, 0.30]:
        if q_star in safety_bands and kappa_typical in safety_bands[q_star]:
            t_critical = safety_bands[q_star][kappa_typical]
            guide_content += f"- **{q_star*100:.0f}% risk tolerance** (κ=0.5): |t*| = {t_critical:.2f}\n"

    guide_content += """
## Validation Recommendations

1. **Cross-validation**: Test framework on subset of known misallocations
2. **Sensitivity analysis**: Vary κ parameter based on local geometric properties
3. **Update parameters**: Recalibrate annually or when infrastructure changes significantly

## Citation

When using this calibration for your region, please cite both the original framework
and acknowledge the use of calibration tools:

```bibtex
@article{voronoi_probabilistic_2025,
  title={The Hidden Cost of Straight Lines: Quantifying Misallocation Risk in Voronoi-Based Service Area Models},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

---
Generated by Voronoi Framework Calibration Tool
"""

    guide_path = os.path.join(output_dir, 'implementation_guide.md')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)

    print(f"  ✓ Saved implementation guide: {guide_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calibrate Voronoi framework for new region')
    parser.add_argument('--data', '-d', required=True,
                       help='CSV file with municipality-facility distance data')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--method', '-m', default='mle', choices=['mle', 'robust'],
                       help='Parameter estimation method (default: mle)')
    parser.add_argument('--config', '-c',
                       help='JSON configuration file (optional)')

    args = parser.parse_args()

    print("=" * 80)
    print("🌍 VORONOI FRAMEWORK - NEW REGION CALIBRATION")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.config}")

    # Load and process data
    df = load_region_data(args.data)
    beta_values = df['Beta_Factor'].values

    # Estimate parameters
    s_param = estimate_geographic_parameter(beta_values, method=args.method)
    lognormal_params = fit_lognormal_parameters(beta_values)

    # Compute safety bands
    safety_bands = compute_safety_bands(s_param)

    # Create outputs
    create_calibration_plots(df, lognormal_params, safety_bands, str(output_dir))
    lookup_table = create_lookup_table(safety_bands, str(output_dir))
    generate_implementation_guide(lognormal_params, safety_bands, str(output_dir))

    # Save calibration results
    calibration_results = {
        'region_data_file': args.data,
        'n_observations': len(df),
        'estimation_method': args.method,
        'parameters': {
            's_geographic': s_param,
            'mu_lognormal': lognormal_params['mu'],
            'sigma_lognormal': lognormal_params['sigma']
        },
        'fit_quality': {
            'ks_statistic': lognormal_params['ks_statistic'],
            'ks_pvalue': lognormal_params['ks_pvalue']
        },
        'beta_statistics': {
            'mean': float(np.mean(beta_values)),
            'std': float(np.std(beta_values)),
            'median': float(np.median(beta_values)),
            'min': float(np.min(beta_values)),
            'max': float(np.max(beta_values))
        }
    }

    results_path = output_dir / 'calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"📊 Processed {len(df)} municipality-facility pairs")
    print(f"📐 Estimated parameters: s = {s_param:.3f}, μ = {lognormal_params['mu']:.3f}, σ = {lognormal_params['sigma']:.3f}")
    print(f"📁 Results saved in: {output_dir.absolute()}")
    print("\n🎯 Next steps:")
    print("  1. Review calibration plots and fit quality")
    print("  2. Use lookup table for practical applications")
    print("  3. Follow implementation guide for deployment")

if __name__ == "__main__":
    main()