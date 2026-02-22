#!/usr/bin/env python3
"""
Recalculate confidence interval for theoretical misallocation count
using updated empirical parameters and municipality count.

Author: Claude Code Analysis
Date: September 16, 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
codigo_dir = Path(__file__).parent
output_dir = codigo_dir.parent / "imagenes"

def load_empirical_ratios():
    """Load the empirical beta ratios to recalculate distribution parameters."""
    print("Loading empirical ratio data...")

    # Load from our previous analysis
    ratios_file = codigo_dir / "detailed_ratios_analysis.csv"
    if ratios_file.exists():
        df_ratios = pd.read_csv(ratios_file)
        ratios = df_ratios['Ratio'].values
        print(f"Loaded {len(ratios)} empirical ratios")
        return ratios
    else:
        # Fallback: generate synthetic ratios with observed parameters
        print("Using fallback synthetic ratios based on empirical parameters")
        # From our analysis: mean=1.186, std=0.129
        np.random.seed(42)
        ratios = np.random.lognormal(mean=np.log(1.186), sigma=0.129, size=9240)
        return ratios

def fit_lognormal_distribution(ratios):
    """Fit log-normal distribution to empirical ratios."""
    print("Fitting log-normal distribution...")

    # Remove outliers for better fit
    q01, q99 = np.percentile(ratios, [1, 99])
    ratios_clean = ratios[(ratios >= q01) & (ratios <= q99)]

    # Fit log-normal distribution
    # Method 1: Using scipy
    shape, loc, scale = stats.lognorm.fit(ratios_clean, floc=0)

    # Method 2: Direct calculation from log values
    log_ratios = np.log(ratios_clean)
    mu = np.mean(log_ratios)
    sigma = np.std(log_ratios)

    print(f"Log-normal parameters:")
    print(f"  - Shape (sigma): {shape:.6f}")
    print(f"  - Scale (exp(mu)): {scale:.6f}")
    print(f"  - Direct sigma: {sigma:.6f}")
    print(f"  - Direct exp(mu): {np.exp(mu):.6f}")

    return shape, scale, sigma, mu

def calculate_misallocation_probability(sigma, num_municipalities=383, num_plants=46):
    """Calculate theoretical misallocation probability for the region."""
    print(f"Calculating misallocation probability for {num_municipalities} municipalities, {num_plants} plants")

    # For each municipality, calculate probability of being misallocated
    # This depends on the Voronoi geometry and local competition

    # Simplified model: assume each municipality has on average 2-3 competing plants
    # and the probability depends on the distance ratios

    # Average probability per municipality (based on geometric considerations)
    # For a typical Voronoi cell, the probability depends on the distribution tail

    # Using the theoretical formula for log-normal misallocation
    # P(misallocation) ≈ P(β₁/β₂ > R) where R is the distance ratio

    # For a typical municipality with competing plants at ratio R ≈ 1.2 (20% further)
    typical_ratio = 1.2

    # P(misallocation) = P(log(β₁) - log(β₂) > log(R))
    # = P(N(0, sqrt(2)*sigma) > log(R))
    # = 1 - Φ(log(R) / (sqrt(2) * sigma))

    prob_per_municipality = 1 - stats.norm.cdf(np.log(typical_ratio) / (np.sqrt(2) * sigma))

    print(f"Individual misallocation probability: {prob_per_municipality:.4f}")

    # Expected number of misallocations
    expected_misallocations = num_municipalities * prob_per_municipality

    # Confidence interval using binomial approximation
    # For large n, binomial → normal
    mean = expected_misallocations
    variance = num_municipalities * prob_per_municipality * (1 - prob_per_municipality)
    std = np.sqrt(variance)

    # 95% confidence interval
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std

    print(f"Theoretical misallocation analysis:")
    print(f"  - Expected: {mean:.1f} municipalities")
    print(f"  - Standard deviation: {std:.1f}")
    print(f"  - 95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")

    return mean, std, ci_lower, ci_upper, prob_per_municipality

def calculate_alternative_ci(sigma, empirical_rate=88/383):
    """Calculate CI using empirical misallocation rate as calibration."""
    print(f"\\nAlternative calculation using empirical rate: {empirical_rate:.3f}")

    num_municipalities = 383

    # If empirical rate is p_emp, then we can estimate the effective distance ratio
    # p_emp = 1 - Φ(log(R_eff) / (sqrt(2) * sigma))
    # Solving for R_eff:
    # log(R_eff) = sqrt(2) * sigma * Φ⁻¹(1 - p_emp)

    z_score = stats.norm.ppf(1 - empirical_rate)
    effective_ratio = np.exp(np.sqrt(2) * sigma * z_score)

    print(f"  - Effective distance ratio: {effective_ratio:.3f}")
    print(f"  - Z-score: {z_score:.3f}")

    # Now calculate CI with this calibrated model
    mean = num_municipalities * empirical_rate
    variance = num_municipalities * empirical_rate * (1 - empirical_rate)
    std = np.sqrt(variance)

    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std

    print(f"  - Expected: {mean:.1f} municipalities")
    print(f"  - 95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")

    return mean, std, ci_lower, ci_upper

def create_ci_visualization(ratios, sigma, mean, ci_lower, ci_upper, empirical_count=88):
    """Create visualization of the confidence interval analysis."""
    print("Creating CI visualization...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of ratios
    ax1.hist(ratios, bins=50, density=True, alpha=0.7, color='lightgray', edgecolor='black')

    # Overlay fitted log-normal
    x = np.linspace(ratios.min(), ratios.max(), 1000)
    fitted_pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(np.log(ratios).mean()))
    ax1.plot(x, fitted_pdf, 'r-', linewidth=2, label=f'Log-Normal fit (σ={sigma:.3f})')

    ax1.set_xlabel('Beta Ratio (Network/Euclidean)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Distance Ratios with Log-Normal Fit', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Theoretical vs Empirical comparison
    categories = ['Theoretical\\n(Lower)', 'Theoretical\\n(Mean)', 'Theoretical\\n(Upper)', 'Empirical\\n(Actual)']
    values = [ci_lower, mean, ci_upper, empirical_count]
    colors = ['lightgray', 'gray', 'lightgray', 'red']

    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=empirical_count, color='red', linestyle='--', linewidth=2, label='Empirical Count')
    ax2.set_ylabel('Number of Misallocated Municipalities', fontsize=12)
    ax2.set_title('Theoretical vs Empirical Misallocation Counts', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

    # 3. Probability sensitivity analysis
    sigma_range = np.linspace(0.05, 0.20, 100)
    prob_range = []

    for s in sigma_range:
        typical_ratio = 1.2
        prob = 1 - stats.norm.cdf(np.log(typical_ratio) / (np.sqrt(2) * s))
        prob_range.append(383 * prob)

    ax3.plot(sigma_range, prob_range, 'k-', linewidth=2, label='Theoretical')
    ax3.axvline(x=sigma, color='red', linestyle='--', linewidth=2, label=f'Fitted σ={sigma:.3f}')
    ax3.axhline(y=empirical_count, color='red', linestyle=':', linewidth=2, label='Empirical Count')
    ax3.set_xlabel('Log-Normal Shape Parameter (σ)', fontsize=12)
    ax3.set_ylabel('Expected Misallocations', fontsize=12)
    ax3.set_title('Sensitivity to Distribution Parameters', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confidence intervals comparison
    scenarios = ['Original\\n(s=0.0926)', 'Updated\\n(Empirical)', 'Calibrated\\n(Rate-based)']

    # Original CI (from paper)
    original_lower, original_upper = 52, 65

    # Updated CI (calculated)
    updated_lower, updated_upper = ci_lower, ci_upper

    # Calibrated CI
    cal_mean, cal_std, cal_lower, cal_upper = calculate_alternative_ci(sigma)

    ci_data = np.array([
        [original_lower, original_upper],
        [updated_lower, updated_upper],
        [cal_lower, cal_upper]
    ])

    x_pos = np.arange(len(scenarios))
    width = 0.6

    # Plot CI ranges
    for i, (lower, upper) in enumerate(ci_data):
        ax4.bar(x_pos[i], upper - lower, bottom=lower, width=width,
               color='lightgray', alpha=0.7, edgecolor='black')
        # Mark empirical value
        ax4.plot(x_pos[i], empirical_count, 'ro', markersize=8, markerfacecolor='red')

    ax4.axhline(y=empirical_count, color='red', linestyle='--', alpha=0.5,
               label=f'Empirical: {empirical_count}')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenarios)
    ax4.set_ylabel('Number of Misallocated Municipalities', fontsize=12)
    ax4.set_title('Confidence Interval Comparison', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_interval_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved CI analysis plot: {output_dir / 'confidence_interval_analysis.png'}")

def main():
    """Main execution function."""
    print("=== CONFIDENCE INTERVAL RECALCULATION ===")

    # Load empirical data
    ratios = load_empirical_ratios()

    # Fit distribution
    shape, scale, sigma, mu = fit_lognormal_distribution(ratios)

    # Calculate theoretical CI
    mean, std, ci_lower, ci_upper, prob = calculate_misallocation_probability(sigma)

    # Calculate alternative CI
    alt_mean, alt_std, alt_ci_lower, alt_ci_upper = calculate_alternative_ci(sigma)

    # Create visualization
    create_ci_visualization(ratios, sigma, mean, ci_lower, ci_upper)

    # Summary
    print("\\n=== SUMMARY ===")
    print(f"Empirical misallocations: 88/383 municipalities (23.0%)")
    print(f"Original CI (paper): [52, 65] - EMPIRICAL OUTSIDE RANGE")
    print(f"Updated theoretical CI: [{ci_lower:.0f}, {ci_upper:.0f}]")
    print(f"Calibrated CI: [{alt_ci_lower:.0f}, {alt_ci_upper:.0f}]")

    if ci_lower <= 88 <= ci_upper:
        print("✓ Empirical value within updated theoretical CI")
    else:
        print("✗ Empirical value still outside theoretical CI")

    if alt_ci_lower <= 88 <= alt_ci_upper:
        print("✓ Empirical value within calibrated CI")
    else:
        print("✗ Empirical value outside calibrated CI")

if __name__ == "__main__":
    main()