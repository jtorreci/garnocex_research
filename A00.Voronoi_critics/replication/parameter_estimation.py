#!/usr/bin/env python3
"""
Parameter Estimation Tool for New Geographic Regions
===================================================

Estimates the geographic complexity parameter 's' for a new region based on
distance ratio data. Provides multiple estimation methods and validation tools.

Usage:
    python parameter_estimation.py --input data.csv --output params.json

Author: Voronoi Framework Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

def load_distance_data(file_path):
    """
    Load distance data from CSV file

    Expected columns:
    - Euclidean_Distance_km: Straight-line distance
    - Network_Distance_km: Actual travel distance
    OR
    - Beta_Factor: Pre-computed ratio
    """
    print(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)

    if 'Beta_Factor' in df.columns:
        beta_values = df['Beta_Factor'].values
        print(f"  ✓ Using pre-computed Beta_Factor column")
    elif 'Euclidean_Distance_km' in df.columns and 'Network_Distance_km' in df.columns:
        beta_values = df['Network_Distance_km'] / df['Euclidean_Distance_km']
        print(f"  ✓ Computed beta factors from distance columns")
    else:
        raise ValueError("File must contain either 'Beta_Factor' or both distance columns")

    # Filter valid ratios
    valid_mask = (beta_values > 0.5) & (beta_values < 10.0) & np.isfinite(beta_values)
    beta_values = beta_values[valid_mask]

    n_removed = len(df) - len(beta_values)
    if n_removed > 0:
        print(f"  ⚠ Removed {n_removed} invalid ratios")

    print(f"  ✓ Final dataset: {len(beta_values)} observations")
    print(f"  ✓ Range: [{np.min(beta_values):.3f}, {np.max(beta_values):.3f}]")

    return beta_values

def estimate_s_mle(beta_values):
    """Maximum Likelihood Estimation of s parameter"""
    print("Method 1: Maximum Likelihood Estimation")

    # Fit log-normal distribution
    sigma_ln, loc_ln, scale_ln = stats.lognorm.fit(beta_values, floc=0)
    mu_ln = np.log(scale_ln)

    # s parameter approximates sigma in log-normal
    s_mle = sigma_ln

    # Compute log-likelihood
    log_likelihood = np.sum(stats.lognorm.logpdf(beta_values, s=sigma_ln, scale=scale_ln))

    # Standard error approximation
    n = len(beta_values)
    fisher_info = n / (sigma_ln**2)
    se_s = 1 / np.sqrt(fisher_info)

    print(f"  ✓ s_MLE = {s_mle:.4f} ± {se_s:.4f}")
    print(f"  ✓ Log-likelihood = {log_likelihood:.2f}")

    return {
        'value': s_mle,
        'standard_error': se_s,
        'log_likelihood': log_likelihood,
        'mu': mu_ln,
        'sigma': sigma_ln,
        'method': 'MLE'
    }

def estimate_s_method_of_moments(beta_values):
    """Method of Moments estimation"""
    print("Method 2: Method of Moments")

    log_beta = np.log(beta_values)

    # Sample moments
    mean_log = np.mean(log_beta)
    var_log = np.var(log_beta, ddof=1)

    # Method of moments estimators
    sigma_mom = np.sqrt(var_log)
    mu_mom = mean_log
    s_mom = sigma_mom

    # Standard error (delta method approximation)
    n = len(beta_values)
    se_s = np.sqrt(var_log / (2 * n))

    print(f"  ✓ s_MoM = {s_mom:.4f} ± {se_s:.4f}")

    return {
        'value': s_mom,
        'standard_error': se_s,
        'mu': mu_mom,
        'sigma': sigma_mom,
        'method': 'Method of Moments'
    }

def estimate_s_robust(beta_values):
    """Robust estimation using quantiles"""
    print("Method 3: Robust Quantile-based Estimation")

    log_beta = np.log(beta_values)

    # Use interquartile range for robust scale estimation
    q75 = np.percentile(log_beta, 75)
    q25 = np.percentile(log_beta, 25)
    median = np.median(log_beta)

    # IQR-based estimate of standard deviation
    sigma_robust = (q75 - q25) / (2 * stats.norm.ppf(0.75))
    s_robust = sigma_robust

    # Bootstrap standard error
    n_bootstrap = 1000
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(log_beta, size=len(log_beta), replace=True)
        boot_q75 = np.percentile(boot_sample, 75)
        boot_q25 = np.percentile(boot_sample, 25)
        boot_s = (boot_q75 - boot_q25) / (2 * stats.norm.ppf(0.75))
        bootstrap_estimates.append(boot_s)

    se_s = np.std(bootstrap_estimates)

    print(f"  ✓ s_Robust = {s_robust:.4f} ± {se_s:.4f}")

    return {
        'value': s_robust,
        'standard_error': se_s,
        'mu': median,
        'sigma': sigma_robust,
        'method': 'Robust Quantile'
    }

def estimate_s_bayesian(beta_values, prior_mean=0.1, prior_var=0.01):
    """Bayesian estimation with informative prior"""
    print("Method 4: Bayesian Estimation")

    log_beta = np.log(beta_values)
    n = len(log_beta)

    # Sample statistics
    sample_var = np.var(log_beta, ddof=1)

    # Conjugate prior for variance (inverse gamma)
    # Convert to precision (inverse variance) for conjugacy
    prior_alpha = prior_mean**2 / prior_var + 2
    prior_beta = prior_mean * (prior_alpha - 1)

    # Posterior parameters
    posterior_alpha = prior_alpha + n / 2
    posterior_beta = prior_beta + (n - 1) * sample_var / 2

    # Posterior mean and variance for sigma^2
    posterior_var_mean = posterior_beta / (posterior_alpha - 1)
    posterior_var_var = posterior_beta**2 / ((posterior_alpha - 1)**2 * (posterior_alpha - 2))

    # Convert back to s parameter
    s_bayes = np.sqrt(posterior_var_mean)
    se_s = np.sqrt(posterior_var_var) / (2 * s_bayes)

    print(f"  ✓ s_Bayes = {s_bayes:.4f} ± {se_s:.4f}")
    print(f"  ✓ Prior: N({prior_mean:.3f}, {prior_var:.3f})")

    return {
        'value': s_bayes,
        'standard_error': se_s,
        'posterior_alpha': posterior_alpha,
        'posterior_beta': posterior_beta,
        'method': 'Bayesian'
    }

def cross_validate_estimates(beta_values, n_folds=5):
    """Cross-validation to assess estimate stability"""
    print(f"Cross-validation with {n_folds} folds")

    # Randomly partition data
    indices = np.random.permutation(len(beta_values))
    fold_size = len(beta_values) // n_folds

    cv_results = {'MLE': [], 'MoM': [], 'Robust': []}

    for fold in range(n_folds):
        # Create train/test split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(beta_values)

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        train_data = beta_values[train_indices]

        # Estimate on training data
        mle_result = estimate_s_mle(train_data)
        mom_result = estimate_s_method_of_moments(train_data)
        robust_result = estimate_s_robust(train_data)

        cv_results['MLE'].append(mle_result['value'])
        cv_results['MoM'].append(mom_result['value'])
        cv_results['Robust'].append(robust_result['value'])

    # Compute CV statistics
    cv_stats = {}
    for method, values in cv_results.items():
        cv_stats[method] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / np.mean(values)  # Coefficient of variation
        }
        print(f"  {method}: mean = {cv_stats[method]['mean']:.4f}, CV = {cv_stats[method]['cv']:.3f}")

    return cv_stats

def validate_estimates(beta_values, estimates):
    """Validate parameter estimates using goodness-of-fit tests"""
    print("Validating parameter estimates")

    validation_results = {}

    for method, params in estimates.items():
        if method == 'Cross-validation':
            continue

        s_val = params['value']
        mu_val = params.get('mu', np.log(beta_values).mean())

        # Create distribution with estimated parameters
        dist = stats.lognorm(s=s_val, scale=np.exp(mu_val))

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(beta_values, dist.cdf)

        # Anderson-Darling test (approximate)
        try:
            transformed = (np.log(beta_values) - mu_val) / s_val
            ad_stat, ad_critical, ad_significance = stats.anderson(transformed, dist='norm')
            ad_pvalue = 1 - ad_significance / 100 if ad_stat < ad_critical[-1] else 0.001
        except:
            ad_stat, ad_pvalue = np.nan, np.nan

        # AIC and BIC
        log_likelihood = np.sum(dist.logpdf(beta_values))
        n = len(beta_values)
        aic = 2 * 2 - 2 * log_likelihood  # 2 parameters
        bic = 2 * np.log(n) - 2 * log_likelihood

        validation_results[method] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'ad_statistic': ad_stat,
            'ad_pvalue': ad_pvalue,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood
        }

        print(f"  {method}: KS p-value = {ks_pvalue:.3f}, AIC = {aic:.1f}")

    return validation_results

def create_diagnostic_plots(beta_values, estimates, output_dir):
    """Create diagnostic plots for parameter estimation"""
    print("Creating diagnostic plots")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Histogram with fitted distributions
    ax1 = axes[0, 0]
    ax1.hist(beta_values, bins=50, density=True, alpha=0.7, color='lightgray', edgecolor='black')

    x = np.linspace(beta_values.min(), beta_values.max(), 1000)
    colors = ['red', 'blue', 'green', 'orange']

    for i, (method, params) in enumerate(estimates.items()):
        if method == 'Cross-validation':
            continue

        s_val = params['value']
        mu_val = params.get('mu', np.log(beta_values).mean())
        dist = stats.lognorm(s=s_val, scale=np.exp(mu_val))

        ax1.plot(x, dist.pdf(x), color=colors[i], linewidth=2,
                label=f'{method}: s={s_val:.3f}')

    ax1.set_xlabel('β Scaling Factor')
    ax1.set_ylabel('Density')
    ax1.set_title('Parameter Estimation Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-Q plot for best method (MLE)
    ax2 = axes[0, 1]
    mle_params = estimates['MLE']
    best_dist = stats.lognorm(s=mle_params['value'], scale=np.exp(mle_params['mu']))
    stats.probplot(beta_values, dist=best_dist, plot=ax2)
    ax2.set_title('Q-Q Plot (MLE Fit)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter estimates with confidence intervals
    ax3 = axes[1, 0]
    methods = [m for m in estimates.keys() if m != 'Cross-validation']
    values = [estimates[m]['value'] for m in methods]
    errors = [estimates[m]['standard_error'] for m in methods]

    bars = ax3.bar(range(len(methods)), values, yerr=errors, capsize=5,
                   color='skyblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45)
    ax3.set_ylabel('Estimated s Parameter')
    ax3.set_title('Parameter Estimates with Standard Errors')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + err + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Cross-validation results
    ax4 = axes[1, 1]
    if 'Cross-validation' in estimates:
        cv_results = estimates['Cross-validation']
        cv_methods = list(cv_results.keys())
        cv_means = [cv_results[m]['mean'] for m in cv_methods]
        cv_stds = [cv_results[m]['std'] for m in cv_methods]

        bars = ax4.bar(range(len(cv_methods)), cv_means, yerr=cv_stds, capsize=5,
                      color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(cv_methods)))
        ax4.set_xticklabels(cv_methods)
        ax4.set_ylabel('Cross-validated s Parameter')
        ax4.set_title('Cross-validation Results')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/parameter_estimation_diagnostics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved diagnostic plots: {plot_path}")

def recommend_best_estimate(estimates, validation_results):
    """Recommend best parameter estimate based on multiple criteria"""
    print("Determining best parameter estimate")

    scores = {}
    methods = [m for m in estimates.keys() if m != 'Cross-validation']

    for method in methods:
        score = 0

        # Criterion 1: Goodness of fit (higher p-value is better)
        ks_pvalue = validation_results[method]['ks_pvalue']
        score += min(ks_pvalue * 10, 5)  # Max 5 points

        # Criterion 2: AIC (lower is better, relative scoring)
        aics = [validation_results[m]['aic'] for m in methods]
        aic_rank = sorted(aics).index(validation_results[method]['aic'])
        score += (len(methods) - aic_rank) * 2  # Max points for best AIC

        # Criterion 3: Standard error (lower is better)
        se_values = [estimates[m]['standard_error'] for m in methods]
        se_rank = sorted(se_values).index(estimates[method]['standard_error'])
        score += (len(methods) - se_rank) * 2

        # Criterion 4: Cross-validation stability (if available)
        if 'Cross-validation' in estimates and method in estimates['Cross-validation']:
            cv_coeff = estimates['Cross-validation'][method]['cv']
            score += max(0, 5 - cv_coeff * 20)  # Penalty for high CV

        scores[method] = score
        print(f"  {method}: score = {score:.1f}")

    best_method = max(scores.keys(), key=lambda x: scores[x])
    print(f"  ✓ Recommended method: {best_method}")

    return best_method, scores

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Estimate s parameter for new geographic region')
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with distance data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file for estimated parameters')
    parser.add_argument('--plots-dir', '-p', default='.',
                       help='Directory for diagnostic plots (default: current)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation (slower but more robust)')

    args = parser.parse_args()

    print("=" * 80)
    print("📐 PARAMETER ESTIMATION FOR NEW GEOGRAPHIC REGION")
    print("=" * 80)

    # Load data
    beta_values = load_distance_data(args.input)

    # Estimate parameters using multiple methods
    print("\nEstimating parameters using multiple methods...")
    estimates = {}

    estimates['MLE'] = estimate_s_mle(beta_values)
    estimates['MoM'] = estimate_s_method_of_moments(beta_values)
    estimates['Robust'] = estimate_s_robust(beta_values)
    estimates['Bayesian'] = estimate_s_bayesian(beta_values)

    # Cross-validation if requested
    if args.cross_validate:
        print("\nPerforming cross-validation...")
        cv_results = cross_validate_estimates(beta_values)
        estimates['Cross-validation'] = cv_results

    # Validate estimates
    print("\nValidating parameter estimates...")
    validation_results = validate_estimates(beta_values, estimates)

    # Create diagnostic plots
    create_diagnostic_plots(beta_values, estimates, args.plots_dir)

    # Recommend best estimate
    best_method, scores = recommend_best_estimate(estimates, validation_results)

    # Prepare output
    output_data = {
        'input_file': args.input,
        'n_observations': len(beta_values),
        'data_summary': {
            'mean_beta': float(np.mean(beta_values)),
            'std_beta': float(np.std(beta_values)),
            'min_beta': float(np.min(beta_values)),
            'max_beta': float(np.max(beta_values))
        },
        'estimates': {method: {
            'value': params['value'],
            'standard_error': params['standard_error'],
            'method': params['method']
        } for method, params in estimates.items() if method != 'Cross-validation'},
        'validation': validation_results,
        'recommendation': {
            'best_method': best_method,
            'best_estimate': estimates[best_method]['value'],
            'best_std_error': estimates[best_method]['standard_error'],
            'scores': scores
        }
    }

    # Add cross-validation results if available
    if 'Cross-validation' in estimates:
        output_data['cross_validation'] = estimates['Cross-validation']

    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ PARAMETER ESTIMATION COMPLETE")
    print("=" * 80)
    print(f"📊 Analyzed {len(beta_values)} distance ratios")
    print(f"🎯 Recommended estimate: s = {estimates[best_method]['value']:.4f} ± {estimates[best_method]['standard_error']:.4f}")
    print(f"📁 Results saved to: {args.output}")
    print(f"📈 Diagnostic plots saved to: {args.plots_dir}")

    print("\n📋 Summary of all estimates:")
    for method, params in estimates.items():
        if method != 'Cross-validation':
            print(f"  {method}: {params['value']:.4f} ± {params['standard_error']:.4f}")

if __name__ == "__main__":
    main()