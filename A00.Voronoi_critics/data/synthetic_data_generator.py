#!/usr/bin/env python3
"""
Synthetic Data Generator for Voronoi Probabilistic Framework
==========================================================

Generates synthetic β scaling factors matching the statistical properties
observed in the Extremadura case study. This enables replication of results
without access to sensitive geographic data.

Based on empirical parameters from paper:
- μ = 0.166 (log-normal location parameter)
- σ = 0.093 (log-normal scale parameter)
- s = 0.093 (geographic complexity parameter)

Author: Voronoi Framework Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os

def generate_beta_factors(n_samples=9240, mu_log=0.166, sigma_log=0.093,
                         seed=42, realistic_bounds=True):
    """
    Generate synthetic network scaling factors β

    Parameters:
    -----------
    n_samples : int
        Number of municipality-facility pairs (default: 9240 from paper)
    mu_log : float
        Log-normal location parameter (default: 0.166 from paper)
    sigma_log : float
        Log-normal scale parameter (default: 0.093 from paper)
    seed : int
        Random seed for reproducibility
    realistic_bounds : bool
        Whether to apply realistic bounds [0.8, 5.0]

    Returns:
    --------
    numpy.ndarray
        Array of synthetic β values
    """
    np.random.seed(seed)

    # Generate log-normal distributed β factors
    beta_values = np.random.lognormal(mu_log, sigma_log, n_samples)

    # Apply realistic bounds if requested
    if realistic_bounds:
        beta_values = np.clip(beta_values, 0.8, 5.0)

    return beta_values

def generate_municipality_data(n_municipalities=383, n_facilities=46,
                             mu_log=0.166, sigma_log=0.093, seed=42):
    """
    Generate complete synthetic dataset with municipality-facility assignments

    Parameters:
    -----------
    n_municipalities : int
        Number of municipalities (default: 383 from Extremadura)
    n_facilities : int
        Number of waste treatment facilities (default: 46 from study)
    mu_log, sigma_log : float
        Log-normal parameters
    seed : int
        Random seed

    Returns:
    --------
    pandas.DataFrame
        Complete synthetic dataset
    """
    np.random.seed(seed)

    # Generate municipality IDs
    municipality_ids = [f"MUN_{i:03d}" for i in range(1, n_municipalities + 1)]

    # Generate facility IDs
    facility_ids = [f"FAC_{i:02d}" for i in range(1, n_facilities + 1)]

    # Create all possible pairs
    all_pairs = []
    for mun_id in municipality_ids:
        for fac_id in facility_ids:
            all_pairs.append((mun_id, fac_id))

    # Sample subset to match paper (9,240 pairs)
    n_pairs = min(9240, len(all_pairs))
    selected_pairs = np.random.choice(len(all_pairs), size=n_pairs, replace=False)

    # Generate data for selected pairs
    data_records = []
    beta_values = generate_beta_factors(n_pairs, mu_log, sigma_log, seed)

    for i, pair_idx in enumerate(selected_pairs):
        mun_id, fac_id = all_pairs[pair_idx]

        # Generate synthetic coordinates (normalized to [0,1])
        mun_x, mun_y = np.random.uniform(0, 1, 2)
        fac_x, fac_y = np.random.uniform(0, 1, 2)

        # Generate Euclidean distance (km)
        euclidean_dist = np.random.uniform(5, 150)  # Realistic range

        # Network distance from β factor
        network_dist = beta_values[i] * euclidean_dist

        # Binary assignment (Voronoi vs optimal)
        # Misallocation occurs when β > threshold
        is_misallocated = beta_values[i] > 1.25  # Conservative threshold

        data_records.append({
            'Municipality_ID': mun_id,
            'Facility_ID': fac_id,
            'Municipality_X': mun_x,
            'Municipality_Y': mun_y,
            'Facility_X': fac_x,
            'Facility_Y': fac_y,
            'Euclidean_Distance_km': euclidean_dist,
            'Network_Distance_km': network_dist,
            'Beta_Factor': beta_values[i],
            'Is_Misallocated': is_misallocated,
            'Voronoi_Assignment': fac_id,
            'Optimal_Assignment': fac_id if not is_misallocated else f"FAC_{np.random.randint(1, n_facilities):02d}"
        })

    return pd.DataFrame(data_records)

def generate_extremadura_like_data(output_dir=".", add_noise=True, seed=42):
    """
    Generate complete Extremadura-like dataset

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    add_noise : bool
        Whether to add realistic noise to parameters
    seed : int
        Random seed
    """
    np.random.seed(seed)

    print("Generating Extremadura-like synthetic dataset...")

    # Base parameters from paper
    base_mu = 0.166
    base_sigma = 0.093
    base_s = 0.093

    # Add realistic parameter variation if requested
    if add_noise:
        mu_noise = np.random.normal(0, 0.01)  # Small variation
        sigma_noise = np.random.normal(0, 0.005)
        s_noise = np.random.normal(0, 0.005)

        mu_log = base_mu + mu_noise
        sigma_log = max(0.01, base_sigma + sigma_noise)  # Ensure positive
        s_param = max(0.01, base_s + s_noise)
    else:
        mu_log = base_mu
        sigma_log = base_sigma
        s_param = base_s

    # Generate complete dataset
    df_complete = generate_municipality_data(
        n_municipalities=383,
        n_facilities=46,
        mu_log=mu_log,
        sigma_log=sigma_log,
        seed=seed
    )

    # Summary statistics
    n_misallocated = df_complete['Is_Misallocated'].sum()
    misallocation_rate = n_misallocated / len(df_complete) * 100

    print(f"Generated dataset summary:")
    print(f"  - Total municipality-facility pairs: {len(df_complete)}")
    print(f"  - Misallocated pairs: {n_misallocated} ({misallocation_rate:.1f}%)")
    print(f"  - Mean β factor: {df_complete['Beta_Factor'].mean():.3f}")
    print(f"  - Std β factor: {df_complete['Beta_Factor'].std():.3f}")
    print(f"  - Parameters: μ={mu_log:.3f}, σ={sigma_log:.3f}, s={s_param:.3f}")

    # Save complete dataset
    output_path = os.path.join(output_dir, "extremadura_anonymized.csv")
    df_complete.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Generate geographic coordinates file
    municipalities = df_complete.groupby('Municipality_ID').first()
    coordinates_df = municipalities[['Municipality_X', 'Municipality_Y']].copy()
    coordinates_df.index.name = 'Municipality_ID'
    coordinates_df.columns = ['Longitude', 'Latitude']  # Normalized coordinates

    coord_path = os.path.join(output_dir, "geographic_coordinates.csv")
    coordinates_df.to_csv(coord_path)
    print(f"Saved: {coord_path}")

    # Generate parameter file
    params = {
        'mu_log': mu_log,
        'sigma_log': sigma_log,
        's_parameter': s_param,
        'n_municipalities': 383,
        'n_facilities': 46,
        'misallocation_rate': misallocation_rate,
        'seed_used': seed
    }

    param_path = os.path.join(output_dir, "generation_parameters.json")
    import json
    with open(param_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {param_path}")

    return df_complete, coordinates_df, params

def create_validation_plots(df, output_dir="."):
    """Create validation plots to verify synthetic data properties"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: β factor distribution
    ax1 = axes[0, 0]
    ax1.hist(df['Beta_Factor'], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Overlay theoretical log-normal
    x = np.linspace(df['Beta_Factor'].min(), df['Beta_Factor'].max(), 1000)
    mu_emp = np.log(df['Beta_Factor']).mean()
    sigma_emp = np.log(df['Beta_Factor']).std()
    theoretical = stats.lognorm.pdf(x, s=sigma_emp, scale=np.exp(mu_emp))
    ax1.plot(x, theoretical, 'r-', linewidth=2, label=f'Log-Normal(μ={mu_emp:.3f}, σ={sigma_emp:.3f})')

    ax1.set_xlabel('β Scaling Factor')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Network Scaling Factors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance relationship
    ax2 = axes[0, 1]
    sample_indices = np.random.choice(len(df), size=min(1000, len(df)), replace=False)
    ax2.scatter(df.iloc[sample_indices]['Euclidean_Distance_km'],
               df.iloc[sample_indices]['Network_Distance_km'],
               alpha=0.6, s=20, color='green')

    # Perfect correlation line
    max_dist = max(df['Euclidean_Distance_km'].max(), df['Network_Distance_km'].max())
    ax2.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='Perfect correlation')

    ax2.set_xlabel('Euclidean Distance (km)')
    ax2.set_ylabel('Network Distance (km)')
    ax2.set_title('Network vs Euclidean Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Misallocation analysis
    ax3 = axes[1, 0]
    misalloc_counts = df['Is_Misallocated'].value_counts()
    colors = ['lightgreen', 'salmon']
    labels = ['Correct Assignment', 'Misallocated']
    ax3.pie(misalloc_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Assignment Quality Distribution')

    # Plot 4: β factor vs misallocation
    ax4 = axes[1, 1]
    correct_beta = df[~df['Is_Misallocated']]['Beta_Factor']
    misalloc_beta = df[df['Is_Misallocated']]['Beta_Factor']

    ax4.hist(correct_beta, bins=30, alpha=0.7, label='Correct', color='lightgreen', density=True)
    ax4.hist(misalloc_beta, bins=30, alpha=0.7, label='Misallocated', color='salmon', density=True)
    ax4.axvline(1.25, color='red', linestyle='--', linewidth=2, label='Threshold (1.25)')

    ax4.set_xlabel('β Scaling Factor')
    ax4.set_ylabel('Density')
    ax4.set_title('β Distribution by Assignment Quality')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "synthetic_data_validation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved validation plots: {plot_path}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Generate synthetic data for Voronoi framework')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--add-noise', action='store_true',
                       help='Add realistic parameter variation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip validation plots generation')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate data
    df, coords, params = generate_extremadura_like_data(
        output_dir=args.output_dir,
        add_noise=args.add_noise,
        seed=args.seed
    )

    # Create validation plots unless disabled
    if not args.no_plots:
        create_validation_plots(df, args.output_dir)

    print(f"\nSynthetic data generation complete!")
    print(f"Files saved in: {args.output_dir}")
    print(f"Ready for reproducibility analysis.")

if __name__ == "__main__":
    main()