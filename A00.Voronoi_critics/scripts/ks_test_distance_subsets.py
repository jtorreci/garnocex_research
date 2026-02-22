#!/usr/bin/env python3
"""
KS test on distance-filtered subsets of beta values.

For the n=383 plant-municipality pairs (each municipality to its assigned plant),
filter by d_euclidean thresholds and show how log-normal fit improves at short distances.

Also test on the k=3 and k=5 nearest Euclidean pairs per municipality.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(".").resolve().parent
TABLES = BASE / "tables"

# Load data
df_euc = pd.read_csv(TABLES / "D_euclidea_plantas_clean.csv")
df_net = pd.read_csv(BASE / "codigo" / "tablas" / "D_real_plantas_clean_corrected.csv")

df_euc.columns = ["municipality", "plant_name", "d_euclidean"]
df_net.columns = ["municipality", "plant_name", "entry_cost", "network_cost", "exit_cost", "d_network"]

df_euc["plant_idx"] = df_euc.groupby("municipality").cumcount()
df_net["plant_idx"] = df_net.groupby("municipality").cumcount()

df = df_euc.merge(df_net[["municipality", "plant_idx", "d_network"]],
                  on=["municipality", "plant_idx"])

# Compute beta
df["beta"] = df["d_network"] / df["d_euclidean"]
df = df[df["beta"] >= 1.0].copy()  # physical constraint

print(f"Total pairs (beta >= 1): {len(df)}")
print(f"Euclidean distance range: {df['d_euclidean'].min():.0f} - {df['d_euclidean'].max():.0f} m")
print()

# --- Method 1: Filter ALL pairs by d_euclidean thresholds ---
print("=" * 75)
print("METHOD 1: All pairs filtered by Euclidean distance threshold")
print("=" * 75)

thresholds_km = [10, 20, 30, 50, 75, 100, "All"]
for thr in thresholds_km:
    if thr == "All":
        subset = df["beta"].values
        label = f"All pairs"
    else:
        subset = df[df["d_euclidean"] <= thr * 1000]["beta"].values
        label = f"d_euc <= {thr} km"
    
    if len(subset) < 10:
        print(f"  {label}: n={len(subset)} (too few)")
        continue
    
    log_beta = np.log(subset)
    m_hat = np.mean(log_beta)
    s_hat = np.std(log_beta, ddof=1)
    
    ks_stat, p_val = stats.kstest(
        subset,
        lambda x: stats.lognorm.cdf(x, s=s_hat, scale=np.exp(m_hat))
    )
    
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    quality = "Good" if p_val > 0.05 else ("Acceptable" if p_val > 0.01 else "Poor")
    
    print(f"  {label:20s}  n={len(subset):6d}  "
          f"KS={ks_stat:.4f}  p={p_str:>8s}  m={m_hat:.3f}  s={s_hat:.3f}  [{quality}]")

print()

# --- Method 2: k-nearest pairs per municipality ---
print("=" * 75)
print("METHOD 2: k-nearest Euclidean pairs per municipality")
print("=" * 75)

for k in [1, 2, 3, 5, 10, 46]:
    label = f"k={k} nearest" if k < 46 else "All (k=46)"
    k_nearest = df.groupby("municipality").apply(
        lambda g: g.nsmallest(k, "d_euclidean")
    ).reset_index(drop=True)
    
    subset = k_nearest["beta"].values
    log_beta = np.log(subset)
    m_hat = np.mean(log_beta)
    s_hat = np.std(log_beta, ddof=1)
    
    ks_stat, p_val = stats.kstest(
        subset,
        lambda x: stats.lognorm.cdf(x, s=s_hat, scale=np.exp(m_hat))
    )
    
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    quality = "Good" if p_val > 0.05 else ("Acceptable" if p_val > 0.01 else "Poor")
    
    max_d = k_nearest["d_euclidean"].max() / 1000
    mean_d = k_nearest["d_euclidean"].mean() / 1000
    
    print(f"  {label:15s}  n={len(subset):6d}  "
          f"KS={ks_stat:.4f}  p={p_str:>8s}  m={m_hat:.3f}  s={s_hat:.3f}  "
          f"d_max={max_d:.0f}km  d_mean={mean_d:.0f}km  [{quality}]")

print()

# --- Method 3: Distribution comparison at short distances ---
print("=" * 75)
print("METHOD 3: Log-Normal vs Gamma vs Weibull at d_euc <= 30km")
print("=" * 75)

short = df[df["d_euclidean"] <= 30000]["beta"].values
print(f"  n = {len(short)}")

for dist_name, dist_obj in [("Log-Normal", stats.lognorm), ("Gamma", stats.gamma), ("Weibull", stats.weibull_min)]:
    params = dist_obj.fit(short, floc=0) if dist_name != "Log-Normal" else None
    
    if dist_name == "Log-Normal":
        log_b = np.log(short)
        m, s = np.mean(log_b), np.std(log_b, ddof=1)
        ks_stat, p_val = stats.kstest(short, lambda x: stats.lognorm.cdf(x, s=s, scale=np.exp(m)))
    elif dist_name == "Gamma":
        ks_stat, p_val = stats.kstest(short, lambda x: dist_obj.cdf(x, *params))
    else:
        ks_stat, p_val = stats.kstest(short, lambda x: dist_obj.cdf(x, *params))
    
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    quality = "Good" if p_val > 0.05 else ("Acceptable" if p_val > 0.01 else "Poor")
    print(f"  {dist_name:12s}  KS={ks_stat:.4f}  p={p_str:>8s}  [{quality}]")
