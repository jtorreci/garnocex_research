#!/usr/bin/env python3
"""
Compute KS test results for Table 3 subsets:
  - Full sample (n=383 nearest-plant pairs)
  - k=2 nearest plants per municipality
  - d_euc <= 10 km
All three distributions: Log-Normal, Gamma, Weibull.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # support_material/
TABLES = BASE / "tables"

# Load data
df_euc = pd.read_csv(TABLES / "D_euclidea_plantas_clean.csv")
df_net = pd.read_csv(BASE / "codigo" / "tablas" / "D_real_plantas_clean_corrected.csv")

df_euc.columns = ["municipality", "plant_name", "d_euclidean"]
df_net.columns = ["municipality", "plant_name",
                   "entry_cost", "network_cost", "exit_cost", "d_network"]

df_euc["plant_idx"] = df_euc.groupby("municipality").cumcount()
df_net["plant_idx"] = df_net.groupby("municipality").cumcount()

df = df_euc.merge(df_net[["municipality", "plant_idx", "d_network"]],
                  on=["municipality", "plant_idx"])
df["beta"] = df["d_network"] / df["d_euclidean"]
df = df[df["beta"] >= 1.0].copy()


def fit_and_test(beta_values, label):
    """Fit LN, Gamma, Weibull and report KS stats + parameters."""
    n = len(beta_values)
    print(f"\n{'='*70}")
    print(f"{label}  (n = {n})")
    print(f"{'='*70}")

    # --- Log-Normal ---
    log_b = np.log(beta_values)
    m_hat = np.mean(log_b)
    s_hat = np.std(log_b, ddof=1)
    ks_ln, p_ln = stats.kstest(
        beta_values,
        lambda x: stats.lognorm.cdf(x, s=s_hat, scale=np.exp(m_hat)))
    p_ln_s = f"{p_ln:.4f}" if p_ln >= 0.0001 else "<0.0001"
    q_ln = "Good" if p_ln > 0.05 else ("Acceptable" if p_ln > 0.01 else "Poor")
    print(f"  Log-Normal  KS={ks_ln:.4f}  p={p_ln_s:>8s}  "
          f"m={m_hat:.3f}  s={s_hat:.3f}  [{q_ln}]")

    # --- Gamma ---
    a_g, loc_g, scale_g = stats.gamma.fit(beta_values, floc=0)
    ks_g, p_g = stats.kstest(
        beta_values,
        lambda x: stats.gamma.cdf(x, a_g, loc=0, scale=scale_g))
    p_g_s = f"{p_g:.4f}" if p_g >= 0.0001 else "<0.0001"
    q_g = "Good" if p_g > 0.05 else ("Acceptable" if p_g > 0.01 else "Poor")
    print(f"  Gamma       KS={ks_g:.4f}  p={p_g_s:>8s}  "
          f"k={a_g:.3f}  theta={scale_g:.3f}  [{q_g}]")

    # --- Weibull ---
    c_w, loc_w, scale_w = stats.weibull_min.fit(beta_values, floc=0)
    ks_w, p_w = stats.kstest(
        beta_values,
        lambda x: stats.weibull_min.cdf(x, c_w, loc=0, scale=scale_w))
    p_w_s = f"{p_w:.4f}" if p_w >= 0.0001 else "<0.0001"
    q_w = "Good" if p_w > 0.05 else ("Acceptable" if p_w > 0.01 else "Poor")
    print(f"  Weibull     KS={ks_w:.4f}  p={p_w_s:>8s}  "
          f"k={c_w:.3f}  lambda={scale_w:.3f}  [{q_w}]")


# --- Subset 1: Full sample (k=1 nearest) ---
nearest = df.groupby("municipality").apply(
    lambda g: g.nsmallest(1, "d_euclidean")).reset_index(drop=True)
fit_and_test(nearest["beta"].values, "Full sample (k=1 nearest)")

# --- Subset 2: k=2 nearest ---
k2 = df.groupby("municipality").apply(
    lambda g: g.nsmallest(2, "d_euclidean")).reset_index(drop=True)
fit_and_test(k2["beta"].values, "k=2 nearest plants")

# --- Subset 3: d_euc <= 10 km (from k=1 nearest) ---
short = nearest[nearest["d_euclidean"] <= 10000]
fit_and_test(short["beta"].values, "d_euc <= 10 km (nearest plant)")
