"""Audit of the lognormal-fit pipeline.

Goals:
    1. Confirm whether the published `detailed_ratios_analysis_filtered.csv`
       (n=9112) reproduces the paper's claim m̂=0.166, ŝ=0.093.
    2. Check whether a proper MLE on muni-planta (n=383) data, with the
       paper's parameter convention, would give predictions consistent
       with the framework's expectations.
    3. Check the 59 vs 61 gap by varying distance-saving thresholds.

The aim is to distinguish 'modelling problem' from 'analysis problem'.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def fit_lognormal_paper_convention(beta):
    """Fit lognormal as the paper appears to do: ln(beta) ~ N(m, s^2).

    Note: the paper's parametrization is m = E[ln beta], s = std(ln beta).
    This corresponds to scipy.stats.lognorm with shape=s and scale=exp(m),
    NOT with location=1 (the paper does not enforce beta>=1 in the fit).
    """
    log_beta = np.log(beta)
    m = log_beta.mean()
    s = log_beta.std(ddof=1)
    # Equivalent scipy lognorm: shape=s, loc=0, scale=exp(m)
    ll = stats.lognorm.logpdf(beta, s, 0.0, np.exp(m)).sum()
    return m, s, ll


def predict_misallocations_lognormal_closed(s, R_values):
    """Theorem 1: P[misalloc | R] = Phi(-ln R / (sqrt(2) s)) summed over munis."""
    p = norm.cdf(-np.log(R_values) / (np.sqrt(2.0) * s))
    return float(p.sum())


def main():
    banner("AUDIT 1 — Reproduce paper's m=0.166, s=0.093 from filtered_ratios")

    pairs_paper = pd.read_csv(DATA / "detailed_ratios_analysis_filtered.csv")
    print(f"  Loaded n={len(pairs_paper):,} muni-muni filtered pairs")
    print(f"  beta: mean={pairs_paper.Ratio.mean():.4f}  std={pairs_paper.Ratio.std():.4f}")

    m, s, ll = fit_lognormal_paper_convention(pairs_paper.Ratio.values)
    print(f"  PAPER convention (no floc): m={m:.4f}  s={s:.4f}  ll={ll:.1f}")
    print(f"  Paper claim:                m=0.166  s=0.093")
    delta_s = abs(s - 0.093)
    print(f"  Match? {'YES' if delta_s < 0.005 else 'NO (delta=%.4f)' % delta_s}")

    # Also try with floc=1 (β>=1 constraint, what I had used)
    s_floc1, loc, scale = stats.lognorm.fit(pairs_paper.Ratio.values, floc=1.0)
    print(f"  With floc=1: s={s_floc1:.4f}, scale={scale:.4f}")
    print(f"  -> floc=1 inflates s by {s_floc1/s:.1f}x (the issue in A.2)")

    banner("AUDIT 2 — Predicted misallocations with paper's s=0.093")

    # R values: d2/d1 to consolidated plants (32 effective)
    df_eu = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv")
    R_values = []
    for muni, grp in df_eu.groupby("InputID"):
        d_min = grp.groupby("TargetID")["Distance"].min().values
        d_sorted = np.sort(d_min)
        if len(d_sorted) >= 2 and d_sorted[0] > 0:
            R_values.append(d_sorted[1] / d_sorted[0])
    R_values = np.array([r for r in R_values if np.isfinite(r)])
    print(f"  R values: n={len(R_values)}, median={np.median(R_values):.4f}")

    pred_paper = predict_misallocations_lognormal_closed(0.093, R_values)
    pred_my_munimuni = predict_misallocations_lognormal_closed(s, R_values)
    pred_my_floc1 = predict_misallocations_lognormal_closed(s_floc1, R_values)
    print(f"  Predicted misallocations (s=0.093, paper):     {pred_paper:.1f}")
    print(f"  Predicted misallocations (s={s:.3f}, MLE no floc): {pred_my_munimuni:.1f}")
    print(f"  Predicted misallocations (s={s_floc1:.3f}, MLE floc=1): {pred_my_floc1:.1f}")
    print(f"  Observed (consolidated): 61")
    print(f"  Paper claim: 52-65 CI contains 59")

    # Bootstrap CI for s=0.093 prediction
    rng = np.random.default_rng(42)
    boot_preds = []
    for _ in range(1000):
        boot = rng.choice(pairs_paper.Ratio.values, size=len(pairs_paper), replace=True)
        m_b, s_b, _ = fit_lognormal_paper_convention(boot)
        boot_preds.append(predict_misallocations_lognormal_closed(s_b, R_values))
    boot_preds = np.array(boot_preds)
    lo, hi = np.percentile(boot_preds, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for prediction with paper-convention s:")
    print(f"  [{lo:.1f}, {hi:.1f}]  (paper claims [52, 65])")

    banner("AUDIT 3 — Lognormal fit on muni-planta (n=383)")

    canon = pd.read_csv(DATA / "municipios_canonical.csv")
    beta_mp = canon["beta_assigned"].dropna().values
    beta_mp = beta_mp[beta_mp >= 1.0]
    print(f"  n = {len(beta_mp)}, mean = {beta_mp.mean():.4f}, std = {beta_mp.std():.4f}")
    m_mp, s_mp, ll_mp = fit_lognormal_paper_convention(beta_mp)
    s_mp_floc1, _, _ = stats.lognorm.fit(beta_mp, floc=1.0)
    print(f"  PAPER convention: m={m_mp:.4f}  s={s_mp:.4f}")
    print(f"  floc=1 convention: s={s_mp_floc1:.4f}")
    pred_mp = predict_misallocations_lognormal_closed(s_mp, R_values)
    pred_mp_floc1 = predict_misallocations_lognormal_closed(s_mp_floc1, R_values)
    print(f"  Predicted misalloc with muni-planta s={s_mp:.3f}: {pred_mp:.1f}")
    print(f"  Predicted misalloc with muni-planta s={s_mp_floc1:.3f}: {pred_mp_floc1:.1f}")

    banner("AUDIT 4 — Investigate 59 vs 61: distance-saving threshold filter")

    # Re-derive misallocations and per-muni saving
    df_eu_p = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv")
    df_re_p = pd.read_csv(DATA / "D_real_plantas_clean.csv")
    df_re_p_min = df_re_p.groupby(["origin_id", "destination_id"])["total_cost"].min().reset_index()
    df_re_p_min.columns = ["origin", "destination", "d_re"]
    df_eu_p_min = df_eu_p.groupby(["InputID", "TargetID"])["Distance"].min().reset_index()
    df_eu_p_min.columns = ["origin", "destination", "d_eu"]

    # Voronoi (eucl) and network (real) assignments per muni
    eu_min_idx = df_eu_p_min.groupby("origin")["d_eu"].idxmin()
    eu_assign = df_eu_p_min.loc[eu_min_idx, ["origin", "destination", "d_eu"]].rename(
        columns={"destination": "voronoi_dest", "d_eu": "d_eu_assigned"})
    re_min_idx = df_re_p_min.groupby("origin")["d_re"].idxmin()
    re_assign = df_re_p_min.loc[re_min_idx, ["origin", "destination", "d_re"]].rename(
        columns={"destination": "network_dest", "d_re": "d_net_optimal"})
    asgn = eu_assign.merge(re_assign, on="origin")

    # Network distance under Voronoi assignment
    df_re_lookup = df_re_p_min.set_index(["origin", "destination"])["d_re"]
    asgn["d_net_voronoi"] = asgn.apply(
        lambda r: df_re_lookup.get((r["origin"], r["voronoi_dest"]), np.nan), axis=1)
    asgn["misallocated_raw"] = (asgn["voronoi_dest"] != asgn["network_dest"]).astype(int)

    # Distance saving (network) when reassigning
    asgn["saving_m"] = asgn["d_net_voronoi"] - asgn["d_net_optimal"]
    print(f"  Total munis: {len(asgn)}")
    print(f"  Raw misallocations (no consolidation): {asgn['misallocated_raw'].sum()}")

    # Apply consolidation
    PLANT_CONSOLIDATION = {
        2: 1, 12: 1, 17: 3, 15: 6, 40: 19, 32: 20,
        23: 22, 26: 22, 28: 22, 34: 27, 35: 27, 37: 27, 38: 36, 43: 36,
    }
    # NOTE: the destinations here are NAMES not numbers. Need to know
    # whether consolidation by ID can be mapped to names. Skip — we apply
    # the canonical 'misallocated' from municipios_canonical.csv directly.
    canon_mis = canon[["municipio", "misallocated"]].copy()
    asgn["_norm"] = asgn["origin"].str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.strip()
    canon_mis["_norm"] = canon_mis["municipio"].str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.strip()
    asgn = asgn.merge(canon_mis[["_norm", "misallocated"]], on="_norm", how="left")
    asgn["misallocated"] = asgn["misallocated"].fillna(0).astype(int)

    print(f"  Consolidated misallocations (from canonical): {asgn['misallocated'].sum()}")

    # Distribution of distance saving among consolidated misallocations
    miss = asgn[asgn["misallocated"] == 1].dropna(subset=["saving_m"])
    print(f"\n  Among the {len(miss)} consolidated misallocations:")
    print(f"  saving_m  median={miss['saving_m'].median():.0f}  "
          f"mean={miss['saving_m'].mean():.0f}  "
          f"min={miss['saving_m'].min():.0f}  max={miss['saving_m'].max():.0f}")
    print(f"  saving<100m   : {(miss['saving_m'] < 100).sum()} munis")
    print(f"  saving<500m   : {(miss['saving_m'] < 500).sum()} munis")
    print(f"  saving<1000m  : {(miss['saving_m'] < 1000).sum()} munis")
    print(f"  saving<2000m  : {(miss['saving_m'] < 2000).sum()} munis")

    # Try thresholds: do any give 59?
    print(f"\n  Threshold sweep (saving >= T):")
    for t in [0, 100, 200, 500, 1000, 1500, 2000, 3000, 5000]:
        n_filt = (miss["saving_m"] >= t).sum()
        print(f"    saving >= {t:>5} m  ->  {n_filt} misallocations  "
              f"{'<-- 59' if n_filt == 59 else ''}")

    print(f"\n  Average saving (paper claims 6.4 km): "
          f"{miss['saving_m'].mean()/1000:.1f} km")


if __name__ == "__main__":
    main()
