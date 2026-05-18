"""Audit of the R values used in the misallocation predictor.

Paper claim: 52-65 misallocations predicted with s=0.093.
My recompute with same s and consolidated R: 25.

Hypothesis: the paper computed R using the 46 PHYSICAL plants
(no consolidation), whereas the ASSIGNMENT analysis (which gives 59)
uses the 32 CONSOLIDATED plants. Inconsistent in the paper itself.

This script tests both R-value definitions and pinpoints the source.
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
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def predict(s, R_values):
    p = norm.cdf(-np.log(R_values) / (np.sqrt(2.0) * s))
    return float(p.sum())


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def main():
    banner("R values: physical plants (46) vs consolidated (32)")

    df_eu = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv")
    print(f"  Loaded {len(df_eu)} euclidean muni-plant rows")
    print(f"  Origins: {df_eu.InputID.nunique()}, "
          f"Plants (TargetID): {df_eu.TargetID.nunique()}")

    # --- (a) R from PHYSICAL plants (no consolidation) ---
    # Each muni-plant pair stays separate. If two physical plants are at
    # distance 0 from same point (like 'Ribera del Fresno' twice), each
    # row is a separate pair.
    R_phys = []
    for muni, grp in df_eu.groupby("InputID"):
        d = grp["Distance"].values
        d_sorted = np.sort(d)
        if len(d_sorted) >= 2 and d_sorted[0] > 0:
            R_phys.append(d_sorted[1] / d_sorted[0])
    R_phys = np.array([r for r in R_phys if np.isfinite(r)])
    print(f"  R_physical (no consolidation): n={len(R_phys)}, "
          f"median={np.median(R_phys):.4f}, mean={R_phys.mean():.4f}, "
          f"<=1.05 count: {(R_phys <= 1.05).sum()}")

    # --- (b) R from CONSOLIDATED plants ---
    # First aggregate by plant name (TargetID). Two rows with same name
    # become one (we take minimum distance to that named plant).
    df_eu_named = df_eu.groupby(["InputID", "TargetID"])["Distance"].min().reset_index()
    R_named = []
    for muni, grp in df_eu_named.groupby("InputID"):
        d = grp["Distance"].values
        d_sorted = np.sort(d)
        if len(d_sorted) >= 2 and d_sorted[0] > 0:
            R_named.append(d_sorted[1] / d_sorted[0])
    R_named = np.array([r for r in R_named if np.isfinite(r)])
    print(f"  R_named (1 entry per plant name): n={len(R_named)}, "
          f"median={np.median(R_named):.4f}, mean={R_named.mean():.4f}, "
          f"<=1.05 count: {(R_named <= 1.05).sum()}")

    # --- (c) R after applying explicit consolidation by name (15 km grouping) ---
    # The PLANT_CONSOLIDATION map uses numeric ids (1..46). The TargetID
    # column gives names. We need a name->canonical mapping. For the
    # documented groups (Cabezuela, Navalmoral, Escurial...), the
    # consolidation merges plants in NEARBY DIFFERENT towns. Hence
    # consolidation reduces effective destination count by approx 14.

    # Approximation: for each muni, group destinations by canonical
    # 'cluster' using a 15-km Euclidean threshold among plant locations.
    # This requires the plant coordinates.
    plantas = pd.read_csv(DATA / "coordenadas_plantas.csv", encoding="latin-1")
    print(f"\n  Loaded {len(plantas)} plant coords (note: only 23 in this CSV; "
          f"the 46 physical plants are elsewhere)")

    # Skip explicit (c) — we do not have a clean 46-plant coord table.
    # Use named consolidation as the proxy.

    # --- Predictions ---
    s_paper = 0.093
    print(f"\n  Predictions with s = {s_paper} (paper convention):")
    print(f"    R_physical:   {predict(s_paper, R_phys):.1f} (n={len(R_phys)})")
    print(f"    R_named:      {predict(s_paper, R_named):.1f} (n={len(R_named)})")

    # Bootstrap CI with R_physical
    rng = np.random.default_rng(42)
    boot = []
    pairs_paper = pd.read_csv(DATA / "detailed_ratios_analysis_filtered.csv")
    log_beta = np.log(pairs_paper.Ratio.values)
    for _ in range(2000):
        idx = rng.choice(len(log_beta), size=len(log_beta), replace=True)
        s_b = log_beta[idx].std(ddof=1)
        boot.append(predict(s_b, R_phys))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI (R_physical, paper-conv s): [{lo:.1f}, {hi:.1f}]")

    boot = []
    for _ in range(2000):
        idx = rng.choice(len(log_beta), size=len(log_beta), replace=True)
        s_b = log_beta[idx].std(ddof=1)
        boot.append(predict(s_b, R_named))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI (R_named,    paper-conv s): [{lo:.1f}, {hi:.1f}]")

    banner("Same with muni-planta beta (n=383, s=0.326)")
    canon = pd.read_csv(DATA / "municipios_canonical.csv")
    beta_mp = canon["beta_assigned"].dropna().values
    s_mp = np.log(beta_mp).std(ddof=1)
    print(f"  s_muni-planta (paper conv) = {s_mp:.4f}")
    print(f"    R_physical: {predict(s_mp, R_phys):.1f}")
    print(f"    R_named:    {predict(s_mp, R_named):.1f}")

    boot = []
    log_mp = np.log(beta_mp)
    for _ in range(2000):
        idx = rng.choice(len(log_mp), size=len(log_mp), replace=True)
        s_b = log_mp[idx].std(ddof=1)
        boot.append(predict(s_b, R_named))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI (R_named, muni-planta s):   [{lo:.1f}, {hi:.1f}]")

    boot = []
    for _ in range(2000):
        idx = rng.choice(len(log_mp), size=len(log_mp), replace=True)
        s_b = log_mp[idx].std(ddof=1)
        boot.append(predict(s_b, R_phys))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI (R_physical, muni-planta s):[{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
