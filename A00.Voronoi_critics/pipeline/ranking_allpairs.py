"""Distributional ranking on the all-pairs (municipality-to-municipality) domain.

tab:ranking_munimuni had no stored source: canonical_numbers.json records only n, m, s
and beta_min for this domain, not the W1 / A2 / KS ranking that the table reports. This
recomputes it with the same fitters and metrics as the predictor-domain ranking in
canonical_numbers.py, so the two tables are produced by identical machinery.

The all-pairs domain carries no misallocation prediction: it characterizes directional
dispersion, and the Theorem 1 predictor is not evaluated on it. Fit metrics only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from run_case import FITTERS, anderson_darling, wasserstein1  # noqa: E402

OD = ROOT / "pipeline" / "od"
CASE = "operational"


def main() -> None:
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                      encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
    MX, MY = mun.X.to_numpy(), mun.Y.to_numpy()

    d = pd.read_csv(OD / f"od_{CASE}_munimuni_tol0.csv", encoding="utf-8").dropna(
        subset=["total_cost"])
    d = d[d.origin_pos != d.destination_pos]
    dd = np.hypot(MX[d.origin_pos.to_numpy()] - MX[d.destination_pos.to_numpy()],
                  MY[d.origin_pos.to_numpy()] - MY[d.destination_pos.to_numpy()])
    v = d.total_cost.to_numpy() / dd
    v = v[np.isfinite(v) & (v > 0)]

    print(f"all-pairs domain: n = {len(v):,}   m = {np.log(v).mean():.4f}   "
          f"s = {np.log(v).std(ddof=1):.4f}   beta_min = {v.min():.4f}")
    assert (v >= 1).all(), "beta < 1 on the all-pairs domain: check the cost convention"
    print("beta >= 1 throughout, as Definition 1 requires.\n")

    rows = []
    for name, dist in FITTERS.items():
        params = dist.fit(v)
        rows.append({
            "distribution": name,
            "W1": round(wasserstein1(v, dist, params), 4),
            "A2": round(anderson_darling(v, dist, params), 1),
            "KS": round(float(stats.kstest(v, dist.cdf, args=params).statistic), 4),
        })
    rows.sort(key=lambda r: r["W1"])

    print(f"{'distribution':30}{'W1':>9}{'A2':>12}{'KS':>9}")
    print("-" * 60)
    for r in rows:
        print(f"{r['distribution']:30}{r['W1']:9.4f}{r['A2']:12.1f}{r['KS']:9.4f}")

    print("\n--- cuerpo LaTeX ---")
    for r in rows:
        a2 = f"{r['A2']:,.1f}".replace(",", "{,}")
        name = (f"\\textbf{{{r['distribution']}}}" if r["distribution"] == "Lognormal"
                else r["distribution"])
        print(f"{name} & ${r['W1']:.3f}$ & ${a2}$ & ${r['KS']:.3f}$ \\\\")

    (ROOT / "pipeline" / "ranking_allpairs.json").write_text(
        json.dumps({"domain": f"{CASE}_munimuni", "n": int(len(v)),
                    "m": float(np.log(v).mean()), "s": float(np.log(v).std(ddof=1)),
                    "beta_min": float(v.min()), "ranking": rows}, indent=2),
        encoding="utf-8")
    print("\nwrote pipeline/ranking_allpairs.json")


if __name__ == "__main__":
    main()
