"""How accurate is the misallocation prediction as a function of pilot-sample size?

tab:precision_tradeoffs claimed +-15 % / +-8 % / +-5 % error for a literature default s,
a 30-50 sample pilot and a 100+ sample stratified calibration. No script produced those
figures. This measures the two pilot rows directly.

Procedure. The reference is the full-data prediction: s_b estimated on all 383 assigned
routes, fed through the Theorem 1 predictor over the observed R values. For a pilot of
size k, draw k municipalities without replacement, estimate s from that pilot alone,
predict, and record the relative error against the reference. The reported band is the
central 90 % of the replicate errors, i.e. the +- figure a practitioner should expect.

The "literature default" row is NOT simulated here: it depends on which default value is
assumed, and inventing one would be the same failure the audit is trying to remove.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from run_case import CONSOLIDATION_KM, predict_lognormal  # noqa: E402

OD = ROOT / "pipeline" / "od"
TOL = "5"
SEED, N_REP = 42, 4000
SIZES = (30, 50, 100, 150, 200, 250, 300)


def canonical():
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                      encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
    MX, MY = mun.X.to_numpy(), mun.Y.to_numpy()
    pl = pd.read_csv(OD / "plantas_roster.csv", encoding="utf-8")
    PX, PY = pl.x.to_numpy(), pl.y.to_numpy()
    par = list(range(len(pl)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            if np.hypot(PX[i] - PX[j], PY[i] - PY[j]) <= CONSOLIDATION_KM * 1000:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
    grp = np.array([find(i) for i in range(len(pl))])
    grp = np.array([{g: k for k, g in enumerate(sorted(set(grp)))}[g] for g in grp])

    od = pd.read_csv(OD / f"od_operational_muniplant_tol{TOL}.csv", encoding="utf-8")
    C = od.pivot_table(index="origin_pos", columns="destination_pos", values="total_cost",
                       aggfunc="min").reindex(index=range(len(mun)),
                                              columns=range(len(pl))).to_numpy()
    beta, R = [], []
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        de_g = pd.Series(de).groupby(grp).min().sort_values()
        g1 = int(de_g.index[0])
        cand = np.where(grp == g1)[0]
        p = cand[np.argmin(de[cand])]
        if de[p] > 0 and np.isfinite(C[i, p]):
            beta.append(C[i, p] / de[p])
            if len(de_g) >= 2:
                R.append(de_g.iloc[1] / de_g.iloc[0])
    return np.array(beta), np.array(R)


def main() -> None:
    beta, R = canonical()
    Rv = R[np.isfinite(R)]
    s_full = float(np.log(beta).std(ddof=1))
    ref = predict_lognormal(s_full, Rv)
    print(f"n = {len(beta)}   s_full = {s_full:.4f}   reference prediction = {ref:.2f}\n")

    rng = np.random.default_rng(SEED)
    out = {}
    print(f"{'pilot k':>8}{'s median':>11}{'pred median':>13}{'90% band of rel. error':>26}")
    print("-" * 60)
    for k in SIZES:
        errs, ss = [], []
        for _ in range(N_REP):
            idx = rng.choice(len(beta), size=k, replace=False)
            s_k = float(np.log(beta[idx]).std(ddof=1))
            ss.append(s_k)
            errs.append(100.0 * (predict_lognormal(s_k, Rv) - ref) / ref)
        errs = np.array(errs)
        lo, hi = np.percentile(errs, [5, 95])
        band = float(max(abs(lo), abs(hi)))
        out[k] = {"s_median": float(np.median(ss)), "rel_err_p5": float(lo),
                  "rel_err_p95": float(hi), "band_pct": round(band, 1)}
        print(f"{k:8d}{np.median(ss):11.4f}{np.median([predict_lognormal(x, Rv) for x in ss]):13.2f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}] -> +-{band:.0f}%':>26}")

    (ROOT / "pipeline" / "calibration_precision.json").write_text(
        json.dumps({"n": int(len(beta)), "s_full": s_full, "reference_prediction": float(ref),
                    "n_replicates": N_REP, "by_pilot_size": out}, indent=2), encoding="utf-8")
    print("\nwrote pipeline/calibration_precision.json")


if __name__ == "__main__":
    main()
