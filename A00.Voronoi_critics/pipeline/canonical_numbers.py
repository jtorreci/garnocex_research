"""Generate the definitive figure set for the manuscript, under configuration D13.

Writes pipeline/canonical_numbers.json. Every number that goes into manuscript.tex
should be traceable to a key in that file.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_case import (CONSOLIDATION_KM, FITTERS, N_BOOTSTRAP, SEED,  # noqa: E402
                      anderson_darling, demojibake, predict_general,
                      predict_lognormal, wasserstein1)

ROOT = Path(__file__).resolve().parent.parent
OD = ROOT / "pipeline" / "od"
TOL = "5"
Q_STAR = 0.10

mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                  encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
mun["name"] = mun.NOMBRE.map(demojibake)
MX, MY = mun.X.to_numpy(), mun.Y.to_numpy()
pl = pd.read_csv(OD / "plantas_roster.csv", encoding="utf-8")
PX, PY = pl.x.to_numpy(), pl.y.to_numpy()

par = list(range(len(pl)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
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
                   aggfunc="min").reindex(index=range(len(mun)), columns=range(len(pl))).to_numpy()

beta_assigned, R, kappa, mis, mis_names = [], [], [], 0, []
centre = None
for i in range(len(mun)):
    de = np.hypot(PX - MX[i], PY - MY[i]); dr = C[i]
    gm = pd.Series(de).groupby(grp).min().sort_values()
    g1, g2 = int(gm.index[0]), int(gm.index[1])
    d1, d2 = float(gm.iloc[0]), float(gm.iloc[1])
    c1 = np.where(grp == g1)[0]; p1 = c1[np.argmin(de[c1])]
    c2 = np.where(grp == g2)[0]; p2 = c2[np.argmin(de[c2])]
    beta_assigned.append(dr[p1] / de[p1] if de[p1] > 0 and np.isfinite(dr[p1]) else np.nan)
    R.append(d2 / d1 if d1 > 0 else np.nan)
    p = np.array([MX[i], MY[i]])
    v1, v2 = np.array([PX[p1], PY[p1]]) - p, np.array([PX[p2], PY[p2]]) - p
    cos = np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    dmid = float(np.linalg.norm(np.array([PX[p1], PY[p1]]) - np.array([PX[p2], PY[p2]])) / 2)
    kappa.append(2 * np.sin(np.arccos(cos) / 2) / dmid if dmid > 0 else np.nan)
    ok = np.isfinite(dr)
    if ok.any():
        gn = int(pd.Series(dr[ok]).groupby(grp[ok]).min().idxmin())
        if gn != g1:
            mis += 1
            mis_names.append(mun.name[i])

beta_assigned = np.array(beta_assigned); R = np.array(R); kappa = np.array(kappa)
b = beta_assigned[np.isfinite(beta_assigned) & (beta_assigned > 0)]
s_b, m_b = float(np.log(b).std(ddof=1)), float(np.log(b).mean())

# pooled muni->facility domain (replaces the old s_a = 0.093 quantity)
de_all = np.hypot(PX[None, :] - MX[:, None], PY[None, :] - MY[:, None])
pooled = (C / de_all).ravel()
pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
s_a, m_a = float(np.log(pooled).std(ddof=1)), float(np.log(pooled).mean())

# muni->muni domains, both networks
mm = {}
for case in ("operational", "full"):
    d = pd.read_csv(OD / f"od_{case}_munimuni_tol0.csv", encoding="utf-8").dropna(subset=["total_cost"])
    d = d[d.origin_pos != d.destination_pos]
    dd = np.hypot(MX[d.origin_pos.to_numpy()] - MX[d.destination_pos.to_numpy()],
                  MY[d.origin_pos.to_numpy()] - MY[d.destination_pos.to_numpy()])
    v = d.total_cost.to_numpy() / dd
    v = v[np.isfinite(v) & (v > 0)]
    mm[case] = {"n": int(len(v)), "m": float(np.log(v).mean()),
                "s": float(np.log(v).std(ddof=1)), "beta_min": float(v.min())}

# distributional ranking
rng = np.random.default_rng(SEED)
Rv = R[np.isfinite(R)]
ranking = []
for name, dist in FITTERS.items():
    params = dist.fit(b)
    if name == "Lognormal":
        point = predict_lognormal(s_b, Rv)
        sb = np.array([np.log(rng.choice(b, len(b), replace=True)).std(ddof=1)
                       for _ in range(N_BOOTSTRAP)])
        preds = np.array([predict_lognormal(x, Rv) for x in sb])
    else:
        point = predict_general(dist, params, Rv, rng)
        preds = np.array([predict_general(dist, dist.fit(rng.choice(b, len(b), replace=True)),
                                          Rv, rng, n_sim=20000) for _ in range(200)])
    lo, hi = np.percentile(preds, [2.5, 97.5])
    ranking.append({"distribution": name, "W1": round(wasserstein1(b, dist, params), 4),
                    "A2": round(anderson_darling(b, dist, params), 4),
                    "KS": round(float(stats.kstest(b, dist.cdf, args=params).statistic), 4),
                    "pred": round(point, 2), "ci_lo": round(float(lo), 2),
                    "ci_hi": round(float(hi), 2), "contains_observed": bool(lo <= mis <= hi),
                    "over_predicts": bool(point > mis)})
ranking.sort(key=lambda r: r["W1"])

z = float(stats.norm.ppf(1 - Q_STAR))
t_star = np.sqrt(2) * s_b / kappa * z
t_signed = -np.log(R) / kappa
inside = int(np.nansum(np.abs(t_signed) < t_star))

out = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "configuration": {"network": "carreteras.shp (IGN roads+streets, no camino/senda)",
                      "snap_tolerance_m": float(TOL), "facilities_layer": "Plantas_Ext.shp",
                      "d_r": "total_cost (entry+network+exit)",
                      "consolidation_km": CONSOLIDATION_KM,
                      "bootstrap": N_BOOTSTRAP, "seed": SEED},
    "counts": {"municipalities": int(len(mun)), "facilities_physical": int(len(pl)),
               "facilities_effective": int(len(set(grp))),
               "routes_muni_facility": int(np.isfinite(C).sum()),
               "misallocated": int(mis),
               "misallocated_pct": round(100 * mis / len(mun), 2)},
    "predictor_domain": {"n": int(len(b)), "m": round(m_b, 4), "s": round(s_b, 4),
                         "beta_min": round(float(b.min()), 4),
                         "beta_median": round(float(np.median(b)), 4),
                         "beta_below_one": int((b < 1).sum())},
    "pooled_muni_facility": {"n": int(len(pooled)), "m": round(m_a, 4), "s": round(s_a, 4)},
    "muni_muni": mm,
    "R": {"n": int(len(Rv)), "median": round(float(np.median(Rv)), 4)},
    "ranking": ranking,
    "safety_bands": {"q_star": Q_STAR, "t_star_median_m": round(float(np.nanmedian(t_star)), 1),
                     "municipalities_inside": inside},
    "misallocated_municipalities": sorted(mis_names),
}
(ROOT / "pipeline" / "canonical_numbers.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({k: v for k, v in out.items()
                  if k not in ("misallocated_municipalities", "ranking")}, indent=2, ensure_ascii=False))
print("\nranking:")
for r in ranking:
    print(f"  {r['distribution']:26s} W1={r['W1']:.4f} A2={r['A2']:8.4f} "
          f"pred={r['pred']:6.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] "
          f"contiene={r['contains_observed']} sobreestima={r['over_predicts']}")
