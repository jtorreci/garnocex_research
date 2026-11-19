"""Descriptive statistics of beta for the table in Section 4.2.

The table shipped in the manuscript was pre-revision: n = 9,112 "excluding beta < 1",
which contradicts the third consequence of Definition 1 (beta >= 1 by construction, so
no filtering is possible or needed). This recomputes it on the canonical routing output,
in the two domains the manuscript actually uses, applying exactly the conventions of
canonical_numbers.py:

    domain (a)  every municipality-to-facility route          (pooled / anisotropy fit)
    domain (b)  each municipality to its assigned facility     (predictor fit)

No filtering of any kind is applied. Emits the LaTeX table body.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OD = ROOT / "pipeline" / "od"
TOL = "5"
CONSOLIDATION_KM = 15.0


def main() -> None:
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

    de_all = np.hypot(PX[None, :] - MX[:, None], PY[None, :] - MY[:, None])

    # ---- domain (a): every muni-facility route ------------------------
    pooled = (C / de_all).ravel()
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]

    # ---- domain (b): each muni to its assigned (consolidated) facility
    assigned = []
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        g1 = int(pd.Series(de).groupby(grp).min().idxmin())
        cand = np.where(grp == g1)[0]
        p = cand[np.argmin(de[cand])]
        if de[p] > 0 and np.isfinite(C[i, p]):
            assigned.append(C[i, p] / de[p])
    assigned = np.array(assigned)

    def describe(v):
        return {
            "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "q1": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
            "max": float(v.max()), "below_one": int((v < 1).sum()),
        }

    a, b = describe(pooled), describe(assigned)
    for name, d in (("domain (a) pooled", a), ("domain (b) assigned", b)):
        print(f"\n{name}")
        for k, v in d.items():
            print(f"  {k:10} {v:,.4f}" if isinstance(v, float) else f"  {k:10} {v:,}")

    assert a["below_one"] == 0 and b["below_one"] == 0, "beta < 1 found: check the cost convention"
    print("\nbeta >= 1 holds in both domains, as Definition 1 requires.")

    rows = [("Count", f"{a['n']:,}", f"{b['n']:,}"),
            ("Mean", f"{a['mean']:.3f}", f"{b['mean']:.3f}"),
            ("Std.\\ Dev.", f"{a['sd']:.3f}", f"{b['sd']:.3f}"),
            ("Minimum", f"{a['min']:.3f}", f"{b['min']:.3f}"),
            ("25\\% (Q1)", f"{a['q1']:.3f}", f"{b['q1']:.3f}"),
            ("Median", f"{a['median']:.3f}", f"{b['median']:.3f}"),
            ("75\\% (Q3)", f"{a['q3']:.3f}", f"{b['q3']:.3f}"),
            ("Maximum", f"{a['max']:.3f}", f"{b['max']:.3f}")]
    print("\n--- cuerpo LaTeX ---")
    for r in rows:
        print(f"{r[0]} & {r[1]} & {r[2]} \\\\")

    (ROOT / "pipeline" / "descriptive_stats.json").write_text(
        json.dumps({"domain_a_pooled": a, "domain_b_assigned": b}, indent=2), encoding="utf-8")
    print("\nwrote pipeline/descriptive_stats.json")


if __name__ == "__main__":
    main()
