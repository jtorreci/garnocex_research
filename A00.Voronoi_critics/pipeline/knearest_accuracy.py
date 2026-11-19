"""Accuracy of k-nearest-Euclidean candidate sets against the network-optimal choice.

Section 5.3 of the manuscript reported 84.6 / 92.1 / 92.2 / 97.6 % for the baseline,
k=3, k=5 and the framework. No script in the repository produces those figures, and the
baseline was inconsistent with the canonical result: with 12.79 % of municipalities
misallocated the Voronoi baseline must be 87.21 % accurate, not 84.6 %.

This recomputes the reproducible part on the canonical routing output, with the same
conventions as canonical_numbers.py:

    for each municipality, the network-optimal destination is the consolidated facility
    group minimising d_r; accuracy at k is the fraction of municipalities whose
    network-optimal group is among their k Euclidean-nearest groups.

k = 1 is the Euclidean Voronoi baseline by definition.
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
KS = (1, 2, 3, 5, 10)


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
    n_groups = len(set(grp))

    od = pd.read_csv(OD / f"od_operational_muniplant_tol{TOL}.csv", encoding="utf-8")
    C = od.pivot_table(index="origin_pos", columns="destination_pos", values="total_cost",
                       aggfunc="min").reindex(index=range(len(mun)),
                                              columns=range(len(pl))).to_numpy()

    hits = {k: 0 for k in KS}
    total = 0
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        de_g = pd.Series(de).groupby(grp).min()          # euclidean distance per group
        dr_g = pd.Series(C[i]).groupby(grp).min()        # network distance per group
        if not np.isfinite(dr_g).any():
            continue
        best_net = int(dr_g.idxmin())
        order = list(de_g.sort_values().index)
        total += 1
        for k in KS:
            if best_net in order[:k]:
                hits[k] += 1

    print(f"municipalities evaluated: {total}   effective facility groups: {n_groups}\n")
    out = {}
    for k in KS:
        acc = 100.0 * hits[k] / total
        out[f"k{k}"] = round(acc, 2)
        tag = "  (Euclidean Voronoi baseline)" if k == 1 else ""
        print(f"  k = {k:2d}   accuracy = {acc:6.2f} %   ({hits[k]}/{total}){tag}")

    miss = total - hits[1]
    print(f"\n  consistency: {miss} misallocated -> baseline {100.0 * hits[1] / total:.2f} % "
          f"= 100 - {100.0 * miss / total:.2f} %")

    (ROOT / "pipeline" / "knearest_accuracy.json").write_text(
        json.dumps({"n": total, "groups": n_groups, "accuracy_pct": out}, indent=2),
        encoding="utf-8")
    print("\nwrote pipeline/knearest_accuracy.json")


if __name__ == "__main__":
    main()
