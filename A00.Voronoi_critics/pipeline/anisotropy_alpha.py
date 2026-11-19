"""Anisotropy coefficient alpha and its value as a classifier of misallocation.

Backs the figures in Observation 1, which had no script: median alpha, 95th percentile,
maximum, the medians split by outcome, and the ROC AUC of alpha as a bare classifier.

    alpha(P) = max_i beta(A_i, P) / min_i beta(A_i, P)

taken over the 46 PHYSICAL facilities, not the 34 consolidated groups. Alpha measures the
directional behaviour of the network as seen from P, which is a property of the road graph
and not of the flow-consolidation convention; this matches how kappa is computed from the
two nearest physical plants. The outcome label (misallocated or not) is, as everywhere
else, defined on the consolidated groups, since that is the decision being made.

Computing alpha over the consolidated groups instead gives median 1.57 and p95 2.66: the
grouping removes the near-identical beta of co-located plants and narrows the spread.
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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with ties handled by mid-ranks."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s), float)
    r[order] = ranks
    pos, neg = labels == 1, labels == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


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

    alphas, mis = [], []
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        # alpha over the physical facilities
        ok = np.isfinite(C[i]) & (de > 0)
        if ok.sum() < 2:
            continue
        b = C[i][ok] / de[ok]
        alphas.append(b.max() / b.min())
        # outcome over the consolidated groups
        de_g = pd.Series(de).groupby(grp).min()
        dr_g = pd.Series(C[i]).groupby(grp).min()
        mis.append(int(de_g.idxmin()) != int(dr_g.idxmin()))

    a = np.array(alphas)
    y = np.array(mis, int)
    print(f"n = {len(a)}   misallocated = {y.sum()}\n")
    print(f"  alpha median      {np.median(a):.2f}")
    print(f"  alpha p95         {np.percentile(a, 95):.2f}")
    print(f"  alpha max         {a.max():.1f}")
    print(f"  median | misallocated   {np.median(a[y == 1]):.2f}")
    print(f"  median | correct        {np.median(a[y == 0]):.2f}")
    A = auc(a, y)
    print(f"  ROC AUC           {A:.3f}")

    out = {"n": int(len(a)), "misallocated": int(y.sum()),
           "alpha_median": round(float(np.median(a)), 3),
           "alpha_p95": round(float(np.percentile(a, 95)), 3),
           "alpha_max": round(float(a.max()), 2),
           "median_misallocated": round(float(np.median(a[y == 1])), 3),
           "median_correct": round(float(np.median(a[y == 0])), 3),
           "roc_auc": round(A, 4)}
    (ROOT / "pipeline" / "anisotropy_alpha.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote pipeline/anisotropy_alpha.json")


if __name__ == "__main__":
    main()
