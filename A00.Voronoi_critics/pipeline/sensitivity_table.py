"""Regenerate tab:sensitivity_config from the stored OD tables.

The five variant rows of the sensitivity table were computed interactively and never
persisted, so the table could not be checked or reproduced. Every variant is derivable
from the OD files in pipeline/od/, so this rebuilds all six rows from them and prints
the LaTeX body, applying the conventions of canonical_numbers.py throughout.

Each row changes exactly one decision with respect to the adopted configuration:

    adopted            operational network, 5 m tolerance, total cost, 15 km consolidation
    no consolidation   as adopted, but every physical facility is its own destination
    no urban classes   roads_nourban network
    unrestricted       full network (tracks and footpaths included)
    zero tolerance     operational network built with 0 m snapping tolerance
    no access legs     as adopted, but d_r is the network component only
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OD = ROOT / "pipeline" / "od"
CONSOLIDATION_KM = 15.0


def groups(PX, PY, consolidate: bool):
    n = len(PX)
    if not consolidate:
        return np.arange(n)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(PX[i] - PX[j], PY[i] - PY[j]) <= CONSOLIDATION_KM * 1000:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
    g = np.array([find(i) for i in range(n)])
    return np.array([{v: k for k, v in enumerate(sorted(set(g)))}[v] for v in g])


def evaluate(od_file: str, cost_col: str, consolidate: bool, mun, pl):
    MX, MY = mun.X.to_numpy(), mun.Y.to_numpy()
    PX, PY = pl.x.to_numpy(), pl.y.to_numpy()
    grp = groups(PX, PY, consolidate)

    od = pd.read_csv(OD / od_file, encoding="utf-8")
    C = od.pivot_table(index="origin_pos", columns="destination_pos", values=cost_col,
                       aggfunc="min").reindex(index=range(len(mun)),
                                              columns=range(len(pl))).to_numpy()

    mis, betas = 0, []
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        de_g = pd.Series(de).groupby(grp).min()
        dr_g = pd.Series(C[i]).groupby(grp).min()
        if not np.isfinite(dr_g).any():
            continue
        euc_best = int(de_g.idxmin())
        net_best = int(dr_g.idxmin())
        if euc_best != net_best:
            mis += 1
        cand = np.where(grp == euc_best)[0]
        p = cand[np.argmin(de[cand])]
        if de[p] > 0 and np.isfinite(C[i, p]):
            betas.append(C[i, p] / de[p])

    b = np.array(betas)
    b = b[np.isfinite(b) & (b > 0)]
    return {"groups": int(len(set(grp))), "misallocated": mis,
            "pct": 100.0 * mis / len(mun), "s": float(np.log(b).std(ddof=1)),
            "n_beta": int(len(b)), "beta_min": float(b.min())}


ROWS = [
    ("Adopted", "od_operational_muniplant_tol5.csv", "total_cost", True),
    ("Facilities not consolidated", "od_operational_muniplant_tol5.csv", "total_cost", False),
    ("Network without urban street classes", "od_roads_nourban_muniplant_tol0.csv", "total_cost", True),
    ("Unrestricted network (tracks, footpaths)", "od_full_muniplant_tol0.csv", "total_cost", True),
    ("Graph built with zero tolerance", "od_operational_muniplant_tol0.csv", "total_cost", True),
    ("Access legs excluded from $d_r$", "od_operational_muniplant_tol5.csv", "network_cost", True),
]


def main() -> None:
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                      encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
    pl = pd.read_csv(OD / "plantas_roster.csv", encoding="utf-8")

    out = {}
    print(f"{'configuration':42}{'groups':>7}{'misalloc':>10}{'pct':>7}{'s':>8}{'beta_min':>10}")
    print("-" * 84)
    for label, f, col, cons in ROWS:
        r = evaluate(f, col, cons, mun, pl)
        out[label] = r
        print(f"{label[:41]:42}{r['groups']:7d}{r['misallocated']:10d}"
              f"{r['pct']:7.1f}{r['s']:8.3f}{r['beta_min']:10.3f}")

    print("\n--- cuerpo LaTeX ---")
    for label, f, col, cons in ROWS:
        r = out[label]
        pct = f"{r['pct']:.1f}"
        bold = label == "Adopted"
        cells = [f"{r['groups']}", f"{r['misallocated']} ({pct}\\%)", f"{r['s']:.3f}",
                 f"{r['beta_min']:.3f}"]
        if bold:
            print(f"\\textbf{{{label}}} & " + " & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\")
        else:
            print(f"{label} & " + " & ".join(cells) + " \\\\")

    (ROOT / "pipeline" / "sensitivity_table.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote pipeline/sensitivity_table.json")


if __name__ == "__main__":
    main()
