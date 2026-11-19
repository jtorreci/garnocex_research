"""Acceptance test for the statistical core of the pipeline.

The OD-based path in run_case.py cannot be validated against the published
numbers yet, because the surviving OD exports cover 301 of 383 municipalities
and 36 of 46 facilities. The published predictor domain comes from the
assignment output (municipios_canonical.csv), which does cover all 383.

So we validate the statistics here, against values that are already in print:

    s_b (predictor domain)      published 0.326
    misallocations              published 61 / 383 = 15.9 %
    Theorem 1 prediction        published 63.5, 95 % CI [58.0, 67.7]

If these reproduce, the fitting / R / predictor / bootstrap code is sound and
only needs complete OD exports to run both network cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_case import (SEED, N_BOOTSTRAP, consolidate, euclid_to_groups,  # noqa: E402
                      geometry_terms, load_points, predict_lognormal)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "analyses" / "data"
PUBLISHED = {"s_b": 0.326, "misallocated": 61, "n": 383,
             "pred": 63.5, "ci": (58.0, 67.7)}


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'}  {label:34s} obtenido={got:9.4f}  "
          f"publicado={want:9.4f}  tol=±{tol}")
    return ok


def main() -> None:
    canon = pd.read_csv(DATA / "municipios_canonical.csv", encoding="utf-8")
    beta = canon.beta_assigned.dropna().to_numpy()
    s_b = float(np.log(beta).std(ddof=1))
    m_b = float(np.log(beta).mean())
    mis = int(canon.misallocated.sum())

    print("=== 1. dominio predictor desde municipios_canonical.csv ===")
    results = [
        check("n municipios", len(canon), PUBLISHED["n"], 0),
        check("s_b = sd(log beta)", s_b, PUBLISHED["s_b"], 0.002),
        check("misasignaciones", mis, PUBLISHED["misallocated"], 0),
    ]
    print(f"        m_b = {m_b:.4f}  (el manuscrito y la carta imprimen 0.064 -- discrepancia "
          f"conocida, m no entra en el Theorem 1)")

    print("\n=== 2. valores de R con la regla de consolidacion de 15 km ===")
    mun, plants = load_points()
    plants = consolidate(plants)
    eu = euclid_to_groups(mun, plants)
    geom = geometry_terms(mun, plants, eu)
    R = geom.R.to_numpy()
    print(f"  instalaciones: {len(plants)} fisicas -> {plants.group.nunique()} logicas")
    print(f"  R: n={len(R)}  mediana={np.median(R):.4f}  min={R.min():.4f}  "
          f"<=1.10: {(R <= 1.10).sum()}")

    print("\n=== 3. prediccion del Theorem 1 ===")
    rng = np.random.default_rng(SEED)
    point = predict_lognormal(s_b, R)
    s_boot = np.array([np.log(rng.choice(beta, len(beta), replace=True)).std(ddof=1)
                       for _ in range(N_BOOTSTRAP)])
    preds = np.array([predict_lognormal(s, R) for s in s_boot])
    lo, hi = np.percentile(preds, [2.5, 97.5])
    print(f"  prediccion puntual = {point:.2f}   IC 95 % = [{lo:.2f}, {hi:.2f}]   "
          f"(B = {N_BOOTSTRAP})")
    results.append(check("prediccion puntual", point, PUBLISHED["pred"], 1.5))
    results.append(check("IC inferior", lo, PUBLISHED["ci"][0], 2.0))
    results.append(check("IC superior", hi, PUBLISHED["ci"][1], 2.0))
    print(f"  contiene el observado ({mis})? {'si' if lo <= mis <= hi else 'NO'}")

    # what the published A2 script actually did: group facilities by host name
    # instead of applying the 15 km rule
    print("\n=== 4. contraste: R agrupando por municipio anfitrion (lo que hacia A2) ===")
    eu_host = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv", encoding="utf-8")
    R_host = []
    for _, grp in eu_host.groupby("InputID"):
        d = np.sort(grp.groupby("TargetID").Distance.min().to_numpy())
        if len(d) >= 2 and d[0] > 0:
            R_host.append(d[1] / d[0])
    R_host = np.array(R_host)
    p_host = predict_lognormal(s_b, R_host)
    print(f"  R: n={len(R_host)}  mediana={np.median(R_host):.4f}  -> prediccion = {p_host:.2f}")
    print(f"  diferencia por aplicar la consolidacion de 15 km: "
          f"{point - p_host * len(R) / max(len(R_host), 1):+.2f} (escalado a n comparable)")

    print("\n" + "=" * 62)
    print(f"RESULTADO: {sum(results)}/{len(results)} comprobaciones superadas")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
