"""Is the negative residual autocorrelation a symptom of misspecification,
or is it what a correctly specified CAR produces?

The conditional residual y_i - E[y_i | y_-i] is not white even under a correct CAR:
the conditioning induces negative autocorrelation that grows with rho. Testing it
against the zero-autocorrelation null is therefore the wrong reference.

Two valid diagnostics are computed instead:

  1. Whitened residuals. With Q = tau (D - rho W) the model precision and Q = L L',
     z = L' (y - mu) has identity covariance under the fitted model. Moran's I on z
     tests directly whether spatial structure survives the model.

  2. Parametric bootstrap calibration of the conditional residual. Simulate replicate
     data from the fitted model, recompute the conditional-residual Moran on each, and
     locate the observed value in that reference distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
K_KNN, SEED, N_REP = 6, 42, 500

import libpysal
import esda


def main() -> None:
    ref = json.loads((ROOT / "pipeline" / "spatial_refit.json").read_text(encoding="utf-8"))
    arr = np.load(ROOT / "pipeline" / "spatial_refit_residuals.npy")
    coords, log_beta = arr[:, :2], arr[:, 2]
    n = len(log_beta)
    mu = ref["log_car"]["mu"]
    rho = ref["log_car"]["rho"]
    sigma = ref["log_car"]["sigma"]
    tau = 1.0 / sigma ** 2

    w = libpysal.weights.KNN.from_array(coords, k=K_KNN)
    w.transform = "r"
    W = w.full()[0]
    D = np.diag(W.sum(axis=1))
    Q = tau * (D - rho * W)

    print(f"n = {n} | mu = {mu:.4f} | rho = {rho:.4f} | sigma = {sigma:.4f}")

    # ---- 1. whitened residuals -------------------------------------
    L = np.linalg.cholesky(Q)
    z = L.T @ (log_beta - mu)
    m_z = esda.Moran(z, w, permutations=9999)
    print("\n1. WHITENED RESIDUALS (the model-consistent diagnostic)")
    print(f"   Moran's I = {m_z.I:.4f}   z = {m_z.z_sim:.3f}   p = {m_z.p_sim:.4g}")
    print(f"   sd(z) = {z.std(ddof=1):.4f}  (should be near 1 if the model fits)")

    # ---- 2. calibrate the conditional residual by parametric bootstrap
    cond_resid = lambda y: y - (mu + rho * (W @ (y - mu)))
    obs_I = float(esda.Moran(cond_resid(log_beta), w, permutations=99).I)
    Sigma = np.linalg.inv(Q)
    Lc = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(SEED)
    rep = np.empty(N_REP)
    for b in range(N_REP):
        y_rep = mu + Lc @ rng.standard_normal(n)
        rep[b] = esda.Moran(cond_resid(y_rep), w, permutations=1).I
    lo, hi = np.percentile(rep, [2.5, 97.5])
    pval = float(np.mean(np.abs(rep - rep.mean()) >= abs(obs_I - rep.mean())))
    print("\n2. CONDITIONAL RESIDUAL, CALIBRATED BY PARAMETRIC BOOTSTRAP")
    print(f"   observed I = {obs_I:.4f}")
    print(f"   reference distribution under the fitted model ({N_REP} replicates):")
    print(f"     mean = {rep.mean():.4f}   95% range = [{lo:.4f}, {hi:.4f}]")
    print(f"   two-sided p against the correct-model reference = {pval:.4f}")
    verdict = ("CONSISTENT with a correctly specified CAR"
               if lo <= obs_I <= hi else "OUTSIDE the reference range")
    print(f"   -> {verdict}")
    print(f"\n   (the naive test against I = 0 gave p = {ref['log_car']['moran_residuals_p']:.4g},")
    print("    which is the wrong reference for a conditional residual)")

    out = {
        "whitened": {"I": float(m_z.I), "z": float(m_z.z_sim), "p": float(m_z.p_sim),
                     "sd": float(z.std(ddof=1))},
        "conditional_calibrated": {"observed_I": obs_I, "ref_mean": float(rep.mean()),
                                   "ref_lo": float(lo), "ref_hi": float(hi),
                                   "p_vs_model": pval, "n_replicates": N_REP,
                                   "consistent": bool(lo <= obs_I <= hi)},
    }
    (ROOT / "pipeline" / "residual_diagnostics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote pipeline/residual_diagnostics.json")


if __name__ == "__main__":
    main()
