"""Refit the spatial models (Moran, log-CAR, BYM2, LOOCV) on the CANONICAL beta.

Section 3 of the manuscript was estimated on the previous beta values, which came
from the degraded facility layer and a zero-tolerance routing graph. Those values
have changed, so every number in Section 3 has to be re-estimated.

Model specifications are copied from analyses/A1_spatial_carbym/{run.py,bym2.py}
so the only thing that differs is the data.

Run with the isolated interpreter that has pymc/arviz/libpysal:
    C:/Users/Usuario/AppData/Local/Temp/voronoi-spatial/Scripts/python.exe refit_spatial.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OD = ROOT / "pipeline" / "od"
TOL = "5"
K_KNN, N_SAMPLES, N_TUNE, N_CHAINS, SEED = 6, 4000, 2000, 4, 42
CONSOLIDATION_KM = 15.0


def banner(m):
    print("\n" + "=" * 68 + f"\n{m}\n" + "=" * 68, flush=True)


# ---------------------------------------------------------------- data
def canonical_beta():
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                      encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
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
                       aggfunc="min").reindex(index=range(len(mun)),
                                              columns=range(len(pl))).to_numpy()
    beta = np.full(len(mun), np.nan)
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i])
        g1 = int(pd.Series(de).groupby(grp).min().idxmin())
        cand = np.where(grp == g1)[0]; p = cand[np.argmin(de[cand])]
        if de[p] > 0 and np.isfinite(C[i, p]):
            beta[i] = C[i, p] / de[p]
    ok = np.isfinite(beta) & (beta > 0)
    return np.column_stack([MX, MY])[ok], beta[ok], mun.NOMBRE.to_numpy()[ok]


import libpysal, esda
import pymc as pm
import arviz as az


def main():
    coords, beta, names = canonical_beta()
    log_beta = np.log(beta)
    n = len(beta)
    print(f"canonical beta: n={n}  m={log_beta.mean():.4f}  s={log_beta.std(ddof=1):.4f}")


    
   

    w = libpysal.weights.KNN.from_array(coords, k=K_KNN)
    w.transform = "r"
    W = w.full()[0]
    Wsym = np.maximum(W, W.T)

    banner("1. Moran's I on log(beta)")
    m_obs = esda.Moran(log_beta, w, permutations=999)
    print(f"  I = {m_obs.I:.4f}   z = {m_obs.z_sim:.3f}   p = {m_obs.p_sim:.4g}")

    banner("2. log-CAR")
    D = np.diag(W.sum(axis=1))
    Dm = np.diag(1 / np.sqrt(np.maximum(W.sum(axis=1), 1e-12)))
    ev = np.linalg.eigvalsh(Dm @ W @ Dm)
    rho_min, rho_max = 1 / ev.min() + 1e-3, 1 / ev.max() - 1e-3
    with pm.Model():
        mu = pm.Normal("mu", mu=log_beta.mean(), sigma=1.0)
        rho_raw = pm.Beta("rho_raw", alpha=2.0, beta=2.0)
        rho = pm.Deterministic("rho", rho_min + (rho_max - rho_min) * rho_raw)
        tau = pm.Gamma("tau", alpha=2.0, beta=1.0)
        Q = tau * (D - rho * W)
        pm.MvNormal("y", mu=mu * np.ones(n), tau=Q, observed=log_beta)
        car = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS, target_accept=0.95,
                        random_seed=SEED, progressbar=False)
    sm = az.summary(car, var_names=["mu", "rho", "tau"])
    print(sm.to_string())
    mu_h = float(sm.loc["mu", "mean"]); rho_h = float(sm.loc["rho", "mean"])
    tau_h = float(sm.loc["tau", "mean"]); sigma_h = float(1 / np.sqrt(tau_h))
    rhat_car = float(sm["r_hat"].max()); ess_car = float(sm["ess_bulk"].min())

    fitted = mu_h + rho_h * (W @ (log_beta - mu_h))
    resid = log_beta - fitted
    m_res = esda.Moran(resid, w, permutations=9999)
    print(f"  sigma = {sigma_h:.4f} | Moran's I residuals = {m_res.I:.4f} (p = {m_res.p_sim:.4g})")

    banner("3. Spatial LOOCV")
    yhat = np.array([mu_h + rho_h * (W[i] @ (log_beta - mu_h)) for i in range(n)])
    rmse_log = float(np.sqrt(np.mean((log_beta - yhat) ** 2)))
    rmse_beta = float(np.sqrt(np.mean((beta - np.exp(yhat)) ** 2)))
    mae_beta = float(np.mean(np.abs(beta - np.exp(yhat))))
    print(f"  RMSE(log) = {rmse_log:.4f}   RMSE(beta) = {rmse_beta:.4f}   MAE(beta) = {mae_beta:.4f}")

    banner("4. BYM2")
    Qicar = np.diag(Wsym.sum(axis=1)) - Wsym
    evals, evecs = np.linalg.eigh(Qicar)
    nz = evals > 1e-10
    ginv = (evecs[:, nz] / evals[nz]) @ evecs[:, nz].T
    c_scale = float(np.exp(np.mean(np.log(np.diag(ginv)))))
    print(f"  Sorbye scaling c = {c_scale:.4f}")
    with pm.Model():
        mu = pm.Normal("mu", mu=log_beta.mean(), sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        phi_z = pm.Normal("phi_z", 0.0, 1.0, shape=n)
        phi = phi_z - pm.math.mean(phi_z)          # not stored: recoverable from phi_z
        theta = pm.Normal("theta", 0.0, 1.0, shape=n)
        eta = mu + sigma * (pm.math.sqrt(rho / c_scale) * phi + pm.math.sqrt(1 - rho) * theta)
        pm.Potential("icar", -0.5 * pm.math.dot(phi, pm.math.dot(Qicar, phi)))
        tau_obs = pm.Gamma("tau_obs", alpha=2.0, beta=1.0)
        pm.Normal("y", mu=eta, tau=tau_obs, observed=log_beta)
        bym = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS, target_accept=0.99,
                        random_seed=SEED, progressbar=False)
    sb = az.summary(bym, var_names=["mu", "sigma", "rho", "tau_obs"])
    # arviz renames the interval columns across versions; take them by position
    lo_col, hi_col = sb.columns[2], sb.columns[3]
    print("  columnas de intervalo:", list(sb.columns))
    print(sb.to_string())
    div = int(bym.sample_stats["diverging"].values.sum())
    print(f"  divergences = {div}")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"n": int(n), "m_log_beta": float(log_beta.mean()),
                 "s_log_beta": float(log_beta.std(ddof=1)), "k_knn": K_KNN},
        "moran_log_beta": {"I": float(m_obs.I), "z": float(m_obs.z_sim), "p": float(m_obs.p_sim)},
        "log_car": {"mu": mu_h, "rho": rho_h, "sigma": sigma_h,
                    "max_rhat": rhat_car, "min_ess": ess_car,
                    "moran_residuals_I": float(m_res.I), "moran_residuals_p": float(m_res.p_sim)},
        "loocv": {"rmse_log": rmse_log, "rmse_beta": rmse_beta, "mae_beta": mae_beta},
        "bym2": {"sorbye_c": c_scale,
                 "mu": float(sb.loc["mu", "mean"]), "sigma": float(sb.loc["sigma", "mean"]),
                 "sigma_hdi": [float(sb.loc["sigma", lo_col]), float(sb.loc["sigma", hi_col])],
                 "rho": float(sb.loc["rho", "mean"]),
                 "rho_hdi": [float(sb.loc["rho", lo_col]), float(sb.loc["rho", hi_col])],
                 "max_rhat": float(sb["r_hat"].max()), "min_ess": float(sb["ess_bulk"].min()),
                 "divergences": div},
    }
    (ROOT / "pipeline" / "spatial_refit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.save(ROOT / "pipeline" / "spatial_refit_residuals.npy",
            np.column_stack([coords, log_beta, fitted, resid]))
    print("\nwrote pipeline/spatial_refit.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
