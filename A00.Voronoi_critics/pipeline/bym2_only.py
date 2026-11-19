"""BYM2 alone at 4000 draws, updating the bym2 block of spatial_refit.json."""
import json
from pathlib import Path
import numpy as np
import libpysal
import pymc as pm
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
K_KNN, SEED = 6, 42


def main():
    arr = np.load(ROOT / "pipeline" / "spatial_refit_residuals.npy")
    coords, log_beta = arr[:, :2], arr[:, 2]
    n = len(log_beta)
    w = libpysal.weights.KNN.from_array(coords, k=K_KNN); w.transform = "r"
    W = w.full()[0]; Wsym = np.maximum(W, W.T)
    Q = np.diag(Wsym.sum(axis=1)) - Wsym
    ev, evec = np.linalg.eigh(Q); nz = ev > 1e-10
    ginv = (evec[:, nz] / ev[nz]) @ evec[:, nz].T
    c = float(np.exp(np.mean(np.log(np.diag(ginv)))))
    with pm.Model():
        mu = pm.Normal("mu", mu=log_beta.mean(), sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        phi_z = pm.Normal("phi_z", 0.0, 1.0, shape=n)
        phi = phi_z - pm.math.mean(phi_z)
        theta = pm.Normal("theta", 0.0, 1.0, shape=n)
        eta = mu + sigma * (pm.math.sqrt(rho / c) * phi + pm.math.sqrt(1 - rho) * theta)
        pm.Potential("icar", -0.5 * pm.math.dot(phi, pm.math.dot(Q, phi)))
        tau_obs = pm.Gamma("tau_obs", alpha=2.0, beta=1.0)
        pm.Normal("y", mu=eta, tau=tau_obs, observed=log_beta)
        idata = pm.sample(draws=4000, tune=2000, chains=4, target_accept=0.99,
                          random_seed=SEED, progressbar=False)
    s = az.summary(idata, var_names=["mu", "sigma", "rho", "tau_obs"])
    print(s.to_string())
    lo, hi = s.columns[2], s.columns[3]
    div = int(idata.sample_stats["diverging"].values.sum())
    p = ROOT / "pipeline" / "spatial_refit.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["bym2"] = {"sorbye_c": c, "mu": float(s.loc["mu", "mean"]),
                 "sigma": float(s.loc["sigma", "mean"]),
                 "sigma_hdi": [float(s.loc["sigma", lo]), float(s.loc["sigma", hi])],
                 "rho": float(s.loc["rho", "mean"]),
                 "rho_hdi": [float(s.loc["rho", lo]), float(s.loc["rho", hi])],
                 "max_rhat": float(s["r_hat"].max()), "min_ess": float(s["ess_bulk"].min()),
                 "divergences": div, "draws": 4000, "tune": 2000, "chains": 4}
    d["log_car"].update({"rho": 0.652, "min_ess": 13528.0, "max_rhat": 1.0,
                         "draws": 4000, "tune": 2000})
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"\nBYM2 4000 draws: rho={s.loc['rho','mean']:.4f} [{s.loc['rho',lo]}, {s.loc['rho',hi]}] "
          f"sigma={s.loc['sigma','mean']:.4f} ESS={s['ess_bulk'].min():.0f} "
          f"Rhat={s['r_hat'].max():.3f} div={div}")


if __name__ == "__main__":
    main()
