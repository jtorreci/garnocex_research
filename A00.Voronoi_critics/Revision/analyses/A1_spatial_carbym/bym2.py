"""Bayesian BYM2 fit on log(beta) — replaces the simulated BYM in
the original manuscript.

BYM2 specification (Riebler et al. 2016, Statistical Methods in Medical
Research, 25(4):1145-1165):

    log(beta_i) = mu + sigma * ( sqrt(rho/c) * phi_i + sqrt(1-rho) * theta_i )

where:
    phi_i  ~ ICAR(W)   spatially structured (intrinsic CAR)
    theta_i ~ N(0, 1)  unstructured
    rho ∈ [0, 1]       mixing weight (1 = fully spatial, 0 = i.i.d.)
    c                  scaling factor for the ICAR (Sørbye 2014) so that
                       the marginal variance of the structured component
                       is approx 1.

This isolates 'how much of the variance is spatially structured (rho)
vs. unstructured (1-rho)'. The original simulated BYM in the manuscript
implied something like rho≈0.5 with no real fit — we now produce a
real posterior for rho.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import libpysal
import pymc as pm
import arviz as az

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT_FIG = HERE.parent / "outputs" / "figures"
OUT_TAB = HERE.parent / "outputs" / "tables"

K_KNN = 6
SEED = 42
N_SAMPLES = 2000
N_TUNE = 2000           # longer warmup for funnel
N_CHAINS = 4
TARGET_ACCEPT = 0.99    # non-centered already, but be conservative


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def compute_icar_scaling(W: np.ndarray) -> float:
    """Sørbye (2014) scaling factor: geometric mean of the diagonal of
    the generalized inverse of the ICAR precision matrix.

    For the ICAR with precision Q = D - W (where D = diag(W·1)), Q is
    rank-deficient (1 zero eigenvalue) so we use the Moore-Penrose
    pseudo-inverse. The scaling c makes Var(phi_i) ~ 1 marginally.
    """
    n = W.shape[0]
    D = np.diag(W.sum(axis=1))
    Q = D - W
    # Pseudoinverse via eigendecomposition (drop the zero eigenvalue)
    eig_vals, eig_vecs = np.linalg.eigh(Q)
    # Smallest eigenvalue should be ~0 (null space = constant vector)
    eig_inv = np.where(eig_vals > 1e-10, 1.0 / eig_vals, 0.0)
    Q_pinv = eig_vecs @ np.diag(eig_inv) @ eig_vecs.T
    diag_vars = np.diag(Q_pinv)
    # Geometric mean of marginal variances (Sørbye scaling)
    log_diag = np.log(diag_vars[diag_vars > 0])
    return float(np.exp(log_diag.mean()))


def main():
    banner("BYM2 fit on log(beta_assigned) — Extremadura, n=383")

    df = pd.read_csv(DATA / "municipios_canonical.csv")
    df = df.dropna(subset=["beta_assigned", "utm_x", "utm_y"]).reset_index(drop=True)
    coords = df[["utm_x", "utm_y"]].values
    log_beta = np.log(df["beta_assigned"].values)
    n = len(log_beta)
    print(f"  n = {n}")

    # --- W matrix (KNN k=6, symmetrized) ---
    w_pysal = libpysal.weights.KNN.from_array(coords, k=K_KNN)
    W_dense = w_pysal.full()[0].astype(float)
    # ICAR requires symmetric W
    W_sym = np.maximum(W_dense, W_dense.T)
    print(f"  W: shape={W_sym.shape}, symmetric edges={int(W_sym.sum() / 2)}")

    # --- Sørbye scaling for the ICAR ---
    c_scale = compute_icar_scaling(W_sym)
    print(f"  Sorbye scaling factor c = {c_scale:.4f}")

    # --- ICAR precision components ---
    D_diag = W_sym.sum(axis=1)
    n_edges = int((W_sym > 0).sum() // 2)

    # Build edge list for pairwise differences
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W_sym[i, j] > 0:
                edges.append((i, j))
    edges = np.array(edges)
    print(f"  Edges (i<j): {len(edges)}")

    # --- PyMC BYM2 model (non-centered for the latent components) ---
    # The previous version had divergences and r_hat>1.01 because of the
    # funnel between sigma and (phi, theta). Non-centered reparametrization:
    # we sample standardized phi_z, theta_z and scale by sigma * sqrt(rho/c)
    # and sigma * sqrt(1-rho) inside eta — same model, much better geometry.
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=log_beta.mean(), sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)

        # Structured component (ICAR), non-centered: phi_z ~ N(0,1), then
        # apply sum-to-zero (ICAR identifiable only up to constant).
        phi_z = pm.Normal("phi_z", mu=0.0, sigma=1.0, shape=n)
        phi = pm.Deterministic("phi", phi_z - phi_z.mean())

        # ICAR potential on phi (pairwise differences across edges).
        diffs = phi[edges[:, 0]] - phi[edges[:, 1]]
        pm.Potential("icar_prior", -0.5 * pm.math.sum(diffs ** 2))

        # Unstructured component, non-centered
        theta_z = pm.Normal("theta_z", mu=0.0, sigma=1.0, shape=n)

        # BYM2 latent random effect, scaled outside (non-centered)
        eta = (
            mu
            + sigma * pm.math.sqrt(rho / c_scale) * phi
            + sigma * pm.math.sqrt(1.0 - rho) * theta_z
        )

        tau_obs = pm.Gamma("tau_obs", alpha=2.0, beta=1.0)
        pm.Normal("y", mu=eta, tau=tau_obs, observed=log_beta)

        idata = pm.sample(
            draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED, progressbar=True,
        )

    # --- Diagnostics ---
    summ = az.summary(idata, var_names=["mu", "sigma", "rho", "tau_obs"], round_to=4)
    print(summ)
    rhat_max = float(az.rhat(idata, var_names=["mu", "sigma", "rho", "tau_obs"]).to_array().max())
    ess_min = float(az.ess(idata, var_names=["mu", "sigma", "rho", "tau_obs"], method="bulk").to_array().min())
    print(f"  Max R-hat: {rhat_max:.4f}, Min ESS: {ess_min:.0f}")

    rho_post = idata.posterior["rho"].values.flatten()
    sigma_post = idata.posterior["sigma"].values.flatten()
    mu_post = float(idata.posterior["mu"].mean())

    # --- Predictive comparison ---
    eta_post = (
        mu_post
        + sigma_post.mean() * (
            np.sqrt(rho_post.mean() / c_scale) * idata.posterior["phi"].mean(("chain", "draw")).values
            + np.sqrt(1.0 - rho_post.mean()) * idata.posterior["theta_z"].mean(("chain", "draw")).values
        )
    )
    beta_pred = np.exp(eta_post)
    beta_obs = np.exp(log_beta)

    # Multiple notions of "agreement" with the observed:
    agree_5pct = float(np.mean(np.abs(beta_pred - beta_obs) / beta_obs < 0.05))
    agree_10pct = float(np.mean(np.abs(beta_pred - beta_obs) / beta_obs < 0.10))
    agree_20pct = float(np.mean(np.abs(beta_pred - beta_obs) / beta_obs < 0.20))
    rmse_beta = float(np.sqrt(np.mean((beta_pred - beta_obs) ** 2)))
    mae_beta = float(np.mean(np.abs(beta_pred - beta_obs)))

    # Agreement in MISALLOCATION CLASSIFICATION (the paper's '95.1% agreement'
    # interpretation): keep Voronoi assignment under the BYM2 model — it
    # depends only on whether sigma_pred[BYM2] keeps the same misallocation
    # flag. Since we don't refit Voronoi here, we report the closest proxy:
    # munis whose BYM2 prediction is within 1 standard deviation (sigma) of
    # the observed beta — those are 'in agreement' with the independence model.
    within_1sigma = float(np.mean(
        np.abs(np.log(beta_pred) - np.log(beta_obs)) < sigma_post.mean()
    ))

    print(f"  Posterior rho mean = {rho_post.mean():.4f}, "
          f"95% CrI = [{np.percentile(rho_post, 2.5):.4f}, "
          f"{np.percentile(rho_post, 97.5):.4f}]")
    print(f"  Posterior sigma mean = {sigma_post.mean():.4f}")
    print(f"  Variance partition: {rho_post.mean()*100:.1f}% spatial, "
          f"{(1-rho_post.mean())*100:.1f}% unstructured")
    print(f"  Agreement |Δβ|/β < 5%/10%/20%: "
          f"{agree_5pct*100:.1f}% / {agree_10pct*100:.1f}% / {agree_20pct*100:.1f}%")
    print(f"  Within 1 sigma (log scale):    {within_1sigma*100:.1f}%")
    print(f"  RMSE β = {rmse_beta:.4f},  MAE β = {mae_beta:.4f}")

    # --- Save ---
    summary_dict = {
        "n_munis": int(n),
        "k_knn": K_KNN,
        "sorbye_scaling": c_scale,
        "posterior": {
            "mu_mean": mu_post,
            "sigma_mean": float(sigma_post.mean()),
            "sigma_ci": [float(np.percentile(sigma_post, 2.5)),
                          float(np.percentile(sigma_post, 97.5))],
            "rho_mean": float(rho_post.mean()),
            "rho_ci": [float(np.percentile(rho_post, 2.5)),
                       float(np.percentile(rho_post, 97.5))],
        },
        "diagnostics": {
            "rhat_max": rhat_max,
            "ess_min": ess_min,
        },
        "agreement": {
            "within_5pct": agree_5pct,
            "within_10pct": agree_10pct,
            "within_20pct": agree_20pct,
            "within_1_sigma_log": within_1sigma,
            "rmse_beta": rmse_beta,
            "mae_beta": mae_beta,
        },
    }
    (HERE / "BYM2_summary.json").write_text(json.dumps(summary_dict, indent=2))
    print(f"\n  Wrote BYM2_summary.json")

    # Append predictions to dataset
    df_out = df.copy()
    df_out["bym2_log_beta_pred"] = eta_post
    df_out["bym2_beta_pred"] = beta_pred
    df_out.to_csv(HERE / "BYM2_results.csv", index=False, encoding="utf-8")

    # Figure: rho posterior + scatter pred vs obs
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(rho_post, bins=50, color="steelblue", alpha=0.8)
    axes[0].axvline(rho_post.mean(), color="r", linestyle="--",
                    label=f"mean = {rho_post.mean():.3f}")
    axes[0].set_xlabel(r"$\rho$ (BYM2 mixing)")
    axes[0].set_ylabel("Posterior density")
    axes[0].set_title(r"BYM2 spatial fraction $\rho$ posterior")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    axes[1].scatter(beta_obs, beta_pred, alpha=0.5, s=20)
    lo = min(beta_obs.min(), beta_pred.min())
    hi = max(beta_obs.max(), beta_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_xlabel(r"$\beta$ observed")
    axes[1].set_ylabel(r"$\beta$ BYM2-predicted")
    axes[1].set_title(f"BYM2 predictions  ({agree_10pct*100:.1f}% within 10%, "
                       f"{within_1sigma*100:.1f}% within 1σ)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "BYM2_diagnostics.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG / "BYM2_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote BYM2_diagnostics.{{pdf,png}}")

    # LaTeX summary
    tex = (
        "% Auto-generated by A1_spatial_carbym/bym2.py\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Bayesian BYM2 fit on $\\log\\beta_i$ "
        f"($n={n}$ municipalities, KNN $k={K_KNN}$).}}\n"
        "\\label{tab:bym2}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Quantity & Posterior mean & 95\\% CrI \\\\\n"
        "\\midrule\n"
        f"$\\mu$ (intercept) & {mu_post:.4f} & --- \\\\\n"
        f"$\\sigma$ (total scale) & {sigma_post.mean():.4f} & "
        f"[{summary_dict['posterior']['sigma_ci'][0]:.4f}, "
        f"{summary_dict['posterior']['sigma_ci'][1]:.4f}] \\\\\n"
        f"$\\rho$ (spatial fraction) & {rho_post.mean():.4f} & "
        f"[{summary_dict['posterior']['rho_ci'][0]:.4f}, "
        f"{summary_dict['posterior']['rho_ci'][1]:.4f}] \\\\\n"
        f"Agreement within $5\\%$ & {agree_5pct*100:.1f}\\% & --- \\\\\n"
        f"max $\\hat R$ & {rhat_max:.4f} & --- \\\\\n"
        f"min ESS & {int(ess_min)} & --- \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (OUT_TAB / "A1_bym2_summary.tex").write_text(tex, encoding="utf-8")
    print(f"  Wrote A1_bym2_summary.tex")

    banner("DONE")


if __name__ == "__main__":
    main()
