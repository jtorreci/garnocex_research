"""Regenerate the four spatial figures of Section 3 from the refitted models.

Re-runs BYM2 (only) to recover the posterior draws of rho and the posterior mean of
eta per municipality, which the summary-only refit did not retain, then draws:

    spatial_analysis_beta_coefficients.pdf
    spatial_diagnostics.pdf
    spatial_residuals.pdf
    spatial_sensitivity_maps.pdf

Run with the isolated interpreter that has pymc/arviz/libpysal.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
K_KNN, SEED = 6, 42

PAL = ["#1B6AA5", "#C1440E", "#6A3D9A", "#2E7D32", "#B8860B"]
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d9d9d9"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "grid.color": GRID,
    "grid.linewidth": 0.5, "figure.dpi": 200, "savefig.bbox": "tight",
})


def style(ax):
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    return ax


def main() -> None:
    import libpysal, esda
    import pymc as pm
    import arviz as az

    ref = json.loads((ROOT / "pipeline" / "spatial_refit.json").read_text(encoding="utf-8"))
    arr = np.load(ROOT / "pipeline" / "spatial_refit_residuals.npy")
    coords, log_beta, fitted_car = arr[:, :2], arr[:, 2], arr[:, 3]
    n = len(log_beta)
    w = libpysal.weights.KNN.from_array(coords, k=K_KNN)
    w.transform = "r"
    W = w.full()[0]
    Wsym = np.maximum(W, W.T)

    mu_c, rho_c = ref["log_car"]["mu"], ref["log_car"]["rho"]
    tau_c = 1.0 / ref["log_car"]["sigma"] ** 2
    Q = tau_c * (np.diag(W.sum(axis=1)) - rho_c * W)
    L = np.linalg.cholesky(Q)
    z = L.T @ (log_beta - mu_c)

    # ---- BYM2, retaining what the figures need -----------------------
    Qicar = np.diag(Wsym.sum(axis=1)) - Wsym
    ev, evec = np.linalg.eigh(Qicar)
    nz = ev > 1e-10
    ginv = (evec[:, nz] / ev[nz]) @ evec[:, nz].T
    c_scale = float(np.exp(np.mean(np.log(np.diag(ginv)))))
    with pm.Model():
        mu = pm.Normal("mu", mu=log_beta.mean(), sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        phi_z = pm.Normal("phi_z", 0.0, 1.0, shape=n)
        phi = phi_z - pm.math.mean(phi_z)          # not stored: recoverable from phi_z
        theta = pm.Normal("theta", 0.0, 1.0, shape=n)
        eta = pm.Deterministic(
            "eta", mu + sigma * (pm.math.sqrt(rho / c_scale) * phi + pm.math.sqrt(1 - rho) * theta))
        pm.Potential("icar", -0.5 * pm.math.dot(phi, pm.math.dot(Qicar, phi)))
        tau_obs = pm.Gamma("tau_obs", alpha=2.0, beta=1.0)
        pm.Normal("y", mu=eta, tau=tau_obs, observed=log_beta)
        idata = pm.sample(draws=4000, tune=2000, chains=4, target_accept=0.99,
                          random_seed=SEED, progressbar=False)
    rho_draws = idata.posterior["rho"].values.ravel()
    eta_hat = idata.posterior["eta"].values.reshape(-1, n).mean(axis=0)
    print(f"BYM2 rho mean = {rho_draws.mean():.4f}  eta range = "
          f"[{eta_hat.min():.3f}, {eta_hat.max():.3f}]")

    stamp = "superseded_2026-07-25"
    (FIG / stamp).mkdir(exist_ok=True)
    for t in ("spatial_analysis_beta_coefficients.pdf", "spatial_diagnostics.pdf",
              "spatial_residuals.pdf", "spatial_sensitivity_maps.pdf"):
        if (FIG / t).exists():
            shutil.copy2(FIG / t, FIG / stamp / t)

    km = coords / 1000.0
    lag = W @ (log_beta - log_beta.mean())

    # ---- 1. spatial analysis of log beta -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    ax = style(axes[0, 0])
    sc = ax.scatter(km[:, 0], km[:, 1], c=log_beta, cmap="viridis", s=11, zorder=2)
    fig.colorbar(sc, ax=ax, label=r"$\log\beta$")
    ax.set_title("(A) spatial distribution"); ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    ax = style(axes[0, 1])
    ax.scatter(log_beta - log_beta.mean(), lag, s=9, color=PAL[0], alpha=0.7, zorder=2)
    sl = np.polyfit(log_beta - log_beta.mean(), lag, 1)[0]
    xs = np.linspace((log_beta - log_beta.mean()).min(), (log_beta - log_beta.mean()).max(), 10)
    ax.plot(xs, sl * xs, color=PAL[1], linewidth=1.6, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.5); ax.axvline(0, color=INK2, linewidth=0.5)
    ax.set_title(rf"(B) Moran scatter, $I$ = {ref['moran_log_beta']['I']:.3f}")
    ax.set_xlabel(r"$\log\beta$ (centred)"); ax.set_ylabel("spatial lag")
    ax = style(axes[1, 0])
    ax.hist(log_beta, bins=32, color=PAL[0], edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_title("(C) distribution"); ax.set_xlabel(r"$\log\beta$"); ax.set_ylabel("municipalities")
    ax = style(axes[1, 1])
    q = np.digitize(log_beta, np.quantile(log_beta, [0.25, 0.5, 0.75]))
    for k in range(4):
        m = q == k
        ax.scatter(km[m, 0], km[m, 1], s=11, color=PAL[k], label=f"Q{k+1}", zorder=2)
    ax.set_title("(D) quartiles"); ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    ax.legend(frameon=False, labelcolor=INK, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "spatial_analysis_beta_coefficients.pdf"); plt.close(fig)
    print("  wrote spatial_analysis_beta_coefficients.pdf")

    # ---- 2. BYM2 diagnostics ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = style(axes[0])
    ax.hist(rho_draws, bins=45, density=True, color=PAL[0], edgecolor="white",
            linewidth=0.3, zorder=2)
    # Annotate the CANONICAL posterior mean of spatial_refit.json, not this run's own
    # sample mean. This script refits BYM2 to recover the draws the density needs, so its
    # sample mean differs from the canonical one by Monte Carlo error (0.625 vs 0.630);
    # printing its own value put the figure in contradiction with Section 3.3.
    rho_canon = float(ref["bym2"]["rho"])
    ax.axvline(rho_canon, color=INK, linestyle="--", linewidth=1.2, zorder=3)
    ax.annotate(rf"mean {rho_canon:.3f}", xy=(rho_canon, ax.get_ylim()[1] * 0.9),
                xytext=(5, 0), textcoords="offset points", color=INK, fontsize=7)
    print(f"  annotated canonical rho = {rho_canon:.4f} "
          f"(this run's sample mean: {rho_draws.mean():.4f})")
    ax.set_title(r"(A) BYM2 mixing parameter $\rho$"); ax.set_xlabel(r"$\rho$"); ax.set_ylabel("density")
    ax = style(axes[1])
    ax.plot(np.exp(log_beta), np.exp(eta_hat), ".", color=PAL[0], markersize=4, alpha=0.7, zorder=2)
    lim = [1.0, float(np.exp(log_beta).max()) * 1.02]
    ax.plot(lim, lim, color=INK, linestyle="--", linewidth=0.9, zorder=3)
    ax.set_title("(B) BYM2 posterior mean vs observed")
    ax.set_xlabel(r"observed $\beta$"); ax.set_ylabel(r"fitted $\beta$")
    fig.tight_layout(); fig.savefig(FIG / "spatial_diagnostics.pdf"); plt.close(fig)
    print("  wrote spatial_diagnostics.pdf")

    # ---- 3. residual diagnostics ------------------------------------
    rd = json.loads((ROOT / "pipeline" / "residual_diagnostics.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = style(axes[0])
    zc = z - z.mean()
    ax.scatter(zc, W @ zc, s=9, color=PAL[2], alpha=0.7, zorder=2)
    sl = np.polyfit(zc, W @ zc, 1)[0]
    xs = np.linspace(zc.min(), zc.max(), 10)
    ax.plot(xs, sl * xs, color=PAL[1], linewidth=1.6, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.5); ax.axvline(0, color=INK2, linewidth=0.5)
    ax.set_title(rf"(A) whitened residuals, $I$ = {rd['whitened']['I']:.3f} ($p$ = {rd['whitened']['p']:.2f})")
    ax.set_xlabel("whitened residual"); ax.set_ylabel("spatial lag")
    ax = style(axes[1])
    ax.plot(log_beta, fitted_car, ".", color=PAL[0], markersize=4, alpha=0.7, zorder=2)
    lim = [log_beta.min(), log_beta.max()]
    ax.plot(lim, lim, color=INK, linestyle="--", linewidth=0.9, zorder=3)
    ax.set_title(rf"(B) spatial LOOCV, RMSE = {ref['loocv']['rmse_log']:.3f}")
    ax.set_xlabel(r"observed $\log\beta$"); ax.set_ylabel(r"predicted $\log\beta$")
    fig.tight_layout(); fig.savefig(FIG / "spatial_residuals.pdf"); plt.close(fig)
    print("  wrote spatial_residuals.pdf")

    # ---- 4. maps -----------------------------------------------------
    resid = log_beta - eta_hat
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    vmin, vmax = float(min(log_beta.min(), eta_hat.min())), float(max(log_beta.max(), eta_hat.max()))
    for ax, vals, title, cmap, kw in (
            (axes[0], log_beta, r"(A) empirical $\log\beta$", "viridis", dict(vmin=vmin, vmax=vmax)),
            (axes[1], eta_hat, r"(B) BYM2 posterior mean", "viridis", dict(vmin=vmin, vmax=vmax)),
            (axes[2], resid, "(C) residuals", "RdBu_r",
             dict(vmin=-np.abs(resid).max(), vmax=np.abs(resid).max()))):
        style(ax)
        sc = ax.scatter(km[:, 0], km[:, 1], c=vals, cmap=cmap, s=9, zorder=2, **kw)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(title); ax.set_xlabel("UTM E (km)")
    axes[0].set_ylabel("UTM N (km)")
    fig.tight_layout(); fig.savefig(FIG / "spatial_sensitivity_maps.pdf"); plt.close(fig)
    print("  wrote spatial_sensitivity_maps.pdf")


if __name__ == "__main__":
    main()
