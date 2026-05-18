"""Generate the three new spatial-dependence figures for the manuscript:

  1. spatial_residuals.pdf
        - (A) Moran scatter of log-CAR posterior residuals
        - (B) spatial LOOCV: predicted vs. observed log(beta)

  2. spatial_sensitivity_maps.pdf
        - (A) empirical log(beta) per municipality
        - (B) BYM2 posterior mean log(beta)
        - (C) residuals = obs - BYM2

  3. spatial_analysis_beta_coefficients.pdf  (UPDATE of v1 figure)
        - (A) spatial distribution of log(beta)
        - (B) Moran scatter of log(beta) on KNN-6 weights
        - (C) histogram with quartile coloring
        - (D) quartile clustering on the map

All figures are written to submission/figures/. Inputs are the already
saved per-muni outputs (no model refit required).
"""

from __future__ import annotations

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
from matplotlib.colors import TwoSlopeNorm
import libpysal
from esda.moran import Moran

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parent
DATA = ANALYSES / "data"
SUB_FIG = ANALYSES.parent / "submission" / "figures"
SUB_FIG.mkdir(parents=True, exist_ok=True)

K_KNN = 6
SEED = 42


def load_data():
    df_logcar = pd.read_csv(HERE / "A1_spatial_results.csv")
    df_bym2 = pd.read_csv(HERE / "BYM2_results.csv")
    df_canon = pd.read_csv(DATA / "municipios_canonical.csv")

    # Merge on municipio (canonical key); A1 has all 382 with coords, BYM2 same
    df = df_logcar.merge(
        df_bym2[["municipio", "bym2_log_beta_pred", "bym2_beta_pred"]],
        on="municipio",
        how="inner",
    )
    print(f"Merged dataset: n = {len(df)}")
    return df


def build_W(df: pd.DataFrame) -> np.ndarray:
    coords = df[["utm_x", "utm_y"]].values
    w = libpysal.weights.KNN.from_array(coords, k=K_KNN)
    w.transform = "r"  # row-standardized
    return w


def fig_residuals(df: pd.DataFrame, w):
    """(A) Moran scatter of log-CAR residuals; (B) LOOCV pred vs obs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- (A) Moran scatter of log-CAR residuals ---
    res = df["residual"].values
    res_centered = res - res.mean()
    lag = libpysal.weights.lag_spatial(w, res_centered)
    np.random.seed(SEED)
    # 9999 permutations for tighter p-estimate; report both perm and analytic
    moran_res = Moran(res, w, permutations=9999)
    print(f"    Moran residuals: I={moran_res.I:+.4f}  z={moran_res.z_norm:+.3f}  "
          f"p_perm={moran_res.p_sim:.4f}  p_norm={moran_res.p_norm:.4f}")

    axes[0].scatter(res_centered, lag, alpha=0.45, s=22, c="steelblue",
                    edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color="gray", linewidth=0.6, linestyle=":")
    axes[0].axvline(0, color="gray", linewidth=0.6, linestyle=":")
    # Slope line
    slope = moran_res.I
    xx = np.linspace(res_centered.min(), res_centered.max(), 50)
    axes[0].plot(xx, slope * xx, "r--", linewidth=1.5,
                  label=f"Moran's $I = {moran_res.I:+.3f}$  ($p = {moran_res.p_sim:.3f}$)")
    axes[0].set_xlabel("log-CAR residual  $r_i = \\log\\beta_i - \\widehat{\\log\\beta_i}$")
    axes[0].set_ylabel("Spatial lag of residuals  $W r$")
    axes[0].set_title("(A)  Moran scatter of log-CAR residuals")
    axes[0].legend(loc="upper left", framealpha=0.9)
    axes[0].grid(alpha=0.25)

    # --- (B) Spatial LOOCV: pred vs obs ---
    obs = df["log_beta"].values
    pred = df["log_beta_loocv_pred"].values
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))

    axes[1].scatter(obs, pred, alpha=0.45, s=22, c="seagreen",
                    edgecolor="white", linewidth=0.4)
    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="1:1")
    axes[1].set_xlabel(r"Observed  $\log\beta_i$")
    axes[1].set_ylabel(r"LOOCV-predicted  $\widehat{\log\beta_i}$")
    axes[1].set_title(f"(B)  Spatial LOOCV  (RMSE $= {rmse:.3f}$, MAE $= {mae:.3f}$)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out = SUB_FIG / "spatial_residuals.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  (Moran I residuals = {moran_res.I:+.3f}, p = {moran_res.p_sim:.3f})")


def fig_sensitivity_maps(df: pd.DataFrame):
    """3-panel map: empirical log(beta), BYM2 posterior mean, residuals."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    obs = df["log_beta"].values
    fit = df["bym2_log_beta_pred"].values
    res = obs - fit
    x = df["utm_x"].values / 1000.0  # km
    y = df["utm_y"].values / 1000.0

    # Common color scale for (A) and (B)
    vmin = min(obs.min(), fit.min())
    vmax = max(obs.max(), fit.max())

    sc0 = axes[0].scatter(x, y, c=obs, s=24, cmap="viridis",
                           vmin=vmin, vmax=vmax,
                           edgecolor="white", linewidth=0.3)
    axes[0].set_title("(A)  Observed  $\\log\\beta_i$")
    axes[0].set_xlabel("UTM east (km)")
    axes[0].set_ylabel("UTM north (km)")
    axes[0].set_aspect("equal", adjustable="datalim")
    cb0 = plt.colorbar(sc0, ax=axes[0], shrink=0.85)
    cb0.set_label(r"$\log\beta$")

    sc1 = axes[1].scatter(x, y, c=fit, s=24, cmap="viridis",
                           vmin=vmin, vmax=vmax,
                           edgecolor="white", linewidth=0.3)
    axes[1].set_title("(B)  BYM2 posterior mean  $\\widehat{\\log\\beta_i}$")
    axes[1].set_xlabel("UTM east (km)")
    axes[1].set_aspect("equal", adjustable="datalim")
    cb1 = plt.colorbar(sc1, ax=axes[1], shrink=0.85)
    cb1.set_label(r"$\widehat{\log\beta}$")

    # Residuals: diverging colormap centered at 0
    abs_max = float(np.abs(res).max())
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    sc2 = axes[2].scatter(x, y, c=res, s=24, cmap="RdBu_r", norm=norm,
                           edgecolor="white", linewidth=0.3)
    axes[2].set_title("(C)  Residuals  $\\log\\beta_i - \\widehat{\\log\\beta_i}$")
    axes[2].set_xlabel("UTM east (km)")
    axes[2].set_aspect("equal", adjustable="datalim")
    cb2 = plt.colorbar(sc2, ax=axes[2], shrink=0.85)
    cb2.set_label("residual")

    fig.tight_layout()
    out = SUB_FIG / "spatial_sensitivity_maps.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  (residual range [{res.min():+.3f}, {res.max():+.3f}])")


def fig_spatial_analysis(df: pd.DataFrame, w):
    """Update the v1 4-panel figure to use log(beta) instead of raw beta."""
    log_beta = df["log_beta"].values
    log_beta_centered = log_beta - log_beta.mean()
    lag = libpysal.weights.lag_spatial(w, log_beta_centered)
    np.random.seed(SEED)
    moran = Moran(log_beta, w, permutations=9999)
    print(f"    Moran log(beta): I={moran.I:+.4f}  z={moran.z_norm:+.3f}  "
          f"p_perm={moran.p_sim:.4f}  p_norm={moran.p_norm:.4f}")

    x = df["utm_x"].values / 1000.0
    y = df["utm_y"].values / 1000.0

    # Quartile coloring
    q = pd.qcut(log_beta, q=4, labels=False)
    quartile_colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (A) Spatial distribution of log(beta)
    sc = axes[0, 0].scatter(x, y, c=log_beta, s=22, cmap="viridis",
                             edgecolor="white", linewidth=0.3)
    axes[0, 0].set_title(r"(A)  Spatial distribution of $\log\beta$")
    axes[0, 0].set_xlabel("UTM east (km)")
    axes[0, 0].set_ylabel("UTM north (km)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    cb = plt.colorbar(sc, ax=axes[0, 0], shrink=0.85)
    cb.set_label(r"$\log\beta$")

    # (B) Moran scatter
    axes[0, 1].scatter(log_beta_centered, lag, alpha=0.5, s=22, c="steelblue",
                        edgecolor="white", linewidth=0.4)
    xx = np.linspace(log_beta_centered.min(), log_beta_centered.max(), 50)
    axes[0, 1].plot(xx, moran.I * xx, "r--", linewidth=1.5,
                     label=f"$I = {moran.I:.3f}$  ($p = {moran.p_sim:.3f}$)")
    axes[0, 1].axhline(0, color="gray", linewidth=0.6, linestyle=":")
    axes[0, 1].axvline(0, color="gray", linewidth=0.6, linestyle=":")
    axes[0, 1].set_xlabel(r"$\log\beta_i$ (centered)")
    axes[0, 1].set_ylabel(r"Spatial lag $W \log\beta$")
    axes[0, 1].set_title(r"(B)  Moran scatter of $\log\beta$  (KNN-6)")
    axes[0, 1].legend(loc="upper left", framealpha=0.9)
    axes[0, 1].grid(alpha=0.25)

    # (C) Histogram with quartile coloring
    for qi, col in enumerate(quartile_colors):
        sel = q == qi
        axes[1, 0].hist(log_beta[sel], bins=20, color=col, alpha=0.8,
                         edgecolor="white", label=f"Q{qi+1}")
    axes[1, 0].set_xlabel(r"$\log\beta$")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(r"(C)  Histogram of $\log\beta$ by quartile")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25, axis="y")

    # (D) Spatial clustering by quartile
    for qi, col in enumerate(quartile_colors):
        sel = q == qi
        axes[1, 1].scatter(x[sel], y[sel], c=col, s=22,
                            edgecolor="white", linewidth=0.3, label=f"Q{qi+1}")
    axes[1, 1].set_xlabel("UTM east (km)")
    axes[1, 1].set_ylabel("UTM north (km)")
    axes[1, 1].set_title(r"(D)  Spatial clustering of $\log\beta$ quartiles")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = SUB_FIG / "spatial_analysis_beta_coefficients.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  (Moran I on log(beta) = {moran.I:+.3f}, p = {moran.p_sim:.3f})")


def main():
    print(f"Loading data...")
    df = load_data()
    w = build_W(df)
    print(f"Built KNN-{K_KNN} spatial weights.")

    print("\nGenerating figures:")
    fig_residuals(df, w)
    fig_sensitivity_maps(df)
    fig_spatial_analysis(df, w)
    print("\nDone.")


if __name__ == "__main__":
    main()
