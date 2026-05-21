# -*- coding: utf-8 -*-
"""
genera_figuras_articulo.py
==========================
Generates publication-quality figures for A01 using recalculated data
(real road-distance assignment, log₂ cost model, consolidated plants).

Figures generated:
  - Fig 8:  Stacked bar charts (cost components by plant and municipality)
  - Fig 9:  Violin plots (cost distribution across 3 analytical levels)
  - Fig 10: Histograms (cost component distributions)
  - Fig 11: Cost model curve (piecewise log₂ with sensitivity)

All figures use grayscale + red accents for journal compatibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import math
import os

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "datos_red_real")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Latex", "Imagenes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Cost model parameters (must match recalcula_costes_red_real.py)
C0 = 40_000
T0 = 5_000
V_TREAT = 0.35
RHO_TRANS = 0.35

GRAY_LIGHT = "0.75"
GRAY_DARK = "0.30"
RED = "#C03030"


def fixed_cost_total(T):
    if T <= 0: return 0.0
    if T < T0: return C0
    return C0 * (math.log2(T / T0) + 1)


def treatment_cost_per_tonne(T):
    if T <= 0: return 0.0
    return fixed_cost_total(T) / T + V_TREAT


# ============================================================
# LOAD DATA
# ============================================================
df_plantas = pd.read_csv(os.path.join(DATA_DIR, "datos_red_real_plantas.csv"))
df_mun = pd.read_csv(os.path.join(DATA_DIR, "datos_red_real_municipios.csv"))

print(f"Data loaded: {len(df_plantas)} plants, {len(df_mun)} municipalities")

# Build per-tonne dataset (weighted expansion for violin/histogram)
tonne_records = []
for _, row in df_mun.iterrows():
    n = max(1, int(row["produccion_t"]))
    tonne_records.extend([{
        "C_trans": row["C_trans"],
        "C_trat": row["C_trat"],
        "C_tot": row["C_tot"],
    }] * n)
df_tonne = pd.DataFrame(tonne_records)
print(f"Per-tonne records: {len(df_tonne)}")


# ============================================================
# FIG 8: STACKED BAR CHARTS
# ============================================================
def generate_stacked_bars():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    datasets = [
        (df_plantas, "Plants", 0.8, "black", 0.6),
        (df_mun, "Municipalities", 1.0, "none", 0.0),
    ]

    for col, (df, label, width, ec, lw) in enumerate(datasets):
        df_s = df.sort_values("C_tot", ascending=False).reset_index(drop=True)
        x = np.arange(len(df_s))
        trat = df_s["C_trat"]
        trans = df_s["C_tot"] - df_s["C_trat"]

        # Absolute bars
        axes[0, col].bar(x, trat, color=GRAY_LIGHT, width=width,
                         label="Treatment", edgecolor=ec, linewidth=lw)
        axes[0, col].bar(x, trans, bottom=trat, color=GRAY_DARK, width=width,
                         label="Transport", edgecolor=ec, linewidth=lw)
        axes[0, col].axhline(trat.mean(), color=RED, ls="--", lw=2,
                             label=f"Mean treatment: {trat.mean():.1f}")
        axes[0, col].axhline(df_s["C_tot"].mean(), color=RED, ls="-", lw=2,
                             label=f"Mean total: {df_s['C_tot'].mean():.1f}")
        axes[0, col].set_title(f"Cost per {label} (\u20ac/t)")
        axes[0, col].set_ylabel("\u20ac/t" if col == 0 else "")
        if col == 0:  # Plants: number them all
            axes[0, col].set_xticks(x)
            axes[0, col].set_xticklabels([str(i+1) for i in x], fontsize=6, rotation=0)
        else:  # Municipalities: too many to label one-by-one, sparse numbering
            tick_pos = np.linspace(0, len(x) - 1, num=10, dtype=int)
            tick_pos = sorted(set(tick_pos.tolist()))
            axes[0, col].set_xticks(tick_pos)
            axes[0, col].set_xticklabels([str(i + 1) for i in tick_pos], fontsize=7, rotation=0)
        axes[0, col].set_xlabel(label)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(axis="y", alpha=0.3)

        # Percentage bars
        pct_trat = 100 * trat / df_s["C_tot"]
        pct_trans = 100 - pct_trat
        axes[1, col].bar(x, pct_trat, color=GRAY_LIGHT, width=width,
                         label="% Treatment", edgecolor=ec, linewidth=lw)
        axes[1, col].bar(x, pct_trans, bottom=pct_trat, color=GRAY_DARK, width=width,
                         label="% Transport", edgecolor=ec, linewidth=lw)
        axes[1, col].axhline(pct_trat.mean(), color=RED, ls="--", lw=2,
                             label=f"Mean % treatment: {pct_trat.mean():.0f}%")
        axes[1, col].set_title(f"Component share per {label}")
        axes[1, col].set_ylabel("%" if col == 0 else "")
        if col == 0:
            axes[1, col].set_xticks(x)
            axes[1, col].set_xticklabels([str(i+1) for i in x], fontsize=6, rotation=0)
        else:
            tick_pos = np.linspace(0, len(x) - 1, num=10, dtype=int)
            tick_pos = sorted(set(tick_pos.tolist()))
            axes[1, col].set_xticks(tick_pos)
            axes[1, col].set_xticklabels([str(i + 1) for i in tick_pos], fontsize=7, rotation=0)
        axes[1, col].set_xlabel(label)
        axes[1, col].set_ylim(0, 105)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(axis="y", alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "barras_apiladas_costes_componentes.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIG 9: VIOLIN PLOTS
# ============================================================
def generate_violin_plots():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True,
                             constrained_layout=True)

    cost_cols = ["C_trans", "C_trat", "C_tot"]
    titles = ["Transportation Cost", "Treatment Cost", "Total Cost"]
    datasets = [df_plantas, df_mun, df_tonne]
    approach_labels = ["By Plant", "By Municipality", "By Tonne"]
    grays = ["0.80", "0.55", "0.30"]

    for ax, col, title in zip(axes, cost_cols, titles):
        parts_data = [ds[col].dropna().values for ds in datasets]

        vp = ax.violinplot(parts_data, positions=range(3), showextrema=False,
                           widths=0.75)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(grays[i])
            body.set_edgecolor("black")
            body.set_linewidth(0.8)
            body.set_alpha(0.85)

        # Quartiles + mean
        overall_mean = np.concatenate(parts_data).mean()
        ax.axhline(overall_mean, color=RED, ls="-", lw=2.5, alpha=0.8)

        for i, data in enumerate(parts_data):
            q1, q2, q3 = np.percentile(data, [25, 50, 75])
            y_lo, y_hi = np.percentile(data, [5, 95])
            ax.vlines(i, y_lo, y_hi, color="steelblue", lw=1.2)
            for yy in [q1, q2, q3]:
                ax.scatter(i, yy, s=50, color="white", edgecolor="black", zorder=3)
            ax.scatter(i, data.mean(), s=50, color=RED, edgecolor="black", zorder=3)

        ax.set_title(title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(approach_labels)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Cost (\u20ac/t)")

        legend_els = [
            Line2D([0], [0], color=RED, lw=2.5,
                   label=f"Overall mean: {overall_mean:.2f}"),
            Line2D([0], [0], color=RED, marker="o", ls="None", ms=7,
                   mfc=RED, mec="black", label="Group mean"),
            Line2D([0], [0], color="white", marker="o", ls="None", ms=7,
                   mfc="white", mec="black", label="Q1, Q2, Q3"),
            Line2D([0], [0], color="steelblue", lw=1.2, label="5\u201395% range"),
        ]
        ax.legend(handles=legend_els, fontsize=8, loc="upper right")

    path = os.path.join(OUTPUT_DIR, "violin_plot.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIG 10: HISTOGRAMS + KDE
# ============================================================
def generate_histograms():
    fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)

    datasets = [df_plantas, df_mun, df_tonne]
    approach_labels = ["By Plant", "By Municipality", "By Tonne"]
    cost_cols = ["C_trans", "C_trat", "C_tot"]
    titles = ["Transportation", "Treatment", "Total"]
    grays = ["0.80", "0.55", "0.30"]

    for i, (df, alabel, gray) in enumerate(zip(datasets, approach_labels, grays)):
        for j, (col, title) in enumerate(zip(cost_cols, titles)):
            ax = axes[i, j]
            data = df[col].dropna()
            ax.hist(data, bins=25, density=True, color=gray, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
            # KDE
            from scipy.stats import gaussian_kde
            if len(data) > 5:
                kde = gaussian_kde(data, bw_method=0.3)
                xr = np.linspace(data.min(), data.max(), 200)
                ax.plot(xr, kde(xr), color="black", lw=1.5)
            mean_val = data.mean()
            ax.axvline(mean_val, color=RED, ls="--", lw=2,
                       label=f"Mean: {mean_val:.2f}")
            ax.set_title(f"{alabel} \u2014 {title}")
            ax.set_xlabel("\u20ac/t")
            ax.set_ylabel("Density" if j == 0 else "")
            ax.legend(fontsize=8)

    # Row 4: KDE overlay comparison
    for j, (col, title) in enumerate(zip(cost_cols, titles)):
        ax = axes[3, j]
        for df, alabel, gray in zip(datasets, approach_labels, grays):
            data = df[col].dropna()
            if len(data) > 5:
                kde = gaussian_kde(data, bw_method=0.3)
                xr = np.linspace(0, data.quantile(0.99), 200)
                ax.plot(xr, kde(xr), color=gray, lw=2.2, label=alabel)
        ax.set_title(f"Comparison \u2014 {title}")
        ax.set_xlabel("\u20ac/t")
        ax.set_ylabel("Density" if j == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "histograms.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIG 11: COST MODEL CURVE (log₂ piecewise)
# ============================================================
def generate_cost_model_curve():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    T = np.linspace(200, 100_000, 1000)

    # Base curves
    C_fix_total = np.array([fixed_cost_total(t) for t in T])
    C_fix_unit = C_fix_total / T

    # Sensitivity: C0
    for C0_alt, ls, label in [(30_000, "--", r"$C_0$ = 30k"), (50_000, "--", r"$C_0$ = 50k")]:
        C_alt = np.array([C0_alt if t < T0 else C0_alt * (math.log2(t/T0) + 1) for t in T])
        ax1.plot(T, C_alt, color="steelblue", ls=ls, lw=1.5, label=label)
        ax2.plot(T, C_alt / T, color="steelblue", ls=ls, lw=1.5, label=label)

    # Sensitivity: T0
    for T0_alt, ls, label in [(3_000, "-.", r"$T_0$ = 3k"), (7_000, "-.", r"$T_0$ = 7k")]:
        C_alt = np.array([C0 if t < T0_alt else C0 * (math.log2(t/T0_alt) + 1) for t in T])
        ax1.plot(T, C_alt, color="forestgreen", ls=ls, lw=1.5, label=label)
        ax2.plot(T, C_alt / T, color="forestgreen", ls=ls, lw=1.5, label=label)

    # Base (on top)
    ax1.plot(T, C_fix_total, color="black", lw=2.5,
             label=rf"Base: $C_0$={C0/1e3:.0f}k, $T_0$={T0/1e3:.0f}k")
    ax2.plot(T, C_fix_unit, color="black", lw=2.5,
             label=rf"Base: $C_0$={C0/1e3:.0f}k, $T_0$={T0/1e3:.0f}k")

    # Variable cost line on right panel
    ax2.axhline(V_TREAT, color=RED, ls=":", lw=1.5,
                label=f"Variable cost v = {V_TREAT}")

    # Reference lines
    for ax in [ax1, ax2]:
        ax.axvline(T0, color=RED, ls=":", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Annual throughput, $T_i$ (t/year)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left" if ax == ax1 else "upper right")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel("Total fixed cost, $C_{\\mathrm{fix}}$ (\u20ac/year)")
    ax2.set_ylabel("Treatment cost per tonne, $C_{\\mathrm{treat},i}$ (\u20ac/t)")
    ax1.set_title("Total Fixed Cost vs. Throughput")
    ax2.set_title("Treatment Cost per Tonne vs. Throughput")

    # Annotate T0
    ax1.text(T0 * 1.15, C0 * 0.6, f"$T_0 = {T0}$ t/y", color=RED, fontsize=9)

    # Overlay actual plant data on right panel
    ax2.scatter(df_plantas["produccion_total"], df_plantas["C_trat"],
                c="black", s=30, alpha=0.7, zorder=5, label="Observed plants")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Ctreat_function.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Generating figures for A01 manuscript...")
    print()
    generate_stacked_bars()
    generate_violin_plots()
    generate_histograms()
    generate_cost_model_curve()
    print()
    print("All figures generated in:", OUTPUT_DIR)
