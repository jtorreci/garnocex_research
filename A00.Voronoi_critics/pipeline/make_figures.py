"""Regenerate the manuscript figures that depend on the calibration.

Reads the canonical OD runs and rewrites the seven affected figures. The four
spatial figures (log-CAR / BYM2 posteriors, residual maps) are NOT regenerated
here: they require the Bayesian refit, which needs the analyses/environment.yml
stack (PyMC, arviz). They are listed at the end as outstanding.

Superseded figures are moved to figures/superseded_<date>/ rather than overwritten.

Scope: this script deliberately produces a SUPERSET of what the manuscript prints. Three of
its outputs are not currently included in the paper --- histogram_ratio_dr_de,
beta_distribution_fitted_curves and distance_improvement_analysis --- because the R2
concision pass removed them: the first two duplicated the Q-Q panel, and the third restated
the misallocation rate that the text already gives. They are still generated on purpose.
They are the natural exploratory views of beta, they are cheap, and keeping them means the
selection of what to print is an editorial decision recorded in the manuscript rather than a
constraint imposed by the code. If a future revision needs one of them back, it is already
there and already consistent with the canonical calibration.

All figures are written as vector PDF. The Q-Q panel was previously a PNG for no reason;
it is a scatter of 383 points per panel and carries no raster content.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_case import CONSOLIDATION_KM, FITTERS, demojibake, predict_lognormal

ROOT = Path(__file__).resolve().parent.parent
OD = ROOT / "pipeline" / "od"
FIG = ROOT / "figures"
TOL = "5"

# Categorical palette, fixed order, validated with the dataviz validator
# (all checks PASS, worst adjacent pair dE 8.1 protan).
PAL = ["#1B6AA5", "#C1440E", "#6A3D9A", "#2E7D32", "#B8860B"]
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d9d9d9"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def style(ax):
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def save(fig, name):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- data
def load():
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv",
                      encoding="utf-8-sig").dropna(subset=["NOMBRE"]).reset_index(drop=True)
    mun["name"] = mun.NOMBRE.map(demojibake)
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
    rows = []
    for i in range(len(mun)):
        de = np.hypot(PX - MX[i], PY - MY[i]); dr = C[i]
        gm = pd.Series(de).groupby(grp).min().sort_values()
        g1, g2 = int(gm.index[0]), int(gm.index[1])
        c1 = np.where(grp == g1)[0]; p1 = c1[np.argmin(de[c1])]
        c2 = np.where(grp == g2)[0]; p2 = c2[np.argmin(de[c2])]
        ok = np.isfinite(dr)
        gn = int(pd.Series(dr[ok]).groupby(grp[ok]).min().idxmin()) if ok.any() else -1
        dr_net = float(np.nanmin(dr)) if ok.any() else np.nan
        p = np.array([MX[i], MY[i]])
        v1, v2 = np.array([PX[p1], PY[p1]]) - p, np.array([PX[p2], PY[p2]]) - p
        cos = np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        dmid = float(np.linalg.norm(np.array([PX[p1], PY[p1]]) - np.array([PX[p2], PY[p2]])) / 2)
        rows.append({"name": mun.name[i], "d_e": de[p1], "d_r": dr[p1],
                     "beta": dr[p1] / de[p1] if de[p1] > 0 else np.nan,
                     "R": gm.iloc[1] / gm.iloc[0] if gm.iloc[0] > 0 else np.nan,
                     "kappa": 2 * np.sin(np.arccos(cos) / 2) / dmid if dmid > 0 else np.nan,
                     "mis": int(gn != g1), "saving": dr[p1] - dr_net})
    return pd.DataFrame(rows), C, mun, pl


df, C, mun, pl = load()
b = df.beta.dropna().to_numpy(); b = b[b > 0]
s_b = float(np.log(b).std(ddof=1))
n_mis = int(df.mis.sum())
print(f"n={len(b)}  s_b={s_b:.4f}  misallocated={n_mis}")

stamp = "superseded_2026-07-25"
(FIG / stamp).mkdir(exist_ok=True)
TARGETS = ["histogram_ratio_dr_de.pdf", "beta_distribution_fitted_curves.pdf",
           "qq_plots_beta_distributions.pdf", "euclidean_vs_real_scatterplot.pdf",
           "distance_improvement_analysis.pdf", "safety_bands_voronoi_risk.pdf",
           "sensitivity_s_parameter.pdf"]
for t in TARGETS:
    if (FIG / t).exists():
        shutil.copy2(FIG / t, FIG / stamp / t)
print(f"  previous versions copied to figures/{stamp}/")

# ------------------------------------------------- 1. histogram of beta
fig, ax = plt.subplots(figsize=(3.4, 2.4))
style(ax)
ax.hist(b, bins=40, color=PAL[0], alpha=0.85, edgecolor="white", linewidth=0.4, zorder=2)
ax.axvline(np.median(b), color=INK, linestyle="--", linewidth=1.2, zorder=3)
ax.annotate(f"median {np.median(b):.2f}", xy=(np.median(b), ax.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords="offset points", color=INK, fontsize=7)
ax.set_xlabel(r"$\beta = d_r / d_e$ (assigned facility)")
ax.set_ylabel("municipalities")
save(fig, "histogram_ratio_dr_de.pdf")

# ------------------------------------- 2. fitted densities over the data
fig, ax = plt.subplots(figsize=(3.4, 2.4))
style(ax)
ax.hist(b, bins=40, density=True, color=GRID, edgecolor="white", linewidth=0.4, zorder=1)
x = np.linspace(1.0, np.percentile(b, 99), 400)
for (name, dist), col in zip(FITTERS.items(), PAL):
    p = dist.fit(b)
    ax.plot(x, dist.pdf(x, *p), color=col, linewidth=1.6, zorder=3,
            label=name.replace(" (Frechet)", ""))
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("density")
ax.legend(frameon=False, labelcolor=INK)
save(fig, "beta_distribution_fitted_curves.pdf")

# ---------------------------------------------------------- 3. QQ plots
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
qs = (np.arange(1, len(b) + 1) - 0.5) / len(b)
emp = np.sort(b)
for ax, ((name, dist), col) in zip(axes.ravel(), zip(FITTERS.items(), PAL)):
    style(ax)
    p = dist.fit(b)
    theo = dist.ppf(qs, *p)
    ax.plot(theo, emp, ".", color=col, markersize=3, alpha=0.7, zorder=2)
    lo, hi = 1.0, np.percentile(b, 99)
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=0.9, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(name.replace(" (Frechet)", ""), color=INK)
    ax.set_xlabel("theoretical"); ax.set_ylabel("empirical")
axes.ravel()[-1].axis("off")
fig.tight_layout()
save(fig, "qq_plots_beta_distributions.pdf")

# ---------------------------------------- 4. euclidean vs network scatter
fig, ax = plt.subplots(figsize=(3.4, 2.6))
style(ax)
ok = df.d_e.notna() & df.d_r.notna()
ax.plot(df.d_e[ok] / 1000, df.d_r[ok] / 1000, ".", color=PAL[0], markersize=3.5,
        alpha=0.6, zorder=2, label="municipality")
lim = float(np.nanmax(df.d_r) / 1000) * 1.02
ax.plot([0, lim], [0, lim], color=INK, linestyle="--", linewidth=0.9, zorder=3)
ax.annotate(r"$d_r = d_e$", xy=(lim * 0.72, lim * 0.72), xytext=(4, -10),
            textcoords="offset points", color=INK2, fontsize=7)
ax.set_xlabel(r"Euclidean distance $d_e$ (km)")
ax.set_ylabel(r"network distance $d_r$ (km)")
save(fig, "euclidean_vs_real_scatterplot.pdf")

# --------------------------------- 5. distance saving of the reassignments
sav = df.loc[df.mis == 1, "saving"].dropna() / 1000
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = style(axes[0])
ax.hist(sav, bins=18, color=PAL[1], alpha=0.9, edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xlabel("distance saved by reassignment (km)")
ax.set_ylabel("municipalities")
ax.annotate(f"median {sav.median():.1f} km", xy=(0.97, 0.9), xycoords="axes fraction",
            ha="right", color=INK, fontsize=7)
ax = style(axes[1])
vals = [len(df) - n_mis, n_mis]
ax.barh([1, 0], vals, color=[PAL[3], PAL[1]], height=0.55, zorder=2)
for y, v in zip([1, 0], vals):
    ax.annotate(f"{v}  ({100*v/len(df):.1f}%)", xy=(v, y), xytext=(5, 0),
                textcoords="offset points", va="center", color=INK, fontsize=7)
ax.set_yticks([1, 0]); ax.set_yticklabels(["Voronoi assignment\nconfirmed", "reassigned by\nnetwork distance"])
ax.set_xlabel("municipalities"); ax.set_xlim(0, len(df) * 1.18)
fig.tight_layout()
save(fig, "distance_improvement_analysis.pdf")

# ---------------------------------------------------- 6. safety bands
fig, ax = plt.subplots(figsize=(3.6, 2.6))
style(ax)
t = np.linspace(0, 12000, 300)
for kap, col in zip(np.percentile(df.kappa.dropna(), [25, 50, 75]), PAL):
    prob = stats.norm.cdf(-kap * t / (np.sqrt(2) * s_b))
    ax.plot(t / 1000, prob, color=col, linewidth=1.7, zorder=3,
            label=rf"$\kappa$ = {kap*1000:.2f} km$^{{-1}}$")
ax.axhline(0.10, color=INK, linestyle="--", linewidth=0.9, zorder=2)
ax.annotate(r"$q^* = 0.10$", xy=(11.5, 0.10), xytext=(0, 5), textcoords="offset points",
            ha="right", color=INK2, fontsize=7)
ax.set_xlabel(r"normal distance to the Voronoi boundary $|t|$ (km)")
ax.set_ylabel("misallocation probability")
ax.set_ylim(0, 0.52)
ax.legend(frameon=False, labelcolor=INK, title=rf"$s_b$ = {s_b:.3f}", title_fontsize=7)
save(fig, "safety_bands_voronoi_risk.pdf")

# ------------------------------------------- 7. sensitivity to s
fig, ax = plt.subplots(figsize=(3.6, 2.6))
style(ax)
Rv = df.R.dropna().to_numpy()
grid = np.linspace(0.05, 0.45, 90)


def poisson_binomial_quantiles(probabilities, quantiles):
    """Exact quantiles for the sum of independent, unequal Bernoulli trials."""
    pmf = np.zeros(len(probabilities) + 1)
    pmf[0] = 1.0
    for probability in probabilities:
        pmf[1:] = pmf[1:] * (1.0 - probability) + pmf[:-1] * probability
        pmf[0] *= 1.0 - probability
    cdf = np.cumsum(pmf)
    return np.searchsorted(cdf, quantiles, side="left")


rate = []
intervals = {0.50: [], 0.80: [], 0.95: []}
for s in grid:
    probabilities = stats.norm.cdf(-np.log(Rv) / (np.sqrt(2.0) * s))
    rate.append(100.0 * probabilities.sum() / len(Rv))
    for coverage in intervals:
        alpha = (1.0 - coverage) / 2.0
        bounds = poisson_binomial_quantiles(probabilities, [alpha, 1.0 - alpha])
        intervals[coverage].append(100.0 * bounds / len(Rv))
rate = np.asarray(rate)
intervals = {coverage: np.asarray(bounds) for coverage, bounds in intervals.items()}

for coverage, alpha in ((0.95, 0.10), (0.80, 0.18), (0.50, 0.28)):
    bounds = intervals[coverage]
    ax.fill_between(grid, bounds[:, 0], bounds[:, 1], color=PAL[0], alpha=alpha,
                    zorder=1, linewidth=0, label=f"{int(100 * coverage)}% predictive interval")
ax.plot(grid, rate, color=PAL[0], linewidth=1.8, zorder=3, label="expected rate")

predicted_at_fit = 100.0 * predict_lognormal(s_b, Rv) / len(Rv)
observed = 100.0 * n_mis / len(df)
ax.plot([s_b], [predicted_at_fit], "o", color=PAL[1], markersize=6, zorder=5)
ax.annotate(f"central prediction\n{predicted_at_fit:.1f}% at $s_b={s_b:.3f}$",
            xy=(s_b, predicted_at_fit), xytext=(9, 9), textcoords="offset points",
            color=INK, fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
ax.axhline(observed, color=INK2, linestyle="--", linewidth=0.9, zorder=2)
ax.annotate(f"observed {observed:.1f}%", xy=(0.445, observed), xytext=(0, -9),
            textcoords="offset points", ha="right", color=INK2, fontsize=7)
ax.set_xlabel(r"dispersion parameter $s$")
ax.set_ylabel("misallocation rate (%)")
ax.legend(frameon=False, loc="upper left")
save(fig, "sensitivity_s_parameter.pdf")

print("\nNOT regenerated (require the Bayesian refit, analyses/environment.yml):")
for f in ("spatial_analysis_beta_coefficients.pdf", "spatial_diagnostics.pdf",
          "spatial_residuals.pdf", "spatial_sensitivity_maps.pdf"):
    print(f"  {f}")
