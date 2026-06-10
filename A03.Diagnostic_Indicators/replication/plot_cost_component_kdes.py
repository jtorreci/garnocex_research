# -*- coding: utf-8 -*-
"""
plot_cost_component_kdes.py
===========================
Overlaid kernel-density estimates of the per-municipality unit cost, split into
transport / treatment / total, for the three scenarios S1 (proximity),
S2 (real observed use) and S3 (cost-optimal). Reads muni_costs_{s1,s2,s3}.csv
exported by compute_three_scenarios.py.

The figure makes two things visible at a glance:
  - transport: S2's pathological long-haul tail and how S3 reshapes it;
  - treatment: the concentration of treatment cost around the optimum under S3;
  - total: the net distributional effect.
Dashed vertical lines mark the production-weighted means.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    def kde_curve(values, xs):
        return gaussian_kde(values)(xs)
except Exception:  # scipy not available -> simple Gaussian KDE fallback
    def kde_curve(values, xs):
        values = np.asarray(values, float)
        n = len(values)
        bw = 1.06 * values.std(ddof=1) * n ** (-1 / 5)  # Silverman
        bw = max(bw, 1e-6)
        diff = (xs[:, None] - values[None, :]) / bw
        return np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})

SCEN = {
    "s1": {"label": "S1 proximity", "color": "#6f6f6f"},
    "s2": {"label": "S2 real",      "color": "#e8820c"},
    "s3": {"label": "S3 optimised", "color": "#2ca02c"},
}
data = {k: pd.read_csv(os.path.join(SCRIPT_DIR, f"muni_costs_{k}.csv")) for k in SCEN}

COMPONENTS = [
    ("transport", "(a) Transport cost"),
    ("treatment", "(b) Treatment cost"),
    ("total",     "(c) Total unit cost"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
for ax, (col, title) in zip(axes, COMPONENTS):
    allv = np.concatenate([data[k][col].values for k in SCEN])
    allv = allv[np.isfinite(allv)]
    # full data range so the pathological tails stay visible
    lo, hi = allv.min(), allv.max()
    pad = 0.04 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 600)
    ymax = 0.0
    for j, (k, meta) in enumerate(SCEN.items()):
        v = data[k][col].values
        v = v[np.isfinite(v)]
        y = kde_curve(v, xs)
        ymax = max(ymax, y.max())
        ax.plot(xs, y, color=meta["color"], lw=1.8, label=meta["label"], zorder=3)
        ax.fill_between(xs, y, color=meta["color"], alpha=0.10, zorder=2)
        wmean = np.average(data[k][col].values, weights=data[k]["prod"].values)
        ax.axvline(wmean, color=meta["color"], lw=1.0, ls="--", alpha=0.85, zorder=4)
    # rug: one row of ticks per scenario below the axis -> individual municipalities, incl. tail
    rug_h = 0.045 * ymax
    for j, (k, meta) in enumerate(SCEN.items()):
        v = data[k][col].values
        v = v[np.isfinite(v)]
        y0 = -(j + 1) * rug_h * 1.3
        ax.plot(v, np.full_like(v, y0), "|", color=meta["color"],
                ms=4, alpha=0.45, markeredgewidth=0.6, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("EUR/t")
    ax.set_yticks([])
    ax.set_ylim(bottom=-(len(SCEN) + 0.5) * rug_h * 1.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Per-municipality unit-cost distributions across scenarios "
             "(dashed lines: production-weighted means)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = os.path.join(SCRIPT_DIR, "cost_component_kdes")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("Saved:", out + ".png / .pdf")
