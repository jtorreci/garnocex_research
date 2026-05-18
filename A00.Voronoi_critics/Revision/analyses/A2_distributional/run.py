"""A.2 - Distributional ranking with Wasserstein and Anderson-Darling.

Replaces Tables 3 and 4 of the manuscript (which use KS p-values
that the reviewer correctly rejects for n=9112).

Pipeline:
    1. Load full muni-muni beta dataset (post-filter beta>=1).
    2. Fit candidate distributions by MLE: Lognormal, Gamma, Weibull,
       Inverse Weibull (Frechet), Generalized Gamma.
    3. Score by:
        - Wasserstein-1 distance W1(F_emp, F_fit)
        - Anderson-Darling statistic
        - KS statistic + p-value (reference only)
    4. For each candidate, simulate predicted misallocation count using
       Theorem 1 with the fitted s, bootstrap CI.
    5. Compare to observed = 61 (consolidated).

Outputs:
    A2_distributional_ranking.csv
    A2_summary.json
    ../outputs/tables/A2_distributional_ranking.tex
    ../outputs/figures/A2_qq_plots.pdf
    ../outputs/figures/A2_predicted_vs_observed.pdf
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
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT_FIG = HERE.parent / "outputs" / "figures"
OUT_TAB = HERE.parent / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

OBSERVED_MISALLOCATIONS = 61   # canonical (consolidated)
N_MUNIS = 383
SEED = 42
N_BOOTSTRAP = 50    # each iter does MC sim (~20k samples), so keep modest


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


# Distribution wrappers -------------------------------------------------
class FitResult:
    def __init__(self, name, dist, params, ll):
        self.name = name
        self.dist = dist
        self.params = params
        self.ll = ll

    def cdf(self, x):
        return self.dist.cdf(x, *self.params)

    def pdf(self, x):
        return self.dist.pdf(x, *self.params)

    def ppf(self, q):
        return self.dist.ppf(q, *self.params)

    def rvs(self, size, rng):
        return self.dist.rvs(*self.params, size=size, random_state=rng)


# Paper convention: ln(beta) ~ N(m, s^2) with NO location constraint.
# This is the convention the published manuscript uses (m=0.166, s=0.093
# on n=9112 reproduces almost exactly, see audit_lognormal.py).
# Forcing floc=1.0 inflates s ~6x and breaks every prediction.

def fit_lognormal(x):
    # Free location: scipy will choose loc to maximise likelihood
    # (typically loc≈0 when β cluster around 1.2)
    s, loc, scale = stats.lognorm.fit(x)
    ll = stats.lognorm.logpdf(x, s, loc, scale).sum()
    return FitResult("Lognormal", stats.lognorm, (s, loc, scale), ll)


def fit_gamma(x):
    a, loc, scale = stats.gamma.fit(x)
    ll = stats.gamma.logpdf(x, a, loc, scale).sum()
    return FitResult("Gamma", stats.gamma, (a, loc, scale), ll)


def fit_weibull(x):
    c, loc, scale = stats.weibull_min.fit(x)
    ll = stats.weibull_min.logpdf(x, c, loc, scale).sum()
    return FitResult("Weibull", stats.weibull_min, (c, loc, scale), ll)


def fit_invweibull(x):
    c, loc, scale = stats.invweibull.fit(x)
    ll = stats.invweibull.logpdf(x, c, loc, scale).sum()
    return FitResult("Inverse Weibull (Frechet)", stats.invweibull, (c, loc, scale), ll)


def fit_gengamma(x):
    a, c, loc, scale = stats.gengamma.fit(x)
    ll = stats.gengamma.logpdf(x, a, c, loc, scale).sum()
    return FitResult("Generalized Gamma", stats.gengamma, (a, c, loc, scale), ll)


FITTERS = [fit_lognormal, fit_gamma, fit_weibull, fit_invweibull, fit_gengamma]
FITTER_BY_NAME = {
    "Lognormal": fit_lognormal,
    "Gamma": fit_gamma,
    "Weibull": fit_weibull,
    "Inverse Weibull (Frechet)": fit_invweibull,
    "Generalized Gamma": fit_gengamma,
}


# Goodness-of-fit metrics -----------------------------------------------
def wasserstein1(emp, fit: FitResult, n_grid: int = 5000):
    """W1 between empirical CDF and fitted CDF, computed by quantile match."""
    qs = np.linspace(1.0 / n_grid, 1.0 - 1.0 / n_grid, n_grid)
    emp_q = np.quantile(emp, qs)
    fit_q = fit.ppf(qs)
    return float(np.mean(np.abs(emp_q - fit_q)))


def ks_stat(emp, fit: FitResult):
    res = stats.kstest(emp, fit.cdf)
    return float(res.statistic), float(res.pvalue)


def anderson_darling_custom(emp, fit: FitResult):
    """A^2 statistic. We compute manually since scipy.stats.anderson is for
    specific dists only.
    """
    x = np.sort(emp)
    n = len(x)
    F = fit.cdf(x)
    # Avoid log(0) by clipping
    eps = 1e-12
    F = np.clip(F, eps, 1 - eps)
    i = np.arange(1, n + 1)
    A2 = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    return float(A2)


# Misallocation prediction ---------------------------------------------
def predict_misallocations_lognormal(s_param: float, R_values: np.ndarray) -> float:
    """Closed-form prediction for Lognormal: sum Phi(-ln R / (sqrt(2) s))."""
    p = stats.norm.cdf(-np.log(R_values) / (np.sqrt(2.0) * s_param))
    return float(p.sum())


def predict_misallocations_general(fr: "FitResult", R_values: np.ndarray,
                                   rng, n_sim: int = 50000) -> float:
    """For any distribution F, P[misalloc | R] = P[β1/β2 > R] with β_i ~ F i.i.d.

    Estimated by Monte Carlo: simulate large iid sample, compute ratios,
    then for each R count the fraction of ratios > R.
    """
    b1 = fr.rvs(n_sim, rng)
    b2 = fr.rvs(n_sim, rng)
    ratios = b1 / b2
    ratios_sorted = np.sort(ratios)
    # P[ratio > R] = (n - searchsorted(ratios, R, side='right')) / n
    n = len(ratios_sorted)
    counts = []
    for r in R_values:
        idx = np.searchsorted(ratios_sorted, r, side="right")
        counts.append((n - idx) / n)
    return float(np.sum(counts))


def load_betas(mode: str) -> tuple[np.ndarray, str]:
    """Load beta values according to the configured mode.

    mode = 'munimuni'   -> 90,300 muni-muni pairs (β=d_re/d_eu, β>=1)
    mode = 'muniplanta' -> 383 muni-planta pairs (one per muni, the assigned plant)
    """
    if mode == "munimuni":
        pairs = pd.read_csv(DATA / "beta_munimuni.csv")
        return pairs["beta"].values, "muni-muni (n=90,300 pairs)"
    elif mode == "muniplanta":
        canon = pd.read_csv(DATA / "municipios_canonical.csv")
        beta = canon["beta_assigned"].dropna().values
        beta = beta[beta >= 1.0]
        return beta, f"muni-planta (n={len(beta)})"
    else:
        raise ValueError(f"Unknown mode {mode!r}")


def run_one_mode(mode: str):
    banner(f"A.2 - DISTRIBUTIONAL RANKING [{mode}]")
    beta, label = load_betas(mode)
    print(f"  Loaded β: {label}")
    print(f"  beta: mean={beta.mean():.4f}  std={beta.std():.4f}  "
          f"min={beta.min():.4f}  max={beta.max():.4f}")

    # R values per muni: ratio of euclidean distance to 2nd-nearest vs nearest
    # CONSOLIDATED plant. Two physical plants in the same group count as one.
    df_eu_full = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv")
    # Plant consolidation map (same as build_canonical_dataset.py)
    PLANT_CONSOLIDATION = {
        2: 1, 12: 1, 17: 3, 15: 6, 40: 19, 32: 20,
        23: 22, 26: 22, 28: 22, 34: 27, 35: 27, 37: 27, 38: 36, 43: 36,
    }
    # The TargetID in this CSV is the plant municipality name, not numeric ID.
    # We reduce to per-muni minimum distance per (consolidated) plant via name dedup.
    R_values = []
    for muni, grp in df_eu_full.groupby("InputID"):
        # Take the minimum distance per plant TargetID (multiple rows per plant possible)
        d_min = grp.groupby("TargetID")["Distance"].min().values
        d_sorted = np.sort(d_min)
        if len(d_sorted) >= 2 and d_sorted[0] > 0:
            R_values.append(d_sorted[1] / d_sorted[0])
    R_values = np.array([r for r in R_values if np.isfinite(r)])
    print(f"  R values per muni (d2/d1, consolidated): n={len(R_values)}, "
          f"median={np.median(R_values):.4f}, mean={R_values.mean():.4f}, "
          f"<=1.10: {(R_values <= 1.10).sum()}")

    # 2. Fit candidate distributions
    banner("2. Fit candidate distributions (MLE)")
    fits = []
    for fitter in FITTERS:
        try:
            fr = fitter(beta)
            fits.append(fr)
            print(f"  + {fr.name}: params={tuple(round(p, 4) for p in fr.params)}, ll={fr.ll:.1f}")
        except Exception as e:
            print(f"  ! {fitter.__name__} failed: {e}")

    # 3. GoF scoring
    banner("3. Goodness of fit")
    rng = np.random.default_rng(SEED)
    rows = []
    for fr in fits:
        w1 = wasserstein1(beta, fr)
        ad = anderson_darling_custom(beta, fr)
        ks_s, ks_p = ks_stat(beta, fr)
        # Predicted misallocations using the FITTED distribution.
        # For Lognormal we use the closed form (Theorem 1) with
        # s = std(log(beta)) under the paper's convention.
        if fr.name == "Lognormal":
            s_param = float(np.log(beta).std(ddof=1))   # paper convention
            pred_point = predict_misallocations_lognormal(s_param, R_values)
        else:
            pred_point = predict_misallocations_general(fr, R_values, rng, n_sim=80000)
            sample = fr.rvs(20000, rng)
            s_param = float(np.std(np.log(np.maximum(sample, 1e-9))))

        # Bootstrap CI: refit the distribution on bootstrap samples.
        preds = [pred_point]
        fitter_fn = FITTER_BY_NAME[fr.name]
        for _ in range(N_BOOTSTRAP):
            beta_b = rng.choice(beta, size=len(beta), replace=True)
            try:
                fr_b = fitter_fn(beta_b)
            except Exception:
                fr_b = fr
            if fr_b.name == "Lognormal":
                s_b = float(np.log(beta_b).std(ddof=1))   # paper convention
                pred_b = predict_misallocations_lognormal(s_b, R_values)
            else:
                pred_b = predict_misallocations_general(fr_b, R_values, rng, n_sim=20000)
            preds.append(pred_b)
        preds = np.array(preds)
        ci_lo, ci_hi = np.percentile(preds, [2.5, 97.5])
        pred_mean = float(pred_point)

        rows.append({
            "distribution": fr.name,
            "params": str(tuple(round(p, 4) for p in fr.params)),
            "loglik": fr.ll,
            "wasserstein1": w1,
            "anderson_darling": ad,
            "ks_stat": ks_s,
            "ks_p": ks_p,
            "s_param_used": s_param,
            "pred_misalloc_mean": pred_mean,
            "pred_misalloc_ci_lo": float(ci_lo),
            "pred_misalloc_ci_hi": float(ci_hi),
            "observed_in_ci": bool(ci_lo <= OBSERVED_MISALLOCATIONS <= ci_hi),
        })
        print(f"  {fr.name}: W1={w1:.4f}  AD={ad:.2f}  KS={ks_s:.4f}  "
              f"pred={pred_mean:.1f} CI=[{ci_lo:.1f}, {ci_hi:.1f}] "
              f"{'CONTAINS' if ci_lo <= OBSERVED_MISALLOCATIONS <= ci_hi else 'misses'} obs={OBSERVED_MISALLOCATIONS}")

    df_results = pd.DataFrame(rows).sort_values("wasserstein1")
    df_results.to_csv(HERE / f"A2_distributional_ranking_{mode}.csv", index=False)
    print(f"\n  Wrote A2_distributional_ranking_{mode}.csv (sorted by W1)")

    # JSON summary
    summary = {
        "mode": mode,
        "label": label,
        "n_pairs": int(len(beta)),
        "observed_misallocations": OBSERVED_MISALLOCATIONS,
        "n_munis_for_R": int(len(R_values)),
        "best_by_wasserstein": str(df_results.iloc[0]["distribution"]),
        "best_by_AD": str(df_results.sort_values("anderson_darling").iloc[0]["distribution"]),
        "results": rows,
    }
    (HERE / f"A2_summary_{mode}.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote A2_summary_{mode}.json")

    # 4. Figures - QQ plots
    banner("4. Figures")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    rng = np.random.default_rng(SEED)
    for ax, fr in zip(axes, fits):
        # theoretical quantiles
        n = len(beta)
        qs = (np.arange(1, n + 1) - 0.5) / n
        theo = fr.ppf(qs)
        emp = np.sort(beta)
        ax.plot(theo, emp, "b.", alpha=0.3, markersize=2)
        lo = max(beta.min(), theo.min())
        hi = min(beta.max(), theo.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"Theoretical quantile ({fr.name})")
        ax.set_ylabel("Empirical quantile (β)")
        w1 = next(r["wasserstein1"] for r in rows if r["distribution"] == fr.name)
        ax.set_title(f"{fr.name}  (W₁={w1:.3f})")
        ax.set_xlim(1.0, np.percentile(beta, 99.5))
        ax.set_ylim(1.0, np.percentile(beta, 99.5))
    if len(fits) < 6:
        axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"A2_qq_plots_{mode}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG / f"A2_qq_plots_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote A2_qq_plots.{{pdf,png}}")

    # Predicted vs observed
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df_results.copy()
    y = np.arange(len(df_sorted))
    # Clamp xerr to be non-negative (could be near-zero due to deterministic predict)
    err_lo = np.maximum(df_sorted["pred_misalloc_mean"] - df_sorted["pred_misalloc_ci_lo"], 0.0)
    err_hi = np.maximum(df_sorted["pred_misalloc_ci_hi"] - df_sorted["pred_misalloc_mean"], 0.0)
    ax.errorbar(
        df_sorted["pred_misalloc_mean"], y,
        xerr=[err_lo, err_hi],
        fmt="o", capsize=4, color="steelblue",
    )
    ax.axvline(OBSERVED_MISALLOCATIONS, color="r", linestyle="--",
               label=f"Observed = {OBSERVED_MISALLOCATIONS}")
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["distribution"])
    ax.set_xlabel("Predicted misallocations (mean ± 95% CI)")
    ax.set_title("Predicted vs observed misallocations by candidate distribution")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"A2_predicted_vs_observed_{mode}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG / f"A2_predicted_vs_observed_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote A2_predicted_vs_observed.{{pdf,png}}")

    # LaTeX table
    lines = [
        f"% Auto-generated by A2_distributional/run.py [{mode}]",
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{Distributional ranking for $\\beta$ ({label}). "
        f"Sorted by Wasserstein-1 distance. Observed misallocations: "
        f"{OBSERVED_MISALLOCATIONS}/{N_MUNIS}.}}",
        f"\\label{{tab:distributional_ranking_{mode}}}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Distribution & $W_1$ & $A^2$ & KS & Predicted misalloc. (95\\% CI) & Contains obs? \\\\",
        "\\midrule",
    ]
    for _, r in df_results.iterrows():
        lines.append(
            f"{r['distribution']} & {r['wasserstein1']:.4f} & "
            f"{r['anderson_darling']:.2f} & {r['ks_stat']:.4f} & "
            f"{r['pred_misalloc_mean']:.1f} "
            f"[{r['pred_misalloc_ci_lo']:.1f}, {r['pred_misalloc_ci_hi']:.1f}] & "
            f"{'$\\checkmark$' if r['observed_in_ci'] else '$\\times$'} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (OUT_TAB / f"A2_distributional_ranking_{mode}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote A2_distributional_ranking_{mode}.tex")

    return df_results, summary


def main():
    results = {}
    for mode in ("munimuni", "muniplanta"):
        df_r, summ = run_one_mode(mode)
        results[mode] = (df_r, summ)

    # Combined comparison summary
    banner("COMBINED — muni-muni vs muni-planta")
    combined = {}
    for mode, (df_r, summ) in results.items():
        combined[mode] = {
            "n_pairs": summ["n_pairs"],
            "best_W1": summ["best_by_wasserstein"],
            "best_AD": summ["best_by_AD"],
            "lognormal_pred": next(r["pred_misalloc_mean"] for r in summ["results"]
                                   if r["distribution"] == "Lognormal"),
            "weibull_pred": next(r["pred_misalloc_mean"] for r in summ["results"]
                                 if r["distribution"] == "Weibull"),
        }
    (HERE / "A2_combined_summary.json").write_text(json.dumps(combined, indent=2))
    print("  Wrote A2_combined_summary.json")
    for mode, c in combined.items():
        print(f"  [{mode:11s}] n={c['n_pairs']:>6}  best W1={c['best_W1']}, "
              f"Lognormal pred={c['lognormal_pred']:.1f}, "
              f"Weibull pred={c['weibull_pred']:.1f}")

    banner("DONE")


if __name__ == "__main__":
    main()
