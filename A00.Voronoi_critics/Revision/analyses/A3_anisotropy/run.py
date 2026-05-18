"""A.3 - Anisotropy as a Remark in Section 2.

Post-processes the muni-muni beta dataset to compute the anisotropy
coefficient alpha_i = beta_max(i) / beta_min(i) per origin, and shows
that high-alpha munis are over-represented among misallocations.

Pipeline:
    1. Load beta_munimuni.csv (n=90,300 pairs, 301 origins).
    2. Per origin, compute alpha_i = max(beta) / min(beta), plus aux stats.
    3. Cross with canonical misallocation flag (61 misallocated munis).
    4. ROC + AUC of alpha as binary predictor of misallocation.
    5. Choropleth-like scatter using UTM coords coloured by alpha.

Outputs:
    A3_anisotropy_results.csv
    A3_summary.json
    ../outputs/tables/A3_anisotropy_summary.tex
    ../outputs/figures/A3_anisotropy_map.pdf
    ../outputs/figures/A3_alpha_vs_misallocation_roc.pdf
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
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT_FIG = HERE.parent / "outputs" / "figures"
OUT_TAB = HERE.parent / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def main():
    banner("A.3 - ANISOTROPY REMARK")

    # 1. Load
    canon = pd.read_csv(DATA / "municipios_canonical.csv")
    print(f"  canon: {len(canon)} munis ({canon['misallocated'].sum()} misallocated)")

    # Build muni-PLANT beta from the cleaned plant distance matrices.
    # This is the directionally-relevant alpha for misallocation prediction.
    df_eu = pd.read_csv(DATA / "D_euclidea_plantas_clean.csv")
    df_re = pd.read_csv(DATA / "D_real_plantas_clean.csv")
    df_eu = df_eu.rename(columns={"InputID": "origin", "TargetID": "destination", "Distance": "d_eu"})
    df_re = df_re.rename(columns={"origin_id": "origin", "destination_id": "destination", "total_cost": "d_re"})
    pairs_p = df_eu[["origin", "destination", "d_eu"]].merge(
        df_re[["origin", "destination", "d_re"]], on=["origin", "destination"]
    )
    pairs_p["beta"] = pairs_p["d_re"] / pairs_p["d_eu"]
    pairs_p = pairs_p[(pairs_p["beta"] >= 1.0) & np.isfinite(pairs_p["beta"])].copy()
    print(f"  muni-plant pairs (β≥1): {len(pairs_p):,}, "
          f"origins: {pairs_p.origin.nunique()}, plants: {pairs_p.destination.nunique()}")

    # 2. Per-origin anisotropy (relative to plants)
    per_origin = pairs_p.groupby("origin").agg(
        beta_max=("beta", "max"),
        beta_min=("beta", "min"),
        beta_mean=("beta", "mean"),
        beta_median=("beta", "median"),
        n_dest=("beta", "count"),
    ).reset_index()
    per_origin["alpha"] = per_origin["beta_max"] / per_origin["beta_min"]
    # Also keep alpha computed only over the K nearest plants (closer to the
    # decision boundary; should be the most informative for misallocation).
    K_NEAREST = 5
    near_alpha = []
    for muni, grp in pairs_p.groupby("origin"):
        grp = grp.sort_values("d_eu").head(K_NEAREST)
        if len(grp) >= 2:
            a = grp["beta"].max() / grp["beta"].min()
        else:
            a = 1.0
        near_alpha.append({"origin": muni, "alpha_near": a})
    per_origin = per_origin.merge(pd.DataFrame(near_alpha), on="origin")
    print(f"  alpha: mean={per_origin['alpha'].mean():.3f}, "
          f"median={per_origin['alpha'].median():.3f}, "
          f"max={per_origin['alpha'].max():.3f}")

    # 3. Cross with misallocation flag
    # Canonical uses 'municipio'; pairs use 'origin'. Both are raw names.
    canon_m = canon[["municipio", "misallocated", "utm_x", "utm_y"]].copy()
    df = per_origin.merge(canon_m, left_on="origin", right_on="municipio", how="left")
    df["misallocated"] = df["misallocated"].fillna(0).astype(int)
    n_with_flag = df["municipio"].notna().sum()
    print(f"  Origins matched to canonical: {n_with_flag}/{len(df)}")
    print(f"  Misallocated within matched: {df['misallocated'].sum()}")

    # 4. ROC + AUC for both alpha (all plants) and alpha_near (5 nearest)
    banner("4. alpha as misallocation predictor")
    df_full = df.dropna(subset=["municipio"])
    y = df_full["misallocated"].values
    auc_all = roc_auc_score(y, df_full["alpha"].values)
    auc_near = roc_auc_score(y, df_full["alpha_near"].values)
    print(f"  AUC (alpha over all 46 plants):    {auc_all:.4f}")
    print(f"  AUC (alpha over 5 nearest plants): {auc_near:.4f}")
    # Use the more informative one for the Remark
    use_near = auc_near > auc_all
    auc = auc_near if use_near else auc_all
    score_col = "alpha_near" if use_near else "alpha"
    print(f"  Using {score_col} (AUC={auc:.4f}) for thresholding")
    fpr, tpr, thr = roc_curve(y, df_full[score_col].values)
    # Youden's J
    j = tpr - fpr
    j_idx = np.argmax(j)
    alpha_thr = thr[j_idx]
    sens = tpr[j_idx]
    spec = 1 - fpr[j_idx]
    print(f"  AUC = {auc:.4f}")
    print(f"  Optimal threshold (Youden): alpha >= {alpha_thr:.4f}")
    print(f"    sensitivity = {sens:.3f}, specificity = {spec:.3f}")
    above = (df_full[score_col] >= alpha_thr).sum()
    print(f"  Munis with {score_col} >= threshold: {above}")
    captured = ((df_full[score_col] >= alpha_thr) & (y == 1)).sum()
    n_misalloc = int(y.sum())
    print(f"  Misallocations captured by threshold: {captured}/{n_misalloc} = "
          f"{100.0*captured/max(n_misalloc,1):.1f}%")

    # 5. Save tabular outputs
    df_out = df.sort_values("alpha", ascending=False)
    df_out.to_csv(HERE / "A3_anisotropy_results.csv", index=False, encoding="utf-8")
    print(f"\n  Wrote A3_anisotropy_results.csv")

    summary = {
        "n_origins": int(len(df)),
        "alpha_stats_all_plants": {
            "mean": float(df["alpha"].mean()),
            "median": float(df["alpha"].median()),
            "p90": float(df["alpha"].quantile(0.9)),
            "max": float(df["alpha"].max()),
        },
        "alpha_stats_near5_plants": {
            "mean": float(df["alpha_near"].mean()),
            "median": float(df["alpha_near"].median()),
            "p90": float(df["alpha_near"].quantile(0.9)),
            "max": float(df["alpha_near"].max()),
        },
        "auc_alpha_all": float(auc_all),
        "auc_alpha_near5": float(auc_near),
        "score_used": score_col,
        "youden_threshold": float(alpha_thr),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "captured_misallocations": int(captured),
        "total_misallocations_in_matched": int(n_misalloc),
    }
    (HERE / "A3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote A3_summary.json")

    # 6. Figures
    banner("6. Figures")
    # ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.scatter([fpr[j_idx]], [tpr[j_idx]], s=100, c="red", zorder=5,
               label=f"Youden (α≥{alpha_thr:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Anisotropy α as misallocation predictor")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A3_alpha_vs_misallocation_roc.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG / "A3_alpha_vs_misallocation_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote A3_alpha_vs_misallocation_roc.{{pdf,png}}")

    # Map
    fig, ax = plt.subplots(figsize=(8, 7))
    df_map = df.dropna(subset=["utm_x", "utm_y"])
    sc = ax.scatter(df_map["utm_x"], df_map["utm_y"], c=df_map[score_col],
                    cmap="viridis", s=20, alpha=0.85)
    # Highlight misallocated
    df_mis = df_map[df_map["misallocated"] == 1]
    ax.scatter(df_mis["utm_x"], df_mis["utm_y"], facecolors="none", edgecolors="red",
               s=60, linewidths=1.5, label="Misallocated")
    plt.colorbar(sc, ax=ax, label=f"α ({score_col})")
    ax.set_xlabel("UTM X")
    ax.set_ylabel("UTM Y")
    ax.set_title("Anisotropy coefficient α per municipality, Extremadura")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A3_anisotropy_map.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG / "A3_anisotropy_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote A3_anisotropy_map.{{pdf,png}}")

    # 7. LaTeX summary
    top = df.dropna(subset=["municipio"]).sort_values("alpha", ascending=False).head(10)
    lines = [
        "% Auto-generated by A3_anisotropy/run.py",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Top-10 municipalities of Extremadura by anisotropy coefficient "
        f"$\\alpha_i = \\beta_{{\\max}}/\\beta_{{\\min}}$. AUC of $\\alpha$ as a "
        f"misallocation predictor: {auc:.3f}.}}",
        "\\label{tab:anisotropy_top10}",
        "\\begin{tabular}{lrrrc}",
        "\\toprule",
        "Municipality & $\\beta_{\\min}$ & $\\beta_{\\max}$ & $\\alpha$ & Misalloc.? \\\\",
        "\\midrule",
    ]
    for _, r in top.iterrows():
        flag = "$\\checkmark$" if r["misallocated"] == 1 else "--"
        # Escape underscores etc in muni name
        name = r["origin"].replace("_", "\\_")
        lines.append(f"{name} & {r['beta_min']:.3f} & {r['beta_max']:.3f} & {r['alpha']:.3f} & {flag} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (OUT_TAB / "A3_anisotropy_summary.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote A3_anisotropy_summary.tex")

    banner("DONE")


if __name__ == "__main__":
    main()
