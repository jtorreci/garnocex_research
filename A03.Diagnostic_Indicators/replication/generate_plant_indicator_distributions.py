from __future__ import annotations

"""Generate reusable plant-level distribution figures for the indicator paper."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "plant_level_indicators.csv"
SUMMARY_CSV = SCRIPT_DIR / "plant_indicator_distribution_summary.csv"
MAIN_FIG_STEM = SCRIPT_DIR / "plant_indicator_distributions"
RANK_FIG_STEM = SCRIPT_DIR / "plant_indicator_rankings"
CORR_SCATTER_STEM = SCRIPT_DIR / "plant_indicator_correlation_scatter"
CORR_HEATMAP_STEM = SCRIPT_DIR / "plant_indicator_correlation_heatmap"
CORR_CSV = SCRIPT_DIR / "plant_indicator_spearman_correlation.csv"

METRICS = [
    {
        "column": "C_i",
        "label": "Unit cost",
        "unit": "EUR/t",
        "color": "#1f77b4",
        "reference": None,
        "xlim": None,
    },
    {
        "column": "D90",
        "label": "Accessibility D90",
        "unit": "km",
        "color": "#7f3c8d",
        "reference": None,
        "xlim": None,
    },
    {
        "column": "Prec",
        "label": "Precision",
        "unit": "",
        "color": "#11a579",
        "reference": 1.0,
        "xlim": (0.0, 1.02),
    },
    {
        "column": "Rec",
        "label": "Recall",
        "unit": "",
        "color": "#3969ac",
        "reference": 1.0,
        "xlim": (0.0, 1.02),
    },
    {
        "column": "Leak_dg",
        "label": "Leakage penalty",
        "unit": "EUR/t",
        "color": "#e73f74",
        "reference": 0.0,
        "xlim": None,
    },
    {
        "column": "IET",
        "label": "Transport efficiency",
        "unit": "%",
        "color": "#f28e2b",
        "reference": 100.0,
        "xlim": None,
    },
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the plant-level indicator table and validate required columns."""
    df = pd.read_csv(path)
    required = {"plant_name", "T_i"} | {metric["column"] for metric in METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path.name}: {missing_str}")
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute compact descriptive statistics for each plotted metric."""
    rows: list[dict[str, float | str]] = []
    for metric in METRICS:
        values = df[metric["column"]].dropna()
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "metric": metric["column"],
                "label": metric["label"],
                "unit": metric["unit"],
                "n_plants": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(values.max()),
                "iqr": float(q3 - q1),
            }
        )
    return pd.DataFrame(rows)


def format_stat(value: float, unit: str) -> str:
    """Format panel statistics with compact precision."""
    if math.isfinite(value) and abs(value) >= 100:
        formatted = f"{value:.0f}"
    elif math.isfinite(value) and abs(value) >= 10:
        formatted = f"{value:.1f}"
    else:
        formatted = f"{value:.2f}"
    return f"{formatted} {unit}" if unit else formatted


def metric_xlim(values: pd.Series, configured: tuple[float, float] | None) -> tuple[float, float]:
    """Derive a padded x-axis range when the metric does not have a fixed one."""
    if configured is not None:
        return configured

    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.1, 1.0)
        return vmin - padding, vmax + padding

    padding = (vmax - vmin) * 0.08
    lower = min(0.0, vmin - padding) if vmin >= 0 else vmin - padding
    upper = vmax + padding
    return lower, upper


def plot_distribution_panel(ax: plt.Axes, values: pd.Series, metric: dict[str, object]) -> None:
    """Draw a compact box-and-strip distribution panel for a single metric."""
    color = str(metric["color"])
    unit = str(metric["unit"])
    xlim = metric_xlim(values, metric["xlim"])
    rng = np.random.default_rng(42)
    y = 1.0 + rng.uniform(-0.09, 0.09, size=len(values))

    ax.axvspan(values.quantile(0.25), values.quantile(0.75), color=color, alpha=0.10, lw=0)
    ax.boxplot(
        values,
        vert=False,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": color, "alpha": 0.18, "edgecolor": color, "linewidth": 1.2},
        medianprops={"color": color, "linewidth": 2.0},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )
    ax.scatter(values, y, s=28, color=color, alpha=0.85, edgecolors="white", linewidths=0.4)

    reference = metric["reference"]
    if reference is not None:
        ax.axvline(float(reference), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)

    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    ax.text(
        0.98,
        0.83,
        f"median {format_stat(median, unit)}\nIQR {format_stat(iqr, unit)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    xlabel = f"{metric['label']} ({unit})" if unit else str(metric["label"])
    ax.set_title(str(metric["label"]), fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(0.72, 1.28)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_rank_panel(ax: plt.Axes, df: pd.DataFrame, metric: dict[str, object]) -> None:
    """Draw a ranked dot plot with only outlier labels to keep the panel readable."""
    column = str(metric["column"])
    color = str(metric["color"])
    ordered = df[["plant_name", "T_i", column]].sort_values(column, ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))

    ax.hlines(y, 0, ordered[column], color=color, alpha=0.20, linewidth=1.0)
    ax.scatter(ordered[column], y, s=26, color=color, alpha=0.90, edgecolors="white", linewidths=0.4)

    reference = metric["reference"]
    if reference is not None:
        ax.axvline(float(reference), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)

    for idx in list(range(min(3, len(ordered)))) + list(range(max(len(ordered) - 3, 0), len(ordered))):
        row = ordered.iloc[idx]
        offset = 4 if idx >= len(ordered) - 3 else -4
        ha = "left" if idx >= len(ordered) - 3 else "right"
        ax.annotate(
            str(row["plant_name"]),
            (float(row[column]), idx),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7,
        )

    xlabel = f"{metric['label']} ({metric['unit']})" if metric["unit"] else str(metric["label"])
    ax.set_title(str(metric["label"]), fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(*metric_xlim(ordered[column], metric["xlim"]))
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def create_distribution_figure(df: pd.DataFrame, output_stem: Path) -> None:
    """Create the main manuscript-oriented multi-panel distribution figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        plot_distribution_panel(ax, df[str(metric["column"])], metric)

    fig.suptitle("Plant-level indicator distributions (n = 32 plants)", fontsize=14, y=1.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def create_ranking_figure(df: pd.DataFrame, output_stem: Path) -> None:
    """Create a companion figure that makes extreme plants easy to identify."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        plot_rank_panel(ax, df, metric)

    fig.suptitle("Plant-level indicator rankings", fontsize=14, y=1.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def create_correlation_outputs(df: pd.DataFrame) -> None:
    """Create scatter/correlation matrix plus a compact heatmap and CSV."""
    metric_columns = [str(metric["column"]) for metric in METRICS]
    labels = [str(metric["label"]) for metric in METRICS]
    colors = [str(metric["color"]) for metric in METRICS]
    data = df[metric_columns].copy()
    corr = data.corr(method="spearman")
    corr.index = labels
    corr.columns = labels
    corr.to_csv(CORR_CSV, float_format="%.6f")

    # Pairwise scatter matrix with rho annotated on every off-diagonal panel.
    n = len(metric_columns)
    fig, axes = plt.subplots(n, n, figsize=(13.5, 13.5), constrained_layout=True)
    for i, row_metric in enumerate(METRICS):
        y = data[str(row_metric["column"])]
        for j, col_metric in enumerate(METRICS):
            ax = axes[i, j]
            x = data[str(col_metric["column"])]
            if i == j:
                ax.hist(x, bins=9, color=colors[j], alpha=0.75, edgecolor="white")
                ax.text(
                    0.04,
                    0.90,
                    "diag",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#444444",
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=22,
                    color=colors[j],
                    alpha=0.70,
                    edgecolors="white",
                    linewidths=0.35,
                )
                rho = float(data[[str(col_metric["column"]), str(row_metric["column"])]].corr(method="spearman").iloc[0, 1])
                ax.text(
                    0.04,
                    0.92,
                    rf"$\rho$={rho:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#d0d0d0"},
                )
            ax.grid(color="#e3e3e3", linewidth=0.6, alpha=0.8)
            ax.tick_params(labelsize=7, length=2)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Plant-level indicator scatter matrix with Spearman rho", fontsize=14, y=1.01)
    fig.savefig(CORR_SCATTER_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(CORR_SCATTER_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Compact heatmap for manuscript/supplement decisions.
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(7.2, 6.1), constrained_layout=True)
    im = heatmap_ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    heatmap_ax.set_xticks(np.arange(n), labels=labels, rotation=45, ha="right", fontsize=8)
    heatmap_ax.set_yticks(np.arange(n), labels=labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            heatmap_ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr.values[i, j]) > 0.55 else "#222222",
            )
    cbar = heatmap_fig.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho", fontsize=9)
    heatmap_ax.set_title("Plant-level indicator Spearman correlation matrix", fontsize=12, pad=10)
    heatmap_fig.savefig(CORR_HEATMAP_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    heatmap_fig.savefig(CORR_HEATMAP_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(heatmap_fig)


def main() -> None:
    """Generate summary statistics and reusable figures for plant indicators."""
    plt.style.use("seaborn-v0_8-whitegrid")
    df = load_data(INPUT_CSV)
    summary = compute_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    create_distribution_figure(df, MAIN_FIG_STEM)
    create_ranking_figure(df, RANK_FIG_STEM)
    create_correlation_outputs(df)

    print(f"Loaded {len(df)} plants from {INPUT_CSV.name}")
    print(f"Summary table: {SUMMARY_CSV.name}")
    print(f"Main figure: {MAIN_FIG_STEM.with_suffix('.png').name} / .pdf")
    print(f"Ranking figure: {RANK_FIG_STEM.with_suffix('.png').name} / .pdf")
    print(f"Correlation scatter: {CORR_SCATTER_STEM.with_suffix('.png').name} / .pdf")
    print(f"Correlation heatmap: {CORR_HEATMAP_STEM.with_suffix('.png').name} / .pdf")
    print(f"Correlation CSV: {CORR_CSV.name}")


if __name__ == "__main__":
    main()
