from __future__ import annotations

"""Generate plant-level dynamic tables and figures for S1/S2/S3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
S1_CSV = SCRIPT_DIR / "plant_level_s1_full.csv"
S2_CSV = SCRIPT_DIR / "plant_level_s2_full.csv"
S3_CSV = SCRIPT_DIR / "plant_level_s3_full.csv"

DELTA_CSV = SCRIPT_DIR / "plant_indicator_deltas.csv"
FIG_STEM = SCRIPT_DIR / "plant_indicator_scenarios_overlay"

METRICS = [
    {"column": "C_i", "label": "Unit cost", "unit": "EUR/t", "color": "#1f77b4", "ref": None},
    {"column": "D90_i", "label": "Accessibility D90", "unit": "km", "color": "#7f3c8d", "ref": None},
    {"column": "Prec_i", "label": "Precision", "unit": "", "color": "#11a579", "ref": 1.0},
    {"column": "Rec_i", "label": "Recall", "unit": "", "color": "#3969ac", "ref": 1.0},
    {"column": "Leak_dg_i", "label": "Leakage penalty", "unit": "EUR/t", "color": "#e73f74", "ref": 0.0},
    {"column": "IET_i", "label": "Transport efficiency", "unit": "%", "color": "#f28e2b", "ref": 100.0},
]

SCENARIO_STYLE = {
    "S1": {"color": "#4d4d4d", "marker": "o"},
    "S2": {"color": "#d95f02", "marker": "^"},
    "S3": {"color": "#1b9e77", "marker": "s"},
}


def load_scenario(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "plant_id" not in df.columns:
        raise ValueError(f"{path.name} missing plant_id")
    df = df.copy()
    df["scenario_tag"] = tag
    return df


def build_delta_table(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    cols = ["plant_id"] + [m["column"] for m in METRICS]
    a = s1[cols].copy().rename(columns={m["column"]: f"{m['column']}_S1" for m in METRICS})
    b = s2[cols].copy().rename(columns={m["column"]: f"{m['column']}_S2" for m in METRICS})
    c = s3[cols].copy().rename(columns={m["column"]: f"{m['column']}_S3" for m in METRICS})

    df = a.merge(b, on="plant_id", how="outer").merge(c, on="plant_id", how="outer")

    for metric in METRICS:
        col = metric["column"]
        df[f"delta_{col}_S2_minus_S1"] = df[f"{col}_S2"] - df[f"{col}_S1"]
        df[f"delta_{col}_S3_minus_S2"] = df[f"{col}_S3"] - df[f"{col}_S2"]

    return df.sort_values("plant_id").reset_index(drop=True)


def derive_xlim(series_list: list[pd.Series]) -> tuple[float, float]:
    values = pd.concat(series_list, ignore_index=True).dropna()
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.1, 1.0)
        return vmin - pad, vmax + pad
    pad = 0.08 * (vmax - vmin)
    lower = min(0.0, vmin - pad) if vmin >= 0 else vmin - pad
    return lower, vmax + pad


def plot_overlay(ax: plt.Axes, metric: dict[str, object], s1: pd.Series, s2: pd.Series, s3: pd.Series) -> None:
    xlim = derive_xlim([s1, s2, s3])
    scenario_data = [
        ("S1", s1.dropna(), 1.25),
        ("S2", s2.dropna(), 1.00),
        ("S3", s3.dropna(), 0.75),
    ]
    rng = np.random.default_rng(42)

    for label, values, ypos in scenario_data:
        color = SCENARIO_STYLE[label]["color"]
        ax.boxplot(
            values,
            vert=False,
            positions=[ypos],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.28, "edgecolor": color, "linewidth": 1.1},
            medianprops={"color": color, "linewidth": 1.8},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
        )
        jitter = ypos + rng.uniform(-0.035, 0.035, size=len(values))
        ax.scatter(
            values,
            jitter,
            s=18,
            color=color,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.25,
        )

    ref = metric["ref"]
    if ref is not None:
        ax.axvline(float(ref), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title(str(metric["label"]), fontsize=11, pad=8)
    ax.set_xlim(*xlim)
    ax.set_yticks([0.75, 1.00, 1.25], labels=["S3", "S2", "S1"])
    unit = str(metric["unit"])
    ax.set_xlabel(f"{metric['label']} ({unit})" if unit else str(metric["label"]), fontsize=9)
    ax.set_ylim(0.55, 1.45)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def create_overlay_figure(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), constrained_layout=True)

    for ax, metric in zip(axes.flat, METRICS):
        col = str(metric["column"])
        plot_overlay(ax, metric, s1[col], s2[col], s3[col])

    handles = [
        plt.Line2D([0], [0], color=SCENARIO_STYLE["S1"]["color"], lw=6, alpha=0.5, label="S1"),
        plt.Line2D([0], [0], color=SCENARIO_STYLE["S2"]["color"], lw=6, alpha=0.5, label="S2"),
        plt.Line2D([0], [0], color=SCENARIO_STYLE["S3"]["color"], lw=6, alpha=0.5, label="S3"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Plant-level indicators across S1, S2, and S3", fontsize=14, y=1.06)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    s1 = load_scenario(S1_CSV, "S1")
    s2 = load_scenario(S2_CSV, "S2")
    s3 = load_scenario(S3_CSV, "S3")

    delta = build_delta_table(s1, s2, s3)
    delta.to_csv(DELTA_CSV, index=False, encoding="utf-8-sig")
    create_overlay_figure(s1, s2, s3, FIG_STEM)

    print(f"Saved: {DELTA_CSV.name}")
    print(f"Saved: {FIG_STEM.with_suffix('.png').name} / .pdf")


if __name__ == "__main__":
    main()
