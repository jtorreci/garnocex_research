from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
RHO_BASE = 0.35
DOT_BASE = 0.50

SCENARIOS = [
    {"scenario_group": "transport", "scenario_name": "rho_025", "rho_trans": 0.25, "dotacion": DOT_BASE},
    {"scenario_group": "transport", "scenario_name": "rho_035", "rho_trans": 0.35, "dotacion": DOT_BASE},
    {"scenario_group": "transport", "scenario_name": "rho_045", "rho_trans": 0.45, "dotacion": DOT_BASE},
    {"scenario_group": "demand", "scenario_name": "dot_045", "rho_trans": RHO_BASE, "dotacion": 0.45},
    {"scenario_group": "demand", "scenario_name": "dot_050", "rho_trans": RHO_BASE, "dotacion": 0.50},
    {"scenario_group": "demand", "scenario_name": "dot_055", "rho_trans": RHO_BASE, "dotacion": 0.55},
]


def run_script(script_name: str, scenario: dict[str, float | str]) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--rho",
        str(scenario["rho_trans"]),
        "--dotacion",
        str(scenario["dotacion"]),
        "--scenario",
        str(scenario["scenario_name"]),
        "--no-plot",
    ]
    subprocess.run(command, check=True)


def count_thresholds_from_assignments(path: Path) -> tuple[int, int, int, int]:
    df = pd.read_csv(path)
    throughputs = df.groupby("plant_id")["prod"].sum()
    return (
        int((throughputs >= 5_000).sum()),
        int((throughputs >= 15_000).sum()),
        int((throughputs >= 40_000).sum()),
        int(throughputs.shape[0]),
    )


def summarize_greedy(path: Path, scenario: dict[str, float | str]) -> dict[str, float | str]:
    df = pd.read_csv(path)
    row = df.loc[df["n_plants"] == 24].iloc[0]
    return {
        "scenario_group": scenario["scenario_group"],
        "scenario_name": scenario["scenario_name"],
        "rho_trans": scenario["rho_trans"],
        "dotacion": scenario["dotacion"],
        "method": "greedy",
        "n_active_plants": int(row["n_plants"]),
        "cost_per_tonne": float(row["cost_per_tonne"]),
        "total_cost_eur": float(row["total_cost_eur"]),
        "max_cost": float(row["max_cost"]),
        "gini": float(row["gini"]),
        "n_above_5000": "",
        "n_above_15000": "",
        "n_above_40000": "",
        "notes": "24-plant pilot within near-optimal plateau",
    }


def summarize_iterative(history_path: Path, assign_path: Path, scenario: dict[str, float | str]) -> dict[str, float | str]:
    df = pd.read_csv(history_path)
    row = df.iloc[0]
    n5000, n15000, n40000, n_active = count_thresholds_from_assignments(assign_path)
    return {
        "scenario_group": scenario["scenario_group"],
        "scenario_name": scenario["scenario_name"],
        "rho_trans": scenario["rho_trans"],
        "dotacion": scenario["dotacion"],
        "method": "iterative",
        "n_active_plants": n_active,
        "cost_per_tonne": float(row["cost_per_tonne"]),
        "total_cost_eur": float(row["total_cost"]),
        "max_cost": float(row["max_cost"]),
        "gini": "",
        "n_above_5000": n5000,
        "n_above_15000": n15000,
        "n_above_40000": n40000,
        "notes": "counts computed from final assignments",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows: list[dict[str, float | str]] = []

    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario['scenario_name']}")
        run_script("analisis_podado.py", scenario)
        run_script("iterative_cost_assignment.py", scenario)

        greedy_path = OUTPUT_DIR / f"pareto_podado_{scenario['scenario_name']}.csv"
        iterative_path = OUTPUT_DIR / f"iterative_optimization_{scenario['scenario_name']}.csv"
        assign_path = OUTPUT_DIR / f"optimal_assignments_{scenario['scenario_name']}.csv"

        summary_rows.append(summarize_greedy(greedy_path, scenario))
        summary_rows.append(summarize_iterative(iterative_path, assign_path, scenario))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False, float_format="%.4f")
    df_summary[df_summary["scenario_group"] == "transport"].to_csv(
        OUTPUT_DIR / "sensitivity_transport.csv", index=False, float_format="%.4f"
    )
    df_summary[df_summary["scenario_group"] == "demand"].to_csv(
        OUTPUT_DIR / "sensitivity_demand.csv", index=False, float_format="%.4f"
    )
    print("Saved sensitivity summary outputs.")


if __name__ == "__main__":
    main()
