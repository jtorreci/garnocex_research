# Reproducibility Package: A05 — Network Rationalization of CDW Treatment Plants

This directory contains all code, data, and outputs needed to reproduce the results of:

**"Network Rationalization of Construction and Demolition Waste Treatment Plants:
Cost-Based Assignment and the Scale--Circularity Nexus---A Case Study of Extremadura, Spain"**

*Companion paper to A00 (Voronoi misallocation framework) and A01 (spatial cost analysis).*

## Key Findings (Extremadura case study)

- **383 municipalities**, **46 treatment plants** consolidated into **32 plant groups**.
- The iterative cost-based reassignment heuristic **stabilizes in 6 iterations** at a **24-plant base-case configuration** with **8 plants receiving zero throughput**.
- System-average cost falls from **12.22 €/t** (A01 diagnostic baseline, per-physical-plant throughput) to **11.34 €/t** (cost-based heuristic), a **7.2 % reduction**. Relative to A05's own 32-plant distance-based baseline of **11.67 €/t**, the corresponding reduction is **2.8 %**.
- Greedy distance-based pruning reaches a **near-optimal plateau between 23 and 25 plants** with cost between **11.36 and 11.37 €/t**.
- **Only 63 of 383 municipalities** change plant assignment under the cost-based heuristic; the rest keep their baseline allocation, so the change is local rather than systemic.
- **23 of 24** surviving plants exceed the **5,000 t/y** viability threshold for mechanical separation, compared to **24 of 32 (75 %)** in the baseline.

## Repository Structure

```
A05.Network_Optimization/
├── README.md                          # this file
├── LICENSE                            # CC-BY 4.0
├── .gitignore
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment
├── data/
│   ├── shp/                           # Extremadura, roads, municipalities, plants
│   │   ├── Extremadura.shp + .{shx, prj, dbf, cpg}
│   │   ├── carreteras.shp + ...
│   │   ├── municipios.shp + ...
│   │   └── plantas.shp + ...
│   └── tablas/
│       ├── codigos_plantas.csv                  # 46 plant id → name
│       ├── datos_voronoi_municipios.csv         # 383 municipality production / population
│       ├── distancias_reales_plantas_municipios.csv # 46x383 road-network distance matrix
│       └── asignacion_municipios_real.csv       # 383 municipalities (list shared with A01)
├── analysis/
│   ├── analisis_podado.py             # greedy progressive pruning (32 → 3 plants)
│   ├── iterative_cost_assignment.py   # iterative cost-based reassignment heuristic
│   ├── run_sensitivity.py             # transport-cost + demand scenario runner
│   ├── optimize_metaheuristics.py     # SA and GA benchmark of greedy pruning (about 4 min)
│   ├── plot_metaheuristics.py         # redraws Figure 4 from the saved benchmark CSVs
│   └── equilibrium_planner/           # fixed point vs planner MILP, random seeds
├── replication/
│   ├── generate_voronoi_map_a05.py    # Figure 1: Voronoi tessellation of the 32 plant groups
│   ├── generate_three_panel_a05.py    # Figure 2: routed flows + cost KDE
│   └── generate_graphical_abstract.py # graphical abstract
├── outputs/                           # canonical CSV outputs of analysis scripts
│   ├── pareto_podado.csv              # greedy pruning system metrics (32 → 3)
│   ├── greedy_assignments_24.csv      # per-municipality assignment at the 24-plant pivot
│   ├── iterative_optimization.csv     # iterative reassignment iteration history
│   ├── optimal_assignments.csv        # per-municipality cost-based assignment
│   ├── sensitivity_summary.csv        # combined sensitivity table
│   ├── sensitivity_transport.csv      # transport-cost scenarios (ρ = 0.25, 0.35, 0.45 €/t·km)
│   ├── sensitivity_demand.csv         # demand scenarios (ρ_CDW = 0.45, 0.50, 0.55 t/inhab·yr)
│   ├── *_<scenario>.csv               # scenario-specific intermediates
│   ├── comparison_metaheuristics.csv  # greedy vs SA/GA by network size
│   ├── unconstrained_ga.csv           # best network of the unconstrained GA
│   └── equilibrium_planner/           # method_comparison.csv and assignments
├── results/
│   └── figures/
│       ├── Voronoi.png                # manuscript Figure 1
│       ├── fig_three_panel_a05.png    # manuscript Figure 2
│       ├── comparison_metaheuristics.png # manuscript Figure 4
│       └── graphical_abstract.png
└── docs/
    ├── methodology.md                 # detailed methodology
    └── data_description.md            # data documentation
```

## Quick Start

### Option 1: pip

```bash
git clone https://github.com/jtorreci/garnocex_research.git
cd garnocex_research/A05.Network_Optimization
pip install -r requirements.txt
python analysis/analisis_podado.py --no-plot           # produces pareto_podado.csv + greedy_assignments_24.csv
python analysis/iterative_cost_assignment.py           # produces optimal_assignments.csv + iterative_optimization.csv
python analysis/run_sensitivity.py                     # produces sensitivity_*.csv
python analysis/equilibrium_planner/fixed_point.py     # equilibrium benchmark (then fixed_point_seeds, planner_milp, compare_methods)
python replication/generate_three_panel_a05.py         # rebuilds Figure 2
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate a05-network
python analysis/analisis_podado.py --no-plot
python analysis/iterative_cost_assignment.py
python analysis/run_sensitivity.py
python replication/generate_three_panel_a05.py
```

### Manuscript figures

All figures are written to `results/figures/` at their printed size and 600 dpi.

```bash
python replication/generate_voronoi_map_a05.py   # Figure 1: Voronoi.png
python replication/generate_three_panel_a05.py   # Figure 2: fig_three_panel_a05.png
python analysis/optimize_metaheuristics.py       # SA/GA search (about 4 min): comparison_metaheuristics.csv,
                                                 # unconstrained_ga.csv and Figure 4
python analysis/plot_metaheuristics.py           # Figure 4 only, from the saved CSVs
```

The map figures read only `data/shp/` and `data/tablas/`. The road attribute
table `carreteras.dbf` is not in the repository (see `data/shp/README.md`);
the scripts use only the road geometries and run without it.

Change on 2026-11-25: `generate_three_panel_a05.py` now looks up municipal
production by normalised name. The distance table carries 82 names with
garbled accents (e.g. "Talav n"); the former exact-name lookup gave those
municipalities zero tonnage, which distorted panel (c) and the plant unit
costs shown on the maps. The analysis CSVs were not affected.

## Key Parameters

| Parameter | Description | Value used |
|-----------|-------------|-----------|
| `C0`      | Base fixed cost (single-operator plant) | 40,000 €/yr |
| `T0`      | Economy-of-scale threshold              | 5,000 t/yr  |
| `v`       | Variable treatment cost                 | 0.35 €/t    |
| `ρ`       | Effective transport rate                | 0.35 €/t·km |
| `ρ_CDW`   | Per-capita CDW generation               | 0.50 t/inhab·yr |
| Consolidation radius | 15 km between physical plants    | yields 32 groups from 46 plants |

## Companion Papers

- **A00 — Voronoi misallocation** (`A00.Voronoi_critics/`): the probabilistic framework that justifies the use of road-network rather than Euclidean distances. In preparation for submission.
- **A01 — Spatial cost analysis** (`A01.Voronoi/`): the diagnostic that calibrates the piecewise logarithmic cost model used here and quantifies the territorial cost penalty. Published in *Waste Management Bulletin* 4(3), 100355 (2026), doi:10.1016/j.wmb.2026.100355.

The three papers share input data (municipality and plant coordinates, the road-network distance matrix); the cost model is calibrated in A01 and inherited here without modification.

## Citation

```bibtex
@article{torrecillaNetworkRationalization2026,
  title   = {Network Rationalization of Construction and Demolition Waste Treatment Plants:
             Cost-Based Assignment and the Scale--Circularity Nexus---A Case Study of
             Extremadura, Spain},
  author  = {Torrecilla-Pinero, J.A. and Ceballos-Mart\'inez, J.M. and Plaza Caballero, P.
             and Cruces L\'opez, A. and Cuartero S\'aez, A.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Contact

- GitHub Issues: <https://github.com/jtorreci/garnocex_research/issues>

## License

Released under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**; see `LICENSE`.

## Acknowledgements

This research was funded by the **GARNOCEX project**, a collaborative agreement between the Junta de Extremadura, the Colegio de Ingenieros Tecnicos de Obras Publicas de Extremadura, four regional CDW management companies, and the Universidad de Extremadura.

---

**Keywords**: construction and demolition waste, network optimization, facility location, economies of scale, circular economy, plant rationalization
