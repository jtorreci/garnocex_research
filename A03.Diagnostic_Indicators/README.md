# Reproducibility Package: A03 — Diagnostic Indicators for CDW Treatment Plant Networks

This directory contains all code and data needed to reproduce the results of:

**"Diagnostic Indicators for CDW Treatment Plant Networks: A Plant-Level and Network-Level Panel Framework"**

*In preparation for submission*

## Key Findings

Three scenarios compare the Extremadura CDW network under different flow configurations,
all evaluated with the same piecewise-log cost model (C₀ = 40,000 €/yr, T₀ = 5,000 t/yr,
ρ = 0.35 €/t·km, v = 0.35 €/t):

| Scenario | C̄ (EUR/t) | Transport | Treatment | Gini | Active plants |
|----------|------------|-----------|-----------|------|---------------|
| S1 — Proximity baseline | **12.22** | 6.03 | 6.19 | 0.187 | 32 |
| S2 — Observed real flows | **12.11** | 6.30 | 5.81 | 0.208 | 36 |
| S3 — Cost-optimal network | **11.34** | 6.42 | 4.92 | 0.197 | 24 |

**Net S1→S3 saving: 7.2% (0.88 EUR/t)**, driven entirely by treatment-cost reduction
through throughput concentration in fewer, larger plants — transport costs actually rise
slightly (+0.39 EUR/t) as some municipalities are routed farther to reach larger plants.

The paper also introduces a **two-panel framework** of 13 diagnostic indicators (7
plant-level, 7 network-level) and demonstrates algebraically that the legacy indices
IET/ICR/IER/ISD are mutually collinear and add no independent information.

## Repository Structure

```
A03.Diagnostic_Indicators/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment
├── LICENSE                            # CC-BY 4.0
├── analysis/
│   ├── reproduce_scenarios.py         # Self-contained reproduction script (START HERE)
│   ├── compute_indicators.py          # Full 13-indicator panel computation (S1)
│   └── compute_three_scenarios.py     # Full three-scenario pipeline (needs confidential S2 Excels)
├── replication/
│   ├── generate_plant_indicator_distributions.py  # Figure 1: plant indicator distributions
│   ├── generate_three_panel_map.py    # Figure 2: three-panel routed-flow map
│   ├── plot_cost_component_kdes.py    # Figure 3: cost component KDEs
│   └── generate_plant_scenario_dynamics.py        # Figure S2: plant indicators across scenarios
├── data/
│   ├── README.md                      # Data source descriptions
│   ├── s1_proximity_baseline.csv      # S1 municipality table (nearest plant by road)
│   └── s2_observed_flows_public.csv   # Anonymised S2 flows (730 rows, 383 municipalities)
├── outputs/                           # Pre-computed results (CSV + figures)
│   ├── three_scenarios_comparison.csv
│   ├── plant_level_indicators.csv     # S1 plant-level panel (input of Figure 1)
│   ├── plant_level_s1_full.csv
│   ├── plant_level_s2_full.csv
│   ├── plant_level_s3_full.csv
│   ├── muni_costs_s1.csv
│   ├── muni_costs_s2.csv
│   ├── muni_costs_s3.csv
│   ├── plant_indicator_deltas.csv
│   ├── plant_indicator_distribution_summary.csv
│   ├── plant_indicator_spearman_correlation.csv
│   ├── plant_indicator_distributions.jpg      # Figure 1
│   ├── fig_three_panel_routed.jpg             # Figure 2
│   ├── cost_component_kdes.jpg                # Figure 3
│   ├── plant_indicator_scenarios_overlay.jpg  # Figure S2
│   └── *.png, *.pdf                           # auxiliary plots (rankings, correlations)
└── docs/
    ├── data_description.md            # Data sources and structure
    └── methodology.md                 # Indicator definitions and pipeline
```

## Quick Start

### Option 1: pip

```bash
# from the repository root
cd A03.Diagnostic_Indicators
pip install -r requirements.txt
python analysis/reproduce_scenarios.py
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate a03-diagnostic-indicators
python analysis/reproduce_scenarios.py
```

## Reproduction Steps

### Headline numbers (fully reproducible from public data)

```bash
python analysis/reproduce_scenarios.py
```

This loads S1 from `data/s1_proximity_baseline.csv`, S2 from
`data/s2_observed_flows_public.csv`, and S3 from
`../../A05.Network_Optimization/outputs/optimal_assignments.csv`, then
computes C̄, Gini, and the transport/treatment split for each scenario and
prints a comparison table. Expected output:

```
  C_bar, production-weighted (EUR/t)    12.22      12.11      11.34
    Transport component (EUR/t)          6.03       6.30       6.42
    Treatment component (EUR/t)          6.19       5.81       4.92
  Gini(C_m)                             0.187      0.208      0.197
```

### Manuscript figures

All figure scripts read from `outputs/` and `data/` and write to `outputs/`
(JPEG, 260 mm wide at 300 dpi, i.e. twice the 130 mm printed width).

```bash
python replication/generate_plant_indicator_distributions.py  # Figure 1
python replication/generate_three_panel_map.py                # Figure 2 (about 1 min)
python replication/plot_cost_component_kdes.py                # Figure 3
python replication/generate_plant_scenario_dynamics.py        # Figure S2
```

The map reads the Extremadura shapefiles shipped with A05
(`../A05.Network_Optimization/data/shp/`), S1 from
`data/s1_proximity_baseline.csv`, S2 from `data/s2_observed_flows_public.csv`
and S3 from `../A05.Network_Optimization/outputs/optimal_assignments.csv`. The
road attribute table `carreteras.dbf` is not in the repository (see
`../A05.Network_Optimization/data/shp/README.md`); only the road geometries
are used and the script runs without it.

Change on 2026-11-25: the map now snaps every municipality centroid and plant
to the largest connected component of the road graph. The road shapefile has
small disconnected fragments (118 components); snapping to one of them could
leave a route with no road path, drawn as a straight segment. With the fix no
route falls back to a straight line.

### Full S2 pipeline (not reproducible without confidential data)

`analysis/compute_three_scenarios.py` contains the complete pipeline including
parsing the 23 `PT_*.xlsx` operational files. It is included for transparency
only; the confidential Excel files are not distributed. See the Data Availability
section below.

## Key Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `C0`      | Base fixed cost | 40,000 €/yr |
| `T0`      | Economy-of-scale threshold | 5,000 t/yr |
| `ρ`       | Transport tariff | 0.35 €/(t·km) |
| `v`       | Variable treatment cost | 0.35 €/t |
| `ρ_CDW`   | CDW generation rate | 0.50 t/inhab·yr |
| EPSG      | Coordinate reference system | 25830 (ETRS89/UTM 30N) |

## Data Availability

- **S1 (proximity baseline)**: fully reproducible from public A01 data.
- **S3 (cost-optimal)**: fully reproducible from public A05 data.
- **S2 (observed operational flows)**: released as anonymised flows in
  `data/s2_observed_flows_public.csv`. Volumes are rescaled to theoretical
  production (551,205 t/yr based on 0.5 t/inhab/yr); raw commercial origin-
  destination tonnages from the regional operational monitoring programme are
  withheld. This is consistent with the data availability statement in the
  manuscript and with the commercial confidentiality agreement with participating
  treatment plant operators.

## Companion Papers

| Package | Description |
|---------|-------------|
| [A01.Voronoi](../A01.Voronoi/) | Spatial cost inequality in CDW management (Waste Management Bulletin, Elsevier) — provides the S1 baseline data |
| [A05.Network_Optimization](../A05.Network_Optimization/) | Network rationalisation via iterative cost-optimal assignment — provides the S3 data |

## Citation

A citation will be added once the corresponding manuscript completes peer review.
(Author and affiliation details are omitted here for double-blind review.)

## License

Released under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**; see `LICENSE`.

---

**Keywords**: construction and demolition waste, diagnostic indicators, treatment plant networks,
economies of scale, Gini coefficient, cost equity, panel framework, Extremadura
