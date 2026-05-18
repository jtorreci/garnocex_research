# Voronoi Misallocation Risk: Reproducibility Package

Reproducibility package for the manuscript *"A Probabilistic Framework for Misallocation Risk in Voronoi Tessellations: Theory and Empirical Validation"*.

## Overview

This repository contains the data, analysis scripts, figure generation code, and a companion QGIS plugin to reproduce all results in the paper and its major revision. The framework derives a closed-form expression for misallocation probability in Voronoi tessellations when network distance replaces Euclidean distance, and provides diagnostic safety bands for identifying unreliable assignments.

## Repository Structure

```
.
├── codigo/                       # Input data + core analysis scripts (v1 submission)
│   ├── coordenadas_{municipios,plantas}.csv         # Geographic coordinates
│   ├── asignacion_municipios_{euclidiana,real}.csv  # Voronoi & network-optimal assignments
│   ├── distancias_euclideas.csv                     # Euclidean distance matrix
│   ├── *_anisotropy_coefficients*.csv               # Per-municipality / per-plant anisotropy
│   ├── {detailed,plant}_ratios_analysis*.csv        # Beta-ratio analyses
│   ├── ks_test_plant_municipality_results.csv       # KS test on plant–municipality dists.
│   ├── misallocated_municipalities.csv              # List of misallocated municipalities
│   ├── safety_bands_lookup_table.csv                # Safety band thresholds (various q*)
│   ├── sensitivity_s_analysis.csv                   # Sensitivity to parameter s
│   ├── calculate_anisotropy.py, calculate_plant_anisotropy.py
│   ├── distancias.py, distributional_robustness_analysis.py
│   └── safety_bands_analysis.py, recalculate_confidence_interval.py
├── tables/                       # Distance matrices + manuscript tables
│   ├── D_{euclidea,real}_{municipios,plantas}_clean.csv  # Distance matrices
│   ├── *.csv                                        # Result tables (multifacility,
│   │                                                  k-nearest capture, statistics, …)
│   └── *.tex                                        # LaTeX-formatted manuscript tables
├── figuras_clean/                # Figure generation scripts (Figures 1–15)
│   ├── generate_figure{1..15}.py                    # One per manuscript figure
│   ├── generate_figure11b_qualitative.py            # Qualitative companion to Fig. 11
│   ├── generate_voronoi_map.py                      # Voronoi tessellation map
│   ├── generate_network_voronoi_map.py              # Network-based service-area map
│   ├── generate_study_area_{map,detail,location}.py # Study-area panels (Fig. 6)
│   ├── generate_qq_plots.py                         # Q-Q diagnostic plots
│   ├── generate_figure_pareto_capture.py            # k-nearest Pareto capture
│   ├── study_area_map.pdf                           # Pre-rendered study-area figure
│   └── FIGURES_DOCUMENTATION.md                     # Per-figure methodology notes
├── scripts/                      # Additional analyses
│   ├── analyze_k_nearest_capture.py                 # k-nearest capture rates
│   ├── analyze_k_nearest_performance.py             # k-nearest accuracy / scalability
│   ├── analyze_s_sensitivity_correct.py             # Corrected s-sensitivity sweep
│   ├── analyze_plant_municipality_ks.py             # KS analysis on plant–muni distances
│   ├── plot_plant_municipality_distributions.py     # Distribution plots
│   ├── diagnose_misallocation_count.py              # Audit of misallocation totals
│   ├── ks_test_distance_subsets.py                  # KS tests by distance subset
│   ├── ks_table3_subsets.py                         # Table 3 subset KS aggregation
│   └── README_ANALYSIS_SCRIPTS.md                   # Per-script documentation
├── Revision/                     # Major-revision artefacts (response to reviewers)
│   ├── README.md                                    # Overview of the revision package
│   └── analyses/
│       ├── A1_spatial_carbym/                       # Bayesian log-CAR + BYM2 fits
│       ├── A2_distributional/                       # Wasserstein-1 + Anderson-Darling
│       ├── A3_anisotropy/                           # Anisotropy coefficient + ROC
│       ├── data/                                    # Canonical input dataset
│       ├── outputs/figures/                         # PDF/PNG figures (A1, A2, A3)
│       ├── outputs/tables/                          # LaTeX summary tables
│       ├── environment.yml, requirements.txt
│       └── README.md
├── qgis_plugin/                  # Voronoi Risk Toolbox (QGIS Processing plugin)
│   ├── voronoi_risk_toolbox/
│   │   ├── algorithms/                              # 5 Processing algorithms:
│   │   │   ├── voronoi_assignment.py                #   1. Euclidean Voronoi assignment
│   │   │   ├── beta_calculator.py                   #   2. Network scaling factor (QNEAT3)
│   │   │   ├── safety_bands.py                      #   3. Risk bands (Theorem 1)
│   │   │   ├── misallocation_detector.py            #   4. Per-feature misallocation flags
│   │   │   └── anisotropy_map.py                    #   5. Anisotropy α = β_max / β_min
│   │   ├── plugin.py, provider.py, __init__.py, metadata.txt
│   ├── docs/{THEORY,USER_MANUAL}.md
│   ├── README.md, INSTALL_AND_TEST.md, CHANGELOG.md, PLUGIN_AUDIT.md
│   └── test_data/
├── extremadura.geojson           # Study region boundary (Extremadura, Spain)
├── distributional_analysis.py    # Distributional comparison (Fig. 13)
├── distributional_sensitivity_s.py  # Sensitivity to s parameter (Fig. 14)
├── make_qq_final.py              # Q-Q plots for real data (Fig. 12)
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Description

### Input Data (`codigo/`)

| File | Description |
|------|-------------|
| `coordenadas_municipios.csv` | Geographic coordinates of 383 municipalities |
| `coordenadas_plantas.csv` | Geographic coordinates of 46 aggregate plants |
| `distancias_euclideas.csv` | Euclidean distance matrix (municipalities × plants) |

### Distance Matrices (`tables/`)

| File | Description |
|------|-------------|
| `D_euclidea_municipios_clean.csv` | Euclidean distances: municipality-to-municipality |
| `D_euclidea_plantas_clean.csv` | Euclidean distances: municipality-to-plant |
| `D_real_municipios_clean.csv` | Network distances: municipality-to-municipality |
| `D_real_plantas_clean.csv` | Network distances: municipality-to-plant |

### Analysis Results (`codigo/`)

| File | Description |
|------|-------------|
| `asignacion_municipios_euclidiana.csv` | Voronoi (Euclidean) facility assignments |
| `asignacion_municipios_real.csv` | Network-optimal facility assignments |
| `misallocated_municipalities.csv` | List of misallocated municipalities |
| `complete_anisotropy_coefficients.csv` | Per-municipality anisotropy index |
| `detailed_ratios_analysis.csv` | Beta ratio analysis (d_net / d_Euclidean) |
| `safety_bands_lookup_table.csv` | Safety-band thresholds for various q* levels |
| `sensitivity_s_analysis.csv` | Sensitivity of predictions to parameter s |

## Quick Start

```bash
# Install dependencies for the paper analyses
pip install -r requirements.txt

# Generate all manuscript figures
cd figuras_clean
python generate_figure1.py     # Histogram of network scaling factor β
python generate_figure2.py     # Violin plots / CDF and theoretical fit
# ... (see FIGURES_DOCUMENTATION.md for full per-figure details)

# Run distributional and sensitivity analyses
python ../distributional_analysis.py          # Fig. 13
python ../distributional_sensitivity_s.py     # Fig. 14
python ../make_qq_final.py                    # Fig. 12

# Run additional analyses (k-nearest, KS subsets, diagnostics)
python ../scripts/analyze_k_nearest_capture.py
python ../scripts/analyze_s_sensitivity_correct.py
python ../scripts/diagnose_misallocation_count.py

# Reproduce the major-revision analyses
cd ../Revision/analyses
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate voronoi-revision
python A1_spatial_carbym/run.py          # Bayesian log-CAR + BYM2
python A2_distributional/run.py          # Wasserstein-1 + Anderson-Darling
python A3_anisotropy/run.py              # Anisotropy coefficient + ROC

# Install the QGIS plugin (see qgis_plugin/INSTALL_AND_TEST.md)
# Copy qgis_plugin/voronoi_risk_toolbox/ into your QGIS plugin folder.
```

## Figure-to-Script Mapping

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `figuras_clean/generate_figure1.py` | Histogram of network scaling factor β |
| Fig. 2 | `figuras_clean/generate_figure2.py` | Violin plot of β distribution comparison |
| Fig. 3 | `figuras_clean/generate_figure3.py` | Plant anisotropy analysis (2×2) |
| Fig. 4 | `figuras_clean/generate_figure4.py` | Municipality assignment changes (real vs Voronoi) |
| Fig. 5 | `figuras_clean/generate_figure5.py` | Q-Q plots of β vs theoretical distributions |
| Fig. 6 | `figuras_clean/generate_figure6.py` (+ `generate_study_area_{map,detail,location}.py`) | Study area + β distribution composite |
| Fig. 7 | *(TikZ in manuscript)* | Conceptual diagram |
| Fig. 8 | `figuras_clean/generate_figure8.py` | Spatial analysis of β coefficients (2×2) |
| Fig. 9 | `figuras_clean/generate_figure9.py` | Spatial sensitivity (CAR/BYM, 3×2) |
| Fig. 10 | `figuras_clean/generate_figure10.py` | Safety bands for Voronoi risk assignment |
| Fig. 11 | `figuras_clean/generate_figure11.py` (+ `generate_figure11b_qualitative.py`) | Computational performance / k-nearest capture |
| Fig. 12 | `figuras_clean/generate_figure12.py` and `make_qq_final.py` | Distance improvement & Q-Q plots |
| Fig. 13 | `figuras_clean/generate_figure13.py` and `distributional_analysis.py` | Distributional comparison |
| Fig. 14 | `distributional_sensitivity_s.py` | Sensitivity of distributional predictions to s |
| Fig. 15 | `figuras_clean/generate_figure15.py` | Euclidean vs network distance correlation |

See `figuras_clean/FIGURES_DOCUMENTATION.md` for the full per-figure methodology, data sources, and output filenames. Auxiliary scripts (`generate_voronoi_map.py`, `generate_network_voronoi_map.py`, `generate_qq_plots.py`, `generate_figure_pareto_capture.py`) produce supporting visualisations used in panels or supplementary materials.

## Major Revision (`Revision/`)

The `Revision/` folder contains the additional analyses produced for the major revision, organised by methodological line:

- **A1 — Spatial Bayesian models** (`Revision/analyses/A1_spatial_carbym/`): Bayesian log-CAR and BYM2 posterior fits on log(β), spatial LOOCV and Moran's I on residuals. Replaces the v1 synthetic calibrated study.
- **A2 — Distributional ranking** (`Revision/analyses/A2_distributional/`): Wasserstein-1 and Anderson-Darling distributional ranking of five candidate distributions on both municipality-to-municipality and municipality-to-facility domains.
- **A3 — Anisotropy** (`Revision/analyses/A3_anisotropy/`): Per-municipality anisotropy coefficient α = β_max / β_min, with ROC against observed misallocations.

Each subfolder ships its own `README.md`, a runnable `run.py`, and canonical inputs in `Revision/analyses/data/`. Pre-generated outputs (figures and LaTeX tables) live under `Revision/analyses/outputs/`.

See [`Revision/README.md`](Revision/README.md) for the full description and the mapping to specific reviewer comments.

## QGIS Plugin (`qgis_plugin/`)

The `qgis_plugin/voronoi_risk_toolbox/` folder packages the full framework as a QGIS Processing plugin — **Voronoi Risk Toolbox** (v0.2.0). It exposes five algorithms that practitioners can run interactively on their own facility / road-network data:

1. **Voronoi Assignment** — Euclidean nearest-facility assignment, distance ratio R.
2. **Beta Calculator** — network scaling factor β = d_network / d_Euclidean (needs QNEAT3).
3. **Safety Bands** — misallocation probability per municipality (Theorem 1), safety-band polygons.
4. **Misallocation Detector** — per-feature misallocation flags and distance savings.
5. **Anisotropy Map** — anisotropy coefficient α = β_max / β_min per origin (needs QNEAT3).

Requirements: QGIS 3.22+ and the [QNEAT3](https://plugins.qgis.org/plugins/QNEAT3/) plugin for the algorithms that compute network distance.

Installation, parameters, and a usage walkthrough are documented in [`qgis_plugin/README.md`](qgis_plugin/README.md), [`qgis_plugin/INSTALL_AND_TEST.md`](qgis_plugin/INSTALL_AND_TEST.md), and [`qgis_plugin/docs/USER_MANUAL.md`](qgis_plugin/docs/USER_MANUAL.md). Theoretical background is summarised in [`qgis_plugin/docs/THEORY.md`](qgis_plugin/docs/THEORY.md).

## Road Network Data

The road network shapefile used to compute network distances (~620 MB) is not included in this repository due to size constraints. The network distances are provided pre-computed in `tables/D_real_*.csv`. The road network data was obtained from the Spatial Data Infrastructure of Extremadura (IDEEx) and OpenStreetMap, and can be reconstructed from those public sources.

## Study Area

Extremadura, Spain: 383 municipalities, 46 aggregate production facilities, 41,635 km².

## Requirements

- Python ≥ 3.8.
- `requirements.txt` — paper analyses (`codigo/`, `figuras_clean/`, `scripts/`).
- `Revision/analyses/requirements.txt` / `Revision/analyses/environment.yml` — major-revision analyses (PyMC, etc.).
- QGIS 3.22+ and the QNEAT3 plugin — only required to run the QGIS plugin.

## License

MIT License. See [LICENSE](LICENSE) for details.
