# Voronoi Misallocation Risk: Reproducibility Package

Reproducibility package for the manuscript *"A Probabilistic Framework for Misallocation Risk in Voronoi Tessellations: Theory and Empirical Validation"*.

## Overview

This repository contains the data, analysis scripts, and figure generation code to reproduce all results in the paper. The framework derives a closed-form expression for misallocation probability in Voronoi tessellations when network distance replaces Euclidean distance, and provides diagnostic safety bands for identifying unreliable assignments.

## Repository Structure

```
.
├── codigo/                  # Core data and analysis scripts
│   ├── *.csv                # Coordinates, assignments, analysis results
│   ├── calculate_anisotropy.py
│   ├── calculate_plant_anisotropy.py
│   ├── distancias.py
│   ├── distributional_robustness_analysis.py
│   ├── safety_bands_analysis.py
│   └── recalculate_confidence_interval.py
├── tables/                  # Distance matrices and result tables
│   ├── D_*_clean.csv        # Euclidean and network distance matrices
│   ├── *.csv                # Summary statistics and performance metrics
│   └── *.tex                # LaTeX-formatted tables for the manuscript
├── figuras_clean/           # Figure generation scripts (Figures 1-15)
│   ├── generate_figure*.py  # One script per manuscript figure
│   └── FIGURES_DOCUMENTATION.md
├── scripts/                 # Additional analysis scripts
│   ├── analyze_k_nearest_*.py
│   ├── analyze_s_sensitivity_correct.py
│   └── README_ANALYSIS_SCRIPTS.md
├── figures/                 # Output directory for generated figures
├── extremadura.geojson      # Study region boundary (Extremadura, Spain)
├── distributional_analysis.py       # Distributional comparison (Fig. 13)
├── distributional_sensitivity_s.py  # Sensitivity to s parameter (Fig. 14)
├── make_qq_final.py                 # Q-Q plots for real data (Fig. 12)
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
| `distancias_euclideas.csv` | Euclidean distance matrix (municipalities x plants) |

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
| `misallocated_municipalities.csv` | List of 61 misallocated municipalities |
| `complete_anisotropy_coefficients.csv` | Per-municipality anisotropy index |
| `detailed_ratios_analysis.csv` | Beta ratio analysis (d_net/d_Euclidean) |
| `safety_bands_lookup_table.csv` | Safety band thresholds for various q* levels |
| `sensitivity_s_analysis.csv` | Sensitivity of predictions to parameter s |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all manuscript figures
cd figuras_clean
python generate_figure1.py    # Study area and beta distributions
python generate_figure2.py    # Beta CDF and theoretical fit
# ... (see FIGURES_DOCUMENTATION.md for full mapping)

# Run distributional robustness analysis
python ../distributional_analysis.py

# Run sensitivity analysis
python ../distributional_sensitivity_s.py
```

## Figure-to-Script Mapping

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `figuras_clean/generate_figure1.py` | Study area with beta distributions |
| Fig. 2 | `figuras_clean/generate_figure2.py` | CDF of beta and log-normal fit |
| Fig. 3 | `figuras_clean/generate_figure3.py` | Misallocation probability P(mis\|R) |
| Fig. 4 | `figuras_clean/generate_figure4.py` | Local geometry approximation |
| Fig. 5 | `figuras_clean/generate_figure5.py` | Safety bands visualization |
| Fig. 6 | `figuras_clean/generate_figure6.py` | Confidence interval for misallocation count |
| Fig. 7 | *(TikZ in manuscript)* | Conceptual diagram |
| Fig. 8 | `figuras_clean/generate_figure8.py` | Spatial distribution of misallocations |
| Fig. 9 | `figuras_clean/generate_figure9.py` | CAR/BYM spatial robustness |
| Fig. 10 | `figuras_clean/generate_figure10.py` | Safety bands with Voronoi overlay |
| Fig. 11 | `figuras_clean/generate_figure11.py` | k-nearest capture performance |
| Fig. 12 | `make_qq_final.py` | Q-Q distributional validation |
| Fig. 13 | `distributional_analysis.py` | Distributional comparison |
| Fig. 14 | `distributional_sensitivity_s.py` | Sensitivity of distributional predictions |
| Fig. 15 | `figuras_clean/generate_figure15.py` | Algorithmic complexity analysis |

## Road Network Data

The road network GeoJSON file (`carreteras.geojson`, ~276 MB) used to compute network distances is not included in this repository due to size constraints. The network distances are provided pre-computed in `tables/D_real_*.csv`. The road network data was obtained from the Spatial Data Infrastructure of Extremadura (IDEEx) and OpenStreetMap.

## Study Area

Extremadura, Spain: 383 municipalities, 46 aggregate production facilities, 41,635 km².

## Requirements

- Python >= 3.8
- See `requirements.txt` for package dependencies

## License

MIT License. See [LICENSE](LICENSE) for details.
