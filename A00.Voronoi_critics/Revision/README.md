# Major Revision: artefacts and analyses

This folder collects the artefacts produced for the **major revision**
of manuscript 2303622 (under review at *Geographical Analysis*). It
complements the original reproducibility package in `../codigo/`,
`../tables/`, `../figuras_clean/`, and `../scripts/`.

The structure mirrors the methodological lines addressed in the
response letter.

## Contents

```
Revision/
├── analyses/
│   ├── A1_spatial_carbym/      # Bayesian log-CAR and BYM2 posterior fits
│   │                             on log(beta). Replaces the v1 synthetic
│   │                             calibrated study. Comment 16, 17.
│   ├── A2_distributional/      # Wasserstein-1 + Anderson-Darling
│   │                             distributional ranking of five candidates
│   │                             on both muni-muni and muni-facility
│   │                             domains. Comment 18.
│   ├── A3_anisotropy/          # Anisotropy coefficient alpha = beta_max/beta_min,
│   │                             with ROC against observed misallocations.
│   ├── outputs/
│   │   ├── figures/            # PNG + PDF figures of A1, A2, A3 outputs
│   │   └── tables/             # LaTeX-formatted summary tables
│   ├── data/                   # Canonical input dataset used by A1-A3
│   │                             (small files; large distance matrices are
│   │                             in ../../codigo/ and ../../tables/)
│   ├── environment.yml         # Conda env spec
│   ├── requirements.txt        # pip requirements
│   └── README.md
```

## What is *not* in this folder

- **The road network shapefile** (`shp/carreteras.*`, ~620 MB) is
  excluded by `.gitignore` due to size. It can be reconstructed from
  public sources (Spatial Data Infrastructure of Extremadura, IDEEx; and
  OpenStreetMap).
- **The large pre-computed distance matrices** (`D_real_municipios_clean.csv`,
  `beta_munimuni.csv`, etc.) are available in the parent reproducibility
  package (`../codigo/`, `../tables/`).
- **The manuscript and response letters** are submitted through the
  journal's editorial system; they are not part of this code repository.

## Reproducing the revision analyses

```bash
# Set up the environment (recommended: conda)
cd Revision/analyses
conda env create -f environment.yml
conda activate voronoi-revision

# A1: Bayesian log-CAR + BYM2 (PyMC + NUTS)
python A1_spatial_carbym/run.py

# A2: Wasserstein-1 + Anderson-Darling distributional ranking
python A2_distributional/run.py

# A3: Anisotropy coefficient and ROC analysis
python A3_anisotropy/run.py
```

Each subfolder contains its own `README.md` with details on inputs,
outputs and the corresponding section of the response letter.

## QGIS Processing Toolbox

The companion QGIS plugin that implements the full framework lives at
[`../qgis_plugin/`](../qgis_plugin/). It packages Voronoi assignment,
network-distance beta calculation, safety bands, misallocation detection
and anisotropy mapping as five Processing algorithms.

See [`../qgis_plugin/README.md`](../qgis_plugin/README.md) for installation,
parameters and a usage walkthrough.
