# Reproducibility Package: A01 — Spatial Cost Analysis of CDW Management

This directory contains all code and data needed to reproduce the results of:

**"Spatial Cost Analysis of Construction and Demolition Waste Management:
Beyond Geometric Proximity in Sparsely Populated Regions (Extremadura, Spain)"**

*Submitted to Waste Management (Elsevier)*

## Key Findings (Extremadura case study)

- **383 municipalities**, **46 treatment plants** consolidated into **32 plant groups**.
- **Municipality-level (unweighted) mean cost**: €15.82/t.
- **Production-weighted system mean cost**: €12.22/t.
- Municipal cost ranges from ≈ €5/t (urban catchments) to ≈ €50/t (remote rural municipalities).
- **8 of 32 plant groups** operate below the **5,000 t/yr economy-of-scale threshold**.
- Consolidating **14 redundant co-located facilities** would save ≈ **€303,000/year (4.5%)** without increasing transport distances.

## Repository Structure

```
A01.Voronoi/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (pip)
├── environment.yml                    # Conda environment
├── LICENSE                            # CC-BY 4.0
├── .gitignore
├── data/                              # Input data (anonymised / aggregated)
│   ├── coordenadas_municipios.csv     # 383 municipality centroids
│   ├── coordenadas_plantas.csv        # 46 plant locations
│   ├── Plantas_con_nombre_municipio.csv
│   ├── Matriz_Municipios.csv          # 383x46 distance matrix
│   ├── distancias_reales_penalizadas_simplificado.csv
│   ├── distancias_euclideas.csv
│   ├── asignacion_municipios_real.csv      # network-based allocation
│   ├── asignacion_municipios_euclidiana.csv # Voronoi reference allocation
│   ├── datos_voronoi_municipios.csv
│   └── plant_anisotropy_coefficients.csv
├── analysis/                          # Analytical scripts producing manuscript results
│   ├── modelo_coste.py                          # Piecewise logarithmic cost model
│   ├── script_asignacion_planta_distancia_real.py    # Network-based allocation
│   ├── script_asignacion_planta_distancia_euclidea.py # Euclidean reference
│   ├── recalcula_costes_red_real.py             # Plant and municipality unit costs
│   ├── script_produccion.py                     # Per-plant throughput
│   ├── script_produccion_municipios.py          # Per-municipality CDW generation
│   ├── sensibilidad_dotacion.py                 # Sensitivity to ρ_CDW (Fig. sensibilidad_dotacion)
│   ├── spatial_analysis_clean.py                # Spatial autocorrelation
│   ├── spatial_sensitivity_analysis.py          # Robustness checks
│   ├── distributional_robustness_analysis.py    # Distributional comparison
│   ├── genera_figuras_articulo.py               # Manuscript figures
│   ├── genera_mapas_articulo.py                 # Manuscript maps
│   └── generate_basic_plots.py                  # Auxiliary plots
├── replication/
│   ├── run_full_analysis.py           # One-click end-to-end pipeline
│   └── procesa_datos_unificado.py     # Master data processing
├── docs/                              # Methodology and data documentation
└── results/                           # Generated figures and tables (gitignored)
    ├── figures/
    └── tables/
```

## Quick Start

### Option 1: pip

```bash
git clone https://github.com/jtorreci/garnocex_research.git
cd garnocex_research/A01.Voronoi
pip install -r requirements.txt
python replication/run_full_analysis.py
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate a01-spatial-cost
python replication/run_full_analysis.py
```

## Main Results Reproduction

### Cost model and headline figures

```bash
python analysis/modelo_coste.py                      # Cost curve and parameters
python analysis/recalcula_costes_red_real.py         # Plant and municipality unit costs
python analysis/script_produccion.py                 # Per-plant throughput
python analysis/script_produccion_municipios.py      # Per-municipality CDW generation
```

### Allocation (network vs. Euclidean reference)

```bash
python analysis/script_asignacion_planta_distancia_real.py
python analysis/script_asignacion_planta_distancia_euclidea.py
```

### Sensitivity analysis (ρ_CDW, robustness)

```bash
python analysis/sensibilidad_dotacion.py             # Fig. sensibilidad_dotacion
python analysis/spatial_sensitivity_analysis.py
python analysis/distributional_robustness_analysis.py
```

### Manuscript figures and maps

```bash
python analysis/genera_figuras_articulo.py
python analysis/genera_mapas_articulo.py
```

## Key Parameters

| Parameter | Description | Value used |
|-----------|-------------|-----------|
| `C0`      | Base fixed cost (single-operator plant) | 40,000 €/yr |
| `T0`      | Economy-of-scale threshold              | 5,000 t/yr  |
| `v`       | Variable treatment cost                 | 0.35 €/t    |
| `ρ_trans` | Effective transport rate                | 0.35 €/t·km |
| `ρ_CDW`   | Per-capita CDW generation               | 0.50 t/inhab·yr |
| `EPSG`    | Coordinate reference system             | 25830 (ETRS89 / UTM 30N) |

## Companion Paper

The methodological companion paper that quantifies the bias of Voronoi-based allocation in this same network is available under [A00.Voronoi_critics](../A00.Voronoi_critics/). The two papers share input data (municipality and plant coordinates, road-network distance matrix) but produce independent analyses.

## Citation

```bibtex
@article{torrecillaSpatialCostAnalysis2026,
  title   = {Spatial Cost Analysis of Construction and Demolition Waste Management:
             Beyond Geometric Proximity in Sparsely Populated Regions (Extremadura, Spain)},
  author  = {Torrecilla-Pinero, J.A. and Ceballos-Martinez, J.M. and Plaza Caballero, P.
             and Cruces Lopez, A. and Cuartero Saez, A.},
  journal = {Waste Management},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

- GitHub Issues: <https://github.com/jtorreci/garnocex_research/issues>

## License

Released under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**; see `LICENSE`.

## Acknowledgements

This research was funded by the **GARNOCEX project**, a collaborative agreement between the Junta de Extremadura, the Colegio de Ingenieros Tecnicos de Obras Publicas de Extremadura, four regional CDW management companies, and the Universidad de Extremadura.

---

**Keywords**: construction and demolition waste, spatial cost analysis, economies of scale, polluter pays principle, network-based routing, facility location, rural depopulation
