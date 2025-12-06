# Reproducibility Package: Voronoi Probabilistic Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17772773.svg)](https://doi.org/10.5281/zenodo.17772773)

This repository contains all code, data, and analysis scripts needed to reproduce the results from:

**"The Hidden Cost of Straight Lines: Quantifying Misallocation Risk in Voronoi-Based Service Area Models"**

*Submitted to Computers, Environment and Urban Systems (CEUS)*

## Key Findings (Extremadura Case Study)

- **383 municipalities**, 46 treatment facilities
- **15.4% misallocation rate** (59 municipalities incorrectly assigned by Euclidean Voronoi)
- Log-Normal distribution best fit: μ = 0.166, σ = 0.093
- Framework achieves **97.6% accuracy** at O(n) complexity

## 📁 Repository Structure

```
voronoi-probabilistic-framework/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── data/
│   ├── synthetic_data_generator.py    # Generate synthetic β factors
│   ├── extremadura_anonymized.csv     # Anonymized real data
│   └── geographic_coordinates.csv     # Municipality coordinates
├── analysis/
│   ├── distributional_robustness_analysis.py  # Table 2 & Fig 3-4
│   ├── safety_bands_analysis.py              # Fig 5-7 & Table 3
│   ├── spatial_analysis.R                    # Moran's I & CAR models
│   └── multi_facility_validation.py          # Trade-off curves
├── results/
│   ├── figures/                       # All publication figures
│   ├── tables/                        # All publication tables
│   └── supplementary/                 # Additional analyses
├── replication/
│   ├── run_full_analysis.py          # One-click reproduction
│   ├── calibration_new_region.py     # Apply to new geographic area
│   └── parameter_estimation.py       # Estimate s for new data
└── docs/
    ├── methodology.md                 # Detailed methodology
    ├── data_description.md            # Data documentation
    └── parameter_guide.md             # Parameter calibration guide
```

## 🚀 Quick Start

### Option 1: One-Click Reproduction
```bash
git clone https://github.com/username/voronoi-probabilistic-framework.git
cd voronoi-probabilistic-framework
pip install -r requirements.txt
python replication/run_full_analysis.py
```

### Option 2: Conda Environment
```bash
conda env create -f environment.yml
conda activate voronoi-framework
python replication/run_full_analysis.py
```

## 📊 Main Results Reproduction

### Table 2: Distributional Comparison
```bash
python analysis/distributional_robustness_analysis.py
# Outputs: distributional_comparison_table.tex
#          qq_plots_comparison.png
#          tail_behavior_comparison.png
```

### Figure 5-7: Safety Band Calibration
```bash
python analysis/safety_bands_analysis.py
# Outputs: safety_bands_calibration_curves.png
#          safety_bands_contour_maps.png
#          safety_bands_practical_examples.png
```

### Spatial Analysis (Requires R)
```bash
Rscript analysis/spatial_analysis.R
# Outputs: moran_test_results.csv
#          spatial_model_comparison.png
```

## 🔧 Apply to New Geographic Region

To apply the framework to your own geographic area:

1. **Prepare your data**: See `data/data_description.md` for format requirements
2. **Estimate parameters**:
   ```bash
   python replication/parameter_estimation.py --input your_data.csv --output calibrated_params.json
   ```
3. **Generate safety bands**:
   ```bash
   python replication/calibration_new_region.py --params calibrated_params.json
   ```

## 📋 Key Parameters

| Parameter | Description | Typical Range | Extremadura Value |
|-----------|-------------|---------------|-------------------|
| `s` | Geographic complexity | 0.03-0.20 | 0.093 |
| `κ` | Geometric parameter | 0.1-2.0 | 0.5 |
| `q*` | Risk threshold | 0.10-0.30 | 0.20 |

## 🌍 Geographic Context Calibration

| Context | s Range | Examples |
|---------|---------|-----------|
| Urban Dense | 0.05-0.08 | Metropolitan cores |
| Flat Plains | 0.03-0.06 | Agricultural areas |
| Moderate Hills | 0.08-0.12 | Extremadura-like |
| Mountainous | 0.10-0.15 | Alpine regions |
| Island/Complex | 0.12-0.20 | Archipelagos |

## 🤝 Contributing

We welcome contributions! Please see `docs/contributing.md` for guidelines.

## 📧 Contact

For questions about the methodology or code:
- GitHub Issues: [Repository Issues](https://github.com/jtorreci/garnocex_research/issues)

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)** - see `LICENSE` file for details.

This license is compatible with Elsevier's preprint policy and allows:
- Sharing and adaptation for any purpose
- Commercial use
- With proper attribution required

## 🏆 Acknowledgments

This research was funded by the **GARNOCEX project**, a collaborative agreement between the Regional Government of Extremadura (Junta de Extremadura), the College of Civil Engineers (Colegio de Ingenieros de Caminos, Canales y Puertos), and the University of Extremadura.

Technical acknowledgments:
- QNEAT3 plugin for network analysis
- SciPy and R spatial analysis communities

---

**Keywords**: Voronoi tessellation, probabilistic modeling, spatial optimization, network analysis, risk assessment, territorial planning
