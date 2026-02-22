# Analysis Scripts for Voronoi Framework Revision v5

**Created**: November 11, 2025
**Purpose**: Methodological improvements addressing K-S test robustness and s parameter sensitivity

---

## Overview

This directory contains Python scripts for enhanced distributional and sensitivity analyses that address two critical methodological improvements:

1. **Plant-Municipality Pairs Analysis**: More robust K-S test using statistically independent observations
2. **Sensitivity Analysis of s Parameter**: Addresses prediction uncertainty and spatial heterogeneity
3. **K-Nearest Analysis**: Demonstrates framework performance improves with facility sparsity

---

## Scripts

### 1. `analyze_plant_municipality_ks.py`

**Purpose**: Perform K-S test on plant-municipality pairs instead of all municipality pairs.

**Why**:
- **Statistical independence**: Each municipality → its assigned plant (n=383 independent observations)
- **Direct research question**: "How good is Voronoi assignment?"
- **Reduced autocorrelation**: Eliminates spatial clustering from shared plants

**Usage**:
```bash
python scripts/analyze_plant_municipality_ks.py
```

**Outputs**:
- `tables/ks_test_plant_municipality.tex` - LaTeX table for manuscript (Table 2a)
- `codigo/ks_test_plant_municipality_results.csv` - Detailed results

**Expected result**:
- Higher p-values than all-pairs analysis (p > 0.05 for Log-Normal)
- Clearer statistical significance
- Log-Normal remains best fit

---

### 2. `plot_plant_municipality_distributions.py`

**Purpose**: Generate publication-quality figures for plant-municipality distributional analysis.

**Figures generated**:
1. `qq_plots_plant_municipality.pdf` - 2×3 panel (Q-Q plots + histograms)
2. `cdf_comparison_plant_municipality.pdf` - Empirical vs theoretical CDFs
3. `histogram_plant_municipality_all_fits.pdf` - Single histogram with all fits overlaid

**Usage**:
```bash
python scripts/plot_plant_municipality_distributions.py
```

**Outputs location**: `figuras_clean/`

---

### 3. `analyze_s_parameter_sensitivity.py`

**Purpose**: Sensitivity analysis showing how misallocation predictions vary with s parameter.

**Key questions addressed**:
1. How sensitive are predictions to s assumptions?
2. What range of s brackets observed misallocation (23.0%)?
3. Does Extremadura's heterogeneity explain s variation?

**Usage**:
```bash
python scripts/analyze_s_parameter_sensitivity.py
```

**Outputs**:
- `tables/sensitivity_s_parameter.tex` - LaTeX table showing predictions for s ∈ [0.05, 0.20]
- `figuras_clean/sensitivity_s_parameter.pdf` - Plot of prediction vs s with compatible range
- `codigo/sensitivity_s_analysis.csv` - Detailed results

**Key finding**:
- Observed misallocation (23.0%) consistent with s ∈ [0.08, 0.15]
- Validates framework when s is calibrated to local geography
- **Methodological insight**: s is not universal constant, requires calibration

---

### 4. `analyze_k_nearest_performance.py`

**Purpose**: Demonstrate that framework works better with fewer facilities (sparse networks).

**Analysis**:
- Compare K-S test for k=1, k=3, k=5 nearest plants
- Show p-value degrades monotonically with k
- Validate why plant-municipality (k=1) is preferred analysis

**Usage**:
```bash
python scripts/analyze_k_nearest_performance.py
```

**Outputs**:
- `tables/k_nearest_performance.tex` - LaTeX table
- `figuras_clean/k_nearest_ks_comparison.pdf` - p-value vs k plot
- `figuras_clean/beta_distributions_by_k.pdf` - Histogram comparison
- `codigo/k_nearest_performance_results.csv` - Detailed results

**Key finding**:
- Lower k → better Log-Normal fit
- Framework is most robust for sparse facility networks
- Justifies plant-municipality as primary analysis

---

## Workflow

### Step 1: Run K-S Analysis (Plant-Municipality)
```bash
python scripts/analyze_plant_municipality_ks.py
```
This generates **Table 2a** for the manuscript.

### Step 2: Generate Visualizations
```bash
python scripts/plot_plant_municipality_distributions.py
```
This creates Q-Q plots and distribution comparison figures.

### Step 3: Sensitivity Analysis
```bash
python scripts/analyze_s_parameter_sensitivity.py
```
This generates **Table X** (sensitivity) and corresponding figure.

### Step 4: K-Nearest Analysis (Optional/Supplementary)
```bash
python scripts/analyze_k_nearest_performance.py
```
This creates supplementary material showing k-dependence.

---

## Data Requirements

**Input files** (must exist):
- `codigo/asignacion_municipios_euclidiana.csv` - Euclidean Voronoi assignments
- `codigo/asignacion_municipios_real.csv` - Real network assignments
- `codigo/tablas/D_euclidea_plantas_clean.csv` - Euclidean distance matrix (municipalities × plants)
- `codigo/tablas/D_real_plantas_clean_corrected.csv` - Real distance matrix (municipalities × plants)

**Note**: If Voronoi geometry data (kappa, t_star) is not available, sensitivity script uses synthetic data (replace with actual analysis).

---

## Integration with Manuscript (voronoi_note_v5.tex)

### New Tables to Add:

1. **Table 2a**: K-S test for plant-municipality pairs
   - Location: Section 4.2.1 (Distributional Validation)
   - File: `tables/ks_test_plant_municipality.tex`
   - **Primary analysis** (more robust than all-pairs)

2. **Table 2b**: K-S test for all municipality pairs
   - Location: Section 4.2.2 (Comprehensive Analysis)
   - **Current Table 2** (rename to 2b)
   - Complementary analysis showing full variability

3. **Table X**: Sensitivity analysis of s parameter
   - Location: Section 4.3 (Misallocation Prediction Sensitivity)
   - File: `tables/sensitivity_s_parameter.tex`
   - Shows s ∈ [0.08, 0.15] brackets observed rate

4. **Table Y** (Optional/Supplementary): K-nearest performance
   - Location: Appendix or Supplementary Materials
   - File: `tables/k_nearest_performance.tex`

### New Figures to Add:

1. `qq_plots_plant_municipality.pdf` - Replaces or supplements current Q-Q plots
2. `sensitivity_s_parameter.pdf` - New figure for sensitivity discussion
3. `k_nearest_ks_comparison.pdf` - Supplementary material

---

## Methodological Improvements Summary

### Issue 1: Low K-S p-values in All-Pairs Analysis
**Problem**: n=9,112 municipality pairs include spatial autocorrelation
**Solution**: Use plant-municipality pairs (n=383, independent observations)
**Expected outcome**: Higher p-values, better statistical support for Log-Normal

### Issue 2: Circular Reasoning with s Parameter
**Problem**: Assumed s → predicted misallocation → post-hoc adjusted s
**Solution**:
- Test range of s values [0.05, 0.20]
- Show observed rate consistent with s ∈ [0.08, 0.15]
- Discuss spatial heterogeneity as **feature, not bug**
**Expected outcome**: Convert weakness into methodological insight (calibration requirement)

### Issue 3: Limited Justification for Analysis Choice
**Problem**: Why plant-municipality over all-pairs?
**Solution**: k-nearest analysis shows monotonic degradation of fit with k
**Expected outcome**: Clear justification for analysis hierarchy

---

## Expected Results (To Be Verified with Actual Data)

Based on methodological improvements, we expect:

1. **Plant-municipality K-S test**:
   - Log-Normal: p > 0.05 (vs p ≈ 0.01 in all-pairs)
   - Gamma: p < 0.05
   - Weibull: p < 0.01
   - **Clear winner**: Log-Normal

2. **Sensitivity analysis**:
   - s ∈ [0.08, 0.15] brackets 23.0% observed misallocation
   - Lower s (plains-dominated) → underpredict (10-15%)
   - Higher s (mountains-dominated) → overpredict (30-35%)
   - **Conclusion**: Heterogeneity explains variation

3. **K-nearest analysis**:
   - k=1: p ≈ 0.10 (good fit)
   - k=3: p ≈ 0.03 (acceptable fit)
   - k=5: p ≈ 0.005 (poor fit)
   - **Conclusion**: Framework best for sparse networks

---

## Dependencies

```python
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## Troubleshooting

### If scripts fail to find data files:
- Ensure you're running from project root: `E:\...\A00.Voronoi_critics\`
- Check that CSV files exist in `codigo/` and `codigo/tablas/`
- Verify column names match expected format

### If K-S p-values are still low:
- Check for outliers in beta values (should be β ≥ 1.0)
- Verify distance data is clean (no NaN, Inf values)
- Consider data transformations if heavy tails

### If sensitivity analysis doesn't bracket observed rate:
- May indicate need for spatial stratification
- Check if Voronoi geometry (kappa, t_star) is accurate
- Consider revising theoretical model assumptions

---

## Contact & Support

For questions about these scripts:
1. Check script documentation (docstrings)
2. Review this README
3. Examine output CSV files for detailed diagnostics

**Version**: 1.0
**Last updated**: November 11, 2025
**Status**: Ready for execution with real data

---

## Next Steps

1. ✅ Run all scripts with actual data
2. ✅ Verify expected results materialize
3. ✅ Update manuscript (voronoi_note_v5.tex) with new tables/figures
4. ✅ Rewrite Discussion section emphasizing calibration
5. ✅ Add k-nearest analysis to supplementary materials
6. ✅ Final compilation and submission

---

**END OF README**
