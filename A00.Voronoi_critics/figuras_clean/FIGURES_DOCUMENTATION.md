# Figures Documentation - Voronoi Spatial Analysis

**Project**: Voronoi vs Network-Based Assignment Analysis
**Context**: Analysis of 15.0% misassignment rate in Voronoi tessellation vs real network distances
**Generated**: September 30, 2025

---

## Figure 1: Histogram of the Network Scaling Factor (β = d_r/d_e)

### Description:
- **Panel A**: Beta distribution for municipality-to-municipality distances
- **Panel B**: Beta distribution for municipality-to-assigned-plant distances

### Methodology:
Calculates the network scaling factor β (beta) as the ratio between real network distance (d_r) and Euclidean distance (d_e). Uses cleaned distance tables with corrected data to ensure β ≥ 1.0. Two separate datasets are analyzed:
1. Municipality-to-municipality distances (larger dataset)
2. Municipality-to-assigned-plant distances (Voronoi assignments only)

### Interpretation:
The beta coefficient quantifies the "network penalty" - how much longer real routes are compared to straight-line distances. Values > 1.0 indicate network constraints. Higher beta values suggest more complex terrain or indirect routing. The distribution shows the variability in network efficiency across the region.

### Statistical Results:
- Municipality-municipality beta: Mean ≈ 1.4-1.6, wide distribution
- Municipality-plant beta: Mean ≈ 1.3-1.5, slightly more concentrated
- Both distributions exclude outliers > 5.0 for visualization

### Output Files:
- `histogram_ratio_dr_de.png`

---

## Figure 2: Violin Plot of Beta Distribution Comparison

### Description:
- **Single Panel**: Violin plots comparing beta distributions between municipality-to-municipality and municipality-to-assigned-plant distances

### Methodology:
Creates violin plots using seaborn to show the probability density of beta coefficients. Filters outliers (β ≤ 5.0) for display while maintaining complete statistics. Includes mean lines and comprehensive statistics for both datasets.

### Interpretation:
Violin plots reveal the shape and spread of beta distributions more clearly than histograms. The width at each y-value represents the density of observations. Municipality-to-municipality distances typically show broader distributions due to the larger sample size and more diverse route types.

### Statistical Results:
- Displays complete statistics including outliers
- Shows density curves for better distribution comparison
- Red dashed lines mark means for visual reference

### Output Files:
- `violin_plot_beta_comparison.png`

---

## Figure 3: Plant Anisotropy Analysis (2x2)

### Description:
- **Panel A**: Histogram of anisotropy coefficients by plant with mean and median
- **Panel B**: Boxplots of anisotropy by municipality assignment groups (1-5, 6-10, 11-20, >20)
- **Panel C**: Scatter plot of municipalities served vs anisotropy coefficient with trend line
- **Panel D**: Min vs Max beta scatter plot with 1:1 isotropy line and reference lines

### Methodology:
Calculates plant-specific anisotropy coefficients as the ratio of maximum to minimum beta values for each plant. Groups plants by number of assigned municipalities (Voronoi assignments). Filters outliers with anisotropy > 10. Includes correlation analysis and trend lines.

### Interpretation:
Anisotropy measures how consistently a plant serves its assigned municipalities. Low anisotropy (close to 1.0) indicates consistent service quality, while high anisotropy suggests some municipalities are much harder to serve. The analysis reveals whether larger plants (serving more municipalities) tend to have higher anisotropy.

### Statistical Results:
- Anisotropy range (filtered): typically 1.0-8.0
- Correlation between municipality count and anisotropy
- Distribution across municipality assignment groups

### Output Files:
- `plant_anisotropy_analysis.png`

---

## Figure 4: Municipality Assignment Changes (Real vs Voronoi)

### Description:
- **Single Panel**: Bar chart showing net change in municipality assignments for each plant when comparing real network-based vs Voronoi assignments

### Methodology:
Compares Voronoi assignments (nearest plant by Euclidean distance) with optimal network assignments (nearest plant by network distance). Calculates net gain/loss for each plant. Red bars indicate plants gaining municipalities, gray bars indicate plants losing municipalities.

### Interpretation:
Reveals the redistribution of municipality assignments when switching from Voronoi to network-based allocation. The 15.0% misassignment rate manifests as specific plants gaining or losing municipalities. Zero-sum game: total gains equal total losses.

### Statistical Results:
- Plants gaining vs losing municipalities
- Magnitude of assignment changes
- Verification of zero-sum balance

### Output Files:
- `municipality_assignment_changes.png`

---

## Figure 5: Q-Q Plots of Beta Coefficients vs Theoretical Distributions (2x3)

### Description:
**Top row** (Municipality-to-Municipality):
- **Panel A**: Q-Q vs Log-Normal distribution
- **Panel B**: Q-Q vs Gamma distribution
- **Panel C**: Q-Q vs Weibull distribution

**Bottom row** (Municipality-to-Assigned-Plant):
- **Panel D**: Q-Q vs Log-Normal distribution
- **Panel E**: Q-Q vs Gamma distribution
- **Panel F**: Q-Q vs Weibull distribution

### Methodology:
Creates quantile-quantile plots comparing empirical beta distributions with fitted theoretical distributions. Uses scipy.stats for distribution fitting and probplot generation. Calculates R-squared values for goodness of fit assessment.

### Interpretation:
Q-Q plots assess how well theoretical distributions fit the empirical beta data. Points lying on the red reference line indicate good fit. Systematic deviations suggest the theoretical distribution doesn't capture the data characteristics. Useful for selecting appropriate statistical models.

### Statistical Results:
- R-squared values for each distribution fit
- Visual assessment of distribution tails and central tendency
- Comparison between municipality-municipality and municipality-plant patterns

### Output Files:
- `qq_plots_beta_distributions.png`

---

## Figure 6: Beta Distribution with Fitted Theoretical Distributions

### Description:
- **Single Panel**: Histogram of beta coefficients with overlaid fitted distribution curves (Log-Normal, Normal, Gamma, Weibull)

### Methodology:
Combines all beta ratios (municipality-municipality and municipality-plant) and fits multiple theoretical distributions. Uses density normalization for proper comparison with probability density functions. Applies Kolmogorov-Smirnov tests for goodness of fit evaluation.

### Interpretation:
Shows which theoretical distribution best fits the empirical beta data. The Log-Normal distribution is highlighted as it typically provides the best fit for network scaling factors. X-axis limited to 4.0 for better visualization while maintaining complete statistics.

### Statistical Results:
- Kolmogorov-Smirnov statistics for each distribution
- Best-fitting distribution identification
- Combined dataset statistics (mean, std, median)

### Output Files:
- `beta_distribution_fitted_curves.png`

---

## Figure 8: Spatial Analysis of Beta Coefficients (2x2)

### Description:
- **Panel A**: Spatial distribution of beta values using UTM coordinates with color scale
- **Panel B**: Moran's I scatter plot for spatial autocorrelation analysis
- **Panel C**: Histogram of beta coefficients with quartile-based coloring
- **Panel D**: Spatial clustering by quartile categories with UTM coordinates

### Methodology:
Uses real UTM coordinates from `coordenadas_municipios.csv` to create spatial maps. Calculates Moran's I statistic for spatial autocorrelation using inverse distance weights. Creates quartile categories for spatial clustering analysis. Maintains equal aspect ratios for accurate geographic representation.

### Interpretation:
Reveals spatial patterns in beta coefficients. Moran's I tests whether nearby municipalities have similar beta values (spatial autocorrelation). Positive Moran's I indicates clustering of similar values, negative indicates dispersion. Spatial patterns can reveal geographic or infrastructure influences on network efficiency.

### Statistical Results:
- Moran's I statistic and significance
- Spatial lag calculations
- Quartile thresholds and distributions

### Output Files:
- `spatial_analysis_beta_coefficients.png`

---

## Figure 9: Spatial Sensitivity Analysis (CAR/BYM Models) (3x2)

### Description:
**Top row**:
- **Panel A**: Original vs CAR-adjusted scatter plot with fit lines
- **Panel B**: Original vs BYM-adjusted scatter plot with fit lines

**Middle row**:
- **Panel C**: UTM spatial map of CAR adjustment differences
- **Panel D**: UTM spatial map of BYM adjustment differences

**Bottom row**:
- **Panel E**: Histogram of adjustment distributions (CAR and BYM)
- **Panel F**: Confusion matrix comparison (TP, TN, FP, FN)

### Methodology:
Simulates Conditional Autoregressive (CAR) and Besag-York-Mollie (BYM) spatial models for sensitivity analysis. CAR model includes spatial lag effects, BYM combines structured and unstructured random effects. Uses spatial weights based on distance for neighbor identification.

### Interpretation:
Tests robustness of findings to different spatial modeling assumptions. CAR and BYM models adjust beta values based on spatial relationships. Comparison shows how sensitive the analysis is to spatial modeling choices. Confusion matrices evaluate classification performance under different models.

### Statistical Results:
- Model fit statistics (R-squared, slopes)
- Spatial adjustment ranges
- Classification accuracy metrics

### Output Files:
- `spatial_sensitivity_analysis.png`

---

## Figure 10: Safety Bands for Voronoi Risk Assignment (1x3)

### Description:
- **Panel A**: 10% risk level safety bands
- **Panel B**: 20% risk level safety bands
- **Panel C**: 30% risk level safety bands

Each panel shows curves for different terrain complexity parameters (s = 0.05 to 0.20) relating critical distance t* (km) to geometric parameter kappa.

### Methodology:
Uses theoretical safety band model: kappa = alpha * (1 + beta * s) * exp(-gamma * t* / (1 + delta * risk)). Terrain parameter s = 0.093 represents Extremadura's moderate terrain complexity. Validates parameter against empirical beta distribution data.

### Interpretation:
Safety bands define risk thresholds for Voronoi assignment reliability. Higher risk levels require larger safety margins. Different terrain types (urban, plains, mountainous) show distinct safety curves. Extremadura's parameter (s = 0.093) is highlighted as the regional reference.

### Statistical Results:
- Terrain complexity parameter validation
- Risk threshold curves for different environments
- Critical distance ranges for safe Voronoi application

### Output Files:
- `safety_bands_voronoi_risk.png`

---

## Figure 11: Computational Performance Analysis (2x3)

### Description:
**Top row**:
- **Panel A**: Computational scalability (log-log plot)
- **Panel B**: Execution time vs total cost scatter plot
- **Panel C**: Efficiency rate bar chart with ratios

**Bottom row**:
- **Panel D**: Misassignment risk by algorithm (%)
- **Panel E**: Algorithm scalability (time per municipality)
- **Panel F**: Normalized performance comparison (efficiency & risk control)

### Methodology:
Compares five algorithms: Voronoi (baseline), k-nearest-3, k-nearest-5, optimal_approx (proposed method), and network_analysis (true optimal). Uses real algorithm simulations with actual data. Measures execution time, assignment quality, and scalability.

### Interpretation:
Evaluates trade-offs between computational efficiency and assignment quality. Voronoi is fastest but has 15.4% misassignment risk. Network analysis is optimal but slower. The proposed optimal_approx method balances efficiency and accuracy with only 2.4% misassignment risk.

### Statistical Results:
- Execution times across municipality counts
- Accuracy metrics and misassignment rates
- Efficiency ratios relative to optimal solution

### Output Files:
- `computational_performance_analysis.png`

---

## Figure 12: Distance Improvement Analysis (1x2)

### Description:
- **Panel A**: Histogram of distance improvements with mean marked
- **Panel B**: Bar chart of correct vs incorrect assignments with counts and percentages

### Methodology:
Compares network-based vs Voronoi assignments by calculating distance improvements for each municipality. Correctly assigned municipalities (where both methods choose the same plant) have zero improvement. Incorrectly assigned municipalities show actual distance savings.

### Interpretation:
Quantifies the practical benefit of using network-based assignment. The 15.0% of incorrectly assigned municipalities experience significant distance savings when switching to network-based assignment. Correctly assigned municipalities validate the Voronoi method's performance where it works well.

### Statistical Results:
- Mean distance improvement for misassigned municipalities
- Assignment accuracy: ~84.6% correct, 15.4% incorrect
- Distribution of improvement magnitudes

### Output Files:
- `distance_improvement_analysis.png`

---

## Figure 13: Distribution of Distance Ratio Improvements

### Description:
- **Single Panel**: Histogram showing the distribution of relative distance improvements when switching from Voronoi to network-based assignment

### Methodology:
Calculates improvement ratios as (Voronoi_distance - Optimal_distance) / Voronoi_distance. Values close to zero indicate minimal improvement (correct assignments), while positive values show relative savings from network-based assignment.

### Interpretation:
Shows the magnitude of relative improvements available through network-based assignment. The 15.0% misassignment rate corresponds to the fraction of municipalities with non-zero improvements. Higher ratio values indicate greater relative benefits from optimal assignment.

### Statistical Results:
- Mean improvement ratio across all municipalities
- Distribution of relative improvements
- Validation of misassignment rate consistency

### Output Files:
- `distance_ratio_improvements.png`

---

## Figure 15: Euclidean vs Real Distance Correlation

### Description:
- **Single Panel**: Scatterplot of euclidean distance (x-axis) vs real network distance (y-axis) for all municipality-plant connections

### Methodology:
Plots all municipality-plant distance pairs with equal aspect ratio. Red dashed line shows perfect correlation (45-degree line) where real distance equals euclidean distance. Points above the line indicate β > 1, which is expected due to network constraints.

### Interpretation:
Demonstrates the fundamental relationship between euclidean and network distances. Strong correlation validates the beta scaling factor approach. Systematic deviation above the 1:1 line quantifies the network penalty imposed by road infrastructure constraints.

### Statistical Results:
- Correlation coefficient between euclidean and real distances
- Mean beta ratio validation
- Distribution of points relative to perfect correlation line

### Output Files:
- `euclidean_vs_real_scatterplot.png`

---

## Summary Statistics

### Key Findings:
- **Misassignment Rate**: 15.0% of municipalities assigned to suboptimal plants by Voronoi method
- **Network Scaling Factor**: Mean β ≈ 1.4-1.6 across different route types
- **Distance Penalty**: Network routes average 40-60% longer than euclidean distances
- **Terrain Complexity**: Extremadura parameter s = 0.093 represents moderate terrain
- **Spatial Autocorrelation**: Moran's I indicates spatial clustering of beta values

### Methodological Notes:
- All analyses use cleaned, corrected distance tables ensuring β ≥ 1.0
- Outlier filtering (β ≤ 5.0) applied for visualization while maintaining complete statistics
- UTM coordinates ensure accurate spatial analysis with equal aspect ratios
- Statistical significance testing included where appropriate

### Data Sources:
- `tablas/D_euclidea_plantas_clean.csv`: Euclidean distances
- `tablas/D_real_plantas_clean_corrected.csv`: Network distances (corrected)
- `tablas/D_euclidea_municipios_clean.csv`: Municipality distances (euclidean)
- `tablas/D_real_municipios_clean.csv`: Municipality distances (network)
- `coordenadas_municipios.csv`: UTM coordinates for spatial analysis

---

**Note**: Figures 7 and 14 were not generated as they were not included in the analysis scope. All other figures (1-6, 8-13, 15) provide comprehensive coverage of the Voronoi vs network-based assignment analysis.