# Methodology Documentation

## Voronoi Probabilistic Framework for Spatial Misallocation Assessment

### Overview

This framework provides a systematic approach to quantify the risk of misallocation when using Voronoi tessellation-based service area models instead of computationally expensive network-based optimization. The methodology enables practitioners to make informed decisions about when Euclidean approximations are sufficient versus when full network analysis is required.

### Core Mathematical Framework

#### 1. Network Scaling Factor β

The foundation of the framework is the network scaling factor β, defined as:

```
β = d_network / d_euclidean
```

Where:
- `d_network`: Actual travel distance via road network
- `d_euclidean`: Straight-line (Euclidean) distance

#### 2. Distributional Modeling

The β factors are modeled using a log-normal distribution:

```
β ~ LogNormal(μ, σ)
```

Where:
- `μ`: Location parameter of the log-normal distribution
- `σ`: Scale parameter (≈ geographic complexity parameter s)

This choice is justified by:
- **Multiplicative nature**: Distance ratios are naturally log-normal
- **Conservative tail behavior**: Overestimates risk in critical regions
- **Empirical validation**: Best fit across multiple goodness-of-fit criteria

#### 3. Probabilistic Risk Assessment

The probability of misallocation for a municipality at distance `t` from the Voronoi boundary is:

```
P(misallocation | t) ≈ Φ(-κ|t|/(√2 s))
```

Where:
- `Φ`: Standard normal cumulative distribution function
- `κ`: Geometric parameter (typically 0.2-2.0)
- `s`: Geographic complexity parameter
- `t`: Signed distance to Voronoi boundary

#### 4. Safety Bands Calibration

Critical distance thresholds are determined by:

```
|t*| = -Φ^(-1)(q*) × √2 × s / κ
```

Where:
- `q*`: Acceptable risk threshold (e.g., 10%, 20%, 30%)
- `Φ^(-1)`: Inverse standard normal CDF

**Decision Rule**:
- If `|t| < |t*|`: Use network-based analysis (high misallocation risk)
- If `|t| > |t*|`: Euclidean approximation acceptable (low risk)

### Parameter Estimation

#### Geographic Complexity Parameter (s)

Multiple estimation methods are provided:

1. **Maximum Likelihood Estimation (MLE)**:
   ```python
   σ_ln, loc_ln, scale_ln = stats.lognorm.fit(beta_values, floc=0)
   s_mle = σ_ln
   ```

2. **Method of Moments**:
   ```python
   log_beta = np.log(beta_values)
   s_mom = np.std(log_beta, ddof=1)
   ```

3. **Robust Quantile-based**:
   ```python
   q75, q25 = np.percentile(log_beta, [75, 25])
   s_robust = (q75 - q25) / (2 × Φ^(-1)(0.75))
   ```

4. **Bayesian with Informative Prior**:
   Uses conjugate inverse-gamma prior for the variance parameter.

#### Geographic Context Classification

Based on estimated `s` values:

| s Range | Context | Description |
|---------|---------|-------------|
| 0.03-0.06 | Urban Dense | High road density, minimal barriers |
| 0.06-0.09 | Flat Plains | Agricultural areas, good connectivity |
| 0.09-0.12 | Moderate Hills | Mixed terrain (Extremadura-like) |
| 0.12-0.15 | Mountainous | Significant elevation changes |
| 0.15-0.20 | Complex/Islands | Archipelagos, very complex topography |

### Validation Framework

#### 1. Distributional Validation

- **Kolmogorov-Smirnov test**: Overall goodness of fit
- **Anderson-Darling test**: Enhanced sensitivity to tail behavior
- **Information criteria**: AIC/BIC for model comparison
- **Q-Q plots**: Visual assessment of distributional assumptions

#### 2. Spatial Validation

- **Moran's I test**: Spatial autocorrelation in β factors
- **Local indicators (LISA)**: Identification of spatial clusters
- **CAR/BYM models**: Sensitivity analysis for spatial dependence

#### 3. Cross-validation

- **k-fold validation**: Stability of parameter estimates
- **Bootstrap resampling**: Confidence intervals
- **Out-of-sample prediction**: Generalization assessment

### Implementation Guidelines

#### For New Geographic Regions

1. **Data Collection**:
   - Minimum 100-200 municipality-facility pairs
   - Both Euclidean and network distances required
   - Geographic coordinates for spatial analysis (optional)

2. **Parameter Estimation**:
   ```bash
   python parameter_estimation.py --input data.csv --output params.json
   ```

3. **Calibration**:
   ```bash
   python calibration_new_region.py --data data.csv --output results/
   ```

4. **Validation**:
   - Review goodness-of-fit statistics
   - Check spatial autocorrelation if coordinates available
   - Validate against known misallocations (if available)

#### Safety Bands Application

1. **Choose risk threshold** (q*): 10%, 20%, or 30%
2. **Estimate geometric parameter** (κ) from Voronoi analysis
3. **Compute critical distance**:
   ```
   |t*| = -Φ^(-1)(q*) × √2 × s / κ
   ```
4. **Apply decision rule** for each municipality

### Computational Considerations

#### Resource Optimization

The framework enables optimal allocation of computational resources:

- **High-priority areas** (`|t| < |t*|`): Require network analysis
- **Low-priority areas** (`|t| > |t*|`): Euclidean approximation sufficient

Typical computational savings: 60-80% reduction in network calculations while maintaining 95%+ accuracy.

#### Scalability

- **Small regions** (<100 municipalities): Direct network optimization feasible
- **Medium regions** (100-1000 municipalities): Framework provides substantial savings
- **Large regions** (>1000 municipalities): Framework essential for practical implementation

### Extension to Multi-facility Problems

The framework extends to multi-facility scenarios:

1. **k-nearest facility assignment**: Consider k closest facilities instead of single nearest
2. **Capacity constraints**: Incorporate facility capacity limits
3. **Hierarchical services**: Different service levels with nested catchment areas

### Theoretical Foundations

#### Distance Ratio Distribution Theory

The log-normal distribution for β factors is theoretically justified by:

1. **Multiplicative central limit theorem**: Product of many small independent factors
2. **Geometric properties**: Road network tortuosity and elevation changes
3. **Scale invariance**: Distribution shape preserved across different scales

#### Risk Assessment Theory

The probabilistic approach is grounded in:

1. **Geometric probability**: Distance-based risk assessment
2. **Spatial point processes**: Facility location as spatial Poisson process
3. **Network topology theory**: Planar graph properties affecting distance ratios

### Limitations and Assumptions

#### Key Assumptions

1. **Spatial independence**: β factors assumed independent (validated via Moran's I)
2. **Log-normal distribution**: May not hold in extreme geographic contexts
3. **Stationary process**: Geographic complexity assumed constant across region
4. **Planar network**: 3D elevation effects approximated in β factors

#### Known Limitations

1. **Temporal stability**: Parameters may change with infrastructure development
2. **Service heterogeneity**: Assumes uniform service requirements
3. **Administrative constraints**: Ignores political/administrative boundaries
4. **Scale effects**: Calibrated for municipal-level analysis

### Future Research Directions

1. **Dynamic frameworks**: Incorporating temporal changes in infrastructure
2. **Multi-modal transportation**: Extending to multiple transport modes
3. **Uncertainty quantification**: Bayesian inference for all parameters
4. **Machine learning integration**: Neural network approaches for complex terrains

---

This methodology provides a robust, scientifically validated approach to spatial service allocation optimization with quantified uncertainty bounds and practical implementation guidelines.