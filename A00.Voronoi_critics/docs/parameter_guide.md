# Parameter Calibration Guide

## Overview

This guide provides comprehensive instructions for calibrating the Voronoi Probabilistic Framework parameters for new geographic regions. Proper parameter calibration is essential for accurate misallocation risk assessment and optimal computational resource allocation.

## Framework Parameters

### Core Parameters

#### 1. Geographic Complexity Parameter (s)

**Definition**: Controls the variance of the log-normal distribution of network scaling factors β.

**Physical Interpretation**:
- Represents the "impedance" of the geographic terrain
- Higher values indicate more complex routing (mountains, islands)
- Lower values indicate simpler routing (urban grids, flat plains)

**Typical Ranges**:
- Urban Dense: s = 0.03-0.06
- Flat Plains: s = 0.06-0.09
- Moderate Hills: s = 0.09-0.12
- Mountainous: s = 0.12-0.15
- Complex/Islands: s = 0.15-0.20

**Estimation Methods**: See [Parameter Estimation](#parameter-estimation) section.

#### 2. Log-Normal Distribution Parameters

**Location Parameter (μ)**:
- Controls the central tendency of β factors
- Typically ranges from 0.1 to 0.3
- Related to average network complexity

**Scale Parameter (σ)**:
- Approximately equals the geographic complexity parameter s
- Controls the spread of β values
- Key parameter for risk assessment

#### 3. Risk Threshold (q*)

**Definition**: Acceptable probability of misallocation.

**Common Values**:
- **Conservative**: q* = 10% (high precision applications)
- **Balanced**: q* = 20% (typical planning scenarios)
- **Permissive**: q* = 30% (preliminary analysis)

**Selection Criteria**:
- Service criticality (emergency services: 10%, waste management: 20-30%)
- Computational budget constraints
- Stakeholder risk tolerance

#### 4. Geometric Parameter (κ)

**Definition**: Captures the geometric properties of the Voronoi tessellation.

**Typical Ranges**:
- Dense facility networks: κ = 1.5-2.0
- Medium density: κ = 0.8-1.5
- Sparse networks: κ = 0.2-0.8

**Estimation**: Derived from Voronoi cell analysis or empirical validation.

## Parameter Estimation

### Method 1: Maximum Likelihood Estimation (MLE)

**Best for**: Standard applications with sufficient data (n > 200)

**Implementation**:
```bash
python replication/parameter_estimation.py \
    --input your_data.csv \
    --output mle_parameters.json \
    --method mle
```

**Advantages**:
- Statistically optimal for large samples
- Well-established theoretical properties
- Provides standard errors

**Disadvantages**:
- Sensitive to outliers
- Requires distributional assumptions

### Method 2: Robust Quantile-Based Estimation

**Best for**: Data with outliers or uncertain quality

**Implementation**:
```bash
python replication/parameter_estimation.py \
    --input your_data.csv \
    --output robust_parameters.json \
    --method robust
```

**Advantages**:
- Resistant to outliers
- No distributional assumptions required
- Stable across different data qualities

**Disadvantages**:
- Less efficient for clean data
- Larger confidence intervals

### Method 3: Bayesian Estimation

**Best for**: Small datasets or when incorporating prior knowledge

**Features**:
- Incorporates expert knowledge through priors
- Provides full uncertainty quantification
- Handles small sample sizes better

**Implementation**:
```python
# Custom prior specification
from parameter_estimation import estimate_s_bayesian

# Use informative prior based on similar regions
s_estimate = estimate_s_bayesian(
    beta_values,
    prior_mean=0.093,  # From similar region
    prior_var=0.01     # Confidence in prior
)
```

### Method Comparison and Selection

#### Decision Matrix

| Criterion | MLE | Robust | Bayesian |
|-----------|-----|--------|----------|
| Sample size > 500 | ✅ Best | ⚠️ Good | ⚠️ Good |
| Sample size < 200 | ⚠️ Caution | ✅ Good | ✅ Best |
| Clean data | ✅ Best | ⚠️ Good | ⚠️ Good |
| Outliers present | ❌ Poor | ✅ Best | ⚠️ Good |
| Prior knowledge | ❌ N/A | ❌ N/A | ✅ Best |
| Speed | ✅ Fast | ✅ Fast | ❌ Slow |

#### Automatic Method Selection

```python
def select_estimation_method(data_size, outlier_fraction, prior_available):
    if prior_available and data_size < 200:
        return "Bayesian"
    elif outlier_fraction > 0.1:
        return "Robust"
    elif data_size > 500:
        return "MLE"
    else:
        return "Robust"  # Conservative default
```

## Geographic Context Calibration

### Step 1: Regional Classification

Use the estimated s parameter to classify your region:

```python
def classify_geographic_context(s_value):
    contexts = {
        (0.00, 0.06): "Urban Dense",
        (0.06, 0.09): "Flat Plains",
        (0.09, 0.12): "Moderate Hills",
        (0.12, 0.15): "Mountainous",
        (0.15, 0.25): "Complex/Islands"
    }

    for (low, high), context in contexts.items():
        if low <= s_value < high:
            return context
    return "Unknown"
```

### Step 2: Context-Specific Adjustments

#### Urban Dense (s = 0.03-0.06)

**Characteristics**:
- High road density
- Multiple route alternatives
- Grid-like patterns

**Calibration Considerations**:
- Use smaller κ values (0.8-1.2) due to dense facility networks
- Consider pedestrian vs. vehicle routing
- Account for traffic congestion effects

**Validation Approach**:
- Cross-validate with known optimal assignments
- Test during different time periods
- Validate against multiple routing algorithms

#### Flat Plains (s = 0.06-0.09)

**Characteristics**:
- Agricultural areas
- Straight roads
- Good connectivity

**Calibration Considerations**:
- Standard κ values (1.0-1.5)
- Seasonal accessibility variations
- Rural vs. highway speed differences

**Special Factors**:
- Weather-dependent road conditions
- Agricultural vehicle considerations
- Load restrictions on rural roads

#### Moderate Hills (s = 0.09-0.12)

**Characteristics**:
- Mixed terrain
- Roads following contours
- Moderate elevation changes

**Calibration Considerations**:
- Standard parameters work well
- Example: Extremadura with s = 0.093, κ = 0.5

**Validation**:
- Compare against known optimal solutions
- Test sensitivity to elevation data quality
- Validate across different seasons

#### Mountainous (s = 0.12-0.15)

**Characteristics**:
- Significant elevation changes
- Winding roads
- Limited alternatives

**Calibration Considerations**:
- Higher κ values (1.5-2.0) due to constrained routing
- Elevation-aware distance calculation essential
- Weather and seasonal factors important

**Special Considerations**:
- Road closure probabilities
- Gradient limitations for different vehicles
- Alternative route availability

#### Complex/Islands (s = 0.15-0.20)

**Characteristics**:
- Water barriers
- Ferry connections
- Extremely complex routing

**Calibration Considerations**:
- Highest κ values (2.0+)
- Multi-modal transportation
- Schedule-dependent connectivity

**Advanced Modeling**:
- Time-dependent accessibility
- Ferry capacity constraints
- Weather-dependent operations

## Validation and Quality Control

### Statistical Validation

#### 1. Goodness-of-Fit Tests

```python
# Kolmogorov-Smirnov test
ks_statistic, ks_pvalue = stats.kstest(beta_values, fitted_distribution.cdf)

# Anderson-Darling test for tail behavior
ad_statistic = anderson_darling_test(beta_values, fitted_distribution)

# Visual validation with Q-Q plots
stats.probplot(beta_values, dist=fitted_distribution, plot=plt)
```

**Acceptance Criteria**:
- KS test p-value > 0.05
- Q-Q plot shows good linear relationship
- Tail behavior matches empirical data

#### 2. Cross-Validation

```python
# K-fold cross-validation
cv_scores = []
for train_idx, test_idx in kfold.split(data):
    train_data = data[train_idx]
    test_data = data[test_idx]

    # Estimate parameters on training data
    s_estimate = estimate_parameter(train_data)

    # Validate on test data
    score = validate_prediction(test_data, s_estimate)
    cv_scores.append(score)

print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

**Quality Metrics**:
- Low coefficient of variation (<0.1)
- Consistent estimates across folds
- Stable performance metrics

### Practical Validation

#### 1. Known Misallocation Testing

If ground truth data is available:

```python
def validate_against_known_misallocations(s_estimate, kappa, q_star):
    # Compute predicted misallocations
    predictions = predict_misallocations(s_estimate, kappa, q_star)

    # Compare with known misallocations
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score
```

#### 2. Sensitivity Analysis

Test parameter stability:

```python
# Parameter sensitivity analysis
s_values = np.linspace(s_estimate * 0.8, s_estimate * 1.2, 20)
results = []

for s_test in s_values:
    prediction_quality = evaluate_predictions(s_test)
    results.append(prediction_quality)

# Plot sensitivity curve
plt.plot(s_values, results)
plt.axvline(s_estimate, color='red', label='Estimated value')
```

### Temporal Validation

#### Stability Over Time

```python
# Test parameter stability over different time periods
for period in time_periods:
    period_data = filter_data_by_time(data, period)
    s_period = estimate_parameter(period_data)

    # Check for significant changes
    if abs(s_period - s_baseline) > 2 * se_baseline:
        print(f"Warning: Parameter shift detected in {period}")
```

## Implementation Workflow

### Phase 1: Data Preparation

1. **Collect distance data** following format requirements
2. **Quality control** and outlier detection
3. **Exploratory analysis** of β factor distribution

```bash
# Generate data summary report
python scripts/data_quality_report.py --input your_data.csv
```

### Phase 2: Parameter Estimation

1. **Choose estimation method** based on data characteristics
2. **Estimate parameters** with uncertainty quantification
3. **Validate estimates** using multiple criteria

```bash
# Comprehensive parameter estimation
python replication/parameter_estimation.py \
    --input your_data.csv \
    --output calibration_results.json \
    --cross-validate \
    --plots-dir validation_plots/
```

### Phase 3: Regional Calibration

1. **Classify geographic context** based on s estimate
2. **Generate safety bands** for chosen risk thresholds
3. **Create lookup tables** for practitioners

```bash
# Full regional calibration
python replication/calibration_new_region.py \
    --data your_data.csv \
    --output-dir calibration_results/ \
    --method robust
```

### Phase 4: Validation and Deployment

1. **Validate predictions** against known cases
2. **Sensitivity analysis** for parameter uncertainty
3. **Generate implementation guide** for practitioners

## Advanced Calibration Techniques

### Multi-Level Modeling

For regions with heterogeneous geography:

```python
# Hierarchical modeling by geographic zones
for zone in geographic_zones:
    zone_data = filter_by_zone(data, zone)
    s_zone = estimate_parameter(zone_data)

    # Store zone-specific parameters
    zone_parameters[zone] = {
        's': s_zone,
        'confidence_interval': compute_ci(zone_data),
        'sample_size': len(zone_data)
    }
```

### Spatial Autocorrelation Adjustment

When spatial dependence is significant:

```python
# Spatial adjustment using CAR model
from spatial_models import ConditionalAutoregressive

car_model = ConditionalAutoregressive(spatial_weights)
adjusted_parameters = car_model.fit(beta_values, coordinates)
```

### Machine Learning Enhancement

For complex terrain patterns:

```python
# Random Forest for non-parametric estimation
from sklearn.ensemble import RandomForestRegressor

# Features: elevation, road density, land use
rf_model = RandomForestRegressor()
rf_model.fit(geographic_features, beta_values)

# Predict s parameter for new locations
s_predicted = rf_model.predict(new_geographic_features)
```

## Parameter Update and Maintenance

### Monitoring Schedule

- **Annual review**: Check for infrastructure changes
- **Major events**: Road construction, natural disasters
- **Data updates**: New municipalities or facilities added

### Update Triggers

```python
def check_parameter_drift(new_data, baseline_parameters):
    new_s = estimate_parameter(new_data)
    baseline_s = baseline_parameters['s']

    # Statistical test for parameter shift
    z_score = (new_s - baseline_s) / baseline_parameters['se']

    if abs(z_score) > 2.576:  # 99% confidence
        return "Significant change detected - recalibration recommended"
    elif abs(z_score) > 1.96:  # 95% confidence
        return "Possible change detected - monitor closely"
    else:
        return "Parameters stable"
```

### Version Control

Maintain parameter history:

```json
{
  "region_id": "extremadura_spain",
  "parameter_history": [
    {
      "version": "1.0",
      "date": "2025-01-01",
      "s_parameter": 0.093,
      "confidence_interval": [0.089, 0.097],
      "sample_size": 9240,
      "method": "MLE"
    }
  ]
}
```

## Troubleshooting Common Issues

### Poor Fit Quality

**Symptoms**: Low p-values in goodness-of-fit tests, poor Q-Q plot alignment

**Solutions**:
1. Check for data quality issues
2. Try robust estimation methods
3. Consider mixture distributions for heterogeneous regions
4. Validate distance calculation methodology

### Unstable Parameter Estimates

**Symptoms**: Large confidence intervals, high coefficient of variation

**Solutions**:
1. Increase sample size
2. Improve data quality
3. Use Bayesian methods with informative priors
4. Consider spatial grouping for sparse data

### Validation Failures

**Symptoms**: Poor prediction performance, high misclassification rates

**Solutions**:
1. Recalibrate with local validation data
2. Adjust risk thresholds
3. Consider region-specific factors
4. Update geometric parameter estimates

---

This parameter calibration guide provides comprehensive instructions for adapting the Voronoi Probabilistic Framework to new geographic regions with reliable, validated parameters for accurate misallocation risk assessment.