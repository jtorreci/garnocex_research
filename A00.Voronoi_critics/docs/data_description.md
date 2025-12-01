# Data Description and Format Requirements

## Overview

This document describes the data formats and requirements for applying the Voronoi Probabilistic Framework to new geographic regions. The framework requires distance ratio data between municipalities and service facilities to calibrate the geographic complexity parameter and generate safety bands.

## Input Data Formats

### Primary Dataset Format

The main dataset should be a CSV file with the following structure:

#### Required Columns

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `Municipality_ID` | string | Unique identifier for each municipality | "MUN_001", "Barcelona" |
| `Facility_ID` | string | Unique identifier for each facility | "FAC_01", "Plant_A" |
| `Euclidean_Distance_km` | float | Straight-line distance in kilometers | 25.7 |
| `Network_Distance_km` | float | Actual travel distance via road network | 32.1 |

#### Optional Columns

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `Municipality_X` | float | Municipality longitude/x-coordinate | -3.7038 |
| `Municipality_Y` | float | Municipality latitude/y-coordinate | 40.4168 |
| `Facility_X` | float | Facility longitude/x-coordinate | -3.6950 |
| `Facility_Y` | float | Facility latitude/y-coordinate | 40.4200 |
| `Beta_Factor` | float | Pre-computed distance ratio (if available) | 1.249 |
| `Is_Misallocated` | boolean | Known misallocation status (if available) | true/false |
| `Voronoi_Assignment` | string | Current Voronoi-based assignment | "FAC_01" |
| `Optimal_Assignment` | string | Known optimal assignment (if available) | "FAC_03" |

### Example CSV Structure

```csv
Municipality_ID,Facility_ID,Euclidean_Distance_km,Network_Distance_km,Municipality_X,Municipality_Y,Facility_X,Facility_Y
MUN_001,FAC_01,15.2,18.7,-3.7038,40.4168,-3.6950,40.4200
MUN_001,FAC_02,22.8,31.4,-3.7038,40.4168,-3.8200,40.3800
MUN_002,FAC_01,8.9,9.8,-3.6800,40.4300,-3.6950,40.4200
MUN_002,FAC_02,18.5,26.2,-3.6800,40.4300,-3.8200,40.3800
```

### Geographic Coordinates File (Optional)

For spatial autocorrelation analysis, provide a separate CSV with municipality coordinates:

```csv
Municipality_ID,Longitude,Latitude
MUN_001,-3.7038,40.4168
MUN_002,-3.6800,40.4300
MUN_003,-3.7500,40.3900
```

## Data Quality Requirements

### Minimum Sample Size

- **Minimum**: 100 municipality-facility pairs
- **Recommended**: 500+ pairs for robust parameter estimation
- **Optimal**: 1000+ pairs for comprehensive validation

### Distance Ratio Constraints

The framework automatically filters invalid ratios, but input data should ideally satisfy:

- `Beta_Factor = Network_Distance / Euclidean_Distance`
- Valid range: 0.8 ≤ β ≤ 5.0
- Typical range: 1.0 ≤ β ≤ 2.5

### Data Quality Checks

The framework performs automatic quality checks:

1. **Missing values**: Removes rows with missing distance data
2. **Invalid ratios**: Filters β < 0.5 or β > 10.0
3. **Duplicate pairs**: Warns about duplicate municipality-facility combinations
4. **Outliers**: Identifies potential data entry errors

## Distance Measurement Guidelines

### Euclidean Distance

- **Definition**: Straight-line distance using spherical geometry
- **Units**: Kilometers (km)
- **Precision**: 3 decimal places recommended
- **Calculation**:
  ```python
  from geopy.distance import great_circle
  euclidean_km = great_circle((lat1, lon1), (lat2, lon2)).kilometers
  ```

### Network Distance

- **Definition**: Shortest path via road network
- **Units**: Kilometers (km), matching Euclidean units
- **Sources**: OpenStreetMap, national road databases, GPS routing
- **Tools**:
  - QGIS with QNEAT3 plugin
  - OpenRouteService API
  - Google Maps Distance Matrix API
  - OSMnx (Python library)

### Common Distance Calculation Issues

1. **Unit mismatches**: Ensure both distances use same units
2. **Projection errors**: Use appropriate coordinate systems
3. **Network completeness**: Ensure road network includes all relevant routes
4. **Temporal consistency**: Use consistent time period for distance measurements
5. **Access restrictions**: Consider vehicle type and access limitations

## Geographic Context Examples

### Urban Dense (s ≈ 0.05)

**Characteristics**:
- High road density
- Grid-like street patterns
- Minimal topographic barriers
- Short alternative routes

**Example regions**: Manhattan, central Barcelona, downtown areas

**Data requirements**:
- Fine-grained road network
- Multiple route options
- Pedestrian vs. vehicle routing considerations

### Flat Plains (s ≈ 0.08)

**Characteristics**:
- Agricultural areas
- Straight roads following property boundaries
- Minimal elevation changes
- Good connectivity between settlements

**Example regions**: Netherlands polders, American Midwest, Pampas

**Data considerations**:
- Seasonal road accessibility
- Agricultural vehicle routing
- Rural vs. highway speed differences

### Moderate Hills (s ≈ 0.093)

**Characteristics**:
- Mixed terrain with moderate elevation changes
- Roads following natural contours
- Some route constraints due to topography
- Balance between direct and circuitous routes

**Example regions**: Extremadura (Spain), Piedmont regions, rolling hills

**Data requirements**:
- Elevation-aware routing
- Consideration of road gradients
- Multiple route alternatives

### Mountainous (s ≈ 0.12)

**Characteristics**:
- Significant elevation changes
- Winding roads following valleys
- Limited route alternatives
- Bridge and tunnel infrastructure

**Example regions**: Alps, Andes foothills, Appalachian mountains

**Special considerations**:
- Weather-dependent accessibility
- Seasonal road closures
- Elevation-based routing constraints

### Complex/Islands (s ≈ 0.15-0.20)

**Characteristics**:
- Water barriers requiring ferries/bridges
- Extremely complex topography
- Limited connectivity between areas
- Multi-modal transportation

**Example regions**: Greek islands, Norwegian fjords, complex archipelagos

**Data challenges**:
- Ferry schedules and routes
- Multi-modal journey planning
- Seasonal accessibility variations

## Data Preparation Workflow

### Step 1: Data Collection

1. **Identify municipalities and facilities** in your region
2. **Obtain coordinates** for all locations
3. **Calculate Euclidean distances** using spherical geometry
4. **Compute network distances** using routing algorithms
5. **Validate distance pairs** for consistency

### Step 2: Quality Control

```python
# Example quality control script
import pandas as pd
import numpy as np

def validate_distance_data(df):
    # Check for missing values
    missing_mask = df[['Euclidean_Distance_km', 'Network_Distance_km']].isnull().any(axis=1)
    print(f"Rows with missing distances: {missing_mask.sum()}")

    # Calculate beta factors
    df['Beta_Factor'] = df['Network_Distance_km'] / df['Euclidean_Distance_km']

    # Identify outliers
    valid_mask = (df['Beta_Factor'] >= 0.8) & (df['Beta_Factor'] <= 5.0)
    print(f"Valid ratios: {valid_mask.sum()}/{len(df)} ({valid_mask.mean()*100:.1f}%)")

    # Statistical summary
    print(f"Beta factor statistics:")
    print(df['Beta_Factor'].describe())

    return df[valid_mask]
```

### Step 3: Format Standardization

```python
# Standardize column names and formats
def standardize_format(df):
    # Ensure required columns exist
    required_cols = ['Municipality_ID', 'Facility_ID',
                    'Euclidean_Distance_km', 'Network_Distance_km']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert to standard types
    df['Municipality_ID'] = df['Municipality_ID'].astype(str)
    df['Facility_ID'] = df['Facility_ID'].astype(str)
    df['Euclidean_Distance_km'] = pd.to_numeric(df['Euclidean_Distance_km'])
    df['Network_Distance_km'] = pd.to_numeric(df['Network_Distance_km'])

    return df
```

## Synthetic Data Generation

For testing and validation purposes, the framework includes synthetic data generation:

```bash
# Generate synthetic data matching Extremadura parameters
python data/synthetic_data_generator.py --output-dir test_data/ --seed 42

# Generate data for different geographic contexts
python data/synthetic_data_generator.py --output-dir urban_test/ --s-param 0.05
python data/synthetic_data_generator.py --output-dir mountain_test/ --s-param 0.15
```

## Privacy and Anonymization

When sharing data for research or validation:

1. **Remove identifying information**: Use generic IDs instead of place names
2. **Aggregate sensitive locations**: Group small municipalities if needed
3. **Normalize coordinates**: Use relative positions instead of absolute coordinates
4. **Document anonymization**: Provide metadata about transformations applied

### Example Anonymization

```python
def anonymize_dataset(df):
    # Replace actual names with generic IDs
    unique_municipalities = df['Municipality_ID'].unique()
    mun_mapping = {old: f"MUN_{i:03d}" for i, old in enumerate(unique_municipalities)}

    unique_facilities = df['Facility_ID'].unique()
    fac_mapping = {old: f"FAC_{i:02d}" for i, old in enumerate(unique_facilities)}

    # Apply mappings
    df['Municipality_ID'] = df['Municipality_ID'].map(mun_mapping)
    df['Facility_ID'] = df['Facility_ID'].map(fac_mapping)

    # Normalize coordinates to [0,1] range if present
    if 'Municipality_X' in df.columns:
        x_min, x_max = df['Municipality_X'].min(), df['Municipality_X'].max()
        y_min, y_max = df['Municipality_Y'].min(), df['Municipality_Y'].max()

        df['Municipality_X'] = (df['Municipality_X'] - x_min) / (x_max - x_min)
        df['Municipality_Y'] = (df['Municipality_Y'] - y_min) / (y_max - y_min)

    return df
```

## Troubleshooting Common Issues

### Distance Calculation Problems

- **Negative distances**: Check coordinate order (lat, lon vs. lon, lat)
- **Unrealistic ratios**: Verify units match between Euclidean and network distances
- **Missing routes**: Ensure road network coverage is complete
- **Projection issues**: Use appropriate coordinate reference system

### Data Format Issues

- **Encoding problems**: Use UTF-8 encoding for international characters
- **Decimal separators**: Ensure consistent use of decimal points vs. commas
- **Missing headers**: Verify CSV header row is present and correct
- **Quote characters**: Handle special characters in municipality names

### Performance Considerations

- **Large datasets**: Consider sampling for initial parameter estimation
- **Memory usage**: Process data in chunks for very large regions
- **Computation time**: Parallel processing for distance calculations
- **Storage format**: Consider Parquet format for large datasets

---

This data description provides comprehensive guidance for preparing and formatting data for the Voronoi Probabilistic Framework. Following these guidelines ensures reliable parameter estimation and accurate risk assessment for your specific geographic region.