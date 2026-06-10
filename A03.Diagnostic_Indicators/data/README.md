# Data — A03 Diagnostic Indicators

This directory contains the data file required to reproduce Scenario 2 results.

## Files

### `s2_observed_flows_public.csv`

Anonymised Scenario 2 (S2) input flows. Columns:

| Column     | Description                                                |
|------------|------------------------------------------------------------|
| `mun_norm` | Normalised municipality name (ASCII, lower-case)           |
| `plant_id` | Treatment plant identifier (integer; negative = plants not in A01) |
| `dist_km`  | Road-network distance municipality → plant (km)            |
| `prod_t`   | Assigned production volume (t/yr)                          |

**730 rows** covering **383 municipalities** and **36 active plants**.

Volumes are rescaled to the theoretical production total of **551,205 t/yr**
(the same denominator used in S1). Raw commercial tonnages from the regional
operational monitoring programme are withheld; see the Data Availability
section in the manuscript and in `../README.md`.

## Scenario data sources

| Scenario | Description | Source |
|----------|-------------|--------|
| S1 — Proximity baseline | Road-distance assignment to 32 plant groups | `../outputs/muni_costs_s1.csv` (derived from A01 public data) |
| S2 — Observed flows | Anonymised regional operational data | `s2_observed_flows_public.csv` (this directory) |
| S3 — Cost-optimal | Iterative cost-based heuristic, 24 plants | `../../A05.Network_Optimization/outputs/optimal_assignments.csv` |

S1 and S3 are **fully reproducible** from the public A01 and A05 data in this
repository. S2 is released as anonymised flows with volumes rescaled to
theoretical production; the underlying commercial origin-destination data are
confidential.
