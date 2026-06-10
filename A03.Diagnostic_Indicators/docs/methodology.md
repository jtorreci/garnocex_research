# Methodology — A03 Diagnostic Indicators

This document describes the analytical pipeline implemented in `analysis/`
and `replication/`. For full derivations and discussion see the manuscript.

---

## 1. Cost Model

Total unit cost for municipality _m_ assigned to plant _i_:

```
C(m, i) = ρ · d(m, i) + C_treat(T_i)
```

where `d(m, i)` is the road-network distance (km) and `C_treat(T_i)` is
the unit treatment cost at plant throughput `T_i`.

**Piecewise-log fixed-cost model** (from companion paper A01):

```
C_fix(T) = C0                          if T < T0
C_fix(T) = C0 · (log₂(T / T0) + 1)    if T ≥ T0

C_treat(T) = C_fix(T) / T + v
```

Parameters: `C0 = 40,000 €/yr`, `T0 = 5,000 t/yr`, `ρ = 0.35 €/(t·km)`,
`v = 0.35 €/t`. Calibrated in A01 from regional cost data.

---

## 2. Three-Scenario Framework

The framework compares three configurations of the same physical network:

### S1 — Proximity Baseline
Each municipality is assigned to its nearest plant by road-network distance.
Provides a theoretical reference representing pure geographic accessibility.
Data: companion paper A01 (383 municipalities, 32 plant groups).

### S2 — Observed Operational Flows
Real multi-destination flows from the regional operational monitoring
programme. Municipalities may send waste to multiple plants. Covers 23 of
46 physical plants; residual production assigned to S1 baseline plant via
conservative full-network closure.
Public input: `data/s2_observed_flows_public.csv` (volumes rescaled to
theoretical production; commercial tonnages withheld).

### S3 — Cost-Optimal Network
Iterative cost-based heuristic from companion paper A05. Each municipality
assigned to the plant minimising total unit cost given current throughput;
converges in 6 iterations. 24 plants remain active.
Data: companion paper A05 (`optimal_assignments.csv`).

---

## 3. Indicator Computation

### Plant-Level Panel

**C_i** (unit total cost): production-weighted average over municipalities
assigned to plant _i_.

**D_i^(90)** (90th-percentile accessibility): production-weighted 90th
percentile of road distances from municipalities to their assigned plant.

**Prec_i, Rec_i** (Voronoi precision and recall): treat the nearest-plant
(Voronoi) assignment as the "baseline prediction" and the actual assignment
as the "true label." Precision = fraction of actual intake that is
Voronoi-optimal; Recall = fraction of Voronoi cell actually captured.

**Leak_dg_i** (cost-penalised leakage): for each municipality in plant _i_'s
Voronoi cell that is routed elsewhere, the excess cost per tonne relative to
routing to plant _i_; averaged over total Voronoi-cell production.

**IET_i** (transport efficiency): ratio of theoretical t-km (Voronoi
assignment) to actual t-km, expressed as a percentage.

### Network-Level Panel

**C̄** (production-weighted mean unit cost):

```
C̄ = Σ_m  prod_m · C_m  /  Σ_m  prod_m
```

**Gini(C_m)**: standard Gini coefficient applied to the unweighted
distribution of municipal unit costs (one value per municipality).

**MAD_C**: production-weighted mean absolute deviation around C̄.

**CVaR₀.₉(D)**: expected road distance in the worst 10% of the
production-weighted distance distribution.

**micro-F₁**: 2TP / (2TP + FP + FN) aggregated across all plants; measures
system-wide alignment between actual and Voronoi-optimal assignment.

**IGET** (global transport efficiency): ratio of total system theoretical
t-km to total actual t-km.

**CV(T_i)**: coefficient of variation (std/mean) of plant throughputs.

---

## 4. Legacy Index Collinearity

The paper demonstrates that the four legacy indices IET, ICR, IER, ISD are
algebraically collinear with IGET:

```
ICR = 10,000 / IGET
IER = ICR
ISD = ICR - 100
```

All four are monotone transforms of IGET (or its reciprocal) and therefore
carry no independent information. They are retained in `compute_indicators.py`
for backward compatibility only.

---

## 5. Reproduction

To reproduce the headline numbers from public data:

```bash
python analysis/reproduce_scenarios.py
```

To recompute the full indicator panels:

```bash
python analysis/compute_three_scenarios.py   # requires confidential S2 Excels
python analysis/compute_indicators.py        # plant-level panel for S1
```

To reproduce the manuscript figures:

```bash
python replication/generate_three_panel_map.py
python replication/generate_plant_indicator_distributions.py
python replication/generate_plant_scenario_dynamics.py
python replication/plot_cost_component_kdes.py
```

Note: map figures require the Extremadura shapefiles distributed with A01 (see
`../../A01.Voronoi/` — shapefiles are not re-distributed in this package).

---

## 6. Dependencies

See `requirements.txt` (pip) or `environment.yml` (conda) in the package root.
Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `geopandas`.
