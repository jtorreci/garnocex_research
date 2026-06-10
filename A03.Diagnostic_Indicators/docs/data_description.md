# Data Description — A03 Diagnostic Indicators

## Overview

The analysis covers the **Construction and Demolition Waste (CDW) treatment
plant network of Extremadura, Spain**: 383 municipalities, up to 46 physical
treatment plants consolidated into 32 effective plant groups.

Three scenarios are compared using the same cost model and indicator framework.

---

## Scenario 1 — Proximity Baseline (S1)

**Source**: A01 public data, derived from the companion paper  
"Spatial Cost Inequality in CDW Management in Sparsely Populated Regions"

- **Municipalities**: 383 (full Extremadura network)
- **Plant groups**: 32 (14 co-located plants consolidated into 8 group leaders)
- **Assignment rule**: each municipality assigned to the nearest active plant
  by road-network distance
- **Production total**: 551,205 t/yr (at 0.5 t/inhab/yr)
- **Pre-computed file in this package**: `outputs/muni_costs_s1.csv`
  (columns: mun, transport, total, prod, treatment)

## Scenario 2 — Observed Operational Flows (S2)

**Source**: regional operational monitoring programme (2019–2022);
anonymised public version in `data/s2_observed_flows_public.csv`

- **Municipalities**: 383 (after conservative full-network closure)
- **Active plants**: 36
- **Assignment rule**: real multi-destination flows from 23 audited plants;
  residual production for unaudited municipalities assigned to S1 proximity
  baseline plant (conservative closure)
- **Production total**: 551,205 t/yr (volumes rescaled to theoretical
  production; raw commercial tonnages withheld)
- **Confidential input**: 23 `PT_*.xlsx` operational data files — NOT
  distributed; see `analysis/compute_three_scenarios.py` for the full
  pipeline (included for transparency)

## Scenario 3 — Cost-Optimal Network (S3)

**Source**: A05 public data, derived from the companion paper  
"Network Rationalization of CDW Treatment Plants"

- **Municipalities**: 382 (one municipality unreachable in iterative heuristic)
- **Active plants**: 24 (out of 32 plant groups)
- **Assignment rule**: iterative cost-based heuristic; each municipality
  assigned to the plant that minimises total unit cost (transport + treatment)
  given current throughput configuration; converges in 6 iterations
- **Production total**: 551,331 t/yr
- **Source file**: `../../A05.Network_Optimization/outputs/optimal_assignments.csv`

---

## Cost Model Parameters (shared across all scenarios)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Minimum fixed cost | C₀ | 40,000 €/yr | Annual fixed cost at minimum scale |
| Economy-of-scale threshold | T₀ | 5,000 t/yr | Reference throughput |
| Transport tariff | ρ | 0.35 €/(t·km) | Effective per-tonne-km rate |
| Variable treatment cost | v | 0.35 €/t | Variable component of treatment |
| CDW generation rate | ρ_CDW | 0.50 t/inhab·yr | Used to derive production |
| Coordinate system | — | EPSG:25830 | ETRS89 / UTM zone 30N |

---

## Indicator Framework

The paper introduces a **two-panel framework** of 13 indicators:

### Plant-Level Panel (7 indicators)

| Indicator | Symbol | Description |
|-----------|--------|-------------|
| Unit total cost | C_i | EUR/t, decomposed into transport + treatment |
| Capacity utilisation | U_i | N/A (no capacity data available) |
| 90th-pct accessibility | D_i^(90) | Production-weighted 90th-pct distance (km) |
| Voronoi precision | Prec_i | Share of actual intake that is Voronoi-optimal |
| Voronoi recall | Rec_i | Share of Voronoi cell actually captured |
| Cost-penalised leakage | Leak_dg_i | EUR/t penalty for sub-optimal routing |
| Transport efficiency | IET_i | % ratio theoretical/actual t-km |

### Network-Level Panel (7 indicators)

| Indicator | Symbol | Description |
|-----------|--------|-------------|
| Mean unit cost | C̄ | Production-weighted average (EUR/t) |
| Cost inequality | Gini(C_m) | Gini coefficient of municipal unit costs |
| Cost dispersion | MAD_C | Mean absolute deviation (EUR/t) |
| Tail accessibility | CVaR₀.₉(D) | Expected distance in worst 10% tail (km) |
| System alignment | micro-F₁ | Production-weighted Voronoi compliance |
| Global transport efficiency | IGET | % ratio system-level theoretical/actual t-km |
| Throughput concentration | CV(T_i) | Coefficient of variation of plant throughputs |

### Legacy Index Collinearity

The paper demonstrates algebraically that the legacy indices IET, ICR, IER,
and ISD are mutually collinear (all derivable from IGET by linear or
hyperbolic transforms) and therefore add no independent information. They are
retained in the full pipeline for backward compatibility only.
