# GARNOCEX Research Repository

Research outputs and reproducibility packages from the **GARNOCEX project** — a collaborative agreement between the Regional Government of Extremadura (Junta de Extremadura), the College of Civil Engineers (Colegio de Ingenieros Técnicos de Obras Públicas), and the University of Extremadura.

## About GARNOCEX

GARNOCEX is a research project focused on the management and application of recycled aggregates, particularly in non-conventional uses. A key subproject involves the geographic analysis of the distribution and efficiency of the waste treatment plant network in the Extremadura region of Spain.

This repository hosts the code, data, and analysis scripts associated with peer-reviewed publications from the project.

## Repository Structure

Each subdirectory corresponds to a specific research paper or analysis:

| Folder | Title | Status |
|--------|-------|--------|
| [A00.Voronoi_critics](./A00.Voronoi_critics/) | A Probabilistic Framework for Misallocation Risk in Voronoi Tessellations: Theory and Empirical Validation | Under review at *Applied Geography* |
| [A01.Voronoi](./A01.Voronoi/) | Spatial Cost Inequality in Construction and Demolition Waste Management in Sparsely Populated Regions: Evidence from Extremadura, Spain | Submitted to *Waste Management Bulletin* |
| [A03.Diagnostic_Indicators](./A03.Diagnostic_Indicators/) | Diagnostic Indicators for CDW Treatment Plant Networks: A Plant-Level and Network-Level Panel Framework | Submitted to *Waste Management & Research* |
| [A05.Network_Optimization](./A05.Network_Optimization/) | Network Rationalization of Construction and Demolition Waste Treatment Plants: Cost-Based Assignment and the Scale--Circularity Nexus---A Case Study of Extremadura, Spain | Submitted to *Resources, Conservation and Recycling* |

## Quick Navigation

### A00.Voronoi_critics

Reproducibility package for the Voronoi probabilistic framework paper analyzing waste treatment plant service areas:
- **Key finding**: 15.4% misallocation rate in Euclidean Voronoi assignments
- **Case study**: 383 municipalities, 46 waste treatment facilities in Extremadura
- **Framework accuracy**: 97.6% at O(n) complexity

[View full documentation →](./A00.Voronoi_critics/README.md)

### A01.Voronoi

Reproducibility package for the spatial cost analysis of CDW management in Extremadura:
- **Headline cost figures**: €15.82/t (municipality-level mean) and €12.22/t (production-weighted system mean), spanning €5/t to ≈ €50/t across 383 municipalities.
- **Cost model**: piecewise logarithmic treatment cost derived from a doubling rule, calibrated against operational data of four regional CDW companies.
- **Rationalisation potential**: consolidating 14 redundant co-located facilities saves ≈ €303,000/yr (4.5%) without increasing transport distances.

[View full documentation →](./A01.Voronoi/README.md)

### A03.Diagnostic_Indicators

Reproducibility package for the two-panel diagnostic indicator framework comparing three flow scenarios:
- **S1→S3 net saving: 7.2% (0.88 €/t)** purely from treatment-cost reduction via throughput concentration; transport costs actually rise slightly.
- **S1 proximity baseline**: C̄ = 12.22 €/t (transport 6.03 + treatment 6.19), Gini = 0.187, 32 plants.
- **S2 real observed flows**: C̄ = 12.11 €/t (transport 6.30 + treatment 5.81), Gini = 0.208, 36 active plants.
- **S3 cost-optimal network**: C̄ = 11.34 €/t (transport 6.42 + treatment 4.92), Gini = 0.197, 24 plants.
- **Framework**: two 7-indicator panels (plant-level + network-level); legacy indices IET/ICR/IER/ISD shown algebraically collinear.

[View full documentation →](./A03.Diagnostic_Indicators/README.md)

### A05.Network_Optimization

Reproducibility package for the network rationalisation paper. Builds on A00's road-network distances and A01's piecewise logarithmic cost model:
- **Iterative cost-based heuristic**: stabilises in 6 iterations at a 24-plant configuration with 8 plants receiving zero throughput.
- **System cost reduction**: 11.34 €/t (cost-based heuristic) vs 12.22 €/t (A01 diagnostic baseline), a 7.2 % saving (2.8 % relative to A05's own 32-plant distance baseline of 11.67 €/t).
- **Local change**: only 65 of 383 municipalities switch plant; the remaining 318 keep their baseline allocation.
- **Scale--circularity nexus**: 23 of 24 surviving plants exceed the 5,000 t/y viability threshold for mechanical separation (vs 24 of 32 in the baseline).

[View full documentation →](./A05.Network_Optimization/README.md)

## Citation

If you use materials from this repository, please cite the corresponding paper. See individual project folders for specific citation information.

## License

Each subproject may have its own license. See the LICENSE file in each folder for details.

## Contact

- GitHub Issues: [Repository Issues](https://github.com/jtorreci/garnocex_research/issues)

---

*This repository is maintained as part of the GARNOCEX project research outputs.*
