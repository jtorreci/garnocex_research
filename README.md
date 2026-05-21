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
| [A01.Voronoi](./A01.Voronoi/) | Spatial Cost Analysis of Construction and Demolition Waste Management: Beyond Geometric Proximity in Sparsely Populated Regions (Extremadura, Spain) | Submitted to *Waste Management* |

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

## Citation

If you use materials from this repository, please cite the corresponding paper. See individual project folders for specific citation information.

## License

Each subproject may have its own license. See the LICENSE file in each folder for details.

## Contact

- GitHub Issues: [Repository Issues](https://github.com/jtorreci/garnocex_research/issues)

---

*This repository is maintained as part of the GARNOCEX project research outputs.*
