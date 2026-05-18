# Research Repository — Anonymous Peer-Review Snapshot

This branch (`peer-review-anonymous`) is an **anonymized snapshot** prepared
to accompany manuscript **2303622** submitted to *Geographical Analysis*.
Author names, affiliations, and funding details have been removed for
peer review and will appear in the post-acceptance release.

## Repository Structure

Each subdirectory corresponds to a specific research paper or analysis:

| Folder | Title | Status |
|--------|-------|--------|
| [A00.Voronoi_critics](./A00.Voronoi_critics/) | A Probabilistic Framework for Misallocation Risk in Voronoi Tessellations: Theory and Empirical Validation | Under major revision |

## Quick Navigation

### A00.Voronoi_critics

Reproducibility package for the Voronoi misallocation risk framework:

- **Key finding:** 15.9% misallocation rate (61 / 383 municipalities) in
  Euclidean Voronoi assignments vs. network-optimal.
- **Case study:** 383 municipalities, 46 aggregate facilities in
  Extremadura, Spain.
- **Safety bands:** reduce network routing requirements by over 98%.
- **Companion software:** a QGIS Processing Toolbox (*Voronoi Risk
  Toolbox*) packaging the full pipeline (see
  [A00.Voronoi_critics/qgis_plugin/](./A00.Voronoi_critics/qgis_plugin/)).
- **Major revision outputs:** Bayesian log-CAR / BYM2 spatial fits,
  Wasserstein-1 / Anderson-Darling distributional ranking, and the
  reproducibility audit (see
  [A00.Voronoi_critics/Revision/](./A00.Voronoi_critics/Revision/)).

[View full documentation →](./A00.Voronoi_critics/README.md)

## Citation

If you use materials from this repository in academic work, please cite
the corresponding paper. Citation information has been anonymized for
peer review and will be made available on acceptance.

## License

Each subproject may have its own license. See the LICENSE file in each
folder for details.
