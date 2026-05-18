# Voronoi Risk Toolbox

[![QGIS](https://img.shields.io/badge/QGIS-3.22+-green.svg)](https://qgis.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](voronoi_risk_toolbox/metadata.txt)

A QGIS Processing Toolbox that quantifies misallocation risk in
Voronoi-based service-area assignments under network distance metrics.

It implements the probabilistic framework described in:

> TBD (2026). *A Probabilistic Framework
> for Misallocation Risk in Voronoi Tessellations: Theory and Empirical
> Validation*. **Geographical Analysis** (under review, manuscript 2303622).

The toolbox lets a practitioner answer questions like:

- For a given network of facilities (e.g. waste plants, hospitals,
  emergency stations), **which municipalities are at risk of being
  misassigned** if we plan with a Euclidean Voronoi tessellation instead
  of a real road-network analysis?
- What is the **probability of misallocation** for each spatial unit?
- Which territories sit in the **safety band** around a Voronoi boundary,
  where the Euclidean assignment is unreliable?
- How **anisotropic** is the road network around each municipality
  (the directional spread of detour coefficients)?

---

## What it does

Five processing algorithms grouped under "Voronoi Risk Analysis":

| # | Algorithm | What it computes | Needs QNEAT3 |
|---|---|---|---|
| 1 | **Voronoi Assignment** | Euclidean nearest-facility assignment, distance ratio *R* | no |
| 2 | **Beta Calculator** | Network scaling factor β = d_network / d_euclidean per route, raw misallocations | **yes** |
| 3 | **Safety Bands** | Misallocation probability per municipality (Theorem 1), safety-band polygons at configurable risk levels | no (uses Voronoi only) |
| 4 | **Misallocation Detector** | Per-feature misallocation flags, distance savings | no (post-processes Alg 2) |
| 5 | **Anisotropy Map** | Anisotropy coefficient α = β_max / β_min per origin | **yes** |

Each algorithm produces:

- A **vector layer** with computed attributes joined to the input geometries.
- An optional **structured CSV audit log** (one row per feature, with
  meta-headers, for reproducibility and downstream auditing).
- For algorithms 1, 2 and 3: optional **polygon outputs** for direct
  visualization (service-area cells, misallocation islands, risk bands).

All polygon outputs share a common municipality-cell tessellation, so
they are **directly superimposable** in QGIS.

---

## Quick start

### Requirements

- QGIS 3.22 LTR or newer.
- [QNEAT3](https://plugins.qgis.org/plugins/QNEAT3/) plugin (required for
  algorithms 2 and 5).
- Python (bundled with QGIS): numpy, scipy.

### Install

1. Download or clone this repository.
2. Copy the `voronoi_risk_toolbox/` directory into your QGIS plugin folder:

   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

3. Restart QGIS.
4. *Plugins → Manage and Install Plugins → Installed →* check
   **Voronoi Risk Toolbox**.
5. The five algorithms appear in *Processing Toolbox → Voronoi Risk Analysis*.

### Test it

The repository ships with reduced test data. See
[INSTALL_AND_TEST.md](INSTALL_AND_TEST.md) for a step-by-step validation
walkthrough using the canonical Extremadura dataset.

---

## Input data

| Layer | Geometry | Required by | Notes |
|---|---|---|---|
| **Municipalities** (or any spatial units) | Point or Polygon | all | Polygon centroids are used internally |
| **Facilities** | Point | all | One feature per facility, ID field required |
| **Road network** | LineString | Alg 2, 5 | Topologically valid graph for QNEAT3 |
| **Regional boundary** | Polygon | optional | Used only to clip polygon outputs cleanly |

CRS expectations: any **metric CRS** (UTM, etc.). Geographic CRS (lat/lon)
will produce distances in degrees and is not recommended.

---

## Documentation

- **[USER_MANUAL.md](docs/USER_MANUAL.md)** — Full manual with parameter
  descriptions and field-by-field documentation of every output for
  every algorithm.
- **[INSTALL_AND_TEST.md](INSTALL_AND_TEST.md)** — Installation guide
  and validation plan using the Extremadura case study.
- **[THEORY.md](docs/THEORY.md)** — A short summary of the underlying
  probabilistic framework (Theorem 1, Lemma 1, Corollary 1).
- **[CHANGELOG.md](CHANGELOG.md)** — Version history.

---

## Visualization recipe

Once you run **Algorithm 2** with all polygon outputs enabled, you have:

1. **Voronoi polygons** — Euclidean assignment per facility.
2. **Network service-area polygons** — real assignment per facility.
3. **Misallocation polygons** — only the cells where the two disagree.

To produce the standard "misallocation map":

```
Layer order (top to bottom):
  Misallocation polygons        (categorical by net_facility_id, opaque)
  Network service-area polygons (categorical by net_facility_id, alpha 60%)
  Voronoi polygons              (categorical by voronoi_facility_id, alpha 30%)
  Regional boundary             (outline only)
```

Use the **same colour palette** for the three first layers, indexed on
the facility ID. The misallocations appear as colour islands inside
discordant Voronoi cells — that is exactly the framework's story
told visually.

For the **safety-band overlay** (Algorithm 3), add the band polygons
on top, semi-transparent, graduated by `q_star` (e.g. red 70% / orange 50%
/ light orange 30%).

---

## Audit logs

Every algorithm accepts an optional `Audit log (CSV)` parameter. When set,
the algorithm writes a structured CSV alongside the output sink, with:

- Comment-header lines (`# algorithm:`, `# n_facilities:`, ...) carrying
  the run metadata.
- One row per processed feature, with all input/output values.
- A `# --- summary ---` block at the end with run statistics.

These logs are designed to be diffed against a canonical ground-truth
dataset, e.g. by `pandas.read_csv(path, comment="#")`.

---

## Citation

If you use this toolbox in academic work, please cite both the paper
and the software. Author fields are marked TBD pending publication of
the companion paper:

```bibtex
@article{anon2026voronoi,
  author  = {{TBD}},
  title   = {A Probabilistic Framework for Misallocation Risk in Voronoi
             Tessellations: Theory and Empirical Validation},
  journal = {Geographical Analysis},
  year    = {2026},
  note    = {Under review, manuscript 2303622}
}

@software{voronoi_risk_toolbox,
  author  = {{TBD}},
  title   = {Voronoi Risk Toolbox - A QGIS Processing plugin for
             misallocation risk assessment},
  year    = {2026},
  url     = {https://github.com/jtorreci/garnocex_research},
  version = {0.2.0}
}
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

## Acknowledgements

Funding and institutional acknowledgements: TBD (to appear in the
post-acceptance release of the companion paper).

Network distance computation relies on the
[QNEAT3](https://plugins.qgis.org/plugins/QNEAT3/) plugin by Clemens Raffler.
