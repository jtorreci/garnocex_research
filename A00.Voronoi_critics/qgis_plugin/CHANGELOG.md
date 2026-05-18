# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-05-09

### Added

- **Optional structured CSV audit log** in every algorithm. Captures
  metadata (run parameters, layer counts) plus one row per processed
  feature with all input/output values, plus a summary block. Designed
  for direct comparison against ground-truth datasets via
  `pd.read_csv(path, comment="#")`.
- **Voronoi service-area polygons** output in Algorithm 1 and Algorithm 2.
  Built by tessellating the **municipality centroids** (not facilities)
  and dissolving cells by `voronoi_facility_id` — sharing the same
  geometric base as the network output, so the two layers are directly
  superimposable.
- **Network service-area polygons** output in Algorithm 2. Same muni-cell
  base, dissolved by `net_facility_id`.
- **Misallocation polygons** output in Algorithm 2. The set of muni
  Voronoi cells whose Euclidean assignment differs from the network
  assignment, undissolved, with full per-cell attributes (β, distances,
  saving, penalty %).
- **Safety band polygons** output in Algorithm 3. One polygon per
  Voronoi ridge per risk level, computed as the buffer of the ridge
  segment by `t* = (√2 s / κ) Φ⁻¹(1 − q*)`. Multiple risk levels via a
  comma-separated string parameter (default `"0.05,0.10,0.20"`).
- **Optional regional boundary** parameter in Algorithms 1, 2 and 3.
  When provided, polygon outputs are clipped to the boundary cleanly.
- **Module `_audit.py`** with shared utilities: `AuditLogger`,
  `parse_od_layer` with field-name auto-detection, `collect_layer_points`
  returning positional ids that match QNEAT3 output, and
  `make_layer_with_pos_id` to feed QNEAT3 the required FROM/TO id field.
- **Documentation:** `README.md` rewritten as a GitHub-ready landing
  page; new `docs/USER_MANUAL.md` with full parameter and field
  reference; new `docs/THEORY.md` with the underlying mathematics.

### Fixed

- **Critical:** QNEAT3 OD matrix call failed with
  *"Valor de parámetro incorrecto para FROM_ID_FIELD"* in v0.1.0.
  Fixed by materialising in-memory point layers with a `_pos_id` integer
  field (0..n-1, in iteration order) and passing it to QNEAT3 as
  `FROM_ID_FIELD` and `TO_ID_FIELD`. The OD output now reliably uses
  positional ids that match the plugin's iteration loop.
- **Critical:** Algorithm 3 (Safety Bands) computed κ at the midpoint
  of segment A_1 A_2 in v0.1.0. Lemma 1 actually requires κ at the
  projection Q of P onto the perpendicular bisector. The new
  implementation computes Q properly and yields correct κ for any P.
- **Critical:** in v0.1.0 the OD lookup in Algorithm 2 and Algorithm 5
  used QGIS feature IDs (`feat.id()`), which are unstable across
  datasets, instead of the positional ids that QNEAT3 emits. Fixed by
  consistent use of `enumerate()` indices and the new `_pos_id` field.
- QNEAT3 output field names auto-detected (legacy variants
  `origin_point_id`, `network_cost`, etc., are handled transparently).
- `Misallocation Detector` no longer duplicates the `is_misallocated`
  and `distance_saving_m` fields when the input already carries them.

### Changed

- All polygon outputs in Algorithm 2 are derived from a single
  municipality-cell tessellation, computed once and reused. This
  guarantees that `Voronoi polygons` and `Network service-area polygons`
  share the same cell grid, making their differences correspond
  exactly to misallocations.

---

## [0.1.0] — 2026-04

### Added

- Initial release with five Processing algorithms:
  Voronoi Assignment, Beta Calculator, Safety Bands,
  Misallocation Detector, Anisotropy Map.
- QNEAT3 integration for network distance computation.
- Per-feature output sinks for each algorithm.
