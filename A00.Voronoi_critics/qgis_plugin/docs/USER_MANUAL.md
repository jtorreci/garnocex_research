# Voronoi Risk Toolbox — User Manual

Detailed reference for the five algorithms of the Voronoi Risk Toolbox.
For installation see [README.md](../README.md). For the underlying
mathematics see [THEORY.md](THEORY.md).

---

## Conventions used in this manual

- **β** (beta): network scaling factor `d_network / d_euclidean` for a
  given route. β ≥ 1 (network paths cannot be shorter than straight lines).
- **R**: Euclidean distance ratio `d_e(P, A_2) / d_e(P, A_1)` between
  the two nearest facilities. R ≥ 1; R = 1 on the Voronoi boundary.
- **κ** (kappa): local curvature parameter (Lemma 1).
- **t\*** (t-star): half-width of the safety band at risk level q\*.
- **Voronoi-assigned**: nearest facility under Euclidean distance.
- **Network-assigned**: nearest facility under road-network distance.
- **Misallocated**: a municipality whose Voronoi-assigned facility
  differs from its network-assigned facility.
- **pos_id**: positional 0-based index used internally to match QNEAT3
  output rows. Stable within a single algorithm run, **not** stable
  across edits to the input layer.

All distance fields are in the linear units of the input CRS (metres
when using a UTM CRS, which is the recommended setup).

---

## Algorithm 1 — Voronoi Assignment

Computes the Euclidean Voronoi assignment for each spatial unit and the
distance ratio R, used downstream by Theorem 1. **No QNEAT3 needed.**

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| Municipalities | Point or Polygon layer | yes | Spatial units to be assigned. Polygon centroids are used internally. |
| Facilities | Point layer | yes | Facility locations. |
| Regional boundary | Polygon layer | no | Optional, for clipping the Voronoi service-area polygons. |
| Facility identifier field | Field | yes | Field in the facilities layer that uniquely identifies each facility. |
| Audit log | File path | no | If given, writes a CSV audit log. |

### Outputs

#### Per-municipality sink

One feature per input municipality, with all original attributes preserved
plus the following new fields:

| Field | Type | Description |
|---|---|---|
| `voronoi_facility_id` | string | Identifier (from the chosen field) of the assigned facility. |
| `voronoi_pos_id` | int | 0-based positional index of the assigned facility in iteration order. |
| `voronoi_distance_eu` | double | Euclidean distance from the muni centroid to the assigned facility. |
| `voronoi_distance_eu_2nd` | double | Euclidean distance to the **second-nearest** facility. |
| `voronoi_R` | double | R = `voronoi_distance_eu_2nd / voronoi_distance_eu`. R = 1 on the Voronoi boundary, R = ∞ in the interior. |

#### Voronoi service-area polygons (optional)

A polygon layer where each feature is the union of the muni Voronoi
cells assigned to a single facility. Built by:

1. Computing the Voronoi tessellation of municipality centroids.
2. (If a boundary is provided) clipping cells to the boundary.
3. Dissolving cells by `voronoi_facility_id`.

| Field | Type | Description |
|---|---|---|
| `voronoi_facility_id` | string | Identifier of the facility that owns this service area. |
| `n_munis_assigned` | int | Number of municipalities Voronoi-assigned to this facility. |

### Tips

- The R value is the input to Theorem 1's misallocation probability; you
  can hand this output directly to Algorithm 3 as the municipalities
  layer.
- Use a **metric CRS** (UTM ETRS89, etc.) — the algorithm computes
  Euclidean distances directly on the layer coordinates.

---

## Algorithm 2 — Beta Calculator

For each municipality, computes the network scaling factor β to its
Voronoi-assigned and network-nearest facilities, and produces the polygon
outputs needed for visual auditing of misallocations. **Requires QNEAT3.**

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| Municipalities | Point or Polygon layer | yes | Spatial units. |
| Facilities | Point layer | yes | Facility locations. |
| Road network | LineString layer | yes | Connected road graph passed to QNEAT3. |
| Regional boundary | Polygon layer | no | Optional, for clipping polygon outputs. |
| Facility identifier field | Field | yes | Field that uniquely identifies each facility. |
| Default speed | Number, km/h | yes (default 50) | Default speed used by QNEAT3 when the network has no speed field. Affects only routing speed; for shortest-distance strategy this is informational. |
| Audit log | File path | no | Optional CSV log. |

### Outputs

#### Per-municipality sink

| Field | Type | Description |
|---|---|---|
| `voronoi_facility_id` | string | Identifier of the Voronoi-assigned (Euclidean-nearest) facility. |
| `d_euclidean_assigned` | double | Euclidean distance to the Voronoi-assigned facility. |
| `d_network_assigned` | double | Network distance (via QNEAT3) to the Voronoi-assigned facility. |
| `beta_assigned` | double | β for the Voronoi-assigned route: `d_network_assigned / d_euclidean_assigned`. |
| `net_facility_id` | string | Identifier of the network-nearest facility (the optimal assignment under road distance). |
| `d_network_nearest` | double | Network distance to the network-nearest facility. |
| `beta_nearest_net` | double | β for the network-optimal route: `d_network_nearest / d_euclidean(net_facility)`. |
| `is_misallocated` | int (0/1) | 1 iff `voronoi_facility_id ≠ net_facility_id`. |
| `distance_saving_m` | double | Network distance saved if the muni is reassigned from its Voronoi facility to its network-nearest facility. 0 for non-misallocated munis. |

#### Voronoi service-area polygons (optional)

Same as Algorithm 1. Provided here for convenience so that a single run
of Algorithm 2 yields all three polygon layers used in visual auditing.

#### Network service-area polygons (optional)

The **real** service areas under road-network distance, i.e. the union
of muni Voronoi cells assigned to each facility by network distance.

Construction:

1. Voronoi tessellation of municipality centroids (same as Alg 1's polygon output).
2. Optional clip to boundary.
3. Dissolve cells by `net_facility_id`.

| Field | Type | Description |
|---|---|---|
| `net_facility_id` | string | Identifier of the facility that owns this real service area. |
| `n_munis_assigned` | int | Number of municipalities network-assigned to this facility. |

#### Misallocation polygons (optional)

The single most useful output for visual auditing: only the muni Voronoi
cells whose Voronoi assignment differs from their network assignment —
i.e., only the misallocated municipalities, **not dissolved**.

| Field | Type | Description |
|---|---|---|
| `voronoi_facility_id` | string | Voronoi-assigned facility for this muni. |
| `net_facility_id` | string | Network-assigned facility for this muni. |
| `d_euclidean_assigned` | double | Euclidean distance to the Voronoi-assigned facility. |
| `d_network_assigned` | double | Network distance to the Voronoi-assigned facility. |
| `d_network_nearest` | double | Network distance to the network-assigned facility. |
| `beta_assigned` | double | β on the Voronoi-assigned route. |
| `beta_nearest_net` | double | β on the network-optimal route. |
| `distance_saving_m` | double | `d_network_assigned − d_network_nearest`. |
| `distance_penalty_pct` | double | `100 × distance_saving_m / d_network_nearest`. The relative cost of staying with the Euclidean Voronoi assignment. |

### Tips

- **Network not connected?** QNEAT3 will silently mark unreachable
  origin-destination pairs as missing. The audit log reports the count
  of missing routes (`# missing_routes:` in the summary block).
- **Internal IDs.** The plugin builds an in-memory point layer with a
  positional `_pos_id` field (0..n-1) and feeds it to QNEAT3 as
  `FROM_ID_FIELD` / `TO_ID_FIELD`. This is invisible to the user but
  matters if you inspect QNEAT3 traces directly.
- **No consolidation.** This algorithm reports the **raw** misallocation
  count — every facility is treated as distinct. If your real-world
  model groups physically nearby facilities (e.g. plants ≤ 15 km apart
  treated as a single cluster), apply that consolidation as a
  post-processing step over the audit CSV.

---

## Algorithm 3 — Safety Bands

Computes the misallocation probability for each spatial unit (Theorem 1)
and produces the family of safety-band polygons around Voronoi
boundaries (Corollary 1). **No QNEAT3 needed** — works purely on
Euclidean geometry.

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| Municipalities | Point or Polygon layer | yes | Spatial units. |
| Facilities | Point layer | yes | Facility locations. |
| Facility identifier field | Field | yes | Identifier field. |
| Regional boundary | Polygon layer | no | Optional, clips the band polygons cleanly. |
| Dispersion parameter s | double | yes (default 0.093) | Scale of `log(β)`. Calibrated empirically per region; see THEORY for typical ranges. |
| Per-municipality risk tolerance q\* | double | yes (default 0.10) | Probability threshold used to flag `in_safety_band` per muni. |
| Risk levels for band polygons | string (CSV) | optional (default `"0.05,0.10,0.20"`) | Comma-separated list of q\* values, each producing one band polygon per Voronoi ridge. |
| Audit log | File path | no | Optional CSV log. |

### Outputs

#### Per-municipality sink

| Field | Type | Description |
|---|---|---|
| `voronoi_facility_id` | string | Identifier of the Voronoi-assigned facility. |
| `voronoi_R` | double | R = d_e(P, A_2) / d_e(P, A_1). |
| `kappa` | double | Local curvature at the muni centroid: `2 sin(θ/2) / d`, where Q is the projection of P onto the bisector of A_1 A_2, d = `||Q − A_1||`, θ is the angle subtended at Q. |
| `safety_band_width_t_star` | double | Half-width of the safety band at the configured q\*: `t* = (√2 s / κ) Φ⁻¹(1 − q*)`. |
| `dist_to_voronoi_border` | double | Euclidean distance from the muni centroid to the Voronoi mediatrix between its two nearest facilities. |
| `misallocation_prob` | double | Theorem 1 closed-form probability: `Φ(−ln R / (√2 s))`. |
| `in_safety_band` | int (0/1) | 1 iff `dist_to_voronoi_border < safety_band_width_t_star`. |

#### Safety band polygons (optional)

One polygon per Voronoi ridge per risk level. Each polygon is the
**buffer** of the ridge segment by t\*(q\*).

| Field | Type | Description |
|---|---|---|
| `q_star` | double | Risk level for this band (one of the values in the input CSV). |
| `t_star_m` | double | Half-width of the band: `(√2 s / κ) Φ⁻¹(1 − q_star)`. |
| `kappa` | double | Local curvature of the ridge, computed at the midpoint of the segment between the two flanking facilities: `κ = 4 / ||A − A'||`. |
| `facility_a` | string | Identifier of the facility on one side of the ridge. |
| `facility_b` | string | Identifier of the facility on the other side. |
| `ridge_length_m` | double | Length of the underlying Voronoi ridge in linear units. |

### Tips

- **Boundary-edge ridges are skipped.** Voronoi ridges that extend to
  infinity (along the convex hull of the facility set) cannot be
  meaningfully buffered; they are silently omitted.
- **Choose s for your region.** Typical ranges from the literature
  (see THEORY): flat Netherlands ≈ 0.03–0.06; mixed Iberian inland
  ≈ 0.08–0.12; mountainous ≈ 0.13–0.20; coastal/fragmented ≈ 0.15–0.25.
  For an unknown region, run Algorithm 2 first, fit a lognormal to the
  resulting `beta_assigned` and use its shape parameter.
- **Symbology.** The standard recipe: graduated render by `q_star`,
  with the highest-risk (smallest q\*) bands drawn last, opaque or
  near-opaque, and lower-risk bands drawn underneath, semi-transparent.
  The visual is a heatmap of misallocation risk concentrated near the
  Voronoi boundaries.

---

## Algorithm 4 — Misallocation Detector

A thin convenience algorithm: takes the per-muni output of Algorithm 2
and adds (or recomputes) the misallocation flag and savings. Useful when
you want to apply a custom filter to the Algorithm 2 output (for example,
after applying facility consolidation rules) and re-derive the totals.

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| Municipalities (with Beta Calculator output) | Point or Polygon layer | yes | A layer carrying the fields `voronoi_facility_id`, `net_facility_id`, `d_network_assigned`, `d_network_nearest`. Typically the output of Algorithm 2. |
| Audit log | File path | no | Optional CSV log. |

### Outputs

| Field | Type | Description |
|---|---|---|
| `is_misallocated` | int (0/1) | 1 iff the two facility IDs differ. (Added if not already present.) |
| `distance_saving_m` | double | `d_network_assigned − d_network_nearest`. (Added if not already present.) |
| `distance_penalty_pct` | double | `100 × distance_saving_m / d_network_nearest`. |

### Tips

- The algorithm preserves all input attributes.
- It will **not duplicate** `is_misallocated` or `distance_saving_m`
  if those fields already exist; in that case it only appends
  `distance_penalty_pct`.
- A summary line is printed to the Processing log with the totals
  (e.g. *"Misallocation detection complete: 88 of 383 municipalities
  misallocated (23.0%)"*).

---

## Algorithm 5 — Anisotropy Map

Computes the directional anisotropy coefficient α = β_max / β_min for
every origin (typically a municipality), using a configurable destination
layer. **Requires QNEAT3.**

### Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| Municipalities | Point or Polygon layer | yes | Origins. |
| Destinations | Point layer | yes | Destinations against which β is computed. May be the facilities (α relative to plants) or another set of munis (α relative to other munis — useful for detecting topographic asymmetries irrespective of facility placement). |
| Road network | LineString layer | yes | Network for QNEAT3. |
| Destination identifier field | Field | yes | Identifier field for destinations. |
| Default speed | Number, km/h | yes (default 50) | QNEAT3 default speed. |
| Audit log | File path | no | Optional CSV log. |

### Outputs

| Field | Type | Description |
|---|---|---|
| `beta_min` | double | Minimum β across all valid destinations from this origin (β ≥ 1 filter applied). |
| `beta_max` | double | Maximum β across all valid destinations. |
| `beta_mean` | double | Arithmetic mean β across all valid destinations. |
| `beta_min_dest_id` | string | Identifier of the destination achieving β_min. |
| `beta_max_dest_id` | string | Identifier of the destination achieving β_max. |
| `anisotropy_alpha` | double | α = β_max / β_min. α = 1 means perfectly isotropic access; α >> 1 means a strong directional bias in the road network. |
| `anisotropy_class` | string | Categorical class: `"Low"` (α < 1.5), `"Medium"` (1.5 ≤ α < 2.5), `"High"` (α ≥ 2.5), `"N/A"` (less than 2 destinations were reachable). |
| `n_dest_used` | int | Number of destinations actually used for this origin (after dropping unreachable rows and routing artifacts where β < 1). |

### Tips

- α is **not** a binary predictor of misallocation by itself; it
  describes a property of the road network, not the assignment outcome.
  See the paper's Remark on anisotropy for the proper interpretation.
- For large origin sets and dense destination sets (e.g. n = 400 munis
  × 400 munis = 160 000 routes), QNEAT3 may take several minutes and
  significant RAM. Reduce the destination set first if needed.

---

## Reading the audit log

When an algorithm is run with an `Audit log` path, it produces a CSV
with three blocks:

```csv
# voronoi_risk_toolbox audit log
# algorithm: beta_calculator
# started: 2026-05-09T16:00:00+00:00
# n_municipalities: 383
# n_facilities: 46
# default_speed_kmh: 50.0
# facility_id_field: id
# od_rows: 17848
mun_pos_id,mun_qgs_id,voronoi_pos_id,voronoi_facility_id, ...
0,0,17,18, ...
1,1,13,14, ...
...
# --- summary ---
# misallocations: 88
# missing_routes: 0
# rows_written: 383
# finished: 2026-05-09T16:01:30+00:00
```

To load it in pandas:

```python
import pandas as pd
df = pd.read_csv("audit.csv", comment="#")
```

To extract metadata:

```python
import re
meta = {}
with open("audit.csv") as f:
    for line in f:
        if not line.startswith("#"):
            break
        m = re.match(r"#\s*(\w+):\s*(.+)", line)
        if m:
            meta[m.group(1)] = m.group(2).strip()
```

---

## Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `Valor de parámetro incorrecto para FROM_ID_FIELD` | QNEAT3 demands FROM/TO id fields. **Already fixed in v0.2.0** with auto-generated positional ids. | Update to v0.2.0+. |
| Distances in degrees instead of metres | Layer is in EPSG:4326 (lat/lon). | Reproject to a metric CRS (UTM 30N is EPSG:25830 for Iberia). |
| `Misallocations` count differs from a published paper figure | A consolidation rule was applied externally (e.g. plants ≤ 15 km treated as one). | The plugin reports raw counts; apply consolidation in post-processing. |
| Many `null` values in `beta_assigned` | Origin-destination pair not reachable on the road network. | Inspect connectivity; consider adjusting the network layer. |
| Safety band polygons miss the territory edges | Convex-hull ridges are infinite and skipped. | This is by design. The interior of the territory is fully covered. |

---

## Reproducibility

The combination of:

- A canonical input dataset (vector layers + chosen parameters),
- The audit-log CSVs from each algorithm,
- The plugin version recorded in metadata.txt,

is sufficient to reproduce every number in the paper. Pin the versions
when archiving:

```
QGIS:               3.34.x
QNEAT3:             1.0.6+
Voronoi Risk Toolbox: 0.2.0
```
