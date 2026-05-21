# -*- coding: utf-8 -*-
"""
Algorithm 2: Beta Calculator

Computes the network scaling factor beta = d_network / d_euclidean
for each municipality, against (a) its Voronoi-Euclidean-assigned facility
and (b) its network-nearest facility. Uses QNEAT3 for shortest-path.

Fixes since v0.1.0 audit:
    B2.1  use positional ids (not feat.id()) when looking up QNEAT3 OD
    B2.2  field names auto-detected (origin_id/total_cost, plus legacy variants)
    B2.4  optional structured CSV audit log

Inputs:
    - Municipalities layer (point or polygon)
    - Facilities layer (point)
    - Network layer (line)
    - Facility ID field
    - Audit log path (optional CSV)

Outputs (sink + optional CSV):
    voronoi_facility_id, d_euclidean_assigned, d_network_assigned,
    beta_assigned, net_facility_id, d_network_nearest, beta_nearest_net,
    is_misallocated, distance_saving_m
"""

from __future__ import annotations

import numpy as np

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessing,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant

from ._audit import (
    AuditLogger, parse_od_layer, collect_layer_points, make_layer_with_pos_id,
)


class BetaCalculatorAlgorithm(QgsProcessingAlgorithm):

    MUNICIPALITIES = "MUNICIPALITIES"
    FACILITIES = "FACILITIES"
    NETWORK = "NETWORK"
    BOUNDARY = "BOUNDARY"
    FACILITY_ID_FIELD = "FACILITY_ID_FIELD"
    DEFAULT_SPEED = "DEFAULT_SPEED"
    AUDIT_LOG = "AUDIT_LOG"
    OUTPUT = "OUTPUT"
    OUTPUT_VORONOI_POLYGONS = "OUTPUT_VORONOI_POLYGONS"
    OUTPUT_NETWORK_POLYGONS = "OUTPUT_NETWORK_POLYGONS"
    OUTPUT_DIFF_POLYGONS = "OUTPUT_DIFF_POLYGONS"

    def name(self):
        return "beta_calculator"

    def displayName(self):
        return "2. Beta Calculator (network scaling factor)"

    def group(self):
        return "Voronoi Risk Analysis"

    def groupId(self):
        return "voronoi_risk_analysis"

    def shortHelpString(self):
        return (
            "Computes the network scaling factor beta = d_network / d_euclidean "
            "for each municipality, against its Voronoi-assigned and "
            "network-nearest facility. Requires QNEAT3.\n\n"
            "Audit log: when an output CSV path is provided, the algorithm "
            "writes a per-municipality structured log (one row per feature "
            "with all input/output values) for downstream comparison with "
            "ground-truth datasets."
        )

    def createInstance(self):
        return BetaCalculatorAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MUNICIPALITIES, "Municipalities (point or polygon)",
            [QgsProcessing.TypeVectorAnyGeometry],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.FACILITIES, "Facilities (point)",
            [QgsProcessing.TypeVectorPoint],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.NETWORK, "Road network (line)",
            [QgsProcessing.TypeVectorLine],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.BOUNDARY, "Regional boundary (polygon, optional — clips polygon outputs)",
            [QgsProcessing.TypeVectorPolygon], optional=True,
        ))
        self.addParameter(QgsProcessingParameterField(
            self.FACILITY_ID_FIELD, "Facility identifier field",
            parentLayerParameterName=self.FACILITIES,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DEFAULT_SPEED, "Default speed (km/h) for QNEAT3",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=50.0, minValue=1.0, maxValue=300.0,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.AUDIT_LOG, "Audit log (CSV, optional)",
            fileFilter="CSV files (*.csv)", optional=True,
            createByDefault=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Beta calculation result"
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_VORONOI_POLYGONS,
            "Voronoi polygons (Euclidean assignment by facility)",
            type=QgsProcessing.TypeVectorPolygon,
            optional=True, createByDefault=True,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_NETWORK_POLYGONS,
            "Network service-area polygons (real assignment by facility)",
            type=QgsProcessing.TypeVectorPolygon,
            optional=True, createByDefault=True,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_DIFF_POLYGONS,
            "Misallocation polygons (Voronoi cells where Euclidean ≠ network)",
            type=QgsProcessing.TypeVectorPolygon,
            optional=True, createByDefault=True,
        ))

    def processAlgorithm(self, parameters, context, feedback):
        source_mun = self.parameterAsSource(parameters, self.MUNICIPALITIES, context)
        source_fac = self.parameterAsSource(parameters, self.FACILITIES, context)
        fac_id_field = self.parameterAsString(parameters, self.FACILITY_ID_FIELD, context)
        default_speed = self.parameterAsDouble(parameters, self.DEFAULT_SPEED, context)
        audit_path = self.parameterAsFileOutput(parameters, self.AUDIT_LOG, context)

        # ---- Collect facility coords + ids by positional iteration ----
        # Critical: positional id (0..n-1) is what QNEAT3 emits as
        # destination_id, NOT the QGIS feat.id().
        fac_coords, fac_pos_ids, fac_fids = collect_layer_points(source_fac)
        fac_coords = np.asarray(fac_coords, dtype=float)

        # Re-read facility ID field values matching positional order.
        fac_user_ids = []
        for feat in source_fac.getFeatures():
            try:
                fac_user_ids.append(feat[fac_id_field])
            except KeyError:
                fac_user_ids.append(None)
        n_fac = len(fac_coords)
        feedback.pushInfo(f"Loaded {n_fac} facilities")

        # ---- Collect municipality coords ----
        mun_coords, mun_pos_ids, mun_fids = collect_layer_points(source_mun)
        mun_coords = np.asarray(mun_coords, dtype=float)
        feedback.pushInfo(f"Loaded {len(mun_coords)} municipalities")

        # ---- Run QNEAT3 OD matrix ----
        # QNEAT3 requires FROM_ID_FIELD and TO_ID_FIELD. We materialize
        # in-memory point layers with a positional integer field that
        # matches our enumeration order; the OD output then uses those
        # ids as origin_id / destination_id, matching `pos` in the loop.
        import processing
        feedback.pushInfo("Computing OD cost matrix via QNEAT3 (this may take a while)...")
        mun_layer = self.parameterAsVectorLayer(parameters, self.MUNICIPALITIES, context)
        fac_layer = self.parameterAsVectorLayer(parameters, self.FACILITIES, context)
        net_layer = self.parameterAsVectorLayer(parameters, self.NETWORK, context)

        feedback.pushInfo("Building positional-id layers for QNEAT3 ...")
        mun_pos_layer, _ = make_layer_with_pos_id(mun_layer, "_pos_id")
        fac_pos_layer, _ = make_layer_with_pos_id(fac_layer, "_pos_id")

        try:
            od_result = processing.run(
                "qneat3:OdMatrixFromLayersAsTable",
                {
                    "INPUT": net_layer,
                    "FROM_POINT_LAYER": mun_pos_layer,
                    "FROM_ID_FIELD": "_pos_id",
                    "TO_POINT_LAYER": fac_pos_layer,
                    "TO_ID_FIELD": "_pos_id",
                    "STRATEGY": 0,                      # shortest distance
                    "DEFAULT_DIRECTION": 2,             # both directions
                    "DEFAULT_SPEED": default_speed,
                    "TOLERANCE": 0,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=context, feedback=feedback,
            )
        except Exception as e:
            feedback.reportError(
                f"QNEAT3 OD Matrix failed: {e}\n"
                "Ensure the QNEAT3 plugin is installed and the network "
                "layer is a topologically valid graph."
            )
            return {}

        od_layer = od_result["OUTPUT"]
        net_distances = parse_od_layer(od_layer)
        feedback.pushInfo(f"OD matrix parsed: {len(net_distances):,} rows")

        # ---- Audit log setup ----
        audit_fields = [
            "mun_pos_id", "mun_qgs_id",
            "voronoi_pos_id", "voronoi_facility_id",
            "d_euclidean_assigned", "d_network_assigned", "beta_assigned",
            "net_pos_id", "net_facility_id",
            "d_network_nearest", "d_euclidean_nearest", "beta_nearest_net",
            "is_misallocated", "distance_saving_m",
        ]
        log = AuditLogger(audit_path or None, audit_fields, algorithm_name="beta_calculator")
        log.set_meta({
            "n_municipalities": len(mun_coords),
            "n_facilities": n_fac,
            "default_speed_kmh": default_speed,
            "facility_id_field": fac_id_field,
            "od_rows": len(net_distances),
        })

        # ---- Output sink fields ----
        out_fields = QgsFields(source_mun.fields())
        out_fields.append(QgsField("voronoi_facility_id", QVariant.String))
        out_fields.append(QgsField("d_euclidean_assigned", QVariant.Double))
        out_fields.append(QgsField("d_network_assigned", QVariant.Double))
        out_fields.append(QgsField("beta_assigned", QVariant.Double))
        out_fields.append(QgsField("net_facility_id", QVariant.String))
        out_fields.append(QgsField("d_network_nearest", QVariant.Double))
        out_fields.append(QgsField("beta_nearest_net", QVariant.Double))
        out_fields.append(QgsField("is_misallocated", QVariant.Int))
        out_fields.append(QgsField("distance_saving_m", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            out_fields, source_mun.wkbType(), source_mun.sourceCrs(),
        )

        n_misalloc = 0
        n_skipped = 0
        total = source_mun.featureCount() or len(mun_coords)
        for pos, feat in enumerate(source_mun.getFeatures()):
            if feedback.isCanceled():
                break
            if pos % 25 == 0:
                feedback.setProgress(int(100 * pos / max(total, 1)))

            mun_xy = mun_coords[pos]

            # Euclidean distances to all facilities
            eu_dists = np.linalg.norm(fac_coords - mun_xy, axis=1)
            voronoi_pos = int(np.argmin(eu_dists))
            d_eu_assigned = float(eu_dists[voronoi_pos])

            # Network distance to Voronoi-assigned facility (using POSITIONAL id)
            d_net_assigned = net_distances.get((pos, voronoi_pos))

            # Network-nearest facility: scan OD entries for this origin
            net_dists_for_mun = np.full(n_fac, np.inf)
            for j in range(n_fac):
                v = net_distances.get((pos, j))
                if v is not None:
                    net_dists_for_mun[j] = v
            net_nearest_pos = int(np.argmin(net_dists_for_mun))
            d_net_nearest = float(net_dists_for_mun[net_nearest_pos])

            beta_assigned = (d_net_assigned / d_eu_assigned
                             if d_net_assigned and d_eu_assigned > 0 else None)
            d_eu_nearest = float(eu_dists[net_nearest_pos])
            beta_nearest = (d_net_nearest / d_eu_nearest
                            if np.isfinite(d_net_nearest) and d_eu_nearest > 0 else None)

            voronoi_id = fac_user_ids[voronoi_pos]
            net_id = fac_user_ids[net_nearest_pos]
            is_mis = int(voronoi_pos != net_nearest_pos)
            n_misalloc += is_mis

            saving = (float(d_net_assigned) - d_net_nearest
                      if is_mis and d_net_assigned is not None and np.isfinite(d_net_nearest)
                      else 0.0)

            if d_net_assigned is None:
                n_skipped += 1

            # Sink feature
            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(feat.geometry())
            attrs = feat.attributes()
            attrs.extend([
                str(voronoi_id) if voronoi_id is not None else None,
                d_eu_assigned,
                float(d_net_assigned) if d_net_assigned is not None else None,
                float(beta_assigned) if beta_assigned is not None else None,
                str(net_id) if net_id is not None else None,
                float(d_net_nearest) if np.isfinite(d_net_nearest) else None,
                float(beta_nearest) if beta_nearest is not None else None,
                int(is_mis),
                float(saving),
            ])
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            # Audit row
            log.row({
                "mun_pos_id": pos,
                "mun_qgs_id": feat.id(),
                "voronoi_pos_id": voronoi_pos,
                "voronoi_facility_id": voronoi_id,
                "d_euclidean_assigned": d_eu_assigned,
                "d_network_assigned": d_net_assigned,
                "beta_assigned": beta_assigned,
                "net_pos_id": net_nearest_pos,
                "net_facility_id": net_id,
                "d_network_nearest": d_net_nearest if np.isfinite(d_net_nearest) else None,
                "d_euclidean_nearest": d_eu_nearest,
                "beta_nearest_net": beta_nearest,
                "is_misallocated": is_mis,
                "distance_saving_m": saving,
            })

        feedback.pushInfo(
            f"Beta calculation complete: {n_misalloc} misallocations / "
            f"{total} municipalities ({100.0*n_misalloc/max(total,1):.1f}%); "
            f"{n_skipped} routes missing in OD matrix."
        )
        log.close({
            "misallocations": n_misalloc,
            "missing_routes": n_skipped,
        })

        result = {self.OUTPUT: dest_id, self.AUDIT_LOG: audit_path or ""}

        # ---- Optional polygon outputs ----
        # Both outputs are dissolutions of the SAME underlying municipality
        # Voronoi tessellation. That way the two polygon layers share the
        # exact same cell grid and their difference is exactly the set of
        # misallocated municipalities.
        boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY, context)

        want_voro = parameters.get(self.OUTPUT_VORONOI_POLYGONS) is not None
        want_net = parameters.get(self.OUTPUT_NETWORK_POLYGONS) is not None
        want_diff = parameters.get(self.OUTPUT_DIFF_POLYGONS) is not None

        if want_voro or want_net or want_diff:
            cells_layer = self._build_municipality_cells(
                parameters, context, feedback, dest_id, boundary_layer,
            )
            if cells_layer is not None:
                if want_voro:
                    v_dest = self._dissolve_cells(
                        parameters, context, feedback, cells_layer,
                        field="voronoi_facility_id",
                        sink_param=self.OUTPUT_VORONOI_POLYGONS,
                        out_field_name="voronoi_facility_id",
                    )
                    if v_dest is not None:
                        result[self.OUTPUT_VORONOI_POLYGONS] = v_dest
                if want_net:
                    n_dest = self._dissolve_cells(
                        parameters, context, feedback, cells_layer,
                        field="net_facility_id",
                        sink_param=self.OUTPUT_NETWORK_POLYGONS,
                        out_field_name="net_facility_id",
                    )
                    if n_dest is not None:
                        result[self.OUTPUT_NETWORK_POLYGONS] = n_dest
                if want_diff:
                    d_dest = self._misallocation_cells(
                        parameters, context, feedback, cells_layer,
                    )
                    if d_dest is not None:
                        result[self.OUTPUT_DIFF_POLYGONS] = d_dest

        return result

    def _build_municipality_cells(self, parameters, context, feedback,
                                    beta_layer_id, boundary_layer):
        """Build a single municipality-cell tessellation (Voronoi of muni
        centroids) carrying both `voronoi_facility_id` and `net_facility_id`
        attributes per cell. Optionally clipped to boundary.

        This is the SHARED base for the Euclidean and Network polygon
        outputs — making the two layers directly superimposable.

        Returns a QgsVectorLayer (in-memory or temp output) ready to be
        dissolved by either field, or None on failure.
        """
        import processing
        from qgis.core import QgsProcessingUtils, QgsWkbTypes

        feedback.pushInfo("Building municipality Voronoi cells (shared base)...")
        beta_layer = QgsProcessingUtils.mapLayerFromString(beta_layer_id, context)
        if beta_layer is None:
            feedback.reportError("Could not resolve the beta-output layer.")
            return None

        is_polygon = beta_layer.wkbType() in (
            QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon,
            QgsWkbTypes.Polygon25D, QgsWkbTypes.MultiPolygon25D,
            QgsWkbTypes.PolygonZ, QgsWkbTypes.MultiPolygonZ,
        )

        if is_polygon:
            # Munis are already polygons — use them directly as cells.
            cells = beta_layer
        else:
            # Munis are points — build Voronoi cells around each muni.
            try:
                v_result = processing.run(
                    "native:voronoipolygons",
                    {
                        "INPUT": beta_layer,
                        "BUFFER": 20,
                        "TOLERANCE": 0,
                        "COPY_ATTRIBUTES": True,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=context, feedback=feedback,
                )
                cells = v_result["OUTPUT"]
            except Exception as e:
                feedback.reportError(f"native:voronoipolygons failed: {e}")
                return None

        # Optional clip to boundary
        if boundary_layer is not None:
            feedback.pushInfo("Clipping cells to boundary...")
            try:
                clip_result = processing.run(
                    "native:clip",
                    {
                        "INPUT": cells,
                        "OVERLAY": boundary_layer,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=context, feedback=feedback,
                )
                cells = clip_result["OUTPUT"]
            except Exception as e:
                feedback.reportError(f"native:clip failed: {e}")

        return cells

    def _misallocation_cells(self, parameters, context, feedback, cells_layer):
        """Output the muni Voronoi cells whose Euclidean assignment differs
        from the network assignment — i.e., the misallocation polygons.

        One feature per misallocated muni, NOT dissolved (so each cell can
        be inspected individually with its β, distances, and saving).
        """
        feedback.pushInfo("Building misallocation polygons (Voronoi != Network cells)...")

        # The cells layer carries all the per-muni attributes from the
        # beta-calculator output, including is_misallocated /
        # voronoi_facility_id / net_facility_id / beta_assigned / saving.
        out_fields = QgsFields()
        out_fields.append(QgsField("voronoi_facility_id", QVariant.String))
        out_fields.append(QgsField("net_facility_id", QVariant.String))
        out_fields.append(QgsField("d_euclidean_assigned", QVariant.Double))
        out_fields.append(QgsField("d_network_assigned", QVariant.Double))
        out_fields.append(QgsField("d_network_nearest", QVariant.Double))
        out_fields.append(QgsField("beta_assigned", QVariant.Double))
        out_fields.append(QgsField("beta_nearest_net", QVariant.Double))
        out_fields.append(QgsField("distance_saving_m", QVariant.Double))
        out_fields.append(QgsField("distance_penalty_pct", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_DIFF_POLYGONS, context,
            out_fields, cells_layer.wkbType(), cells_layer.sourceCrs(),
        )

        n_misalloc = 0
        for feat in cells_layer.getFeatures():
            try:
                v_id = feat["voronoi_facility_id"]
                n_id = feat["net_facility_id"]
            except KeyError:
                continue
            if v_id is None or n_id is None:
                continue
            if str(v_id) == str(n_id):
                continue
            n_misalloc += 1

            # Pull other useful attributes if present
            def _get(name):
                try:
                    val = feat[name]
                    return None if val is None else float(val)
                except (KeyError, TypeError, ValueError):
                    return None

            d_eu = _get("d_euclidean_assigned")
            d_net_v = _get("d_network_assigned")
            d_net_n = _get("d_network_nearest")
            beta_v = _get("beta_assigned")
            beta_n = _get("beta_nearest_net")

            saving = None
            penalty = None
            if d_net_v is not None and d_net_n is not None:
                saving = d_net_v - d_net_n
                penalty = (saving / d_net_n * 100.0) if d_net_n > 0 else None

            new_feat = QgsFeature(out_fields)
            new_feat.setGeometry(feat.geometry())
            new_feat.setAttributes([
                str(v_id), str(n_id),
                d_eu, d_net_v, d_net_n,
                beta_v, beta_n,
                saving, penalty,
            ])
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

        feedback.pushInfo(f"  -> {n_misalloc} misallocation polygons written")
        return dest_id

    def _dissolve_cells(self, parameters, context, feedback,
                          cells_layer, field, sink_param, out_field_name):
        """Dissolve the shared muni cells by `field` and write to the sink.
        Adds `n_munis_assigned` count. Returns dest_id or None.
        """
        import processing

        feedback.pushInfo(f"Dissolving cells by {field}...")
        try:
            d_result = processing.run(
                "native:dissolve",
                {
                    "INPUT": cells_layer,
                    "FIELD": [field],
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=context, feedback=feedback,
            )
            dissolved = d_result["OUTPUT"]
        except Exception as e:
            feedback.reportError(f"native:dissolve by {field} failed: {e}")
            return None

        # Count munis per facility id from the cells layer
        counts = {}
        for f in cells_layer.getFeatures():
            try:
                k = str(f[field])
                counts[k] = counts.get(k, 0) + 1
            except KeyError:
                pass

        out_fields = QgsFields()
        out_fields.append(QgsField(out_field_name, QVariant.String))
        out_fields.append(QgsField("n_munis_assigned", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters, sink_param, context,
            out_fields, dissolved.wkbType(), dissolved.sourceCrs(),
        )
        for feat in dissolved.getFeatures():
            try:
                fid = feat[field]
            except KeyError:
                fid = None
            new_feat = QgsFeature(out_fields)
            new_feat.setGeometry(feat.geometry())
            new_feat.setAttributes([
                str(fid) if fid is not None else None,
                int(counts.get(str(fid), 0)),
            ])
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
        return dest_id

