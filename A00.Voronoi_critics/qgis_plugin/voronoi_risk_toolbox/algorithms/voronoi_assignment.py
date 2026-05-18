# -*- coding: utf-8 -*-
"""
Algorithm 1: Voronoi Assignment

Given a layer of spatial units (municipalities) and a layer of facilities,
compute the Euclidean Voronoi assignment for each unit. No QNEAT3 needed.

Outputs (sink + optional CSV audit log):
    voronoi_facility_id
    voronoi_distance_eu          d_e to nearest facility
    voronoi_distance_eu_2nd      d_e to 2nd-nearest facility
    voronoi_R                    distance ratio d_e(P, A2) / d_e(P, A1)
"""

from __future__ import annotations

import numpy as np

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessing,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant

from ._audit import AuditLogger, collect_layer_points


class VoronoiAssignmentAlgorithm(QgsProcessingAlgorithm):

    MUNICIPALITIES = "MUNICIPALITIES"
    FACILITIES = "FACILITIES"
    BOUNDARY = "BOUNDARY"
    FACILITY_ID_FIELD = "FACILITY_ID_FIELD"
    AUDIT_LOG = "AUDIT_LOG"
    OUTPUT = "OUTPUT"
    OUTPUT_POLYGONS = "OUTPUT_POLYGONS"

    def name(self):
        return "voronoi_assignment"

    def displayName(self):
        return "1. Voronoi Assignment"

    def group(self):
        return "Voronoi Risk Analysis"

    def groupId(self):
        return "voronoi_risk_analysis"

    def shortHelpString(self):
        return (
            "Assigns each municipality to its nearest facility under "
            "Euclidean distance (Voronoi tessellation). Computes the "
            "distance ratio R = d(P, A_2)/d(P, A_1) used for "
            "misallocation probability estimation."
        )

    def createInstance(self):
        return VoronoiAssignmentAlgorithm()

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
            self.BOUNDARY, "Regional boundary (polygon, optional)",
            [QgsProcessing.TypeVectorPolygon], optional=True,
        ))
        self.addParameter(QgsProcessingParameterField(
            self.FACILITY_ID_FIELD, "Facility identifier field",
            parentLayerParameterName=self.FACILITIES,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.AUDIT_LOG, "Audit log (CSV, optional)",
            fileFilter="CSV files (*.csv)", optional=True, createByDefault=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Voronoi assignment result"
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_POLYGONS,
            "Voronoi polygons (Euclidean tessellation)",
            type=QgsProcessing.TypeVectorPolygon,
            optional=True,
            createByDefault=True,
        ))

    def processAlgorithm(self, parameters, context, feedback):
        source_mun = self.parameterAsSource(parameters, self.MUNICIPALITIES, context)
        source_fac = self.parameterAsSource(parameters, self.FACILITIES, context)
        fac_id_field = self.parameterAsString(parameters, self.FACILITY_ID_FIELD, context)
        audit_path = self.parameterAsFileOutput(parameters, self.AUDIT_LOG, context)

        fac_coords, fac_pos_ids, _ = collect_layer_points(source_fac)
        fac_coords = np.asarray(fac_coords, dtype=float)
        fac_user_ids = []
        for feat in source_fac.getFeatures():
            try:
                fac_user_ids.append(feat[fac_id_field])
            except KeyError:
                fac_user_ids.append(None)
        n_fac = len(fac_coords)
        feedback.pushInfo(f"Loaded {n_fac} facilities")

        out_fields = QgsFields(source_mun.fields())
        out_fields.append(QgsField("voronoi_facility_id", QVariant.String))
        out_fields.append(QgsField("voronoi_pos_id", QVariant.Int))
        out_fields.append(QgsField("voronoi_distance_eu", QVariant.Double))
        out_fields.append(QgsField("voronoi_distance_eu_2nd", QVariant.Double))
        out_fields.append(QgsField("voronoi_R", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            out_fields, source_mun.wkbType(), source_mun.sourceCrs(),
        )

        log = AuditLogger(
            audit_path or None,
            ["mun_pos_id", "mun_qgs_id", "voronoi_pos_id", "voronoi_facility_id",
             "voronoi_distance_eu", "voronoi_distance_eu_2nd", "voronoi_R"],
            algorithm_name="voronoi_assignment",
        )
        log.set_meta({
            "n_facilities": n_fac,
            "facility_id_field": fac_id_field,
        })

        total = source_mun.featureCount()
        n_processed = 0
        for pos, feat in enumerate(source_mun.getFeatures()):
            if feedback.isCanceled():
                break
            if pos % 25 == 0:
                feedback.setProgress(int(100 * pos / max(total, 1)))

            geom = feat.geometry()
            if geom.type().value == 2:    # polygon
                pt = geom.centroid().asPoint()
            else:
                pt = geom.asPoint()
            mun_xy = np.array([pt.x(), pt.y()])

            dists = np.linalg.norm(fac_coords - mun_xy, axis=1)
            sorted_idx = np.argsort(dists)
            voronoi_pos = int(sorted_idx[0])
            d1 = float(dists[voronoi_pos])
            d2 = float(dists[sorted_idx[1]]) if n_fac > 1 else d1
            R = d2 / d1 if d1 > 0 else 1.0
            voronoi_id = fac_user_ids[voronoi_pos]

            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(geom)
            attrs = feat.attributes()
            attrs.extend([
                str(voronoi_id) if voronoi_id is not None else None,
                voronoi_pos, d1, d2, float(R),
            ])
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            log.row({
                "mun_pos_id": pos,
                "mun_qgs_id": feat.id(),
                "voronoi_pos_id": voronoi_pos,
                "voronoi_facility_id": voronoi_id,
                "voronoi_distance_eu": d1,
                "voronoi_distance_eu_2nd": d2,
                "voronoi_R": R,
            })
            n_processed += 1

        feedback.pushInfo(f"Voronoi assignment complete: {n_processed} municipalities")
        log.close({"processed": n_processed})

        # ---- Optional Voronoi service-area polygons ----
        result = {self.OUTPUT: dest_id, self.AUDIT_LOG: audit_path or ""}
        if parameters.get(self.OUTPUT_POLYGONS) is not None:
            boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY, context)
            poly_dest = self._make_voronoi_polygons(
                parameters, context, feedback,
                assigned_layer_id=dest_id,
                boundary_layer=boundary_layer,
            )
            if poly_dest is not None:
                result[self.OUTPUT_POLYGONS] = poly_dest

        return result

    def _make_voronoi_polygons(self, parameters, context, feedback,
                                assigned_layer_id, boundary_layer):
        """Build the Euclidean Voronoi service-area polygons by:

        1. Computing the Voronoi tessellation of the MUNICIPALITY centroids
           (one cell per muni, copying all attributes including the
           voronoi_facility_id assigned in the main loop).
        2. Optionally clipping the cells to the boundary polygon.
        3. Dissolving cells by voronoi_facility_id → one polygon per facility.

        Same geometric basis as the network service-area output of
        Algorithm 2 — making the two layers directly superimposable.
        """
        import processing
        from qgis.core import QgsProcessingUtils, QgsWkbTypes

        feedback.pushInfo("Building Voronoi service-area polygons (muni cells dissolved by facility)...")

        assigned_layer = QgsProcessingUtils.mapLayerFromString(assigned_layer_id, context)
        if assigned_layer is None:
            feedback.reportError("Could not resolve the assignment-output layer.")
            return None

        is_polygon = assigned_layer.wkbType() in (
            QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon,
            QgsWkbTypes.Polygon25D, QgsWkbTypes.MultiPolygon25D,
            QgsWkbTypes.PolygonZ, QgsWkbTypes.MultiPolygonZ,
        )

        # Step 1: muni-cell tessellation
        if is_polygon:
            cells = assigned_layer
        else:
            try:
                v_result = processing.run(
                    "native:voronoipolygons",
                    {
                        "INPUT": assigned_layer,
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

        # Step 2: optional clip to boundary
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

        # Step 3: dissolve by voronoi_facility_id
        feedback.pushInfo("Dissolving cells by voronoi_facility_id...")
        try:
            d_result = processing.run(
                "native:dissolve",
                {
                    "INPUT": cells,
                    "FIELD": ["voronoi_facility_id"],
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=context, feedback=feedback,
            )
            dissolved = d_result["OUTPUT"]
        except Exception as e:
            feedback.reportError(f"native:dissolve failed: {e}")
            return None

        # Count munis per facility id
        counts = {}
        for f in cells.getFeatures():
            try:
                k = str(f["voronoi_facility_id"])
                counts[k] = counts.get(k, 0) + 1
            except KeyError:
                pass

        # Write to sink
        out_fields = QgsFields()
        out_fields.append(QgsField("voronoi_facility_id", QVariant.String))
        out_fields.append(QgsField("n_munis_assigned", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_POLYGONS, context,
            out_fields, dissolved.wkbType(), dissolved.sourceCrs(),
        )
        for feat in dissolved.getFeatures():
            try:
                fid = feat["voronoi_facility_id"]
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
