# -*- coding: utf-8 -*-
"""
Algorithm 5: Anisotropy Map

Computes the directional anisotropy coefficient for each municipality:
    alpha_i = beta_max(A_i) / beta_min(A_i)
where beta_min and beta_max are the minimum and maximum detour
coefficients across all routes from A_i to the destination layer.

Fixes since v0.1.0 audit:
    B5.1  use positional ids (not feat.id()) when looking up QNEAT3 OD
    B5.2  field names auto-detected (origin_id/total_cost variants)
    B5.4  optional structured CSV audit log
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


class AnisotropyMapAlgorithm(QgsProcessingAlgorithm):

    MUNICIPALITIES = "MUNICIPALITIES"
    DESTINATIONS = "DESTINATIONS"     # destinations to compute α against (plants OR other munis)
    NETWORK = "NETWORK"
    DEST_ID_FIELD = "DEST_ID_FIELD"
    DEFAULT_SPEED = "DEFAULT_SPEED"
    AUDIT_LOG = "AUDIT_LOG"
    OUTPUT = "OUTPUT"

    def name(self):
        return "anisotropy_map"

    def displayName(self):
        return "5. Anisotropy Map"

    def group(self):
        return "Voronoi Risk Analysis"

    def groupId(self):
        return "voronoi_risk_analysis"

    def shortHelpString(self):
        return (
            "Computes the directional anisotropy coefficient alpha for each "
            "municipality:\n\n"
            "  alpha_i = beta_max(A_i) / beta_min(A_i)\n\n"
            "where beta = d_network / d_euclidean over all destinations.\n\n"
            "alpha = 1 -> isotropic (route detour is identical in all "
            "directions; Voronoi assignment is robust).\n"
            "alpha >> 1 -> highly anisotropic (Voronoi misassignment likely).\n\n"
            "DESTINATIONS may be plants (alpha relative to facilities) or "
            "another set of points (alpha relative to other munis, etc.). "
            "Uses QNEAT3 for network distances."
        )

    def createInstance(self):
        return AnisotropyMapAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MUNICIPALITIES, "Municipalities (point or polygon)",
            [QgsProcessing.TypeVectorAnyGeometry],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DESTINATIONS, "Destinations (point) — plants OR other munis",
            [QgsProcessing.TypeVectorPoint],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.NETWORK, "Road network (line)",
            [QgsProcessing.TypeVectorLine],
        ))
        self.addParameter(QgsProcessingParameterField(
            self.DEST_ID_FIELD, "Destination identifier field",
            parentLayerParameterName=self.DESTINATIONS,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DEFAULT_SPEED, "Default speed (km/h) for QNEAT3",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=50.0, minValue=1.0, maxValue=300.0,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.AUDIT_LOG, "Audit log (CSV, optional)",
            fileFilter="CSV files (*.csv)", optional=True, createByDefault=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Anisotropy analysis result"
        ))

    def processAlgorithm(self, parameters, context, feedback):
        source_mun = self.parameterAsSource(parameters, self.MUNICIPALITIES, context)
        source_dest = self.parameterAsSource(parameters, self.DESTINATIONS, context)
        dest_id_field = self.parameterAsString(parameters, self.DEST_ID_FIELD, context)
        default_speed = self.parameterAsDouble(parameters, self.DEFAULT_SPEED, context)
        audit_path = self.parameterAsFileOutput(parameters, self.AUDIT_LOG, context)

        # ---- Destinations ----
        dest_coords, _, _ = collect_layer_points(source_dest)
        dest_coords = np.asarray(dest_coords, dtype=float)
        dest_user_ids = []
        for feat in source_dest.getFeatures():
            try:
                dest_user_ids.append(feat[dest_id_field])
            except KeyError:
                dest_user_ids.append(None)
        n_dest = len(dest_coords)
        feedback.pushInfo(f"Loaded {n_dest} destinations")

        # ---- Municipalities ----
        mun_coords, _, _ = collect_layer_points(source_mun)
        mun_coords = np.asarray(mun_coords, dtype=float)
        feedback.pushInfo(f"Loaded {len(mun_coords)} municipalities")

        # ---- QNEAT3 OD ----
        import processing
        feedback.pushInfo("Computing OD cost matrix via QNEAT3...")
        mun_layer = self.parameterAsVectorLayer(parameters, self.MUNICIPALITIES, context)
        dest_layer = self.parameterAsVectorLayer(parameters, self.DESTINATIONS, context)
        net_layer = self.parameterAsVectorLayer(parameters, self.NETWORK, context)

        feedback.pushInfo("Building positional-id layers for QNEAT3 ...")
        mun_pos_layer, _ = make_layer_with_pos_id(mun_layer, "_pos_id")
        dest_pos_layer, _ = make_layer_with_pos_id(dest_layer, "_pos_id")

        try:
            od_result = processing.run(
                "qneat3:OdMatrixFromLayersAsTable",
                {
                    "INPUT": net_layer,
                    "FROM_POINT_LAYER": mun_pos_layer,
                    "FROM_ID_FIELD": "_pos_id",
                    "TO_POINT_LAYER": dest_pos_layer,
                    "TO_ID_FIELD": "_pos_id",
                    "STRATEGY": 0,
                    "DEFAULT_DIRECTION": 2,
                    "DEFAULT_SPEED": default_speed,
                    "TOLERANCE": 0,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=context, feedback=feedback,
            )
        except Exception as e:
            feedback.reportError(f"QNEAT3 OD Matrix failed: {e}")
            return {}

        od_layer = od_result["OUTPUT"]
        net_distances = parse_od_layer(od_layer)
        feedback.pushInfo(f"OD matrix parsed: {len(net_distances):,} rows")

        # ---- Output sink ----
        out_fields = QgsFields(source_mun.fields())
        out_fields.append(QgsField("beta_min", QVariant.Double))
        out_fields.append(QgsField("beta_max", QVariant.Double))
        out_fields.append(QgsField("beta_mean", QVariant.Double))
        out_fields.append(QgsField("beta_min_dest_id", QVariant.String))
        out_fields.append(QgsField("beta_max_dest_id", QVariant.String))
        out_fields.append(QgsField("anisotropy_alpha", QVariant.Double))
        out_fields.append(QgsField("anisotropy_class", QVariant.String))
        out_fields.append(QgsField("n_dest_used", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            out_fields, source_mun.wkbType(), source_mun.sourceCrs(),
        )

        log = AuditLogger(
            audit_path or None,
            ["mun_pos_id", "mun_qgs_id", "n_dest_used",
             "beta_min", "beta_max", "beta_mean",
             "beta_min_dest_id", "beta_max_dest_id",
             "anisotropy_alpha", "anisotropy_class"],
            algorithm_name="anisotropy_map",
        )
        log.set_meta({
            "n_municipalities": len(mun_coords),
            "n_destinations": n_dest,
            "destination_id_field": dest_id_field,
            "default_speed_kmh": default_speed,
        })

        total = source_mun.featureCount() or len(mun_coords)
        for pos, feat in enumerate(source_mun.getFeatures()):
            if feedback.isCanceled():
                break
            if pos % 25 == 0:
                feedback.setProgress(int(100 * pos / max(total, 1)))

            mun_xy = mun_coords[pos]

            # beta to each destination via positional ids
            betas = []
            beta_dest_ids = []
            for j in range(n_dest):
                d_net = net_distances.get((pos, j))
                if d_net is None:
                    continue
                d_eu = float(np.linalg.norm(dest_coords[j] - mun_xy))
                if d_eu <= 0 or d_net <= 0:
                    continue
                beta = d_net / d_eu
                if beta < 1.0:    # routing artifact
                    continue
                betas.append(beta)
                beta_dest_ids.append(dest_user_ids[j])

            beta_min = beta_max = beta_mean = alpha = None
            min_id = max_id = None
            if len(betas) >= 2:
                arr = np.asarray(betas)
                imin = int(np.argmin(arr))
                imax = int(np.argmax(arr))
                beta_min = float(arr[imin])
                beta_max = float(arr[imax])
                beta_mean = float(arr.mean())
                alpha = beta_max / beta_min if beta_min > 0 else None
                min_id = beta_dest_ids[imin]
                max_id = beta_dest_ids[imax]
            elif len(betas) == 1:
                beta_min = beta_max = beta_mean = float(betas[0])
                alpha = 1.0
                min_id = max_id = beta_dest_ids[0]

            if alpha is None:
                aclass = "N/A"
            elif alpha < 1.5:
                aclass = "Low"
            elif alpha < 2.5:
                aclass = "Medium"
            else:
                aclass = "High"

            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(feat.geometry())
            attrs = feat.attributes()
            attrs.extend([
                beta_min, beta_max, beta_mean,
                str(min_id) if min_id is not None else None,
                str(max_id) if max_id is not None else None,
                alpha, aclass, len(betas),
            ])
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            log.row({
                "mun_pos_id": pos,
                "mun_qgs_id": feat.id(),
                "n_dest_used": len(betas),
                "beta_min": beta_min,
                "beta_max": beta_max,
                "beta_mean": beta_mean,
                "beta_min_dest_id": min_id,
                "beta_max_dest_id": max_id,
                "anisotropy_alpha": alpha,
                "anisotropy_class": aclass,
            })

        feedback.pushInfo("Anisotropy mapping complete")
        log.close({})

        return {self.OUTPUT: dest_id, self.AUDIT_LOG: audit_path or ""}
