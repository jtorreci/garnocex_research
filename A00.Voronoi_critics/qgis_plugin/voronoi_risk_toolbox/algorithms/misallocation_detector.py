# -*- coding: utf-8 -*-
"""
Algorithm 4: Misallocation Detector

Reads the output of Algorithm 2 (Beta Calculator) and reports per-feature
metrics on misallocation: distance saving, percent penalty.

Inputs:
    - Layer with fields: voronoi_facility_id, net_facility_id,
                         d_network_assigned, d_network_nearest

Outputs (sink + optional CSV audit log):
    - is_misallocated, distance_saving_m, distance_penalty_pct
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFileDestination,
    QgsProcessing,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant

from ._audit import AuditLogger


class MisallocationDetectorAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    AUDIT_LOG = "AUDIT_LOG"
    OUTPUT = "OUTPUT"

    def name(self):
        return "misallocation_detector"

    def displayName(self):
        return "4. Misallocation Detector"

    def group(self):
        return "Voronoi Risk Analysis"

    def groupId(self):
        return "voronoi_risk_analysis"

    def shortHelpString(self):
        return (
            "Identifies misallocated municipalities (Voronoi-Euclidean assignment "
            "different from network-nearest). Requires the output of Algorithm 2 "
            "as input.\n\n"
            "Misallocation distance saving and penalty percentage are computed "
            "per feature. The summary message reports the aggregate "
            "misallocation rate."
        )

    def createInstance(self):
        return MisallocationDetectorAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, "Municipalities (with Beta Calculator output)",
            [QgsProcessing.TypeVectorAnyGeometry],
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.AUDIT_LOG, "Audit log (CSV, optional)",
            fileFilter="CSV files (*.csv)", optional=True, createByDefault=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Misallocation detection result"
        ))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        audit_path = self.parameterAsFileOutput(parameters, self.AUDIT_LOG, context)

        field_names = [f.name() for f in source.fields()]
        required = ["voronoi_facility_id", "net_facility_id",
                    "d_network_assigned", "d_network_nearest"]
        missing = [f for f in required if f not in field_names]
        if missing:
            feedback.reportError(
                f"Missing required fields: {missing}. "
                "Run Algorithm 2 (Beta Calculator) first."
            )
            return {}

        out_fields = QgsFields(source.fields())
        # Don't duplicate fields if they already exist
        existing = set(field_names)
        if "is_misallocated" not in existing:
            out_fields.append(QgsField("is_misallocated", QVariant.Int))
        if "distance_saving_m" not in existing:
            out_fields.append(QgsField("distance_saving_m", QVariant.Double))
        out_fields.append(QgsField("distance_penalty_pct", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            out_fields, source.wkbType(), source.sourceCrs(),
        )

        log = AuditLogger(
            audit_path or None,
            ["mun_qgs_id", "voronoi_facility_id", "net_facility_id",
             "d_network_assigned", "d_network_nearest", "is_misallocated",
             "distance_saving_m", "distance_penalty_pct"],
            algorithm_name="misallocation_detector",
        )

        total = source.featureCount()
        n_misallocated = 0

        for i, feat in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            if i % 25 == 0:
                feedback.setProgress(int(100 * i / max(total, 1)))

            voronoi_id = feat["voronoi_facility_id"]
            net_id = feat["net_facility_id"]
            d_net_assigned = feat["d_network_assigned"]
            d_net_nearest = feat["d_network_nearest"]

            is_mis = 1 if str(voronoi_id) != str(net_id) else 0
            n_misallocated += is_mis

            saving = 0.0
            penalty = 0.0
            if is_mis and d_net_assigned and d_net_nearest:
                saving = float(d_net_assigned) - float(d_net_nearest)
                penalty = (saving / float(d_net_nearest) * 100.0
                           if d_net_nearest > 0 else 0.0)

            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(feat.geometry())
            attrs = list(feat.attributes())
            if "is_misallocated" not in existing:
                attrs.append(is_mis)
            if "distance_saving_m" not in existing:
                attrs.append(saving)
            attrs.append(penalty)
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            log.row({
                "mun_qgs_id": feat.id(),
                "voronoi_facility_id": voronoi_id,
                "net_facility_id": net_id,
                "d_network_assigned": d_net_assigned,
                "d_network_nearest": d_net_nearest,
                "is_misallocated": is_mis,
                "distance_saving_m": saving,
                "distance_penalty_pct": penalty,
            })

        feedback.pushInfo(
            f"Misallocation detection complete: {n_misallocated} of "
            f"{total} municipalities misallocated "
            f"({100.0 * n_misallocated / max(total, 1):.1f}%)"
        )
        log.close({
            "misallocations": n_misallocated,
            "total": total,
            "rate_pct": round(100.0 * n_misallocated / max(total, 1), 2),
        })

        return {self.OUTPUT: dest_id, self.AUDIT_LOG: audit_path or ""}
