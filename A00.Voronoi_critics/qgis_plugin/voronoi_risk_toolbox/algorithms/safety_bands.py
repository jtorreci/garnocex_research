# -*- coding: utf-8 -*-
"""
Algorithm 3: Safety Bands

Computes safety band widths around Voronoi boundaries based on the
probabilistic framework (Theorem 1 + Lemma 1 + Corollary 1).

Theorem 1:
    P[misallocation | R] = Phi( -ln(R) / (sqrt(2) s) )
where R = d_e(P, A_2) / d_e(P, A_1) is the Euclidean distance ratio
to the two nearest facilities.

Lemma 1 (local approximation near border):
    ln(R(P)) approx kappa * t
    kappa = 2 * sin(theta/2) / d
where Q is the nearest point on the Voronoi border to P,
d = d_e(Q, A_1) = d_e(Q, A_2),
theta is the angle subtended by A_1, A_2 at Q,
t is the signed normal distance from P to the border (negative
into A_1's cell).

Corollary 1 (safety band width for risk q*):
    |t*(q*)| = (sqrt(2) s / kappa) * Phi^{-1}(1 - q*)

Fix since v0.1.0 audit: B3.1 - kappa now computed at Q (the projection
of P onto the perpendicular bisector of A_1 A_2), not at the segment
midpoint. The previous implementation was correct only when P lay on
the perpendicular through the midpoint of A_1 A_2.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessing,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from ._audit import AuditLogger, collect_layer_points


class SafetyBandsAlgorithm(QgsProcessingAlgorithm):

    MUNICIPALITIES = "MUNICIPALITIES"
    FACILITIES = "FACILITIES"
    FACILITY_ID_FIELD = "FACILITY_ID_FIELD"
    BOUNDARY = "BOUNDARY"
    S_PARAMETER = "S_PARAMETER"
    RISK_TOLERANCE = "RISK_TOLERANCE"
    BAND_LEVELS = "BAND_LEVELS"
    AUDIT_LOG = "AUDIT_LOG"
    OUTPUT = "OUTPUT"
    OUTPUT_BAND_POLYGONS = "OUTPUT_BAND_POLYGONS"

    def name(self):
        return "safety_bands"

    def displayName(self):
        return "3. Safety Bands"

    def group(self):
        return "Voronoi Risk Analysis"

    def groupId(self):
        return "voronoi_risk_analysis"

    def shortHelpString(self):
        return (
            "Computes the misallocation probability and safety-band width "
            "for each municipality using Theorem 1 / Lemma 1 / Corollary 1.\n\n"
            "Geographic complexity parameter s:\n"
            "  Flat / low terrain     0.03-0.06\n"
            "  Urban dense            0.05-0.08\n"
            "  Mixed (Extremadura)    0.08-0.12\n"
            "  Mountainous            0.13-0.20\n"
            "  Coastal / fragmented   0.15-0.25\n\n"
            "in_safety_band = 1  ->  the Euclidean Voronoi assignment is "
            "unreliable for this municipality and a network-based check is "
            "recommended."
        )

    def createInstance(self):
        return SafetyBandsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MUNICIPALITIES, "Municipalities (point or polygon)",
            [QgsProcessing.TypeVectorAnyGeometry],
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.FACILITIES, "Facilities (point)",
            [QgsProcessing.TypeVectorPoint],
        ))
        self.addParameter(QgsProcessingParameterField(
            self.FACILITY_ID_FIELD, "Facility identifier field",
            parentLayerParameterName=self.FACILITIES,
        ))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.BOUNDARY, "Regional boundary (polygon, optional — clips band polygons)",
            [QgsProcessing.TypeVectorPolygon], optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.S_PARAMETER, "Dispersion parameter s",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.093, minValue=0.01, maxValue=0.50,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RISK_TOLERANCE, "Per-municipality risk tolerance q* (flag)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.10, minValue=0.001, maxValue=0.499,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.BAND_LEVELS,
            "Risk levels for band polygons (comma-separated)",
            defaultValue="0.05,0.10,0.20",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.AUDIT_LOG, "Audit log (CSV, optional)",
            fileFilter="CSV files (*.csv)", optional=True, createByDefault=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Safety band analysis result (per municipality)"
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_BAND_POLYGONS,
            "Safety band polygons (one per Voronoi ridge × risk level)",
            type=QgsProcessing.TypeVectorPolygon,
            optional=True, createByDefault=True,
        ))

    def processAlgorithm(self, parameters, context, feedback):
        source_mun = self.parameterAsSource(parameters, self.MUNICIPALITIES, context)
        source_fac = self.parameterAsSource(parameters, self.FACILITIES, context)
        fac_id_field = self.parameterAsString(parameters, self.FACILITY_ID_FIELD, context)
        s = self.parameterAsDouble(parameters, self.S_PARAMETER, context)
        q_star = self.parameterAsDouble(parameters, self.RISK_TOLERANCE, context)
        audit_path = self.parameterAsFileOutput(parameters, self.AUDIT_LOG, context)

        # ---- Facilities ----
        fac_coords, _, _ = collect_layer_points(source_fac)
        fac_coords = np.asarray(fac_coords, dtype=float)
        fac_user_ids = []
        for feat in source_fac.getFeatures():
            try:
                fac_user_ids.append(feat[fac_id_field])
            except KeyError:
                fac_user_ids.append(None)
        n_fac = len(fac_coords)
        feedback.pushInfo(f"Loaded {n_fac} facilities")

        # ---- Output sink ----
        out_fields = QgsFields(source_mun.fields())
        out_fields.append(QgsField("voronoi_facility_id", QVariant.String))
        out_fields.append(QgsField("voronoi_R", QVariant.Double))
        out_fields.append(QgsField("kappa", QVariant.Double))
        out_fields.append(QgsField("safety_band_width_t_star", QVariant.Double))
        out_fields.append(QgsField("dist_to_voronoi_border", QVariant.Double))
        out_fields.append(QgsField("misallocation_prob", QVariant.Double))
        out_fields.append(QgsField("in_safety_band", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            out_fields, source_mun.wkbType(), source_mun.sourceCrs(),
        )

        # ---- Audit log ----
        log = AuditLogger(
            audit_path or None,
            ["mun_pos_id", "mun_qgs_id", "voronoi_pos_id", "voronoi_facility_id",
             "second_facility_pos_id",
             "d1", "d2", "R", "kappa", "theta_rad",
             "dist_to_border", "t_star", "misallocation_prob", "in_safety_band"],
            algorithm_name="safety_bands",
        )
        log.set_meta({
            "n_facilities": n_fac,
            "s_parameter": s,
            "risk_tolerance": q_star,
        })

        z_threshold = norm.ppf(1.0 - q_star)
        sqrt2_s = np.sqrt(2.0) * s

        total = source_mun.featureCount() or 0
        n_in_band = 0
        for pos, feat in enumerate(source_mun.getFeatures()):
            if feedback.isCanceled():
                break
            if total > 0 and pos % 25 == 0:
                feedback.setProgress(int(100 * pos / total))

            geom = feat.geometry()
            if geom.type().value == 2:
                pt = geom.centroid().asPoint()
            else:
                pt = geom.asPoint()
            P = np.array([pt.x(), pt.y()])

            # Two nearest facilities to P
            dists = np.linalg.norm(fac_coords - P, axis=1)
            sorted_idx = np.argsort(dists)
            v_pos = int(sorted_idx[0])
            s_pos = int(sorted_idx[1]) if n_fac > 1 else v_pos
            A1 = fac_coords[v_pos]
            A2 = fac_coords[s_pos]
            d1 = float(dists[v_pos])
            d2 = float(dists[s_pos]) if n_fac > 1 else d1
            R = d2 / d1 if d1 > 0 else 1.0
            voronoi_id = fac_user_ids[v_pos]

            # Misallocation probability via Theorem 1 (closed form)
            if R > 0 and s > 0:
                prob = float(norm.cdf(-np.log(R) / sqrt2_s))
            else:
                prob = 0.5

            # ---- Geometry near the Voronoi border (Lemma 1) ----
            # Q = projection of P onto the perpendicular bisector of A1 A2
            seg = A2 - A1
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                kappa = 0.0
                theta = 0.0
                dist_border = 0.0
                t_star = float("inf")
            else:
                seg_unit = seg / seg_len
                mid = 0.5 * (A1 + A2)
                # signed distance from P to the bisector along the segment direction
                t_along_seg = float(np.dot(P - mid, seg_unit))
                Q = P - t_along_seg * seg_unit
                d = float(np.linalg.norm(Q - A1))    # = ||Q-A2|| by construction
                # angle theta at Q between Q->A1 and Q->A2
                if d > 0:
                    v1 = (A1 - Q) / d
                    v2 = (A2 - Q) / d
                    cos_theta = float(np.clip(v1 @ v2, -1.0, 1.0))
                    theta = float(np.arccos(cos_theta))
                    kappa = 2.0 * float(np.sin(theta / 2.0)) / d
                else:
                    theta = np.pi
                    kappa = 0.0
                # Distance from P to the bisector (= |t_along_seg|)
                dist_border = abs(t_along_seg)
                t_star = (sqrt2_s / kappa) * z_threshold if kappa > 0 else float("inf")

            in_band = 1 if dist_border < t_star else 0
            n_in_band += in_band

            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(geom)
            attrs = feat.attributes()
            attrs.extend([
                str(voronoi_id) if voronoi_id is not None else None,
                float(R), float(kappa),
                float(t_star) if np.isfinite(t_star) else None,
                float(dist_border), float(prob), int(in_band),
            ])
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            log.row({
                "mun_pos_id": pos,
                "mun_qgs_id": feat.id(),
                "voronoi_pos_id": v_pos,
                "voronoi_facility_id": voronoi_id,
                "second_facility_pos_id": s_pos,
                "d1": d1, "d2": d2, "R": R,
                "kappa": kappa, "theta_rad": theta,
                "dist_to_border": dist_border,
                "t_star": t_star if np.isfinite(t_star) else None,
                "misallocation_prob": prob,
                "in_safety_band": in_band,
            })

        feedback.pushInfo(
            f"Safety band analysis complete: {n_in_band} municipalities in safety band "
            f"(s={s}, q*={q_star})"
        )
        log.close({
            "in_band_count": n_in_band,
            "s_parameter": s,
            "risk_tolerance": q_star,
        })

        result = {self.OUTPUT: dest_id, self.AUDIT_LOG: audit_path or ""}

        # ---- Optional band polygons ----
        if parameters.get(self.OUTPUT_BAND_POLYGONS) is not None:
            band_levels_str = self.parameterAsString(parameters, self.BAND_LEVELS, context)
            try:
                band_levels = sorted({
                    float(x.strip())
                    for x in band_levels_str.split(",")
                    if x.strip()
                })
            except ValueError:
                feedback.pushWarning(
                    f"Could not parse risk levels '{band_levels_str}'; "
                    f"using default [0.05, 0.10, 0.20]"
                )
                band_levels = [0.05, 0.10, 0.20]
            band_levels = [q for q in band_levels if 0 < q < 0.5]

            boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY, context)
            band_dest = self._make_band_polygons(
                parameters, context, feedback,
                fac_coords, fac_user_ids, s, band_levels, boundary_layer,
                source_crs=source_mun.sourceCrs(),
            )
            if band_dest is not None:
                result[self.OUTPUT_BAND_POLYGONS] = band_dest

        return result

    def _make_band_polygons(self, parameters, context, feedback,
                              fac_coords, fac_user_ids, s, levels,
                              boundary_layer, source_crs):
        """Compute the safety-band polygons.

        For each Voronoi ridge between facilities A_i and A_j:
            kappa  = 4 / ||A_i - A_j||             (Lemma 1 at midpoint of A_iA_j)
            t*(q) = sqrt(2) * s / kappa * Phi^{-1}(1 - q)
        The band geometry is the buffer of the ridge segment by t*.

        Infinite ridges (convex-hull edges) are skipped. The boundary
        polygon, if provided, clips the bands cleanly.
        """
        from scipy.spatial import Voronoi
        from scipy.stats import norm

        feedback.pushInfo(
            f"Building safety-band polygons (s={s}, levels={levels})..."
        )

        if len(fac_coords) < 3:
            feedback.reportError("At least 3 facilities are required for Voronoi ridges.")
            return None

        try:
            vor = Voronoi(fac_coords)
        except Exception as e:
            feedback.reportError(f"scipy.spatial.Voronoi failed: {e}")
            return None

        # Boundary geometry (union of all features) for clipping
        boundary_geom = None
        if boundary_layer is not None:
            try:
                feats = list(boundary_layer.getFeatures())
                if feats:
                    boundary_geom = QgsGeometry(feats[0].geometry())
                    for f in feats[1:]:
                        boundary_geom = boundary_geom.combine(f.geometry())
            except Exception as e:
                feedback.pushWarning(f"Could not read boundary layer: {e}")
                boundary_geom = None

        # Output sink
        out_fields = QgsFields()
        out_fields.append(QgsField("q_star", QVariant.Double))
        out_fields.append(QgsField("t_star_m", QVariant.Double))
        out_fields.append(QgsField("kappa", QVariant.Double))
        out_fields.append(QgsField("facility_a", QVariant.String))
        out_fields.append(QgsField("facility_b", QVariant.String))
        out_fields.append(QgsField("ridge_length_m", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_BAND_POLYGONS, context,
            out_fields, QgsWkbTypes.Polygon, source_crs,
        )

        sqrt2_s = np.sqrt(2.0) * s
        n_written = 0
        n_infinite = 0
        n_empty_after_clip = 0

        for k, (i, j) in enumerate(vor.ridge_points):
            v_idx = vor.ridge_vertices[k]
            if -1 in v_idx:
                n_infinite += 1
                continue
            p1 = vor.vertices[v_idx[0]]
            p2 = vor.vertices[v_idx[1]]

            seg_AA = float(np.linalg.norm(fac_coords[i] - fac_coords[j]))
            if seg_AA <= 0:
                continue
            kappa = 4.0 / seg_AA       # representative at A_iA_j midpoint
            ridge_line = QgsGeometry.fromPolylineXY([
                QgsPointXY(float(p1[0]), float(p1[1])),
                QgsPointXY(float(p2[0]), float(p2[1])),
            ])
            ridge_len = float(ridge_line.length())

            for q_star in levels:
                z = float(norm.ppf(1.0 - q_star))
                t_star = (sqrt2_s / kappa) * z
                if t_star <= 0:
                    continue
                buffer_geom = ridge_line.buffer(t_star, 8)
                if boundary_geom is not None:
                    buffer_geom = buffer_geom.intersection(boundary_geom)
                    if buffer_geom is None or buffer_geom.isEmpty():
                        n_empty_after_clip += 1
                        continue

                new_feat = QgsFeature(out_fields)
                new_feat.setGeometry(buffer_geom)
                new_feat.setAttributes([
                    float(q_star), float(t_star), float(kappa),
                    str(fac_user_ids[i]), str(fac_user_ids[j]),
                    ridge_len,
                ])
                sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
                n_written += 1

        feedback.pushInfo(
            f"  Wrote {n_written} band polygons "
            f"({len(vor.ridge_points)} ridges total, "
            f"{n_infinite} infinite skipped, "
            f"{n_empty_after_clip} empty after boundary clip)"
        )
        return dest_id
