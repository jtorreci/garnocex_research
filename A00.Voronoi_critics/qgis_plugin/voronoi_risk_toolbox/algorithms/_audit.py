# -*- coding: utf-8 -*-
"""
Shared utilities for the Voronoi Risk Toolbox algorithms.

Provides:
    AuditLogger     - structured CSV log writer for per-feature outputs.
    parse_od_layer  - field-name-tolerant parser for QNEAT3 OD matrix output.
    iter_points     - geometry-tolerant point extraction (point or polygon centroid).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qgis.core import (
    QgsWkbTypes,
    QgsField,
    QgsFeature,
    QgsVectorLayer,
    QgsProject,
)
from qgis.PyQt.QtCore import QVariant


# ----------------------------------------------------------------------
# Audit logging — write a CSV alongside the output sink so the user can
# diff against canonical ground truth (e.g. municipios_canonical.csv).
# ----------------------------------------------------------------------
class AuditLogger:
    """Write a per-row audit log to a CSV file, plus a one-line header
    summary for the run.

    Usage:
        log = AuditLogger(path, fieldnames=["muni_id", "beta", ...])
        log.set_meta({"algorithm": "beta_calculator", "n_facilities": 46})
        for ... in features:
            log.row({...})
        log.close()
    """

    def __init__(
        self,
        path: Optional[str],
        fieldnames: List[str],
        algorithm_name: str = "",
    ):
        self.path = path
        self.fieldnames = fieldnames
        self.algorithm_name = algorithm_name
        self._fp = None
        self._writer = None
        self._meta: Dict[str, Any] = {}
        self._row_count = 0

        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fp = open(self.path, "w", newline="", encoding="utf-8")
            # Write a leading meta block as comment lines (CSV reader-friendly)
            self._fp.write(f"# voronoi_risk_toolbox audit log\n")
            self._fp.write(f"# algorithm: {self.algorithm_name}\n")
            self._fp.write(f"# started: {datetime.now(timezone.utc).isoformat()}\n")
            self._writer = None    # written lazily after meta is set

    def set_meta(self, meta: Dict[str, Any]) -> None:
        self._meta.update(meta)
        if self._fp:
            for k, v in meta.items():
                self._fp.write(f"# {k}: {v}\n")

    def _ensure_writer(self) -> None:
        if self._writer is None and self._fp is not None:
            self._writer = csv.DictWriter(self._fp, fieldnames=self.fieldnames)
            self._writer.writeheader()

    def row(self, data: Dict[str, Any]) -> None:
        self._row_count += 1
        if self._writer is None and self._fp is not None:
            self._ensure_writer()
        if self._writer is not None:
            # Coerce types and fill missing with ""
            row = {k: ("" if data.get(k) is None else data.get(k)) for k in self.fieldnames}
            self._writer.writerow(row)

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self._fp:
            if summary:
                self._fp.write("# --- summary ---\n")
                for k, v in summary.items():
                    self._fp.write(f"# {k}: {v}\n")
            self._fp.write(f"# rows_written: {self._row_count}\n")
            self._fp.write(f"# finished: {datetime.now(timezone.utc).isoformat()}\n")
            self._fp.close()
            self._fp = None


# ----------------------------------------------------------------------
# QNEAT3 OD parsing — field names have changed across versions.
# ----------------------------------------------------------------------
_OD_ORIGIN_CANDIDATES = ("origin_id", "origin_point_id", "origin", "InputID", "from_id")
_OD_DEST_CANDIDATES = ("destination_id", "destination_point_id", "destination", "TargetID", "to_id")
_OD_COST_CANDIDATES = ("total_cost", "network_cost", "entry_cost", "cost")


def detect_od_fields(od_layer) -> Tuple[str, str, str]:
    """Find the origin/destination/cost field names actually present
    in a QNEAT3 OD matrix layer. Raises KeyError if none match.
    """
    fields = [f.name() for f in od_layer.fields()]

    def pick(candidates: Iterable[str]) -> str:
        for c in candidates:
            if c in fields:
                return c
        raise KeyError(f"No matching OD field found among {fields} "
                       f"(expected one of {list(candidates)})")

    return pick(_OD_ORIGIN_CANDIDATES), pick(_OD_DEST_CANDIDATES), pick(_OD_COST_CANDIDATES)


def parse_od_layer(od_layer) -> Dict[Tuple[Any, Any], float]:
    """Parse a QNEAT3 OD-as-table layer into {(origin_id, dest_id): cost}.

    Tries known field-name variants. The QNEAT3 origin_id / destination_id
    are 0-based positional indices into the input FROM_POINT_LAYER /
    TO_POINT_LAYER, not QGIS feature IDs.
    """
    f_origin, f_dest, f_cost = detect_od_fields(od_layer)
    out: Dict[Tuple[Any, Any], float] = {}
    for feat in od_layer.getFeatures():
        try:
            o = feat[f_origin]
            d = feat[f_dest]
            c = feat[f_cost]
        except KeyError:
            continue
        if o is None or d is None or c is None:
            continue
        out[(o, d)] = float(c)
    return out


# ----------------------------------------------------------------------
# Point extraction — uniform across point/polygon source layers.
# ----------------------------------------------------------------------
def feature_centroid_xy(feature) -> Tuple[float, float]:
    """Return (x, y) for a feature, using centroid for polygons."""
    geom = feature.geometry()
    if geom.type() == QgsWkbTypes.PolygonGeometry:
        pt = geom.centroid().asPoint()
    else:
        pt = geom.asPoint()
    return float(pt.x()), float(pt.y())


def collect_layer_points(source) -> Tuple[List[Tuple[float, float]], List[int], List[Any]]:
    """Iterate a feature source once and return:
        coords           list of (x, y)
        positional_ids   list of 0-based positions (matches QNEAT3 origin/destination_id)
        qgs_feature_ids  list of QGIS feat.id() values (FID)

    The triple of lists is index-aligned. Use positional_ids when looking
    things up in a QNEAT3 OD matrix; use qgs_feature_ids when joining back
    to the original layer.
    """
    coords: List[Tuple[float, float]] = []
    pos_ids: List[int] = []
    fids: List[Any] = []
    for pos, feat in enumerate(source.getFeatures()):
        coords.append(feature_centroid_xy(feat))
        pos_ids.append(pos)
        fids.append(feat.id())
    return coords, pos_ids, fids


# ----------------------------------------------------------------------
# Temporary in-memory layer with a positional ID field — used to feed
# QNEAT3, which requires FROM_ID_FIELD / TO_ID_FIELD.
# ----------------------------------------------------------------------
def make_layer_with_pos_id(
    source_layer,
    id_field_name: str = "_pos_id",
) -> Tuple["QgsVectorLayer", List[Any]]:
    """Materialise an in-memory vector layer with the same geometry as the
    source plus an integer field `id_field_name` set to 0..n-1 in iteration
    order. Returns (mem_layer, list_of_input_qgs_fids).

    QNEAT3 requires an explicit FROM_ID_FIELD / TO_ID_FIELD. Without one,
    it errors with 'Valor de parametro incorrecto para FROM_ID_FIELD'.
    We give it our own positional id and the OD output uses those ids,
    which matches our positional iteration in the caller.
    """
    geom_type = QgsWkbTypes.displayString(source_layer.wkbType())
    crs = source_layer.sourceCrs()
    # Use the source's wkb type string for the in-memory provider URI
    if "Polygon" in geom_type or "polygon" in geom_type.lower():
        provider_uri = "Point"   # We'll use centroids
    elif "Line" in geom_type or "line" in geom_type.lower():
        provider_uri = "Point"   # not expected for FROM/TO points anyway
    else:
        provider_uri = "Point"

    uri = f"{provider_uri}?crs={crs.authid()}&field={id_field_name}:integer"
    mem = QgsVectorLayer(uri, f"_pos_id_layer_{id_field_name}", "memory")
    pr = mem.dataProvider()

    fids = []
    new_features = []
    for pos, feat in enumerate(source_layer.getFeatures()):
        geom = feat.geometry()
        if geom.type() == QgsWkbTypes.PolygonGeometry:
            geom = geom.centroid()
        new_feat = QgsFeature(mem.fields())
        new_feat.setGeometry(geom)
        new_feat.setAttributes([pos])
        new_features.append(new_feat)
        fids.append(feat.id())

    pr.addFeatures(new_features)
    mem.updateExtents()
    return mem, fids
