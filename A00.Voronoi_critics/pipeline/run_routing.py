"""Headless QNEAT3 OD-matrix runs for the Voronoi misallocation framework.

Must be launched with the QGIS interpreter, not the system python:

    "C:/Program Files/QGIS 3.40.9/bin/python-qgis-ltr.bat" run_routing.py \
        --network operational --pairs muni-plant --tolerance 0

Design notes
------------
* Joins are by positional integer id, never by place name (decision D10 in
  docs/DECISIONES.md): the historical pipeline lost 82 municipalities to mojibake
  in exactly that step. Names travel in a side mapping file.
* ENTRY_COST_CALCULATION_METHOD = Planar, because the working CRS is EPSG:25830.
  The parameter's own help says "only use with projected CRS"; the historical run
  left it at the Ellipsoidal default.
* STRATEGY = Shortest Path, so speed parameters do not enter the result.
* TOLERANCE is the graph-building snap tolerance, not a maximum distance. It is
  exposed here because carreteras.shp has ~234 connected components under
  TOLERANCE = 0, and spurious disconnections are a candidate explanation for the
  handful of extreme beta values that dominate the calibrated dispersion.
* The four cost components (entry / network / exit / total) are all retained:
  Definition 1 of the manuscript defines d_r as the total.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QGIS_PREFIX = r"C:/Program Files/QGIS 3.40.9/apps/qgis-ltr"
PROFILE_PLUGINS = r"C:/Users/Usuario/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "analyses" / "data"
SHP = DATA / "shp"
OUT = ROOT / "pipeline" / "od"

NETWORKS = {
    "operational": SHP / "carreteras.shp",          # RT_VIARIA minus {Camino, Senda}
    "full": ROOT / "network" / "network_full.gpkg",  # RT_VIARIA complete
    "roads_nourban": ROOT / "network" / "network_roads_nourban.gpkg",  # roads only, no urban streets
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def demojibake(s: str) -> str:
    """Repair a cp850/latin-1 round trip when its signature is present."""
    out = str(s)
    # Signature characters of the two corruptions seen in this project: UTF-8 read as
    # latin-1 leaves Ã/Â, and cp850 read as latin-1 turns a/i/n-tilde into nbsp/¡/¤.
    for _ in range(2):
        if not any(c in out for c in "ÃÂ¡¤\xa0"):
            break
        best = out
        for enc in ("utf-8", "cp850"):
            try:
                cand = out.encode("latin-1").decode(enc)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "\ufffd" not in cand:
                best = cand
                break
        if best == out:
            break
        out = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=sorted(NETWORKS))
    ap.add_argument("--pairs", required=True, choices=["muni-plant", "muni-muni"])
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--plants", choices=["real", "degraded"], default="real",
                    help="degraded reproduces the historical plantas.shp for diagnosis only")
    args = ap.parse_args()

    net_path = NETWORKS[args.network]
    if not net_path.exists():
        raise SystemExit(f"network layer not found: {net_path}")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.network}_{args.pairs.replace('-', '')}_tol{args.tolerance:g}"
    if args.plants == "degraded":
        tag += "_degraded"

    # ---- QGIS bootstrap -------------------------------------------------
    from qgis.core import (Qgis, QgsApplication, QgsCoordinateReferenceSystem,
                           QgsFeature, QgsField, QgsGeometry, QgsPointXY,
                           QgsVectorLayer)
    from qgis.PyQt.QtCore import QVariant

    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    sys.path.append(QGIS_PREFIX + "/python/plugins")
    sys.path.append(PROFILE_PLUGINS)
    import processing
    from processing.core.Processing import Processing
    Processing.initialize()
    from QNEAT3.Qneat3Provider import Qneat3Provider
    provider = Qneat3Provider()                     # keep the reference alive
    QgsApplication.processingRegistry().addProvider(provider)
    provider.loadAlgorithms()
    print(f"QGIS {Qgis.QGIS_VERSION} | QNEAT3 algorithms: "
          f"{len([a for a in QgsApplication.processingRegistry().algorithms() if a.id().startswith('qneat3')])}")

    crs = QgsCoordinateReferenceSystem("EPSG:25830")

    def point_layer(name, rows):
        """rows: list of (pos_id, label, x, y). pos_id is what QNEAT3 reports back."""
        lyr = QgsVectorLayer("Point?crs=EPSG:25830", name, "memory")
        dp = lyr.dataProvider()
        dp.addAttributes([QgsField("pos_id", QVariant.Int), QgsField("label", QVariant.String)])
        lyr.updateFields()
        feats = []
        for pid, label, x, y in rows:
            f = QgsFeature(lyr.fields())
            f.setAttributes([int(pid), str(label)])
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
            feats.append(f)
        dp.addFeatures(feats)
        lyr.updateExtents()
        return lyr

    # ---- origins: the 383 municipalities --------------------------------
    muni_rows, seen = [], set()
    with (ROOT / "codigo" / "coordenadas_municipios.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            nom = (r.get("NOMBRE") or "").strip()
            if not nom or nom in seen:
                continue
            seen.add(nom)
            muni_rows.append((len(muni_rows), demojibake(nom), float(r["X"]), float(r["Y"])))
    origins = point_layer("municipios", muni_rows)
    print(f"origins: {origins.featureCount()} municipalities")

    # ---- destinations ---------------------------------------------------
    if args.pairs == "muni-plant":
        # Canonical facility layer: Plantas_Ext.shp, the GARNOCEX layer named by the
        # historical assignment script (PLANTAS_LAYER_NAME = "Plantas_Ext"). All 46
        # facilities carry their own coordinates. Do NOT use plantas.shp: 30 of its 46
        # records were filled in with the host municipality centroid (its own `source`
        # column says so), displacing them by a median of 2.4 km and up to 8.6 km.
        src = "Plantas_Ext.shp" if args.plants == "real" else "plantas.shp"
        plants = QgsVectorLayer(str(SHP / src), src, "ogr")
        if not plants.isValid():
            raise SystemExit("Plantas_Ext.shp could not be read")
        dest_rows, roster = [], []
        for i, f in enumerate(plants.getFeatures()):
            p = f.geometry().asPoint()
            muni = demojibake(f["MUNICIPIO"] if args.plants == "real" else f["mun"])
            deno = demojibake(f["DENOMINACI"] if args.plants == "real" else f["name"])
            deno = deno.replace("\\n", " ").strip()
            dest_rows.append((i, f"plant{i}:{muni}", p.x(), p.y()))
            roster.append((i, deno, muni, p.x(), p.y()))
        destinations = point_layer("Plantas_Ext", dest_rows)
        # Stable roster so every downstream step joins on pos_id, never on a name.
        roster_path = OUT / f"plantas_roster_{args.plants}.csv"
        with roster_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pos_id", "denominacion", "municipio", "x", "y"])
            w.writerows(roster)
        print(f"facility roster -> {roster_path.name} ({len(roster)} facilities)")
    else:
        dest_rows = list(muni_rows)
        destinations = point_layer("municipios_dest", dest_rows)
    print(f"destinations: {destinations.featureCount()} | expected routes: "
          f"{origins.featureCount() * destinations.featureCount():,}")

    params = {
        "INPUT": str(net_path),
        "FROM_POINT_LAYER": origins,
        "FROM_ID_FIELD": "pos_id",
        "TO_POINT_LAYER": destinations,
        "TO_ID_FIELD": "pos_id",
        "STRATEGY": 0,                      # shortest path: distance, speed irrelevant
        "ENTRY_COST_CALCULATION_METHOD": 1,  # planar: the CRS is projected
        "DEFAULT_DIRECTION": 2,             # both directions -> symmetric d_r
        "DEFAULT_SPEED": 50,
        "TOLERANCE": args.tolerance,
        "OUTPUT": "TEMPORARY_OUTPUT",
    }
    print(f"running qneat3:OdMatrixFromLayersAsTable on {net_path.name} "
          f"(TOLERANCE={args.tolerance:g}) ...")
    t0 = time.time()
    res = processing.run("qneat3:OdMatrixFromLayersAsTable", params)
    elapsed = time.time() - t0
    od = res["OUTPUT"]
    fields = [f.name() for f in od.fields()]
    print(f"done in {elapsed/60:.1f} min | rows={od.featureCount():,} | fields={fields}")

    # ---- export ---------------------------------------------------------
    name_o = {pid: lab for pid, lab, _, _ in muni_rows}
    name_d = {pid: lab for pid, lab, _, _ in dest_rows}
    csv_path = OUT / f"od_{tag}.csv"
    n = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["origin_pos", "destination_pos", "origin_label", "destination_label",
                    "entry_cost", "network_cost", "exit_cost", "total_cost"])
        for f in od.getFeatures():
            rec = {k: f[k] for k in fields}
            o = rec.get("origin_id")
            d = rec.get("destination_id")
            if o is None or d is None:
                continue
            w.writerow([o, d, name_o.get(int(o), ""), name_d.get(int(d), ""),
                        rec.get("entry_cost"), rec.get("network_cost"),
                        rec.get("exit_cost"), rec.get("total_cost")])
            n += 1
    print(f"wrote {csv_path} ({n:,} rows)")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qgis": Qgis.QGIS_VERSION,
        "algorithm": "qneat3:OdMatrixFromLayersAsTable",
        "network_case": args.network,
        "network_layer": str(net_path),
        "network_sha256": sha256(net_path),
        "pairs": args.pairs,
        "n_origins": origins.featureCount(),
        "n_destinations": destinations.featureCount(),
        "rows_written": n,
        "elapsed_min": round(elapsed / 60, 2),
        "parameters": {k: v for k, v in params.items()
                       if k not in ("FROM_POINT_LAYER", "TO_POINT_LAYER", "OUTPUT")},
        "output_csv": csv_path.name,
        "output_sha256": sha256(csv_path),
        "crs": crs.authid(),
        "notes": "d_r = total_cost = entry + network + exit (Definition 1). "
                 "Joins by pos_id, never by name (D10).",
    }
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("network_case", "pairs", "rows_written", "elapsed_min")}, indent=2))
    app.exitQgis()


if __name__ == "__main__":
    main()
