"""Faithful replication of the historical assignment script, to see whether the
published 61 misallocations can be reproduced.

It mirrors A01.Voronoi/scripts/script_asignacion_planta_distancia_real.py exactly:
  * engine   : native:shortestpathpointtolayer (QGIS native), not QNEAT3
  * cost     : feature.geometry().length() -- the length of the path geometry, which
               starts and ends at the tied points, so access legs are excluded
  * origins  : one run per facility, with the municipality layer as END_POINTS
  * layers   : Plantas_Ext (facilities) and Municipios_Ext (municipalities)

Run with the QGIS interpreter and pass the network to test:
    python-qgis-ltr.bat replicate_published.py --network <path-to-network>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QGIS_PREFIX = r"C:/Program Files/QGIS 3.40.9/apps/qgis-ltr"
ROOT = Path(__file__).resolve().parent.parent
SHP = ROOT / "analyses" / "data" / "shp"
CONVENIO = Path(r"Z:/Dropbox/Universidad/Investigacion/Convenios/Convenio_aridos/Plantas comunidades/Extremadura")
OUT = ROOT / "pipeline" / "od"

NETWORKS = {
    "operational": SHP / "carreteras.shp",
    "roads_nourban": ROOT / "network" / "network_roads_nourban.gpkg",
    "full": ROOT / "network" / "network_full.gpkg",
    "red_viaria": CONVENIO / "Red_Viaria.shp",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=sorted(NETWORKS))
    ap.add_argument("--from", dest="i0", type=int, default=0)
    ap.add_argument("--to", dest="i1", type=int, default=10_000)
    args = ap.parse_args()
    net = NETWORKS[args.network]
    if not net.exists():
        raise SystemExit(f"not found: {net}")

    from qgis.core import Qgis, QgsApplication, QgsVectorLayer
    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    sys.path.append(QGIS_PREFIX + "/python/plugins")
    import processing
    from processing.core.Processing import Processing
    Processing.initialize()
    print(f"QGIS {Qgis.QGIS_VERSION} | network: {net.name}")

    plants = QgsVectorLayer(str(SHP / "Plantas_Ext.shp"), "Plantas_Ext", "ogr")
    munis = QgsVectorLayer(str(CONVENIO / "Municipios_Ext.shp"), "Municipios_Ext", "ogr")
    network = QgsVectorLayer(str(net), "net", "ogr")
    for lyr, name in ((plants, "Plantas_Ext"), (munis, "Municipios_Ext"), (network, "network")):
        if not lyr.isValid():
            raise SystemExit(f"invalid layer: {name}")
    speed_field = "velocidad" if any(f.name().lower() == "velocidad"
                                     for f in network.fields()) else None
    print(f"facilities={plants.featureCount()} municipalities={munis.featureCount()} "
          f"speed_field={speed_field}")

    from qgis.core import QgsProject
    QgsProject.instance().addMapLayer(munis)

    rows = []
    t0 = time.time()
    for i, pf in enumerate(plants.getFeatures()):
        if not (args.i0 <= i < args.i1):
            continue
        params = {
            "INPUT": network,
            "STRATEGY": 0,
            "START_POINT": pf.geometry().asPoint(),
            "END_POINTS": munis.id(),
            "DEFAULT_SPEED": 50,
            "OUTPUT": "TEMPORARY_OUTPUT",
        }
        if speed_field:
            params["SPEED_FIELD"] = speed_field
        try:
            res = processing.run("native:shortestpathpointtolayer", params)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  facility {i}: FAILED {exc}")
            continue
        for f in res["OUTPUT"].getFeatures():
            rows.append({"plant_pos": i, "muni": f["NOMBRE"],
                         "cost": f["cost"], "distance_m": f.geometry().length()})
        if True:
            print(f"  {i+1}/{plants.featureCount()} facilities, {time.time()-t0:.0f}s", flush=True)

    path = OUT / f"replica_native_{args.network}_{args.i0}_{args.i1}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["plant_pos", "muni", "cost", "distance_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}: {len(rows):,} rows in {(time.time()-t0)/60:.1f} min")
    app.exitQgis()


if __name__ == "__main__":
    main()
