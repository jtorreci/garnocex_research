"""Class composition of the IGN transport network, by length.

The manuscript states that tracks (camino) and footpaths (senda) make up 81.3 % of the
unrestricted network, 70.4 % and 10.8 % respectively. That figure had no reproducible
source: network/MANIFEST.json records the surface breakdown (firme) but the published
network_full.gpkg does not carry the road-class field, so the claim could not be checked
from the published artefacts.

This recomputes it from the IGN source layers named in MANIFEST.json and writes the result
into pipeline/network_composition.json, so the number is derivable without the source.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "Caceres": Path(r"Z:\Dropbox\Universidad\Investigacion\Biblio_matcher\Projects\Garnocex"
                    r"\Pablo Plaza\SHP\Transporte\RT_CACERES\RT_CACERES\RT_VIARIA\rt_tramo_vial.shp"),
    "Badajoz": Path(r"Z:\Dropbox\Universidad\Investigacion\Biblio_matcher\Projects\Garnocex"
                    r"\Pablo Plaza\SHP\Transporte\RT_BADAJOZ\RT_BADAJOZ\RT_VIARIA\rt_tramo_vial.shp"),
}
CRS_METRIC = "EPSG:25830"
EXCLUDED = ("camino", "senda")


def main() -> None:
    parts = []
    for prov, p in SOURCES.items():
        g = gpd.read_file(p)
        if g.crs is None:
            g = g.set_crs("EPSG:4258")
        g = g.to_crs(CRS_METRIC)
        g["length_m"] = g.geometry.length
        g["provincia"] = prov
        parts.append(g)
        print(f"{prov}: {len(g):,} segments, {g.length_m.sum()/1000:,.1f} km")

    g = pd.concat(parts, ignore_index=True)

    # find the field holding the road class
    cand = [c for c in g.columns
            if g[c].dtype == object and c.lower() not in ("geometry", "provincia")
            and g[c].astype(str).str.lower().isin(["camino", "senda"]).any()]
    if not cand:
        print("\ncolumns:", [c for c in g.columns])
        raise SystemExit("no column contains the values 'camino'/'senda'")
    col = cand[0]
    print(f"\nclass field: {col!r}\n")

    by = (g.groupby(g[col].astype(str))["length_m"].sum() / 1000.0).sort_values(ascending=False)
    total = float(by.sum())
    print(f"{'class':38}{'km':>12}{'%':>8}")
    print("-" * 58)
    rows = {}
    for k, km in by.items():
        pct = 100.0 * km / total
        rows[k] = {"km": round(float(km), 1), "pct": round(pct, 2)}
        print(f"{k[:37]:38}{km:12.1f}{pct:8.2f}")

    excl = sum(v["pct"] for k, v in rows.items() if k.strip().lower() in EXCLUDED)
    print("-" * 58)
    print(f"{'TOTAL':38}{total:12.1f}{100.0:8.2f}")
    print(f"\ncamino + senda = {excl:.2f} % of network length")

    (ROOT / "pipeline" / "network_composition.json").write_text(
        json.dumps({"sources": {k: str(v) for k, v in SOURCES.items()},
                    "class_field": col, "total_km": round(total, 1),
                    "excluded_classes": list(EXCLUDED),
                    "excluded_pct": round(excl, 2), "by_class": rows}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print("\nwrote pipeline/network_composition.json")


if __name__ == "__main__":
    main()
