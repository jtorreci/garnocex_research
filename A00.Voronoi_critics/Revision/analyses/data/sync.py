"""Sync canonical input data for the revision analyses.

Pulls authoritative copies from A01.Voronoi (assignment files, shapefiles)
and from A00.Voronoi_critics/codigo (distributional + anisotropy datasets).
Writes a manifest with SHA-256 hashes so we can detect upstream changes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE = .../A00.Voronoi_critics/paper1_geographical_analysis/Revision/analyses/data
A00_ROOT = HERE.parents[3]                   # .../A00.Voronoi_critics
PROJECT_ROOT = A00_ROOT.parent  # parent directory of A00 (anonymized for peer review)
A01_SCRIPTS = PROJECT_ROOT / "A01.Voronoi" / "scripts"
A01_SHP = PROJECT_ROOT / "A01.Voronoi" / "shp"
CODIGO = A00_ROOT / "codigo"

# Files: list of (source_path, dest_filename)
FILES = [
    # Assignment files: authoritative copy in A01 (A00/codigo has a stale copy)
    (A01_SCRIPTS / "asignacion_municipios_euclidiana.csv", "asignacion_municipios_euclidiana.csv"),
    (A01_SCRIPTS / "asignacion_municipios_real.csv",       "asignacion_municipios_real.csv"),
    # Distributional + anisotropy datasets: from A00/codigo
    (CODIGO / "detailed_ratios_analysis_filtered.csv",      "detailed_ratios_analysis_filtered.csv"),
    (CODIGO / "complete_anisotropy_coefficients.csv",       "complete_anisotropy_coefficients.csv"),
    (CODIGO / "plant_anisotropy_coefficients_filtered.csv", "plant_anisotropy_coefficients_filtered.csv"),
    # Coordinates
    (CODIGO / "coordenadas_municipios.csv",                 "coordenadas_municipios.csv"),
    (CODIGO / "coordenadas_plantas.csv",                    "coordenadas_plantas.csv"),
    # Full distance matrices (optional, for re-running pipelines)
    (CODIGO / "tablas" / "D_real_municipios_clean.csv",     "D_real_municipios_clean.csv"),
    (CODIGO / "tablas" / "D_euclidea_municipios_clean.csv", "D_euclidea_municipios_clean.csv"),
    (CODIGO / "tablas" / "D_real_plantas_clean.csv",        "D_real_plantas_clean.csv"),
    (CODIGO / "tablas" / "D_euclidea_plantas_clean.csv",    "D_euclidea_plantas_clean.csv"),
]

# Shapefile sets — copy all sidecar files (.shp, .shx, .dbf, .prj, .cpg)
SHP_SETS = ["municipios", "plantas", "Extremadura", "carreteras"]
SHP_EXTS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_one(src: Path, dst: Path, manifest: dict) -> bool:
    if not src.exists():
        manifest["missing"].append(str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest["files"][str(dst.relative_to(HERE))] = {
        "source": str(src),
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
    }
    return True


def main() -> None:
    manifest = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "A01_scripts": str(A01_SCRIPTS),
            "A01_shp": str(A01_SHP),
            "A00_codigo": str(CODIGO),
        },
        "files": {},
        "missing": [],
    }

    print(f"Syncing from {PROJECT_ROOT}\n")

    for src, dst_name in FILES:
        dst = HERE / dst_name
        if copy_one(src, dst, manifest):
            print(f"  + {dst_name}  ({dst.stat().st_size:,} bytes)")
        else:
            print(f"  ! MISSING: {src}")

    shp_dir = HERE / "shp"
    shp_dir.mkdir(exist_ok=True)
    for stem in SHP_SETS:
        for ext in SHP_EXTS:
            src = A01_SHP / f"{stem}{ext}"
            dst = shp_dir / f"{stem}{ext}"
            if copy_one(src, dst, manifest):
                print(f"  + shp/{stem}{ext}")
            elif ext in (".shp", ".shx", ".dbf"):
                print(f"  ! MISSING: {src}")

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if manifest["missing"]:
        print(f"\nWARNING: {len(manifest['missing'])} file(s) missing — see manifest.json")
    else:
        print("\nOK: all files synced")


if __name__ == "__main__":
    main()
