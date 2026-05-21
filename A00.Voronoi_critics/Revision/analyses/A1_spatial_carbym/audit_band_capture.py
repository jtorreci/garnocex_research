"""Audit how well the safety bands (regional anisotropy s_a = 0.093,
variable-width Lemma 1) capture the empirical 61 misallocated municipalities.

Reports:
    - capture rate at each q* (fraction of misallocated munis whose centroid
      falls inside the band of that risk level)
    - list of misallocated munis that fall OUTSIDE all three bands
      (q*=5% is the loosest), interpretable as "anomalies" likely due to
      geographic features imposing extreme local anisotropy
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import Voronoi
from scipy.stats import norm
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SHP = DATA / "shp"

PLANT_CONSOLIDATION = {
    2: 1, 12: 1, 17: 3, 15: 6, 40: 19, 32: 20,
    23: 22, 26: 22, 28: 22,
    34: 27, 35: 27, 37: 27,
    38: 36, 43: 36,
}
S_REGIONAL = 0.093
RISK_LEVELS = [0.05, 0.10, 0.20]


def consolidate_plant(pid):
    return PLANT_CONSOLIDATION.get(pid, pid)


def variable_width_band(seg_pts, A1, A2, s, q_star, n_samples=80):
    p0, p1 = seg_pts[0], seg_pts[1]
    L = np.linalg.norm(p1 - p0)
    if L < 1.0:
        return None
    ts = np.linspace(0.0, 1.0, n_samples)
    Q = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    v1 = A1[None, :] - Q
    v2 = A2[None, :] - Q
    d = np.linalg.norm(v1, axis=1)
    d = np.where(d < 1.0, 1.0, d)
    cos_theta = np.sum(v1 * v2, axis=1) / (d * np.linalg.norm(v2, axis=1))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    kappa = 2.0 * np.sin(theta / 2.0) / d
    kappa = np.maximum(kappa, 1e-7)
    t_star = (np.sqrt(2.0) * s / kappa) * norm.ppf(1.0 - q_star)
    tangent = np.diff(Q, axis=0)
    tangent = np.vstack([tangent, tangent[-1:]])
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tnorm = np.where(tnorm < 1e-9, 1.0, tnorm)
    tangent = tangent / tnorm
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    upper = Q + normal * t_star[:, None]
    lower = Q - normal * t_star[:, None]
    poly = Polygon(np.vstack([upper, lower[::-1]]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty else poly


def voronoi_pad(plant_pts, boundary):
    minx, miny, maxx, maxy = boundary.bounds
    span = max(maxx - minx, maxy - miny)
    far = 5 * span
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    pad = np.array([
        [cx - far, cy], [cx + far, cy],
        [cx, cy - far], [cx, cy + far],
        [cx - far, cy - far], [cx + far, cy + far],
        [cx - far, cy + far], [cx + far, cy - far],
    ])
    return np.vstack([plant_pts, pad])


def compute_bands(boundary, plant_pts, n_canonical, q_levels, s):
    all_pts = voronoi_pad(plant_pts, boundary)
    vor = Voronoi(all_pts)
    bands = {q: [] for q in q_levels}
    pts = vor.points
    for (i, j), ridge_v in zip(vor.ridge_points, vor.ridge_vertices):
        if i >= n_canonical or j >= n_canonical:
            continue
        if -1 in ridge_v or len(ridge_v) < 2:
            continue
        v0 = vor.vertices[ridge_v[0]]
        v1 = vor.vertices[ridge_v[-1]]
        seg = LineString([v0, v1])
        if not seg.intersects(boundary):
            continue
        seg_pts = np.array([v0, v1])
        for q in q_levels:
            band = variable_width_band(seg_pts, pts[i], pts[j], s, q)
            if band is None:
                continue
            band_clipped = band.intersection(boundary)
            if not band_clipped.is_empty:
                bands[q].append(band_clipped)
    return {q: unary_union(bands[q]).intersection(boundary) if bands[q] else None
            for q in q_levels}


def main():
    munis_shp = gpd.read_file(SHP / "municipios.shp")
    boundary = unary_union(munis_shp.geometry)

    plantas = gpd.read_file(SHP / "plantas.shp")
    plantas["canonical_id"] = plantas["Id"].apply(consolidate_plant)
    canonical = plantas[plantas["Id"] == plantas["canonical_id"]].reset_index(drop=True)
    plant_pts = np.array([(p.x, p.y) for p in canonical.geometry])

    df = pd.read_csv(DATA / "municipios_canonical.csv")
    df_mis = df[df["misallocated"] == 1].copy()
    print(f"Total municipalities: {len(df)}")
    print(f"Misallocated: {len(df_mis)}")
    print(f"Effective plants: {len(canonical)}")

    bands = compute_bands(boundary, plant_pts, len(canonical), RISK_LEVELS, S_REGIONAL)

    # Test each misallocated muni against each band
    print("\nCapture rates (cumulative):")
    print(f"  {'q*':>6} {'inside':>8} {'fraction':>10}")
    for q in [0.20, 0.10, 0.05]:
        band = bands[q]
        if band is None:
            continue
        inside = df_mis.apply(
            lambda r: band.contains(Point(r["utm_x"], r["utm_y"])), axis=1
        )
        n_in = int(inside.sum())
        print(f"  {q*100:>5.0f}% {n_in:>8d} {n_in/len(df_mis):>10.1%}")

    # Munis outside the loosest band (q*=5%)
    band_05 = bands[0.05]
    if band_05 is not None:
        df_mis["inside_5pct"] = df_mis.apply(
            lambda r: band_05.contains(Point(r["utm_x"], r["utm_y"])), axis=1
        )
        outside = df_mis[~df_mis["inside_5pct"]].copy()
        print(f"\n--- Misallocated munis OUTSIDE q*=5% safety band: {len(outside)} ---")
        outside = outside.sort_values("utm_y", ascending=False)
        for _, r in outside.iterrows():
            print(f"  {r['municipio']:35s}  voronoi=plant{int(r['voronoi_plant']):2d}  "
                  f"net=plant{int(r['network_plant']):2d}  "
                  f"x={r['utm_x']/1000:5.0f}km  y={r['utm_y']/1000:5.0f}km")

    # Also report the corollary: munis correctly assigned but inside 5% band
    df_ok = df[df["misallocated"] == 0].copy()
    if band_05 is not None:
        df_ok["inside_5pct"] = df_ok.apply(
            lambda r: band_05.contains(Point(r["utm_x"], r["utm_y"])), axis=1
        )
        n_ok_in = int(df_ok["inside_5pct"].sum())
        print(f"\nCorrectly assigned munis INSIDE q*=5% band: {n_ok_in} / {len(df_ok)} "
              f"({n_ok_in/len(df_ok):.1%})")
        print(f"  → these are ' false alarms' under the q*=5% threshold "
              f"(flagged as risky, but actually correctly assigned in the network)")


if __name__ == "__main__":
    main()
