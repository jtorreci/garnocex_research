"""Regenerate the spatial-analysis and study-area figures for the final revision.

The script uses only persisted analysis outputs and authoritative municipal polygons:
- Figure 2 preserves a 1:1 UTM scale in both map panels.
- Figure 5 uses equal-width panels and reconstructs the Cáceres–Badajoz boundary
  from the union of municipality geometries by NUTS-3 province code.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import shapefile
from scipy.spatial import cKDTree, Voronoi
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "analyses" / "data"
PIPE = ROOT / "pipeline"
PAL = ["#1B6AA5", "#C1440E", "#6A3D9A", "#2E7D32"]
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d9d9d9"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def style(ax):
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def knn_weights(coords: np.ndarray, k: int = 6) -> np.ndarray:
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    W = np.zeros((len(coords), len(coords)))
    rows = np.repeat(np.arange(len(coords)), k)
    W[rows, idx[:, 1:].ravel()] = 1.0 / k
    return W


def set_map_scale(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def spatial_analysis() -> None:
    ref = json.loads((PIPE / "spatial_refit.json").read_text(encoding="utf-8"))
    arr = np.load(PIPE / "spatial_refit_residuals.npy")
    coords, log_beta = arr[:, :2], arr[:, 2]
    km = coords / 1000.0
    W = knn_weights(coords)
    centered = log_beta - log_beta.mean()
    lag = W @ centered
    pad_x = 0.035 * np.ptp(km[:, 0])
    pad_y = 0.035 * np.ptp(km[:, 1])
    xlim = (km[:, 0].min() - pad_x, km[:, 0].max() + pad_x)
    ylim = (km[:, 1].min() - pad_y, km[:, 1].max() + pad_y)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    ax = style(axes[0, 0])
    sc = ax.scatter(km[:, 0], km[:, 1], c=log_beta, cmap="viridis", s=11, zorder=2)
    fig.colorbar(sc, ax=ax, label=r"$\log\beta$", fraction=0.046, pad=0.04)
    ax.set_title("(A) spatial distribution")
    ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    set_map_scale(ax, xlim, ylim)

    ax = style(axes[0, 1])
    ax.scatter(centered, lag, s=9, color=PAL[0], alpha=0.7, zorder=2)
    slope = np.polyfit(centered, lag, 1)[0]
    xs = np.linspace(centered.min(), centered.max(), 10)
    ax.plot(xs, slope * xs, color=PAL[1], linewidth=1.6, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.5); ax.axvline(0, color=INK2, linewidth=0.5)
    ax.set_title(rf"(B) Moran scatter, $I$ = {ref['moran_log_beta']['I']:.3f}")
    ax.set_xlabel(r"$\log\beta$ (centred)"); ax.set_ylabel("spatial lag")

    ax = style(axes[1, 0])
    ax.hist(log_beta, bins=32, color=PAL[0], edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_title("(C) distribution")
    ax.set_xlabel(r"$\log\beta$"); ax.set_ylabel("municipalities")

    ax = style(axes[1, 1])
    q = np.digitize(log_beta, np.quantile(log_beta, [0.25, 0.5, 0.75]))
    for k in range(4):
        mask = q == k
        ax.scatter(km[mask, 0], km[mask, 1], s=11, color=PAL[k],
                   label=f"Q{k+1}", zorder=2)
    ax.set_title("(D) quartiles")
    ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    ax.legend(frameon=False, labelcolor=INK, ncol=2)
    set_map_scale(ax, xlim, ylim)

    fig.tight_layout()
    fig.savefig(FIG / "spatial_analysis_beta_coefficients.pdf")
    plt.close(fig)


def polygon_parts(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def draw_polygon(ax, geom, *, facecolor="none", edgecolor=INK, linewidth=0.8,
                 alpha=1.0, zorder=1):
    for poly in polygon_parts(geom):
        x, y = poly.exterior.xy
        ax.fill(np.asarray(x) / 1000, np.asarray(y) / 1000,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                alpha=alpha, zorder=zorder)


def draw_line(ax, geom, **kwargs):
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        x, y = geom.xy
        ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, **kwargs)
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            draw_line(ax, part, **kwargs)


def region_clip_patch(region, ax):
    verts, codes = [], []
    for poly in polygon_parts(region):
        xy = np.asarray(poly.exterior.coords) / 1000
        verts.extend(xy)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    patch = PathPatch(MplPath(np.asarray(verts), codes), transform=ax.transData)
    return patch


def load_provinces():
    reader = shapefile.Reader(str(DATA / "shp" / "municipios.shp"), encoding="latin1")
    fields = [f[0] for f in reader.fields[1:]]
    i_prov = fields.index("CODNUT3")
    groups = {"ES431": [], "ES432": []}  # Badajoz, Cáceres
    for sr in reader.iterShapeRecords():
        code = str(sr.record[i_prov]).strip()
        if code in groups:
            geom = shape(sr.shape.__geo_interface__)
            if not geom.is_valid:
                geom = geom.buffer(0)
            groups[code].append(geom)
    badajoz = unary_union(groups["ES431"])
    caceres = unary_union(groups["ES432"])
    region = unary_union([badajoz, caceres])
    # Cartographic simplification at 100 m preserves the authoritative municipal
    # union while avoiding multi-megabyte paths at manuscript scale.
    internal = badajoz.boundary.difference(region.boundary.buffer(500)).simplify(100)
    return (badajoz.simplify(100, preserve_topology=True),
            caceres.simplify(100, preserve_topology=True),
            region.simplify(100, preserve_topology=True),
            internal)


def load_points():
    with (ROOT / "codigo" / "coordenadas_municipios.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    muni = np.array([(float(r["X"]), float(r["Y"])) for r in rows if r.get("X") and r.get("Y")])
    with (PIPE / "od" / "plantas_roster.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    plants = np.array([(float(r["x"]), float(r["y"])) for r in rows])
    return muni, plants


def finite_voronoi_segments(points):
    vor = Voronoi(points)
    center = points.mean(axis=0)
    radius = np.ptp(points, axis=0).max() * 2
    for pids, vids in zip(vor.ridge_points, vor.ridge_vertices):
        vids = np.asarray(vids)
        if np.all(vids >= 0):
            yield vor.vertices[vids]
        else:
            v = vids[vids >= 0][0]
            tangent = points[pids[1]] - points[pids[0]]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = points[pids].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            yield np.array([vor.vertices[v], vor.vertices[v] + direction * radius])


def study_area() -> None:
    badajoz, caceres, region, shared = load_provinces()
    muni, plants = load_points()
    minx, miny, maxx, maxy = region.bounds
    xlim = (minx / 1000 - 5, maxx / 1000 + 5)
    ylim = (miny / 1000 - 5, maxy / 1000 + 5)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.9),
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    ax = axes[0]
    draw_polygon(ax, region, facecolor="#E8E8E8", edgecolor=INK, linewidth=1.0)
    # The municipal unions identify the provinces for label placement, but
    # their shared boundary is topologically fragmented in the source layer.
    # Do not draw a misleading or visually discontinuous provincial line.
    for geom, label in ((caceres, "CÁCERES"), (badajoz, "BADAJOZ")):
        pt = geom.representative_point()
        ax.text(pt.x / 1000, pt.y / 1000, label, ha="center", va="center",
                fontsize=8, color="#555555", fontstyle="italic", zorder=4)
    ax.set_title("(A) study area location")
    ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    set_map_scale(ax, xlim, ylim)

    ax = axes[1]
    clip = region_clip_patch(region, ax)
    for seg in finite_voronoi_segments(plants):
        line, = ax.plot(seg[:, 0] / 1000, seg[:, 1] / 1000,
                        color="#BBBBBB", linewidth=0.45, alpha=0.65, zorder=1)
        line.set_clip_path(clip)
    ax.scatter(muni[:, 0] / 1000, muni[:, 1] / 1000, c="#5B9BD5",
               s=7, alpha=0.65, zorder=3, label="Municipalities")
    ax.scatter(plants[:, 0] / 1000, plants[:, 1] / 1000, c="#2E7D32",
               s=28, marker="^", edgecolors="black", linewidths=0.35,
               zorder=5, label="CDW facilities")
    draw_polygon(ax, region, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.set_title("(B) municipalities and CDW facilities")
    ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    ax.legend(loc="lower right", frameon=False)
    set_map_scale(ax, xlim, ylim)

    ax = axes[2]
    zones = np.where(muni[:, 1] < 4_323_000, 0,
                     np.where(muni[:, 1] < 4_400_000, 1, 2))
    labels = ["Southern / plains proxy", "Central / piedmont proxy",
              "Northern / mountain proxy"]
    colors = ["#7CB342", "#E5A520", "#8B6914"]
    for k, (label, color) in enumerate(zip(labels, colors)):
        m = zones == k
        ax.scatter(muni[m, 0] / 1000, muni[m, 1] / 1000,
                   c=color, s=9, alpha=0.72, zorder=3,
                   label=f"{label} ({m.sum()})")
    ax.axhline(4323, color="#888888", linewidth=0.6, linestyle=":", zorder=2)
    ax.axhline(4400, color="#888888", linewidth=0.6, linestyle=":", zorder=2)
    draw_polygon(ax, region, edgecolor=INK, linewidth=0.8, zorder=5)
    ax.set_title("(C) north–south calibration strata (terrain proxy)")
    ax.set_xlabel("UTM E (km)"); ax.set_ylabel("UTM N (km)")
    ax.legend(loc="lower right", frameon=False)
    set_map_scale(ax, xlim, ylim)

    fig.tight_layout()
    fig.savefig(FIG / "study_area_map.pdf")
    plt.close(fig)


if __name__ == "__main__":
    spatial_analysis()
    study_area()
    print("wrote spatial_analysis_beta_coefficients.pdf")
    print("wrote study_area_map.pdf")
