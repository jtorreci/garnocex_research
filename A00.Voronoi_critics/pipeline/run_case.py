"""Recomputation pipeline for one network case.

Regenerates every empirical number of the manuscript from a pair of OD exports,
so that the two network definitions can be run through an identical pipeline:

    case "full"  -> RT_VIARIA complete (133,903 km), sensitivity case (a)
    case "paved" -> Red_Viaria / carreteras.shp (20,790 km), operational case (b)

Conventions preserved from the published analysis:
  * beta = total_cost / d_euclidean, where total_cost = entry + network + exit.
    With the access legs included, beta >= 1 holds by construction (the routed
    path is a planar curve joining the two points, so it cannot be shorter than
    the straight line). No artefact filtering is required.
  * s = sd(log beta) with no location constraint. This is the parameter that
    enters Theorem 1; forcing floc=1.0 inflates it roughly sixfold.
  * Facility consolidation: facilities within CONSOLIDATION_KM of each other are
    one logical facility. Applied to the misallocation count AND to the R values
    that feed the predictor (the published A2 script declared the rule but never
    applied it to R).

Usage:
    python run_case.py --case paved
    python run_case.py --case full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata as ud
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "analyses" / "data"
SHP = DATA / "shp"

CONSOLIDATION_KM = 15.0
N_BOOTSTRAP = 4000          # matches the "B = 4,000" claimed in the manuscript
N_MC = 80_000               # Monte Carlo draws for non-lognormal candidates
SEED = 42

# Each case names the OD exports it consumes. Fresh QNEAT3 runs should drop
# their exports here and be registered in this table.
CASES = {
    "paved": {
        "label": "Red_Viaria / carreteras.shp (paved, 20,790 km)",
        "network": "carreteras.shp",
        "od_plant_real": DATA / "D_real_plantas_clean.csv",
        "od_plant_eucl": DATA / "D_euclidea_plantas_clean.csv",
        "od_muni_real": DATA / "D_real_municipios_clean.csv",
        "od_muni_eucl": DATA / "D_euclidea_municipios_clean.csv",
    },
    "full": {
        "label": "RT_VIARIA complete (133,903 km)",
        "network": "network_full.gpkg",
        "od_plant_real": None,          # pending the fresh run
        "od_plant_eucl": None,
        "od_muni_real": ROOT / "codigo" / "Matriz_Municipios.csv",
        "od_muni_eucl": None,
    },
}


# --------------------------------------------------------------------------
# Name handling. Every join is done on a canonical key, never on raw names:
# the historical pipeline lost 82 municipalities to mojibake in exactly this
# step. The key folds accents, the postposed article and known corruptions.
# --------------------------------------------------------------------------
_MOJIBAKE = re.compile(r"[ÃÂ-¡¤]")


def demojibake(s: str) -> str:
    """Repair a cp850/latin-1 round trip, but only when the telltale marks are
    present, so that already-correct names are left untouched."""
    out = str(s)
    for _ in range(2):
        if not _MOJIBAKE.search(out):
            break
        best = out
        for enc_out in ("utf-8", "cp850"):
            try:
                cand = out.encode("latin-1").decode(enc_out)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "�" not in cand and len(_MOJIBAKE.findall(cand)) < len(_MOJIBAKE.findall(out)):
                best = cand
                break
        if best == out:
            break
        out = best
    return out


def key(s: str) -> str:
    s = ud.normalize("NFKD", demojibake(s))
    s = "".join(c for c in s if not ud.combining(c)).lower().strip()
    m = re.match(r"^(.*),\s*(la|el|las|los)$", s)
    if m:
        s = f"{m.group(2)} {m.group(1)}"
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def read_od(path: Path) -> pd.DataFrame:
    """Read a QNEAT3 OD export, tolerating both the ';'/',' and ','/'.' dialects."""
    head = path.open("rb").read(4096).decode("utf-8-sig", errors="replace")
    sep = ";" if head.count(";") > head.count(",") else ","
    dec = "," if sep == ";" else "."
    df = pd.read_csv(path, sep=sep, decimal=dec, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {"inputid": "origin_id", "targetid": "destination_id", "distance": "d_euclidean"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ("origin_id", "destination_id"):
        df[f"{c}_key"] = df[c].map(key)
    return df


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Geometry: coordinates, facility consolidation, R and kappa
# --------------------------------------------------------------------------
def load_points():
    import geopandas as gpd
    mun = pd.read_csv(ROOT / "codigo" / "coordenadas_municipios.csv", encoding="utf-8-sig")
    mun = mun.dropna(subset=["NOMBRE"])[["NOMBRE", "X", "Y"]]
    mun["name"] = mun.NOMBRE.map(demojibake)
    mun["key"] = mun.NOMBRE.map(key)
    # Canonical facility layer: Plantas_Ext.shp (GARNOCEX), all 46 with their own
    # coordinates. plantas.shp must not be used here: 30 of its records carry the host
    # municipality centroid instead of the facility, per its own `source` column.
    pl = gpd.read_file(SHP / "Plantas_Ext.shp")
    plants = pd.DataFrame({
        "Id": range(len(pl)),
        "mun": pl["MUNICIPIO"].map(demojibake),
        "X": pl.geometry.x.to_numpy(),
        "Y": pl.geometry.y.to_numpy(),
    })
    plants["key"] = plants["mun"].map(key)
    if mun.key.duplicated().any() or len(mun) != mun.key.nunique():
        raise SystemExit("municipality keys are not unique -- fix the normaliser first")
    return mun.reset_index(drop=True), plants.reset_index(drop=True)


def consolidate(plants: pd.DataFrame, km: float = CONSOLIDATION_KM) -> pd.DataFrame:
    """Single-link grouping of facilities closer than `km`: one logical facility."""
    xy = plants[["X", "Y"]].to_numpy()
    n = len(xy)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(xy[i] - xy[j])) <= km * 1000:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    plants = plants.copy()
    plants["group"] = [find(i) for i in range(n)]
    plants["group"] = plants.group.map({g: k for k, g in enumerate(sorted(set(plants.group)))})
    return plants


def euclid_to_groups(mun: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance from every municipality to every logical facility."""
    rows = []
    for m in mun.itertuples():
        d = np.hypot(plants.X.to_numpy() - m.X, plants.Y.to_numpy() - m.Y)
        per_group = pd.DataFrame({"group": plants.group.to_numpy(), "d": d}) \
            .groupby("group", as_index=False).d.min()
        per_group["muni_key"] = m.key
        rows.append(per_group)
    return pd.concat(rows, ignore_index=True)


def geometry_terms(mun, plants, eu_groups):
    """R = d2/d1 over logical facilities, and kappa = 2 sin(theta/2)/d (Lemma 1).

    Two conventions, deliberately different, because consolidation is well defined
    for one and not for the other:

      * R uses the distance to the *nearest physical plant* of each logical group.
        No representative point is needed, so consolidation introduces no arbitrariness.
      * kappa needs an actual bisector, so theta and the facility separation are taken
        from the two nearest *physical* plants of the two competing groups. Collapsing
        a group to its centroid would invent a facility that does not exist.
    """
    out = []
    xy = plants[["X", "Y"]].to_numpy()
    grp = plants.group.to_numpy()
    for m in mun.itertuples():
        g = eu_groups[eu_groups.muni_key == m.key].sort_values("d")
        if len(g) < 2 or g.iloc[0].d <= 0:
            continue
        g1, g2 = int(g.iloc[0].group), int(g.iloc[1].group)
        d1, d2 = float(g.iloc[0].d), float(g.iloc[1].d)
        p = np.array([m.X, m.Y])
        dist_all = np.hypot(xy[:, 0] - m.X, xy[:, 1] - m.Y)
        a1 = xy[np.where(grp == g1)[0][np.argmin(dist_all[grp == g1])]]
        a2 = xy[np.where(grp == g2)[0][np.argmin(dist_all[grp == g2])]]
        v1, v2 = a1 - p, a2 - p
        cos = np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        theta = float(np.arccos(cos))
        d_mid = float(np.linalg.norm(a1 - a2) / 2)
        kappa = 2 * np.sin(theta / 2) / d_mid if d_mid > 0 else np.nan
        out.append({"muni_key": m.key, "name": m.name, "nearest_group": g1,
                    "second_group": g2, "d1": d1, "d2": d2, "R": d2 / d1,
                    "theta": theta, "kappa": kappa})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Distributional fits
# --------------------------------------------------------------------------
FITTERS = {
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Inverse Weibull (Frechet)": stats.invweibull,
    "Generalized Gamma": stats.gengamma,
}


def fit_all(beta: np.ndarray) -> dict:
    fits = {}
    for name, dist in FITTERS.items():
        try:
            fits[name] = dist.fit(beta)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  ! {name} failed to fit: {exc}")
    return fits


def wasserstein1(emp, dist, params, n_grid=5000):
    qs = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)
    return float(np.mean(np.abs(np.quantile(emp, qs) - dist.ppf(qs, *params))))


def anderson_darling(emp, dist, params):
    x = np.sort(emp)
    n = len(x)
    F = np.clip(dist.cdf(x, *params), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n)


def predict_lognormal(s: float, R: np.ndarray) -> float:
    """Theorem 1, summed over municipalities."""
    return float(stats.norm.cdf(-np.log(R) / (np.sqrt(2.0) * s)).sum())


def predict_general(dist, params, R, rng, n_sim=N_MC) -> float:
    ratios = np.sort(dist.rvs(*params, size=n_sim, random_state=rng)
                     / dist.rvs(*params, size=n_sim, random_state=rng))
    idx = np.searchsorted(ratios, R, side="right")
    return float(np.sum((n_sim - idx) / n_sim))


def rank_distributions(beta, R, rng) -> pd.DataFrame:
    fits = fit_all(beta)
    rows = []
    for name, params in fits.items():
        dist = FITTERS[name]
        if name == "Lognormal":
            s_hat = float(np.log(beta).std(ddof=1))
            point = predict_lognormal(s_hat, R)
            # bootstrap on s only: cheap, so the full B = 4,000 is affordable
            s_boot = np.array([np.log(rng.choice(beta, len(beta), replace=True)).std(ddof=1)
                               for _ in range(N_BOOTSTRAP)])
            preds = np.array([predict_lognormal(s, R) for s in s_boot])
        else:
            s_hat = float(np.log(np.maximum(dist.rvs(*params, size=20000, random_state=rng),
                                            1e-9)).std(ddof=1))
            point = predict_general(dist, params, R, rng)
            preds = np.array([predict_general(dist, dist.fit(rng.choice(beta, len(beta),
                                                                       replace=True)),
                                              R, rng, n_sim=20000)
                              for _ in range(min(N_BOOTSTRAP, 200))])
        lo, hi = np.percentile(preds, [2.5, 97.5])
        rows.append({"distribution": name, "params": str(tuple(round(p, 5) for p in params)),
                     "W1": wasserstein1(beta, dist, params),
                     "A2": anderson_darling(beta, dist, params),
                     "KS": float(stats.kstest(beta, dist.cdf, args=params).statistic),
                     "s_used": s_hat, "pred": point, "ci_lo": float(lo), "ci_hi": float(hi),
                     "n_bootstrap": int(len(preds))})
    return pd.DataFrame(rows).sort_values("W1").reset_index(drop=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--q-star", type=float, default=0.10,
                    help="risk tolerance for the safety bands")
    args = ap.parse_args()
    cfg = CASES[args.case]
    out = ROOT / "pipeline" / "outputs" / args.case
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"=== case {args.case}: {cfg['label']} ===")
    missing = [k for k in ("od_plant_real", "od_plant_eucl") if cfg[k] is None]
    if missing:
        raise SystemExit(f"case '{args.case}' still needs: {', '.join(missing)}. "
                         "Register the fresh OD exports in CASES and re-run.")

    mun, plants = load_points()
    plants = consolidate(plants)
    n_groups = plants.group.nunique()
    print(f"facilities: {len(plants)} physical -> {n_groups} logical "
          f"({CONSOLIDATION_KM:.0f} km rule)")

    real = read_od(Path(cfg["od_plant_real"]))
    eucl = read_od(Path(cfg["od_plant_eucl"]))
    real["k"] = real.groupby(["origin_id_key", "destination_id_key"]).cumcount()
    eucl["k"] = eucl.groupby(["origin_id_key", "destination_id_key"]).cumcount()
    od = real.merge(eucl[["origin_id_key", "destination_id_key", "k", "d_euclidean"]],
                    on=["origin_id_key", "destination_id_key", "k"], how="inner")
    od = od[od.d_euclidean > 0].copy()
    cost = od.total_cost if "total_cost" in od else od.network_cost
    od["beta"] = cost / od.d_euclidean
    print(f"muni->facility pairs: {len(od):,} | origins {od.origin_id_key.nunique()} "
          f"| beta min={od.beta.min():.4f} | beta<1: {(od.beta < 1).sum()}")

    # predictor domain: the assigned (Euclidean-nearest logical) facility per municipality
    eu_groups = euclid_to_groups(mun, plants)
    geom = geometry_terms(mun, plants, eu_groups)
    host_key_by_group = plants.groupby("group").apply(
        lambda g: set(g["key"]), include_groups=False).to_dict()
    assigned = []
    for row in geom.itertuples():
        cand = od[(od.origin_id_key == row.muni_key)
                  & (od.destination_id_key.isin(host_key_by_group[row.nearest_group]))]
        if len(cand):
            assigned.append({"muni_key": row.muni_key, "beta_assigned": float(cand.beta.min()),
                             "d_network": float(cand.total_cost.min())
                             if "total_cost" in cand else float(cand.network_cost.min())})
    assigned = pd.DataFrame(assigned)
    dom = geom.merge(assigned, on="muni_key", how="inner")
    s_pred = float(np.log(dom.beta_assigned).std(ddof=1))
    m_pred = float(np.log(dom.beta_assigned).mean())
    print(f"predictor domain: n={len(dom)}  m={m_pred:.4f}  s={s_pred:.4f}")

    # misallocation: cheapest logical facility by network vs Euclidean-nearest
    net_best = od.groupby("origin_id_key").apply(
        lambda g: g.loc[g.beta.mul(g.d_euclidean).idxmin(), "destination_id_key"],
        include_groups=False)
    group_of_host = {k: int(g) for g, ks in host_key_by_group.items() for k in ks}
    mis = 0
    for row in dom.itertuples():
        nb = net_best.get(row.muni_key)
        if nb is not None and group_of_host.get(nb, -1) != row.nearest_group:
            mis += 1
    print(f"misallocated: {mis} / {len(dom)} = {100 * mis / len(dom):.2f} %")

    ranking = rank_distributions(dom.beta_assigned.to_numpy(), dom.R.to_numpy(), rng)
    print("\ndistributional ranking (predictor domain):")
    print(ranking[["distribution", "W1", "A2", "KS", "pred", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # safety bands at the requested risk tolerance
    z = stats.norm.ppf(1 - args.q_star)
    dom["t_star_m"] = np.sqrt(2) * s_pred / dom.kappa * z
    dom["t_signed_m"] = -np.log(dom.R) / dom.kappa
    dom["in_band"] = dom.t_signed_m.abs() < dom.t_star_m
    print(f"\nsafety bands at q*={args.q_star}: median t*={dom.t_star_m.median():,.0f} m "
          f"| municipalities inside: {int(dom.in_band.sum())}")

    dom.to_csv(out / "per_municipality.csv", index=False, encoding="utf-8")
    ranking.to_csv(out / "distributional_ranking.csv", index=False, encoding="utf-8")
    summary = {
        "case": args.case, "label": cfg["label"], "network": cfg["network"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "consolidation_km": CONSOLIDATION_KM,
        "facilities_physical": int(len(plants)), "facilities_logical": int(n_groups),
        "n_municipalities": int(len(dom)),
        "beta_min": float(od.beta.min()), "beta_below_one": int((od.beta < 1).sum()),
        "predictor_domain": {"m": m_pred, "s": s_pred},
        "misallocated": int(mis), "misallocated_pct": round(100 * mis / len(dom), 3),
        "lognormal_prediction": ranking.loc[ranking.distribution == "Lognormal",
                                            ["pred", "ci_lo", "ci_hi"]].round(2)
                                       .to_dict("records")[0],
        "q_star": args.q_star,
        "median_t_star_m": float(dom.t_star_m.median()),
        "inputs": {k: (str(v) if v else None) for k, v in cfg.items() if k.startswith("od_")},
        "input_sha256": {k: sha256(Path(v)) for k, v in cfg.items()
                         if k.startswith("od_") and v},
        "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"\nwrote {out}/per_municipality.csv, distributional_ranking.csv, summary.json")


if __name__ == "__main__":
    main()
