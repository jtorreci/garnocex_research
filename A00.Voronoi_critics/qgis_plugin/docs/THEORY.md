# Theoretical background

A short summary of the probabilistic framework implemented by this
toolbox. For the full derivation, formal proofs and empirical
validation, see TBD (2026),
*Geographical Analysis* (under review, manuscript 2303622).

---

## The problem

Given a set of facilities {A_1, ..., A_n} and a set of spatial units P,
the **Euclidean Voronoi assignment** sends each P to its
straight-line-nearest A_i. This is fast, deterministic and elegant. But
in real territories, what matters is the **road-network distance**
d_r(P, A_i), not the Euclidean d_e(P, A_i).

A unit P is **misallocated** by the Voronoi model if its road-nearest
facility differs from its Euclidean-nearest one. The framework asks:
**what is the probability of misallocation, and where on the territory
is it concentrated?**

---

## The model

For each pair P–A_i, define the **network scaling factor**

```
β(A_i, P) = d_r(A_i, P) / d_e(A_i, P)   ≥ 1
```

We model β as **lognormally distributed** across routes:

```
ln β ~ N(m, s²)
```

where s is a regional dispersion parameter capturing the heterogeneity
of the road network (terrain, density, infrastructure quality).

Empirically, lognormal beats Gamma, Weibull, Frechet and Generalized
Gamma in Wasserstein-1 distance to the empirical CDF (cf. paper §4.3).
Theoretically, the lognormal arises naturally from the multiplicative
central limit theorem applied to chained multiplicative impedance
factors in road networks.

---

## Theorem 1 — Misallocation probability

For a unit P sitting in the Voronoi cell of A_1, with R = d_e(P, A_2) /
d_e(P, A_1) the Euclidean distance ratio to its second-nearest facility:

```
ℙ[P misallocated | R]  =  Φ( −ln R / (√2 · s) )
```

where Φ is the standard normal CDF.

This is exact under the lognormal model with i.i.d. β across routes.

The plugin implements this in **Algorithm 3** as `misallocation_prob`.

### Interpretation

- On the Voronoi border (R = 1): probability is 1/2, by symmetry.
- Deep inside a cell (R → ∞): probability → 0.
- The curve is parameterized by s — wider networks (higher s) flatten
  the transition.

---

## Lemma 1 — Local geometry near the boundary

The R parameter is hard to compute analytically far from the boundary.
But near it, R simplifies.

Let Q be a point on the Voronoi border between A_1 and A_2, P = Q + h
with ‖h‖ small, t = signed normal distance from P to the border, d =
d_e(Q, A_1) = d_e(Q, A_2), and θ the angle subtended by A_1, A_2 at Q.
Then to first order:

```
ln R(P) ≈ κ · t,    with    κ = 2 sin(θ/2) / d
```

κ is the **local curvature parameter**. It depends on:

- d, the distance from the boundary point Q to the flanking facilities
- θ, the angle at which A_1 and A_2 are seen from Q

Geometric intuition: tightly packed facilities at wide angles give large
κ and steep probability gradients across the border; widely spaced
facilities at narrow angles give small κ and broad transition zones.

The plugin implements this in **Algorithm 3** as `kappa`. For the band
polygons it uses the simplified value at the midpoint of A_1 A_2,
where Q is at distance ‖A_1 − A_2‖ / 2 from each facility and θ = π,
giving:

```
κ_midpoint = 4 / ‖A_1 − A_2‖
```

This is an excellent approximation for visual purposes. The
per-municipality `kappa` is computed at the actual projection of P onto
the bisector and is fully accurate.

---

## Corollary 1 — Safety bands

Combining Theorem 1 and Lemma 1 gives the operational definition of the
**safety band** at risk tolerance q\*:

```
t*(q*)  =  (√2 · s / κ) · Φ⁻¹(1 − q*)
```

Any point closer than t\* to a Voronoi boundary has misallocation
probability greater than q\*. The strip {x : |t(x)| < t\*} is the
**safety band** — the territory where the Euclidean Voronoi assignment
is unreliable at level q\*.

The plugin computes:

- Per-municipality `safety_band_width_t_star` and `in_safety_band` flag
  in the per-muni sink of **Algorithm 3**.
- A polygon layer of safety bands at multiple risk levels in the
  optional `Safety band polygons` output of **Algorithm 3**.

---

## Anisotropy (Remark)

For each origin A_i, the directional spread of detour coefficients is
captured by the **anisotropy coefficient**:

```
α(A_i) = β_max(A_i) / β_min(A_i)
```

where the max and min are taken across all routes from A_i. α = 1 means
the route detour is identical in all directions (isotropic access);
α >> 1 means the network has strong directional bias.

The plugin computes α in **Algorithm 5**.

α is **not** a binary predictor of misallocation by itself: empirically
its AUC against the misallocation flag is around 0.5. What α does is
identify the **necessary condition** for misallocation:

- α = 1 ⇒ Voronoi assignment is robust (no misallocation).
- α > 1 ⇒ Voronoi misallocation is *possible*.

The quantitative prediction is given by Theorem 1, not by α directly.

---

## Calibration of s

The dispersion parameter s is the only free knob in the framework. It
must be calibrated for each region. Two routes:

### Direct calibration

Run **Algorithm 2** to produce per-route β values. Fit a lognormal
without forcing the location parameter:

```python
import numpy as np
import pandas as pd
log_beta = np.log(pd.read_csv("audit.csv", comment="#").beta_assigned.dropna())
m = log_beta.mean()
s = log_beta.std(ddof=1)
```

Use that s in **Algorithm 3**.

### Reference ranges from the literature

| Terrain | Typical s | Examples |
|---|---|---|
| Flat, dense network | 0.03 – 0.06 | Netherlands, Denmark, US Midwest |
| Urban dense | 0.05 – 0.08 | Metropolitan cores |
| Mixed topography (Iberian inland) | 0.08 – 0.12 | Extremadura, Castilla |
| Mountainous | 0.13 – 0.20 | Swiss Alps, Pyrenees |
| Coastal / fragmented | 0.15 – 0.25 | Norway, Greece |

For an unknown region, start with the literature range that best
matches the topography and refine empirically with a pilot run.

---

## Spatial dependence

Real β values are not i.i.d. across municipalities — they are
spatially autocorrelated (mountains cluster; flat plains cluster). The
paper handles this with a **log-CAR** model on `log(β)` plus a **BYM2**
decomposition of the variance into spatially structured and unstructured
components.

The plugin does **not** implement the spatial models — they are not
required for the per-feature predictions. They are run separately in
the analyses pipeline that accompanies the paper. The plugin's output
is sufficient to feed those analyses.

---

## Notes on the model assumptions

1. **β > 0 always.** The lognormal support is (0, ∞). The framework
   naturally respects β ≥ 1; violations in audit data (β < 1) are
   filtered out as routing artifacts.
2. **Two-facility approximation.** Theorem 1 is derived for the
   pairwise case. For three-facility intersections (Voronoi vertices),
   the formula is a conservative upper bound on the misallocation
   probability — using Theorem 1 over-estimates risk slightly, which
   is the safe direction for planning applications.
3. **Independence.** β values are treated as independent across routes
   in the per-feature predictor. Spatial correlation is real but does
   not bias the predictor toward optimism — see the paper's BYM2
   decomposition.
