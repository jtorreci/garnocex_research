# Plugin audit — Voronoi Risk Toolbox

**Fecha:** 2026-05-09
**Estado de partida:** scaffolds funcionales con bugs concretos.
**Estado actual:** v0.2.0, los 3 bugs identificados están **corregidos**;
plugin listo para test. Ver `INSTALL_AND_TEST.md`.

## Cambios v0.1.0 → v0.2.0 (este documento)

- Nuevo módulo `algorithms/_audit.py`:
  - `AuditLogger` para log estructurado en CSV (uno por algoritmo).
  - `parse_od_layer` con detección automática de campos de QNEAT3
    (variantes `origin_id` / `origin_point_id` / `InputID`, etc.).
  - `collect_layer_points` con ids posicionales (0..n-1) que casan con
    los `origin_id`/`destination_id` de QNEAT3.
- B2.1 / B5.1 corregidos: ya no se usa `feat.id()` para indexar la OD.
- B2.2 corregido: nombres de campo OD detectados por `detect_od_fields`.
- B3.1 corregido: κ se evalúa en Q (proyección de P sobre la mediatriz),
  no en el midpoint del segmento A1A2.
- Todos los algoritmos exponen un parámetro `Audit log (CSV, optional)`.
- `Misallocation Detector` no duplica campos si ya existen en el input.

A continuación el detalle del audit original.

---

Resultado de leer los 5 archivos de `voronoi_risk_toolbox/algorithms/` y
revisar consistencia con la teoría del paper.

## Resumen

| Algoritmo | Estado | Bugs críticos |
|---|---|---|
| 1. Voronoi Assignment | OK | — |
| 2. Beta Calculator | **BUG** | id mismatch QGIS↔QNEAT3, field names |
| 3. Safety Bands | **BUG** | math de κ y t* incorrecta |
| 4. Misallocation Detector | OK | — (depende de #2 corregido) |
| 5. Anisotropy Map | **BUG** | mismos id/field issues que #2 |

---

## 1. Voronoi Assignment — OK

Cálculo Euclidiano simple. Sin dependencia de red. Salida correcta:
`voronoi_facility_id`, `voronoi_distance_eu`, `voronoi_distance_eu_2nd`,
`voronoi_R`. Listo para testear.

**Test sugerido:**
- Cargar capas de Extremadura.
- Ejecutar y comparar `voronoi_facility_id` con
  `codigo/asignacion_municipios_euclidiana.csv`. Deberían coincidir
  exactamente para los 383 municipios.

---

## 2. Beta Calculator — BUGS CRÍTICOS

### Bug B2.1 — id mismatch QGIS feature.id() vs QNEAT3 origin_id

**Líneas 238, 247, 252.** El código usa `feat.id()` (QGIS internal feature ID)
como clave para buscar en `net_distances`, pero QNEAT3 emite `origin_id`
basado en orden de iteración (0..n-1), no en el feature ID.

Resultado: salvo que las features estén indexadas exactamente 0..n-1, todos
los lookups devuelven `None` y β no se calcula.

**Fix:** enumerar municipios al iterar y guardar la posición:

```python
mun_pos_to_id = {}
for pos, feat in enumerate(source_mun.getFeatures()):
    mun_pos_to_id[pos] = feat.id()

# Y al consultar OD:
d_net_assigned = net_distances.get((pos, voronoi_idx), None)
```

### Bug B2.2 — Field names de QNEAT3

**Líneas 205-207.** El código asume campos `origin_id`, `destination_id`,
`total_cost`. La salida real de `qneat3:OdMatrixFromLayersAsTable` (versión
≥1.0.6) usa `origin_id`, `destination_id`, `total_cost` — confirmado.

Pero hay variantes históricas (`origin_point_id`, `destination_point_id`,
`network_cost`) en versiones anteriores. **Verificar en testing**: si falla,
probar nombres alternativos.

### Bug B2.3 — Falta `OUTPUT` con sink correcto

**Línea 220-223.** El sink se crea con `source_mun.wkbType()`, que está bien.
Pero el código no maneja el caso `dest_id` siendo `None` si el sink no se
crea correctamente. No es bloqueante; solo que el error sería críptico.

### Mejoras sugeridas

- Manejar el caso de municipios sin ruta a alguna planta (red desconectada):
  hoy guarda `None` en `beta_assigned`. Mejor explicitar campo `route_status`.
- Permitir DEFAULT_SPEED como parámetro (hoy hardcoded a 50 km/h).
- Reportar progreso vía `feedback.pushInfo` cada 50 municipios.

**Test sugerido tras fix:**
- Comparar `beta_assigned` con `codigo/detailed_ratios_analysis_filtered.csv`.
  Por municipio, debería coincidir el β a la planta Voronoi-asignada.

---

## 3. Safety Bands — BUG MATEMÁTICO

### Bug B3.1 — κ mal definida para P general

**Líneas 181-205.** El código aproxima κ usando solo el midpoint del segmento
A1A2:

```python
kappa = 2.0 / d_mid   # con d_mid = ||A1-A2||/2
```

Esto es correcto SOLO en el midpoint del segmento (donde Q coincide con el
midpoint del segmento). Para P arbitrario, Q es la proyección de P sobre la
mediatriz, y `d = d_e(Q, A_1)` depende de la posición de Q a lo largo de la
mediatriz, no es constante.

**Definición correcta (Lemma 1 del paper):**

Dado P en la celda Voronoi de A1, sea Q el punto de la frontera Voronoi más
cercano a P (proyección de P sobre la mediatriz de A1-A2 si los dos vecinos
son A1, A2). Entonces:

- `d = d_e(Q, A_1) = d_e(Q, A_2) = sqrt( (||A1-A2||/2)^2 + (proj_perp_dist)^2 )`
- `θ` = ángulo entre `(Q-A1)` y `(Q-A2)` desde Q
- `κ = 2 sin(θ/2) / d`

**Fix:** computar Q como proyección sobre la mediatriz, luego `d` y `θ`:

```python
mid = (A1 + A2) / 2.0
seg_unit = (A2 - A1) / np.linalg.norm(A2 - A1)
# Proyección de P sobre la mediatriz: P_perp = P - ((P - mid)·seg_unit) seg_unit
t_along_seg = np.dot(P - mid, seg_unit)
Q = P - t_along_seg * seg_unit
d = np.linalg.norm(Q - A1)
v1 = (A1 - Q) / d
v2 = (A2 - Q) / d
cos_theta = np.dot(v1, v2)
theta = np.arccos(np.clip(cos_theta, -1, 1))
kappa = 2.0 * np.sin(theta / 2.0) / d
```

### Bug B3.2 — Distancia P → frontera Voronoi

**Líneas 215-220.** El código proyecta sobre `seg_unit` (dirección A1→A2),
pero la frontera Voronoi es la mediatriz, **perpendicular** a A1A2. La
distancia perpendicular desde P a la mediatriz **es** `|t_along_seg|`
(el componente de `P - mid` a lo largo de seg_unit). Esto sí es correcto
geométricamente. La etiqueta del comentario es engañosa pero el cálculo
funciona.

### Bug B3.3 — Solo considera dos plantas más cercanas

Para fronteras con tres o más plantas convergentes (esquinas Voronoi),
la geometría cambia. El paper se restringe al caso de dos vecinos en el
Lemma; documentar la limitación en `shortHelpString`.

**Test sugerido tras fix:**
- Para cada municipio, comparar `misallocation_prob` con la probabilidad
  predicha por la fórmula Φ(-ln R / (√2 s)) con `voronoi_R` del Algorithm 1.
  Deberían coincidir.

---

## 4. Misallocation Detector — OK

Lógica correcta: lee `voronoi_facility_id` y `net_facility_id` del output
del #2, compara, computa savings. Sin dependencias de red.

**Test sugerido:**
- Tras correr #1, #2, #4: contar `is_misallocated`. Debe dar 59 (15.4%) si
  se usan las capas completas de Extremadura.

---

## 5. Anisotropy Map — BUGS CRÍTICOS

### Bug B5.1 — Mismo problema de id que B2.1

**Líneas 174, 207.** Mismo issue: `mun_id = feat.id()` no coincide con
`origin_id` de QNEAT3. El dict `od_data` se llena pero el lookup falla.

### Bug B5.2 — destination_id como índice

**Línea 209.** `dest_idx < len(fac_coords)` — el código asume que
`destination_id` es 0..n_fac-1. Igual que en #2, es plausible pero hay que
verificarlo en testing.

### Bug B5.3 — Coste computacional

OD completa muni × planta: 388 × 46 ≈ 17,800 rutas. Tolerable.
Pero si en el futuro el algoritmo se generaliza a muni × muni (388²=150K),
QNEAT3 tarda mucho. No es bloqueante para el test actual.

**Test sugerido tras fix:**
- Comparar `anisotropy_alpha` con `codigo/complete_anisotropy_coefficients.csv`.
  Ojo: este último usa muni-muni (NumPlants suspicious), revisar si el
  algoritmo del plugin computa muni-planta o muni-muni.

---

## Priorización de fixes

1. **B2.1, B2.2, B5.1, B5.2** (id/field mismatch QNEAT3) — bloquean el plugin entero.
2. **B3.1** (math de κ) — sin esto, los safety bands son incorrectos.
3. **B3.3, B5.3** (limitaciones documentables) — solo documentación.

## Plan de testing

1. Tú nos das una capa de prueba pequeña (5 municipios, 3 plantas, red
   recortada).
2. Yo pongo los fixes B2.1 y B2.2.
3. Tú ejecutas Algorithm 2; comparamos con CSV ground truth.
4. Iteramos hasta cuadrar.
5. Repetimos para cada algoritmo.

## Test data layers

Para que puedas testear, **necesito que pongas en `test_data/` capas de
QGIS (.gpkg o .shp)** con:

- `municipalities.gpkg` (puntos o polígonos, ID estable)
- `facilities.gpkg` (puntos, ID estable)
- `boundary.gpkg` (polígono Extremadura)
- `network.gpkg` (líneas, red vial)

O, si las tienes en otra ubicación, dime la ruta y las configuramos como
referencia. No las copio yo porque no sé dónde las tienes (no están en
el repo).
