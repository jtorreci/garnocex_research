# Voronoi Risk Toolbox — Install & Test Guide

Versión: 0.2.0 (post-audit, 2026-05-09)

Esta guía te permite instalar el plugin en QGIS, ejecutar los 5 algoritmos
con las capas de Extremadura y validar las salidas contra el dataset
canónico de la revisión.

---

## 1. Requisitos previos

| Componente | Versión mínima | Comprobación |
|---|---|---|
| QGIS | 3.22 LTR | `Help → About` |
| Python (en QGIS) | 3.9+ | viene con QGIS |
| QNEAT3 plugin | 1.0.6+ | Plugin Manager |
| numpy, scipy | viene con QGIS | — |

QNEAT3 se instala desde *Plugins → Manage and Install Plugins → All →
buscar "QNEAT3"*. Es obligatorio para los algoritmos 2 y 5.

---

## 2. Instalación del plugin

1. Localiza la carpeta de plugins de tu perfil QGIS:

   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

2. Copia la carpeta completa
   `paper1_geographical_analysis/Revision/qgis_plugin/voronoi_risk_toolbox/`
   dentro de esa ruta.

3. Reinicia QGIS.

4. *Plugins → Manage and Install Plugins → Installed → activar*
   "Voronoi Risk Toolbox".

5. Los algoritmos aparecen en *Processing Toolbox →
   "Voronoi Risk Analysis"* (5 entradas numeradas).

Si en algún momento modificas un .py del plugin, basta con
*Plugins → Plugin Reloader → Voronoi Risk Toolbox* (instala antes el
"Plugin Reloader" desde el repo experimental). No hace falta reiniciar QGIS.

---

## 3. Capas de prueba

Las 4 capas necesarias están en
`paper1_geographical_analysis/Revision/analyses/data/shp/`:

| Capa | Geometría | Notas |
|---|---|---|
| `municipios.shp` | Polygon | 388 polígonos, campo `NAMEUNIT` |
| `plantas.shp` | Point | 46 plantas, identificador en `id` o equivalente |
| `Extremadura.shp` | Polygon | boundary regional |
| `carreteras.shp` | LineString | red vial (input para QNEAT3) |

Cárgalas en QGIS con *Layer → Add Layer → Add Vector Layer*. CRS esperado:
EPSG:25830 (UTM ETRS89 30N) o cualquier CRS métrico.

Antes de correr nada, comprueba el campo de identificación de
`plantas.shp`: abre la tabla de atributos. Anota el nombre del campo de id
de planta (probablemente `id`, `Id`, o similar). Lo necesitarás como
parámetro.

---

## 4. Plan de pruebas (paso a paso)

> **Tip:** en cada algoritmo deja activado **Audit log (CSV, optional)** y
> apunta a un fichero como `C:\Users\Usuario\Desktop\test_alg1.csv`. Eso te
> dejará un CSV estructurado con cada fila procesada para diff vs.
> ground truth.

### 4.1. Algoritmo 1 — Voronoi Assignment

Sólo Euclídeo, no necesita QNEAT3. Es la prueba más rápida.

| Parámetro | Valor |
|---|---|
| Municipalities | `municipios` |
| Facilities | `plantas` |
| Boundary | `Extremadura` (opcional) |
| Facility identifier field | (campo id de la tabla de plantas) |
| Audit log | `test_alg1.csv` |
| Output | `Memory layer` o GPKG temporal |

**Resultado esperado:** layer con campos nuevos `voronoi_facility_id`,
`voronoi_pos_id`, `voronoi_distance_eu`, `voronoi_distance_eu_2nd`,
`voronoi_R`. 388 features (uno por muni).

**Validación:** abrir el CSV de audit y compararlo con
`analyses/data/municipios_canonical.csv`:

```python
import pandas as pd
plug = pd.read_csv("test_alg1.csv", comment="#")
canon = pd.read_csv("../paper1_geographical_analysis/Revision/analyses/data/municipios_canonical.csv")
# El plugin reporta voronoi_pos_id (0-based posicional). Tras emparejar
# por nombre, voronoi_pos_id+1 debería igualar voronoi_plant_raw del canónico.
```

---

### 4.2. Algoritmo 2 — Beta Calculator

Requiere QNEAT3. Tarda más (1-3 minutos para Extremadura completa).

| Parámetro | Valor |
|---|---|
| Municipalities | `municipios` |
| Facilities | `plantas` |
| Road network | `carreteras` |
| Facility identifier field | (id de plantas) |
| Default speed | `50` km/h |
| Audit log | `test_alg2.csv` |
| Output | `test_alg2_output.gpkg` |

**Resultado esperado:** layer con `beta_assigned`, `is_misallocated`,
`distance_saving_m`, etc. **88 misallocations** sin consolidación
(el plugin no consolida; eso lo hace el pipeline analítico aparte).

**Validación contra canónico:**
```python
plug = pd.read_csv("test_alg2.csv", comment="#")
canon = pd.read_csv(".../municipios_canonical.csv")
# Tras emparejar muni por nombre:
# - plug.beta_assigned ≈ canon.beta_assigned  (deberían coincidir hasta 4 decimales)
# - plug.is_misallocated.sum() = 88  (raw, sin consolidación)
# - sum(canon.misallocated)  = 61  (con consolidación, ≠ del plugin)
```

> **Nota sobre consolidación:** el plugin emite los números RAW (88 misallocations).
> Para alinear con la tabla canónica de 61, el usuario debería aplicar la consolidación
> de plantas (umbral 15 km) en post-procesado. Es una decisión consciente: el plugin
> es una herramienta general, no asume reglas específicas de Extremadura.

---

### 4.3. Algoritmo 3 — Safety Bands

| Parámetro | Valor |
|---|---|
| Municipalities | `municipios` |
| Facilities | `plantas` |
| Facility identifier field | (id de plantas) |
| Dispersion parameter s | `0.093` (Extremadura) |
| Risk tolerance q* | `0.10` |
| Audit log | `test_alg3.csv` |
| Output | `test_alg3_output.gpkg` |

**Resultado esperado:** campos `kappa`, `safety_band_width_t_star`,
`misallocation_prob`, `in_safety_band`. Con s=0.093 y q*=0.10, esperamos
del orden de 50-100 munis dentro de la safety band.

**Validación:** compare manualmente algunas filas:
- Para un muni con R ≈ 1 (cerca de frontera), `misallocation_prob`
  debe estar cerca de 0.5.
- Para un muni con R ≈ 2 (centro de celda), prob ≈ 0.0.

> **Cambio matemático respecto a v0.1.0:** la fórmula de κ ahora se evalúa
> en Q (proyección de P sobre la mediatriz) en lugar de en el midpoint de
> A1A2. Para munis cerca del midpoint los números son iguales; para munis
> alejados, κ cambia (correcto).

---

### 4.4. Algoritmo 4 — Misallocation Detector

Toma como entrada el output del algoritmo 2.

| Parámetro | Valor |
|---|---|
| Municipalities (with Beta Calculator output) | `test_alg2_output` |
| Audit log | `test_alg4.csv` |
| Output | `test_alg4_output.gpkg` |

**Resultado esperado:** la salida del 4 es esencialmente la del 2 con un
campo extra `distance_penalty_pct`. Útil sobre todo para visualizar
en mapas (categorizar por penalty %).

---

### 4.5. Algoritmo 5 — Anisotropy Map

Requiere QNEAT3. Más caro: 17,800 rutas (388 × 46) o más si usas munis
como destino. Puede tardar 5-10 minutos.

**Modo A — anisotropía respecto a plantas (recomendado para el paper):**

| Parámetro | Valor |
|---|---|
| Municipalities | `municipios` |
| Destinations | `plantas` |
| Road network | `carreteras` |
| Destination identifier field | (id de plantas) |
| Default speed | `50` |
| Audit log | `test_alg5_plants.csv` |

**Modo B — anisotropía respecto a otros municipios (para anisotropy paper futuro):**

| Parámetro | Valor |
|---|---|
| Municipalities | `municipios` |
| Destinations | `municipios` (same layer) |
| ... | ... |
| Audit log | `test_alg5_munis.csv` |

**Resultado esperado:** campos `beta_min`, `beta_max`, `anisotropy_alpha`,
`anisotropy_class`. Validar α contra
`analyses/data/complete_anisotropy_coefficients.csv` (modo B) o
`plant_anisotropy_coefficients_filtered.csv` (modo A, computado al revés).

---

## 5. Audit log: estructura

Cada algoritmo, cuando se le da un path en `Audit log`, escribe un CSV con:

```csv
# voronoi_risk_toolbox audit log
# algorithm: beta_calculator
# started: 2026-05-09T13:42:00+00:00
# n_municipalities: 388
# n_facilities: 46
# default_speed_kmh: 50.0
# facility_id_field: id
# od_rows: 17848
mun_pos_id,mun_qgs_id,voronoi_pos_id,voronoi_facility_id,...
0,0,17,18,...
1,1,13,14,...
...
# --- summary ---
# misallocations: 88
# missing_routes: 0
# rows_written: 388
# finished: 2026-05-09T13:43:30+00:00
```

Las líneas con `#` son comentarios. `pd.read_csv(path, comment="#")` los
salta automáticamente.

Identificadores presentes:
- `mun_pos_id` (0-based, orden de iteración) — sirve para casar con
  filas del CSV canónico si están en el mismo orden.
- `mun_qgs_id` — feature ID interno de QGIS (puede no ser estable).

Para comparar con `municipios_canonical.csv` lo cómodo es **emparejar por
nombre del muni** (campo del polígono, p.ej. `NAMEUNIT`), tras
normalización (acentos, formato "La X" vs "X. La"). Hay un helper en
`analyses/data/build_canonical_dataset.py` (`normalize_name`) que puedes
reusar.

---

## 6. Reporte de validación

Cuando hayas pasado los 5 algoritmos, el ideal es producir un mini reporte:

```python
# valida_plugin.py
import pandas as pd
from pathlib import Path

REV = Path(".../Revision/analyses/data")
canon = pd.read_csv(REV / "municipios_canonical.csv")

# Algoritmo 2
plug2 = pd.read_csv("test_alg2.csv", comment="#")
print("Misallocations plugin:", plug2.is_misallocated.sum())
print("Misallocations canon (raw):", (canon.voronoi_plant_raw != canon.network_plant_raw).sum())
print("Misallocations canon (consolidated):", canon.misallocated.sum())

# Diff de beta_assigned
m = plug2.merge(canon, left_on="voronoi_facility_id", right_on="voronoi_plant_raw")  # ajustar al field real
print("Mean abs diff beta:", (m.beta_assigned_x - m.beta_assigned_y).abs().mean())
```

Si la diferencia media de β es < 0.01 y los misallocations raw cuadran
en 88, **el plugin está validado**. Mándame el CSV de audit y el reporte
de diferencias y cierro la fase B.

---

## 7. Bugs conocidos / limitaciones

- El plugin no aplica consolidación de plantas. Eso es deliberado: la
  consolidación es específica del caso Extremadura (umbral 15 km). Para
  reproducir la cifra publicada (61 misallocations consolidadas), aplica
  la consolidación post-output con `analyses/data/build_canonical_dataset.py`.
- En el algoritmo 5 con muni-muni (388 × 388 = 150K rutas), QNEAT3 puede
  tardar 30+ minutos o agotar memoria. Para validación rápida, usa muni→
  planta primero (modo A).
- El campo `voronoi_facility_id` se serializa como string. Si en tu
  `plantas.shp` el campo id es entero, el cast a string puede no
  coincidir 100% con valores numéricos del CSV canónico. Compara
  por `voronoi_pos_id` (entero) en su lugar.

---

## 8. Reportar issues

Si algo falla:
1. Captura el log de QGIS Processing (panel de mensajes).
2. Adjunta el CSV de audit (las líneas `#` traen meta crucial).
3. Indica versión QGIS, QNEAT3 y SO.

Lo registramos en `qgis_plugin/PLUGIN_AUDIT.md` como nuevo bug.
