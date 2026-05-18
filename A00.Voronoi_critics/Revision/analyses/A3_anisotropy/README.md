# A3 — Anisotropy as a Remark in Section 2

**Responde a:** Reviewer 1, minor comment 2 (define isotropic space).

## Estado actual

Los α_i ya están calculados en
`codigo/complete_anisotropy_coefficients.csv` para 388 municipios y
`codigo/plant_anisotropy_coefficients_filtered.csv` para 46 plantas.

Aquí solo postprocesamos: estadísticos, mapa coroplético, y test de
discriminación contra la lista de misallocations.

## Pipeline

1. Cargar `complete_anisotropy_coefficients.csv` (Municipality, MaxRatio,
   MinRatio, AnisotropyCoefficient, NumPlants).
2. Cargar `asignacion_municipios_euclidiana.csv` y
   `asignacion_municipios_real.csv`. Marcar misallocated = (Voronoi != Network).
3. Estadísticos: media, mediana, IQR, máx/mín de α.
4. Predictor binario: `α > umbral` ⇒ predict misallocated.
   - Calcular ROC y AUC.
   - Reportar el umbral óptimo (Youden) y la tasa de captura.
5. Tabla top-10 / bottom-10 municipios por α con su misallocation flag.
6. Mapa coroplético sobre Extremadura (requiere coordenadas o, si no hay
   polígonos, scatter sobre coordenadas UTM con tamaño/color por α).

## Encuadre en el manuscrito

En la response letter ya está el texto:

> "...we introduce this as an Observation with empirical evidence from our
> Extremadura data... A detailed treatment of anisotropy-based misallocation
> prediction will be presented in a forthcoming paper."

Queda añadir como Remark en §2: definición + tabla resumen + frase de cierre
con AUC. Ni un párrafo más — protege el paper futuro.

## Salidas

- `outputs/tables/A3_anisotropy_summary.tex`
- `outputs/figures/A3_anisotropy_map.pdf`
- `outputs/figures/A3_alpha_vs_misallocation_roc.pdf`
- `A3_anisotropy_results.csv`
- `A3_summary.json` — AUC, umbral óptimo, capture rate
