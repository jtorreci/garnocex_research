# A1 — Spatial CAR/BYM (real, not simulated)

**Responde a:** Reviewer 1, comments 16 y 17.

## Problema actual

El análisis CAR/BYM publicado en el manuscrito está implementado en
`reproducibility/analysis/spatial_sensitivity_analysis.py`, pero **simula los
datos** (`simulate_municipal_data_with_spatial_structure`). Los números 95.1%
de acuerdo BYM, 85.9% CAR, RMSE 0.017 etc. **no provienen de un ajuste real**.

Si el reviewer mira el código, lo descubre. Hay que rehacerlo con los β
empíricos reales por municipio, ajustar log-CAR de verdad, y reportar los
números que salgan.

## Pipeline

1. **Cargar β empírico por municipio** desde `detailed_ratios_analysis_filtered.csv`.
   Agregamos a una observación por municipio: β medio o β a la planta
   Voronoi-asignada (decidir tras inspección).
2. **Construir matriz de pesos espaciales W** vía `libpysal`. Usamos
   k-nearest-neighbours (k=6) sobre coordenadas UTM. Alternativa
   distance-band si la red es muy heterogénea.
3. **Moran's I sobre β** (`esda.Moran`) — verificar el 0.373 reportado.
4. **Log-CAR bayesiano (PyMC)**:

   ```
   log(beta_i) | rest ~ N( mu + rho * sum_j w_ij (log beta_j - mu), tau^{-1} )
   ```

   Cadenas: 4 × 2000 muestras tras 1000 de warmup, NUTS, target_accept=0.95.

5. **Diagnostics**:
   - Trace plots, R-hat, ESS (vía `arviz`).
   - Residuos posteriores: `r_i = log beta_i - E[log beta_i | data]`.
   - **Moran's I sobre los residuos** — clave para responder al comment 17.
6. **Spatial LOOCV**: para cada municipio i, refit excluyendo i y vecinos
   Queen (o KNN), predecir log β_i, computar RMSE/MAE.
7. **CAR estándar (no log) como contraste**: mismo modelo sin transformación,
   contar fracción de muestras posteriores con β_i < 0 — argumento para
   adoptar log-CAR (comment 16).

## Plan B si PyMC se atasca

Caer a frecuentista vía `spreg.GM_Error_Het` (SAR error model). No es CAR
estricto, pero la inferencia espacial es defendible. Decisión tras 1 día de
debugging PyMC.

## Salidas

- `outputs/tables/A1_spatial_results.tex` — tabla principal para §4 del manuscrito
- `outputs/figures/A1_moran_residuals.pdf` — Moran scatter de β y de residuos
- `outputs/figures/A1_loocv_errors.pdf` — histograma errors LOOCV
- `outputs/figures/A1_carbym_maps.pdf` — mapas de β observado vs predicho vs residuo
- `A1_spatial_results.csv` — datos por municipio
- `A1_summary.json` — métricas resumen (Moran I antes/después, ρ posterior, RMSE LOOCV)
