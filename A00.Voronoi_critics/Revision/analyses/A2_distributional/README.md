# A2 — Distributional ranking with Wasserstein and Anderson-Darling

**Responde a:** Reviewer 1, comments 8, 9, 18.

## Problema actual

Las tablas 3 y 4 del manuscrito reportan p-valores KS para tamaños muestrales
de 383 y 9112. Como el reviewer indica, KS rechaza H₀ para cualquier
desviación con n grande. La tabla actual reporta "Poor" para todas las
distribuciones, pero usa el ranking de KS-statistic como decisión.

Sustituimos por Wasserstein-1 (W₁) sobre las CDFs, complementado con
Anderson-Darling (más sensible en colas).

## Pipeline

1. Cargar β filtrado (n=9112).
2. Ajustar por MLE: Lognormal, Gamma, Weibull, Fréchet (Inverse Weibull),
   Generalized Gamma. Usar `scipy.stats` y validar con bootstrap.
3. Para cada distribución computar:
   - W₁(F_emp, F_fit) vía `scipy.stats.wasserstein_distance` o `POT`.
   - Anderson-Darling statistic (`scipy.stats.anderson` o
     `scipy.stats.anderson_ksamp`).
   - KS-statistic + p-value (referencia, no decisión).
4. Para cada distribución, predicción puntual de número de misallocations
   (vía la fórmula del Theorem 1 con s estimado) y bootstrap CI.
5. Tabla final con: parámetros MLE, W₁, AD, KS, predicted misallocations
   (95% CI), observed=59.

## Argumentos esperados

- Lognormal: W₁ mínimo, AD aceptable, predice 52-65 (contiene 59).
- Weibull: subestima sistemáticamente (predice <59), peligroso en planning.
- Gamma: intermedio.
- Fréchet/Generalized Gamma: comprobar; pueden dar mejor cola.

Si algún competidor (e.g. Generalized Gamma) bate a Lognormal en W₁ y CI,
hay que decidir si cambiamos la distribución base del paper o defendemos
lognormal por simplicidad y CLT multiplicativo. Esa decisión queda para
A.4 después de ver resultados.

## Salidas

- `outputs/tables/A2_distributional_ranking.tex` — sustituye Tables 3 y 4
- `outputs/figures/A2_qq_plots.pdf`
- `outputs/figures/A2_predicted_vs_observed.pdf` — bar chart con CI por distribución
- `A2_distributional_results.csv`
