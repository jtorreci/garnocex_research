# Revision Analyses — Geographical Analysis 2303622

Backend computacional de la revisión major. Los scripts aquí producen los
resultados reales que sustituyen los actualmente simulados en el manuscrito y
responden a los puntos 16, 17 y 18 del Reviewer 1.

## Estructura

```
analyses/
├── data/                  Datos de entrada (copiados de ../../codigo/)
├── A1_spatial_carbym/     Log-CAR bayesiano + Moran residuals + LOOCV espacial
├── A2_distributional/     Wasserstein + Anderson-Darling + ranking distribucional
├── A3_anisotropy/         Post-procesamiento del coeficiente alpha_i
└── outputs/
    ├── figures/           PDFs/PNGs para el manuscrito
    └── tables/            .tex tables para el manuscrito
```

## Setup

```powershell
# Crear entorno (Windows / conda)
conda env create -f environment.yml
conda activate voronoi-rev

# Alternativa: pip
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si `pymc` da problemas de instalación en Windows, ver Plan B en
`A1_spatial_carbym/README.md` (caída a `spreg.GM_Error_Het` frecuentista).

## Orden de ejecución

1. Copiar datos a `data/` (un script en `data/sync.py`).
2. Ejecutar `A1_spatial_carbym/run.py`.
3. Ejecutar `A2_distributional/run.py`.
4. Ejecutar `A3_anisotropy/run.py`.
5. Inspeccionar `outputs/`.

Las tres ramas A1/A2/A3 son independientes y pueden correr en paralelo.

## Mapeo a comments del Reviewer 1

| Análisis | Responde a |
|---|---|
| A1 — Moran sobre log-CAR residuals | Comment 17 (tautología) |
| A1 — Spatial LOOCV | Comment 17 (capacidad predictiva) |
| A1 — log-CAR vs CAR | Comment 16 (β < 0) |
| A2 — Wasserstein ranking | Comment 18 (KS inválido) |
| A2 — Anderson-Darling | Comment 18 |
| A2 — Predicted misallocations CI | Minor 9 (Weibull conservadurismo) |
| A3 — alpha_i map + table | Minor 2 (isotropic space, anisotropy Remark) |
