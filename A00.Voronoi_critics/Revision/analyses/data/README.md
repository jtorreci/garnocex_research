# Data inputs for revision analyses

Datos copiados/sincronizados desde `../../../codigo/` al arrancar el pipeline.
Mantenemos copia local para que las analyses sean reproducibles incluso si
`codigo/` evoluciona.

## Ficheros esperados

| Fichero | Origen | Uso |
|---|---|---|
| `detailed_ratios_analysis_filtered.csv` | `codigo/` | β empírico por par muni-planta (n=9112) |
| `complete_anisotropy_coefficients.csv` | `codigo/` | α_i por municipio (n=388) |
| `plant_anisotropy_coefficients_filtered.csv` | `codigo/` | α por planta (n=46) |
| `coordenadas_municipios.csv` | `codigo/` | UTM x, y por municipio |
| `coordenadas_plantas.csv` | `codigo/` | UTM x, y por planta |
| `asignacion_municipios_euclidiana.csv` | `codigo/` | Voronoi assignment ground truth |
| `asignacion_municipios_real.csv` | `codigo/` | Network assignment ground truth |

## Sincronizar

```powershell
python sync.py
```

`sync.py` copia los ficheros de `codigo/` con timestamp y verifica integridad
(SHA-256). Re-ejecutar tras cualquier actualización en `codigo/`.

## Coordenadas

Los CSV de coordenadas usan UTM ETRS89 huso 30N (EPSG:25830) o similar.
Verificar con `python -c "import pandas as pd; print(pd.read_csv('coordenadas_municipios.csv').head())"`.
