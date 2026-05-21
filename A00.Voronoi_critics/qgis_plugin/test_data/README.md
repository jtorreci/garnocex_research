# Test data for plugin validation

Coloca aquí las capas que el plugin necesita para testearse. Si las tienes en
otro lugar, indica la ruta absoluta y arrancamos con esas.

## Capas requeridas

| Fichero | Geometría | Atributos clave |
|---|---|---|
| `municipalities.gpkg` | Punto o Polígono | ID estable (string) |
| `facilities.gpkg` | Punto | ID de planta (string) |
| `boundary.gpkg` (opcional) | Polígono | — |
| `network.gpkg` | Línea | (campos opcionales: speed, direction) |

## Set reducido recomendado para iteración rápida

Para ciclos rápidos de debug (1–2 minutos en lugar de 10):

- 5 municipios cualesquiera de Extremadura
- 3 plantas
- Red vial recortada al rectángulo envolvente +5 km

Con ese subset podemos depurar id/field mismatches sin esperar a la red
completa.

## Set completo

Las capas reales de Extremadura (388 munis, 46 plantas, red completa) para
validación final. Con tres plantas y cinco municipios ya hemos depurado los
bugs estructurales; con el set completo verificamos coincidencia con los CSV
ground truth (`detailed_ratios_analysis_filtered.csv` etc.).

## Coordinate Reference System

Idealmente UTM ETRS89 30N (EPSG:25830). Si están en EPSG:4326, el plugin
seguirá funcionando pero las distancias estarán en grados — antes del
testing, reproyectar a un CRS métrico.
