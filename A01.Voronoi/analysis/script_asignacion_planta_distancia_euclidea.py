import pandas as pd
from qgis.core import QgsProject
import os
from collections import defaultdict

# =================================================================
# =================== CONFIGURACIÓN DE USUARIO ====================
# =================================================================

# 1. Nombres de las capas
PLANTAS_LAYER_NAME = "Plantas_Ext"
MUNICIPIOS_LAYER_NAME = "Municipios_Ext"

# 2. Nombres de los campos ID
PLANTAS_ID_FIELD = "Id"
MUNICIPIOS_ID_FIELD = "NOMBRE"

# 3. Ruta de guardado del nuevo CSV
OUTPUT_CSV_PATH = 'C:/Users/Usuario/Dropbox/Universidad/Investigacion/Convenios/Convenio_aridos/Plantas comunidades/Extremadura/scripts/tablas_distancias/distancias_euclideas.csv'

# =================================================================
# ================= FIN DE LA CONFIGURACIÓN =======================
# =================================================================

print("Iniciando script de cálculo de DISTANCIAS EUCLÍDEAS.")

# --- 1. Cargar capas ---
project = QgsProject.instance()
plantas_layer = project.mapLayersByName(PLANTAS_LAYER_NAME)[0]
municipios_layer = project.mapLayersByName(MUNICIPIOS_LAYER_NAME)[0]
print("Capas cargadas.")

# --- 1b. Generar IDs únicos para plantas con nombres duplicados ---
# (Mantenemos esta lógica para que los archivos sean consistentes)
print("Verificando nombres de plantas duplicados...")
contador_nombres = defaultdict(int)
ids_unicos_plantas = {} 

for feature in plantas_layer.getFeatures():
    nombre_original = feature[PLANTAS_ID_FIELD]
    feature_id = feature.id()
    
    contador_nombres[nombre_original] += 1
    
    if contador_nombres[nombre_original] > 1:
        nombre_unico = f"{nombre_original}_{contador_nombres[nombre_original]}"
        print (f"Encontrada planta repetida en {nombre_original}")
    else:
        nombre_unico = nombre_original
        
    ids_unicos_plantas[feature_id] = nombre_unico
print("IDs únicos para las plantas generados.")

# --- 2. Calcular las distancias euclídeas ---
all_results = []
num_plantas = plantas_layer.featureCount()
print(f"Procesando {num_plantas} plantas...")

# Iteramos por cada planta
for i, planta_feature in enumerate(plantas_layer.getFeatures()):
    planta_id = ids_unicos_plantas[planta_feature.id()]
    geom_planta = planta_feature.geometry()
    
    print(f"  Procesando planta {i+1}/{num_plantas}: {planta_id}...")
    
    # Y para cada planta, iteramos por todos los municipios
    for municipio_feature in municipios_layer.getFeatures():
        municipio_id = municipio_feature[MUNICIPIOS_ID_FIELD]
        geom_municipio = municipio_feature.geometry()
        
        # La función .distance() calcula la distancia euclídea en las unidades de la capa (metros)
        distancia = geom_planta.distance(geom_municipio)
        
        all_results.append({
            'origin_id': planta_id,
            'destination_id': municipio_id,
            'distance_m': distancia
            # No hay columna de coste, ya que no aplica aquí
        })

# --- 3. Guardar los resultados ---
if all_results:
    df_results = pd.DataFrame(all_results)
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"\n¡ÉXITO! Archivo de distancias euclídeas guardado en: {OUTPUT_CSV_PATH}")
else:
    print("\nError: No se obtuvieron resultados.")