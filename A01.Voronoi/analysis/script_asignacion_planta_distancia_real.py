import pandas as pd
from qgis.core import QgsProject, QgsProcessingException
import os
from collections import defaultdict

# =================================================================
# =================== CONFIGURACIÓN DE USUARIO ====================
# =================================================================

# 1. Nombres de las capas
PLANTAS_LAYER_NAME = "Plantas_Ext"
MUNICIPIOS_LAYER_NAME = "Municipios_Ext"
RED_UNIFICADA_LAYER_NAME = "Red_Viaria" # Tu capa ya simplificada sin caminos

# 2. Nombres de los campos ID
PLANTAS_ID_FIELD = "MUNICIPIO"
MUNICIPIOS_ID_FIELD = "NOMBRE"

# 3. Nombre del campo de velocidad (debe existir en 'Red_Viaria')
SPEED_FIELD_NAME = "Velocidad" 

# 4. Velocidad por defecto (si algún tramo no tiene valor en el campo de velocidad)
DEFAULT_SPEED_KMH = 50

# 5. Ruta de guardado del CSV
OUTPUT_CSV_PATH = 'C:/Users/Usuario/Dropbox/Universidad/Investigacion/Convenios/Convenio_aridos/Plantas comunidades/Extremadura/scripts/distancias_reales_penalizadas_simplificado.csv'

# =================================================================
# ================= FIN DE LA CONFIGURACIÓN =======================
# =================================================================

print("Iniciando script final (v2) de análisis de red.")

# --- 1. Cargar capas y obtener ID de la capa de municipios ---
project = QgsProject.instance()
plantas_layer = project.mapLayersByName(PLANTAS_LAYER_NAME)[0]
municipios_layer = project.mapLayersByName(MUNICIPIOS_LAYER_NAME)[0]
municipios_layer_id = municipios_layer.id() # <-- Obtenemos el ID aquí
red_unificada = project.mapLayersByName(RED_UNIFICADA_LAYER_NAME)[0]
print("Capas cargadas.")

# --- 1b. Generar IDs únicos para plantas con nombres duplicados ---
print("Verificando nombres de plantas duplicados...")
contador_nombres = defaultdict(int)
ids_unicos_plantas = {} 

for feature in plantas_layer.getFeatures():
    nombre_original = feature[PLANTAS_ID_FIELD]
    feature_id = feature.id()
    
    contador_nombres[nombre_original] += 1
    
    if contador_nombres[nombre_original] > 1:
        nombre_unico = f"{nombre_original}_{contador_nombres[nombre_original]}"
    else:
        nombre_unico = nombre_original
        
    ids_unicos_plantas[feature_id] = nombre_unico
print("IDs únicos para las plantas generados en memoria.")

# --- 2. Calcular las rutas más rápidas ---
all_results = []
num_plantas = plantas_layer.featureCount()
print(f"Procesando {num_plantas} plantas.")

for i, planta_feature in enumerate(plantas_layer.getFeatures()):
    planta_id = ids_unicos_plantas[planta_feature.id()]
    print(f"  Procesando planta {i+1}/{num_plantas}: {planta_id}...")
    
    try:
        path_params = {
            'INPUT': red_unificada,
            'STRATEGY': 0, 
            'START_POINT': planta_feature.geometry().asPoint(),
            'END_POINTS': municipios_layer_id, # <-- CORRECCIÓN: Usamos el ID en lugar del objeto
            'SPEED_FIELD': SPEED_FIELD_NAME,
            'DEFAULT_SPEED': DEFAULT_SPEED_KMH,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        shortest_path_result = processing.run("native:shortestpathpointtolayer", path_params)
        
        output_layer = shortest_path_result['OUTPUT']
        for feature in output_layer.getFeatures():
            all_results.append({
                'origin_id': planta_id,
                'destination_id': feature[MUNICIPIOS_ID_FIELD],
                'total_cost_s': feature['cost'],
                'distance_m': feature.geometry().length()
            })
    except QgsProcessingException as e:
        print(f"    -> AVISO: No se pudo procesar la planta {planta_id}. Error: {e}")

# --- 3. Guardar los resultados ---
if all_results:
    df_results = pd.DataFrame(all_results)
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"\n¡ÉXITO! Archivo guardado en: {OUTPUT_CSV_PATH}")
else:
    print("\nError: No se obtuvieron resultados.")