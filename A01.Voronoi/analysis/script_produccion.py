# -*- coding: utf-8 -*-
from qgis.core import *
from qgis.utils import iface
import math
import os

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
PLANTAS_LAYER_NAME = "Plantas"
MUNICIPIOS_LAYER_NAME = "Municipios_Ext"  # Capa de puntos
VORONOI_LAYER_NAME = "Voronoi_Plantas_Recortado"

OUTPUT_FOLDER = r"E:/Dropbox/Universidad/Investigacion/Convenios/Convenio_aridos/Plantas comunidades/Extremadura"
OUTPUT_LAYER_NAME = "Voronoi_Plantas_Resultados"

# Parámetros
DOTACION = 0.5  # t/hab/año
COSTE_TRANSPORTE_POR_TKM = 0.35  # €/t·km
COSTO_BASE_FIJO = 40000  # €/año
UMBRAL_ESCALA = 50000  # t/año

# --------------------------------------------------
# OBTENER CAPAS
# --------------------------------------------------
def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise Exception(f"❌ Capa no encontrada: '{name}'")
    return layers[0]

try:
    plantas_layer = get_layer(PLANTAS_LAYER_NAME)
    municipios_layer = get_layer(MUNICIPIOS_LAYER_NAME)
    voronoi_layer = get_layer(VORONOI_LAYER_NAME)
except Exception as e:
    print(e)
    raise

# Verificar que municipios es de puntos
if municipios_layer.geometryType() != QgsWkbTypes.PointGeometry:
    raise Exception("❌ La capa de municipios debe ser de puntos.")

# Verificar campo 'Nombre' en Voronoi
if voronoi_layer.fields().indexOf("Nombre") == -1:
    raise Exception("❌ El campo 'Nombre' no existe en la capa de Voronoi.")

# --------------------------------------------------
# FUNCIONES DE CÁLCULO
# --------------------------------------------------
def coste_fijo_total(produccion, costo_base=COSTO_BASE_FIJO, umbral=UMBRAL_ESCALA):
    if produccion <= 0:
        return 0.0
    if produccion < umbral:
        return costo_base
    else:
        log_ratio = math.log2(produccion / umbral)
        return costo_base * (log_ratio + 1)

# --------------------------------------------------
# AÑADIR CAMPOS Y REFRESCAR (con nombres ≤10 caracteres)
# --------------------------------------------------
voronoi_layer.startEditing()

fields_to_add = [
    ("Produccion", QVariant.Double),  # Producción anual (t)
    ("Transp_tkm", QVariant.Double),  # Transporte (t·km)
    ("Coste_Var", QVariant.Double),   # Coste variable (€/t)
    ("Coste_Fijo", QVariant.Double),  # Coste fijo (€/t)
    ("Coste_Tot", QVariant.Double)    # Coste total (€/t) → <=10 chars
]

provider = voronoi_layer.dataProvider()
existing_fields = [field.name() for field in provider.fields()]

for name, field_type in fields_to_add:
    if name not in existing_fields:
        print(f"🔍 Añadiendo campo: {name}")
        provider.addAttributes([QgsField(name, field_type, "double", 12, 2)])

# ✅ FORZAR REFRESCO DE CAMPOS
voronoi_layer.updateFields()

# Verificar índices
field_indices = {}
for name, _ in fields_to_add:
    idx = voronoi_layer.fields().indexOf(name)
    if idx == -1:
        raise Exception(f"❌ No se pudo crear el campo '{name}'")
    field_indices[name] = idx

print("✅ Todos los campos están disponibles.")

# --------------------------------------------------
# CÁLCULO POR POLÍGONO DE VORONOI
# --------------------------------------------------
produccion_por_planta = {}

for voronoi_feat in voronoi_layer.getFeatures():
    geom_v = voronoi_feat.geometry()
    nombre_planta = voronoi_feat["Nombre"]
    produccion_total = 0.0
    transporte_total = 0.0

    # Buscar geometría de la planta
    planta_geom = None
    for p_feat in plantas_layer.getFeatures():
        if p_feat["Nombre"] == nombre_planta:
            planta_geom = p_feat.geometry().asPoint()
            break
    if not planta_geom:
        print(f"⚠️  No se encontró planta para {nombre_planta}")
        continue

    # Seleccionar municipios (puntos) dentro del polígono
    for mun_feat in municipios_layer.getFeatures():
        geom_mun = mun_feat.geometry()
        if geom_mun and geom_v.contains(geom_mun):  # Punto dentro del polígono
            try:
                poblacion = mun_feat["HBT"]  # Campo de población
            except:
                continue

            produccion = poblacion * DOTACION
            punto_mun = geom_mun.asPoint()
            distancia_km = math.sqrt(
                (punto_mun.x() - planta_geom.x())**2 +
                (punto_mun.y() - planta_geom.y())**2
            ) / 1000.0  # de m a km
            transporte = produccion * distancia_km

            produccion_total += produccion
            transporte_total += transporte

    produccion_por_planta[nombre_planta] = produccion_total

    # Calcular costes
    if produccion_total > 0.1:
        coste_variable = (transporte_total * COSTE_TRANSPORTE_POR_TKM) / produccion_total
        cfijo_total = coste_fijo_total(produccion_total)
        coste_fijo = cfijo_total / produccion_total
        coste_total = coste_variable + coste_fijo
    else:
        coste_variable = coste_fijo = coste_total = 0.0

    # Asignar atributos
    voronoi_feat.setAttribute(field_indices["Produccion"], produccion_total)
    voronoi_feat.setAttribute(field_indices["Transp_tkm"], transporte_total)
    voronoi_feat.setAttribute(field_indices["Coste_Var"], coste_variable)
    voronoi_feat.setAttribute(field_indices["Coste_Fijo"], coste_fijo)
    voronoi_feat.setAttribute(field_indices["Coste_Tot"], coste_total)

    voronoi_layer.updateFeature(voronoi_feat)

voronoi_layer.commitChanges()
print("✅ Cálculos completados y guardados en la capa.")

# --------------------------------------------------
# GUARDAR EN DISCO
# --------------------------------------------------
def save_layer_to_disk(layer, folder, base_name, format_type="GPKG"):
    if format_type == "GPKG":
        path = os.path.join(folder, f"{base_name}.gpkg")
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
    else:
        path = os.path.join(folder, base_name)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "ESRI Shapefile"
        options.fileEncoding = "UTF-8"

    res = QgsVectorFileWriter.writeAsVectorFormatV3(layer, path, QgsProject.instance().transformContext(), options)
    if res[0] != QgsVectorFileWriter.NoError:
        raise Exception(f"❌ Error guardando {base_name}: {res[1]}")
    print(f"💾 Guardado: {path}")
    return path

save_layer_to_disk(voronoi_layer, OUTPUT_FOLDER, OUTPUT_LAYER_NAME, "GPKG")

# --------------------------------------------------
# AÑADIR AL PROYECTO
# --------------------------------------------------
QgsProject.instance().addMapLayer(voronoi_layer)
print(f"✨ Capa '{OUTPUT_LAYER_NAME}' añadida al proyecto.")

# --------------------------------------------------
# MOSTRAR RESUMEN
# --------------------------------------------------
print("\n" + "="*70)
print("📊 RESUMEN DE RESULTADOS")
print("="*70)
print(f"{'Planta':<20} {'Prod (t)':<12} {'C_fijo (€/t)':<14} {'C_tot (€/t)':<12}")
print("-" * 70)
for nombre, produccion in produccion_por_planta.items():
    if produccion > 0:
        cfijo_total = coste_fijo_total(produccion)
        cfijo_unit = cfijo_total / produccion
        print(f"{nombre:<20} {produccion:<12.1f} {cfijo_unit:<14.2f} {'?':<12}")
print("="*70)