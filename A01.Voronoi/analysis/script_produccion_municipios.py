# -*- coding: utf-8 -*-
from qgis.core import *
from qgis.utils import iface
import math

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
MUNICIPIOS_LAYER_NAME = "Municipios_Ext"
VORONOI_LAYER_NAME = "Voronoi_Plantas_Recortado"
PLANTAS_LAYER_NAME = "Plantas"

# Parámetros del modelo
DOTACION = 0.5  # t/hab/año
COSTE_TRANSPORTE_POR_TKM = 0.35  # €/t·km
COSTO_BASE_FIJO = 40000  # €/año (1 operario + 1 molino)
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
    municipios_layer = get_layer(MUNICIPIOS_LAYER_NAME)
    voronoi_layer = get_layer(VORONOI_LAYER_NAME)
    plantas_layer = get_layer(PLANTAS_LAYER_NAME)
except Exception as e:
    print(e)
    raise

# --------------------------------------------------
# FUNCIONES DE CÁLCULO
# --------------------------------------------------
def coste_fijo_total(produccion, costo_base=COSTO_BASE_FIJO, umbral=UMBRAL_ESCALA):
    """
    Modelo logarítmico: C_fijo_total = C0 * (log2(T/T0) + 1) si T >= T0
    """
    if produccion <= 0:
        return 0.0
    if produccion < umbral:
        return costo_base
    else:
        log_ratio = math.log2(produccion / umbral)
        return costo_base * (log_ratio + 1)

def distance_km(p1, p2):
    """Distancia euclídea en km"""
    return math.sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2) / 1000.0

# --------------------------------------------------
# AÑADIR CAMPOS CON NOMBRES ≤ 10 CARACTERES
# --------------------------------------------------
municipios_layer.startEditing()

fields_to_add = [
    ("Prod", QVariant.Double),      # Producción anual (t)
    ("Dist_km", QVariant.Double),   # Distancia a planta (km)
    ("Tn_km", QVariant.Double),     # Transporte (t·km)
    ("Planta", QVariant.String),    # Planta asignada
    ("C_Trans", QVariant.Double),   # Coste transporte (€/t)
    ("C_Trat", QVariant.Double),    # Coste tratamiento (€/t)
    ("C_Tot", QVariant.Double)      # Coste total (€/t)
]

provider = municipios_layer.dataProvider()
existing_fields = [field.name() for field in provider.fields()]

for name, field_type in fields_to_add:
    if name not in existing_fields:
        print(f"🔍 Añadiendo campo: {name}")
        if field_type == QVariant.Double:
            provider.addAttributes([QgsField(name, field_type, "double", 12, 2)])
        else:
            provider.addAttributes([QgsField(name, field_type, "string", 100)])

# ✅ FORZAR REFRESCO DE CAMPOS
municipios_layer.updateFields()

# Verificar índices
field_indices = {}
for name, _ in fields_to_add:
    idx = municipios_layer.fields().indexOf(name)
    if idx == -1:
        raise Exception(f"❌ Error crítico: no se pudo crear el campo '{name}'")
    field_indices[name] = idx

print("✅ Todos los campos están disponibles.")

# --------------------------------------------------
# PASO 1: Calcular producción por planta (agregación por Voronoi)
# --------------------------------------------------
# Diccionario: Planta -> Producción total
produccion_por_planta = {}

for mun_feat in municipios_layer.getFeatures():
    geom_mun = mun_feat.geometry()
    if not geom_mun or geom_mun.isNull():
        continue

    # Buscar planta asignada mediante Voronoi
    voronoi_feat = None
    for v_feat in voronoi_layer.getFeatures():
        if v_feat.geometry().contains(geom_mun):
            voronoi_feat = v_feat
            break
    if not voronoi_feat:
        continue

    nombre_planta = voronoi_feat["Nombre"]
    try:
        poblacion = mun_feat["HBT"]
    except:
        poblacion = 0

    produccion = poblacion * DOTACION

    if nombre_planta not in produccion_por_planta:
        produccion_por_planta[nombre_planta] = 0.0
    produccion_por_planta[nombre_planta] += produccion

print("📊 Producción por planta calculada.")

# --------------------------------------------------
# PASO 2: Calcular coste fijo por planta (modelo logarítmico)
# --------------------------------------------------
coste_fijo_por_planta = {}  # C_Trat (€/t)

for planta, produccion in produccion_por_planta.items():
    cfijo_total = coste_fijo_total(produccion)
    if produccion > 0:
        coste_fijo_unit = cfijo_total / produccion
    else:
        coste_fijo_unit = 0.0
    coste_fijo_por_planta[planta] = coste_fijo_unit

print("✅ Costes de tratamiento por planta calculados.")

# --------------------------------------------------
# PASO 3: Asignar datos a cada municipio
# --------------------------------------------------
for mun_feat in municipios_layer.getFeatures():
    geom_mun = mun_feat.geometry()
    if not geom_mun or geom_mun.isNull():
        continue
    centroide = geom_mun.centroid().asPoint()

    # Buscar planta asignada
    voronoi_feat = None
    for v_feat in voronoi_layer.getFeatures():
        if v_feat.geometry().contains(geom_mun):
            voronoi_feat = v_feat
            break
    if not voronoi_feat:
        continue

    nombre_planta = voronoi_feat["Nombre"]

    # Buscar geometría de la planta
    planta_geom = None
    for p_feat in plantas_layer.getFeatures():
        if p_feat["Nombre"] == nombre_planta:
            planta_geom = p_feat.geometry().asPoint()
            break
    if not planta_geom:
        continue

    # Calcular distancia
    dist_km = distance_km(centroide, planta_geom)

    # Obtener población
    try:
        poblacion = mun_feat["HBT"]
    except:
        poblacion = 0

    produccion = poblacion * DOTACION
    transporte = produccion * dist_km

    # Costes por tonelada
    if produccion > 0:
        c_trans = (transporte * COSTE_TRANSPORTE_POR_TKM) / produccion
    else:
        c_trans = 0.0

    c_trat = coste_fijo_por_planta.get(nombre_planta, 0.0)
    c_tot = c_trans + c_trat

    # Asignar atributos
    mun_feat.setAttribute(field_indices["Prod"], produccion)
    mun_feat.setAttribute(field_indices["Dist_km"], dist_km)
    mun_feat.setAttribute(field_indices["Tn_km"], transporte)
    mun_feat.setAttribute(field_indices["Planta"], nombre_planta)
    mun_feat.setAttribute(field_indices["C_Trans"], c_trans)
    mun_feat.setAttribute(field_indices["C_Trat"], c_trat)
    mun_feat.setAttribute(field_indices["C_Tot"], c_tot)

    municipios_layer.updateFeature(mun_feat)

municipios_layer.commitChanges()
print("✅ Municipios actualizados con todos los datos: producción, transporte y costes por tonelada.")