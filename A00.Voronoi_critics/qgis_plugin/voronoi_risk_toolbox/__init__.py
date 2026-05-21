# -*- coding: utf-8 -*-
"""
Voronoi Risk Toolbox — QGIS Processing Provider Plugin

Implements the probabilistic framework for misallocation risk assessment
in Voronoi tessellations under network distance metrics.

Reference:
    TBD (2026) "A Probabilistic Framework for Misallocation
    Risk in Voronoi Tessellations", Geographical Analysis.
"""


def classFactory(iface):
    from .plugin import VoronoiRiskPlugin
    return VoronoiRiskPlugin(iface)
