# -*- coding: utf-8 -*-
"""
Plugin entry point — registers the Processing provider with QGIS.
"""

from qgis.core import QgsApplication
from .provider import VoronoiRiskProvider


class VoronoiRiskPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = VoronoiRiskProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
