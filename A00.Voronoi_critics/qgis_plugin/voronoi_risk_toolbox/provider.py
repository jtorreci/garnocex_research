# -*- coding: utf-8 -*-
"""
Processing provider — groups all Voronoi Risk algorithms.
"""

from qgis.core import QgsProcessingProvider

from .algorithms.voronoi_assignment import VoronoiAssignmentAlgorithm
from .algorithms.beta_calculator import BetaCalculatorAlgorithm
from .algorithms.safety_bands import SafetyBandsAlgorithm
from .algorithms.misallocation_detector import MisallocationDetectorAlgorithm
from .algorithms.anisotropy_map import AnisotropyMapAlgorithm


class VoronoiRiskProvider(QgsProcessingProvider):

    def id(self):
        return "voronoi_risk"

    def name(self):
        return "Voronoi Risk Toolbox"

    def longName(self):
        return (
            "Voronoi Risk Toolbox — Probabilistic misallocation risk "
            "assessment for Voronoi tessellations under network metrics"
        )

    def icon(self):
        return QgsProcessingProvider.icon(self)

    def loadAlgorithms(self):
        self.addAlgorithm(VoronoiAssignmentAlgorithm())
        self.addAlgorithm(BetaCalculatorAlgorithm())
        self.addAlgorithm(SafetyBandsAlgorithm())
        self.addAlgorithm(MisallocationDetectorAlgorithm())
        self.addAlgorithm(AnisotropyMapAlgorithm())
