"""
Bounding-box sub-package.

Exports
-------
BoundingBoxInfo, BoundingBoxBuilder
    Generic CARLA actor bbox handling.
SignVisualProfile, TrafficInfrastructureBBox
    Specialised traffic-light / traffic-sign bbox handling.
"""

from .bbox_builder import BoundingBoxBuilder, BoundingBoxInfo
from .infra_bbox import SignVisualProfile, TrafficInfrastructureBBox

__all__ = [
    "BoundingBoxBuilder",
    "BoundingBoxInfo",
    "SignVisualProfile",
    "TrafficInfrastructureBBox",
]
