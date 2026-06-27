"""
lib.utils
=========

Shared utility classes used across the PCCR library.

Modules
-------
attribute_utils
    :class:`AttributeClassifier` — nuScenes attribute inference from CARLA actors.
coordinate_utils
    :class:`CoordinateConverter` — CARLA ↔ nuScenes coordinate-frame transforms.
depth_utils
    :class:`DepthPointCloudAssembler` — depth-image → LiDAR-frame point cloud.
image_utils
    :class:`ImageProcessor` — image encoding, saving, and metadata helpers.
logging_utils
    :func:`setup_logging`, :func:`log_print` — consistent coloured logging.
visibility_utils
    :class:`VisibilityCalculator` — nuScenes annotation visibility level.
"""

from .attribute_utils import AttributeClassifier
from .coordinate_utils import CoordinateConverter
from .depth_utils import DepthPointCloudAssembler
from .image_utils import ImageProcessor
from .logging_utils import setup_logging, log_print
from .visibility_utils import VisibilityCalculator

__all__ = [
    "AttributeClassifier",
    "CoordinateConverter",
    "DepthPointCloudAssembler",
    "ImageProcessor",
    "setup_logging",
    "log_print",
    "VisibilityCalculator",
]
